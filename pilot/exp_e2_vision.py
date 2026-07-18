#!/usr/bin/env python
"""
Phase C E2 (vision): native (non-strawman) baselines vs RGA-KD.
ResNet-56 -> ResNet-20 / CIFAR-100, full budget curve.

Baselines added beyond E1:
  EL2N      = ||softmax(z_S) - onehot(y)||_2  (top-k)            [Paul et al. 2021]
  GraNd     = ||g_S(x)|| KD-gradient norm     (top-k)            [Paul et al. 2021]
  TAGCOS    = k-means on projected KD grads, nearest-to-centroid medoids   [2024]
  GRAFT     = feature MaxVol  == feature_coverage (D-optimal on penult. feats)
  DPP       = greedy log-det on projected KD grads == GradSpan_KD coverage
Plus RGA_KD, random for anchor.

Decision gate §7.4: RGA-KD must beat GRAFT & TAGCOS at aggressive budget.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_rga_diag_real import (
    resnet20, CountSketch, relational_residual, effective_rank, select_coverage,
    load_cifar100, train_teacher, get_feat_logits, kd_train_student, gpu_augment,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=10000)
    ap.add_argument('--proj_dim', type=int, default=2048)
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=4.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--teacher_epochs', type=int, default=50)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--retrain_epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--budgets', default='1000,2000,4000')
    ap.add_argument('--out', default='pilot/E2_VISION_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time(); device = torch.device('cuda')
    print(f"[env] {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0); np.random.seed(0)

    Xtr, ytr, Xte, yte = load_cifar100(device)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    rng = np.random.RandomState(0)
    pool_idx = rng.choice(Xtr.shape[0], args.N, replace=False)
    Xpool, ypool = Xtr[pool_idx], ytr[pool_idx]; n = args.N
    print(f"[data] pool={n} test={Xte.shape[0]}")

    teacher, te_acc = train_teacher(Xtr, ytr, Xte, yte, device, args.teacher_epochs, args.bs,
                                    ckpt=f'pilot/teacher_resnet56_c100_e{args.teacher_epochs}.pt')
    fT_pool, zT_pool = get_feat_logits(teacher, Xpool, device)
    PT_pool = F.softmax(zT_pool / args.temp, dim=-1)
    teacher_correct = (zT_pool.argmax(1) == ypool).float()

    student = resnet20().to(device)
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    for ep in range(args.warmup_epochs):
        student.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            logp = F.log_softmax(student(gpu_augment(Xpool[idx])) / args.temp, dim=-1)
            loss = (args.temp ** 2) * F.kl_div(logp, PT_pool[idx], reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
    student.eval()
    fS_pool, zS_pool = get_feat_logits(student, Xpool, device)
    PS_pool = F.softmax(zS_pool / args.temp, dim=-1)

    rho = (PS_pool - PT_pool)
    gS = torch.einsum('nc,nd->ncd', rho, fS_pool).reshape(n, -1)
    grand = gS.norm(dim=1)                                  # GraNd (KD-grad norm)
    onehot = F.one_hot(ypool, num_classes=zS_pool.shape[1]).float()
    el2n = (F.softmax(zS_pool, dim=-1) - onehot).norm(dim=1)  # EL2N

    proj = lambda f, s: CountSketch(f.shape[1], args.proj_dim, s, device)(f)
    GS = proj(gS, 101); GT = proj(fT_pool, 202); GfS = proj(fS_pool, 303)
    anchor_idx = torch.tensor(rng.choice(n, args.anchors, replace=False), device=device)
    r, _, _, cka = relational_residual(GS, GT, anchor_idx)
    _, R_energy = effective_rank(GS)
    rel_resid_feat = GT - GS
    gate = teacher_correct > 0.5
    idx_all = torch.arange(n, device=device)
    GS_np = GS.cpu().numpy()
    print(f"[E0] R(energy90)={R_energy} cka={cka:.3f} r={float(r.mean()):.3f}")

    def coreset(method, budget, srng):
        budget = min(budget, n - 1)
        if method == 'random':
            sel = srng.choice(n, budget, replace=False)
        elif method == 'EL2N':
            sel = torch.topk(el2n, budget).indices.cpu().numpy()
        elif method == 'GraNd':
            sel = torch.topk(grand, budget).indices.cpu().numpy()
        elif method == 'TAGCOS':
            km = MiniBatchKMeans(n_clusters=budget, random_state=int(srng.randint(1 << 30)),
                                 batch_size=2048, n_init=3, max_iter=100).fit(GS_np)
            cen = torch.tensor(km.cluster_centers_, device=device, dtype=GS.dtype)
            # nearest pool sample to each centroid
            d = torch.cdist(cen, GS)  # (budget, n)
            sel = d.argmin(dim=1).cpu().numpy()
            sel = np.unique(sel)
            if len(sel) < budget:  # fill duplicates' slots with random unused
                rest = np.setdiff1d(np.arange(n), sel)
                srng.shuffle(rest); sel = np.concatenate([sel, rest[:budget - len(sel)]])
        elif method == 'GRAFT':       # feature MaxVol
            sel = idx_all[select_coverage(GfS, budget)].cpu().numpy()
        elif method == 'DPP':         # = GradSpan-KD coverage on KD grads
            sel = idx_all[select_coverage(GS, budget)].cpu().numpy()
        elif method == 'RGA_KD':
            cand = idx_all[gate]; rc = r[gate]
            poolk = min(len(cand), max(budget * 3, budget + 1))
            top = cand[torch.argsort(rc, descending=True)[:poolk]]
            cov = select_coverage(rel_resid_feat[top], min(budget, len(top)))
            sel = top[torch.tensor(cov, device=device)].cpu().numpy()
        return torch.tensor(np.array(sel).astype(int)[:budget], device=device)

    budgets = [int(b) for b in args.budgets.split(',')]
    arms = ['RGA_KD', 'GRAFT', 'DPP', 'TAGCOS', 'EL2N', 'GraNd', 'random']
    results = {}
    for b in budgets:
        results[str(b)] = {}
        for arm in arms:
            accs = []
            for s in range(args.seeds):
                srng = np.random.RandomState(2000 + s)
                sel = coreset(arm, b, srng)
                acc = kd_train_student(PT_pool, Xpool[sel], PT_pool[sel], Xte, yte,
                                       args.temp, 2000 + s, device, args.retrain_epochs, args.bs)
                accs.append(acc)
            accs = np.array(accs)
            results[str(b)][arm] = {'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
                                    'n': args.seeds, 'budget': b}
            print(f"  [b{b}] {arm:10s} acc={accs.mean():.4f}±{accs.std():.4f}")

    out = {'setup': {'dataset': 'CIFAR-100', 'pair': 'ResNet-56->ResNet-20', 'N': n,
                     'seeds': args.seeds, 'budgets': budgets, 'teacher_test_acc': te_acc,
                     'eff_rank_energy90': R_energy},
           'E2': results, 'wall_clock_sec': time.time() - t0}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
