#!/usr/bin/env python
"""
Phase C E0 + E1 (vision): ResNet-56 -> ResNet-20 / CIFAR-100.

E0: eff-rank R of projected student KD-grad, teacher-student relational CKA, r distribution (full pool).
E1 (GATE): space/criterion shoot-out over a full budget curve. Arms:
  RGA-KD (gate + D-optimal coverage on relational residual), GradSpan-KD (D-optimal on KD-grad),
  feature_coverage (D-optimal on penultimate features = late-feature-space selection),
  random, kdloss_topk, low_r (falsification sanity), nogate_highr (RGA without teacher gate).
Reports test acc mean±std (seed variance) AND end-to-end wall-clock = selection cost + training.

Reuses primitives from pilot_rga_diag_real.py. Decision gate per EXPERIMENT_PLAN.md / §7.3.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_rga_diag_real import (
    resnet20, CountSketch, relational_residual, effective_rank, select_coverage,
    load_cifar100, train_teacher, evaluate, get_feat_logits, kd_train_student,
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
    ap.add_argument('--budgets', default='500,1000,2000,4000')
    ap.add_argument('--out', default='pilot/E1_VISION_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device('cuda')
    print(f"[env] {torch.cuda.get_device_name(0)} torch={torch.__version__}")
    torch.manual_seed(0); np.random.seed(0)

    Xtr, ytr, Xte, yte = load_cifar100(device)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)
    rng = np.random.RandomState(0)
    pool_idx = rng.choice(Xtr.shape[0], args.N, replace=False)
    Xpool = Xtr[pool_idx]; ypool = ytr[pool_idx]
    n = args.N
    print(f"[data] pool={n} test={Xte.shape[0]}")

    teacher, te_acc = train_teacher(Xtr, ytr, Xte, yte, device, args.teacher_epochs, args.bs,
                                    ckpt=f'pilot/teacher_resnet56_c100_e{args.teacher_epochs}.pt')
    print(f"[teacher] test_acc={te_acc:.3f}")

    # ---------- selection-phase signatures (timed) ----------
    sel_t0 = time.time()
    fT_pool, zT_pool = get_feat_logits(teacher, Xpool, device)
    PT_pool = F.softmax(zT_pool / args.temp, dim=-1)
    teacher_correct = (zT_pool.argmax(1) == ypool).float()

    # warm up student
    student = resnet20().to(device)
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    from pilot_rga_diag_real import gpu_augment
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
    kd_loss_i = (PT_pool * (PT_pool.clamp_min(1e-12).log() - F.log_softmax(zS_pool / args.temp, -1))).sum(1)

    rho = (PS_pool - PT_pool)
    gS = torch.einsum('nc,nd->ncd', rho, fS_pool).reshape(n, -1)   # student KD grad (last layer)

    def proj(feat, seed):
        return CountSketch(feat.shape[1], args.proj_dim, seed, device)(feat)

    GS = proj(gS, 101); GT = proj(fT_pool, 202)
    anchor_idx = torch.tensor(rng.choice(n, min(args.anchors, n), replace=False), device=device)
    r, KSc, KTc, cka = relational_residual(GS, GT, anchor_idx)
    R_part, R_energy = effective_rank(GS)
    rel_resid_feat = GT - GS
    GfS = proj(fS_pool, 303)  # projected student features for feature_coverage arm
    sel_setup_sec = time.time() - sel_t0
    print(f"[E0] cka={cka:.3f} r={float(r.mean()):.3f}±{float(r.std()):.3f} "
          f"corr(r,kdloss)={float(np.corrcoef(r.cpu(), kd_loss_i.cpu())[0,1]):.3f} "
          f"R(energy90)={R_energy} R(part)={R_part:.1f}  setup={sel_setup_sec:.1f}s")

    e0 = {'cka': cka, 'r_mean': float(r.mean()), 'r_std': float(r.std()),
          'eff_rank_energy90': R_energy, 'eff_rank_participation': R_part,
          'teacher_test_acc': te_acc, 'teacher_pool_correct': float(teacher_correct.mean()),
          'student_warm_acc': float((zS_pool.argmax(1) == ypool).float().mean()),
          'selection_setup_sec': sel_setup_sec}

    # ---------- E1: arms ----------
    gate = teacher_correct > 0.5
    idx_all = torch.arange(n, device=device)

    def coreset(method, budget, seed_rng):
        budget = min(budget, n - 1)
        ts = time.time()
        if method == 'random':
            sel = seed_rng.choice(n, budget, replace=False)
        elif method == 'kdloss_topk':
            sel = torch.topk(kd_loss_i, budget).indices.cpu().numpy()
        elif method == 'GradSpan_KD':
            sel = idx_all[select_coverage(GS, budget)].cpu().numpy()
        elif method == 'feature_coverage':
            sel = idx_all[select_coverage(GfS, budget)].cpu().numpy()
        elif method == 'low_r':
            cand = idx_all[gate]; rc = r[gate]
            sel = cand[torch.argsort(rc)[:budget]].cpu().numpy()
        elif method in ('RGA_KD', 'nogate_highr'):
            if method == 'RGA_KD':
                cand = idx_all[gate]; rc = r[gate]
            else:
                cand = idx_all; rc = r
            poolk = min(len(cand), max(budget * 3, budget + 1))
            top = cand[torch.argsort(rc, descending=True)[:poolk]]
            cov = select_coverage(rel_resid_feat[top], min(budget, len(top)))
            sel = top[torch.tensor(cov, device=device)].cpu().numpy()
        sel = torch.tensor(np.array(sel).astype(int)[:budget], device=device)
        return sel, time.time() - ts

    budgets = [int(b) for b in args.budgets.split(',')]
    arms = ['RGA_KD', 'GradSpan_KD', 'feature_coverage', 'random', 'kdloss_topk', 'low_r', 'nogate_highr']
    results = {}
    for b in budgets:
        results[str(b)] = {}
        for arm in arms:
            accs, sel_secs, train_secs = [], [], []
            for s in range(args.seeds):
                srng = np.random.RandomState(1000 + s)
                sel, sel_sec = coreset(arm, b, srng)
                tt = time.time()
                acc = kd_train_student(PT_pool, Xpool[sel], PT_pool[sel], Xte, yte,
                                       args.temp, 1000 + s, device, args.retrain_epochs, args.bs)
                accs.append(acc); sel_secs.append(sel_sec); train_secs.append(time.time() - tt)
            accs = np.array(accs)
            # end-to-end wall-clock per run = selection setup (shared) + per-arm selection + training
            e2e = sel_setup_sec + float(np.mean(sel_secs)) + float(np.mean(train_secs))
            results[str(b)][arm] = {
                'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()), 'n': args.seeds,
                'budget': b, 'sel_sec': float(np.mean(sel_secs)),
                'train_sec': float(np.mean(train_secs)),
                'e2e_wallclock_sec': e2e,  # incl shared selection setup (eNTK/relational/teacher-infer)
            }
            print(f"  [b{b}] {arm:17s} acc={accs.mean():.4f}±{accs.std():.4f}  "
                  f"sel={np.mean(sel_secs):.2f}s train={np.mean(train_secs):.1f}s e2e={e2e:.1f}s")

    out = {
        'setup': {'dataset': 'CIFAR-100', 'pair': 'ResNet-56->ResNet-20', 'grad_proxy': 'last-layer',
                  'N': n, 'n_test': int(Xte.shape[0]), 'proj_dim': args.proj_dim,
                  'anchors': int(anchor_idx.shape[0]), 'temp': args.temp, 'seeds': args.seeds,
                  'retrain_epochs': args.retrain_epochs, 'budgets': budgets},
        'E0': e0, 'E1': results, 'total_wall_clock_sec': time.time() - t0,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
