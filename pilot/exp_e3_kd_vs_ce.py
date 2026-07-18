#!/usr/bin/env python
"""
Phase C E3 (vision): is RGA's advantage KD-special?
ResNet-56 -> ResNet-20 / CIFAR-100.

Same coreset selection methods, then train the student two ways:
  (KD) distill from teacher soft labels (tau^2 KL)
  (CE) plain cross-entropy on TRUE labels (no teacher)
Compare RGA-KD's advantage over loss/random selection under KD vs CE.
Decision §7.5: if KD and CE show the same relative advantage, drop the "KD-special" claim.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_rga_diag_real import (
    resnet20, CountSketch, relational_residual, effective_rank, select_coverage,
    load_cifar100, train_teacher, get_feat_logits, evaluate, gpu_augment,
)


def train_student(mode, coreset_X, coreset_PT, coreset_y, Xte, yte, T, seed, device, epochs, bs):
    torch.manual_seed(seed); np.random.seed(seed)
    s = resnet20(100).to(device)
    opt = torch.optim.SGD(s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = coreset_X.shape[0]; bs = min(bs, n)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = gpu_augment(coreset_X[idx])
            if mode == 'KD':
                logp = F.log_softmax(s(xb) / T, dim=-1)
                loss = (T * T) * F.kl_div(logp, coreset_PT[idx], reduction='batchmean')
            else:  # CE on true labels
                loss = F.cross_entropy(s(xb), coreset_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return evaluate(s, Xte, yte, device)


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
    ap.add_argument('--out', default='pilot/E3_KDvsCE_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time(); device = torch.device('cuda')
    print(f"[env] {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0); np.random.seed(0)
    Xtr, ytr, Xte, yte = load_cifar100(device)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    rng = np.random.RandomState(0)
    pool_idx = rng.choice(Xtr.shape[0], args.N, replace=False)
    Xpool, ypool = Xtr[pool_idx], ytr[pool_idx]; n = args.N
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
    onehot = F.one_hot(ypool, 100).float()
    el2n = (F.softmax(zS_pool, -1) - onehot).norm(dim=1)
    rho = (PS_pool - PT_pool)
    gS = torch.einsum('nc,nd->ncd', rho, fS_pool).reshape(n, -1)
    proj = lambda f, s: CountSketch(f.shape[1], args.proj_dim, s, device)(f)
    GS = proj(gS, 101); GT = proj(fT_pool, 202)
    anchor_idx = torch.tensor(rng.choice(n, args.anchors, replace=False), device=device)
    r, _, _, _ = relational_residual(GS, GT, anchor_idx)
    rel_resid_feat = GT - GS
    gate = teacher_correct > 0.5; idx_all = torch.arange(n, device=device)

    def coreset(method, budget, srng):
        if method == 'random':
            return torch.tensor(srng.choice(n, budget, replace=False), device=device)
        if method == 'EL2N':
            return torch.topk(el2n, budget).indices
        # RGA_KD
        cand = idx_all[gate]; rc = r[gate]
        poolk = min(len(cand), max(budget * 3, budget + 1))
        top = cand[torch.argsort(rc, descending=True)[:poolk]]
        cov = select_coverage(rel_resid_feat[top], min(budget, len(top)))
        return top[torch.tensor(cov, device=device)]

    budgets = [int(b) for b in args.budgets.split(',')]
    arms = ['RGA_KD', 'EL2N', 'random']
    results = {}
    for b in budgets:
        results[str(b)] = {}
        for mode in ['KD', 'CE']:
            results[str(b)][mode] = {}
            for arm in arms:
                accs = []
                for s in range(args.seeds):
                    srng = np.random.RandomState(3000 + s)
                    sel = coreset(arm, b, srng)
                    acc = train_student(mode, Xpool[sel], PT_pool[sel], ypool[sel],
                                        Xte, yte, args.temp, 3000 + s, device, args.retrain_epochs, args.bs)
                    accs.append(acc)
                accs = np.array(accs)
                results[str(b)][mode][arm] = {'acc_mean': float(accs.mean()), 'acc_std': float(accs.std())}
                print(f"  [b{b}][{mode}] {arm:8s} acc={accs.mean():.4f}±{accs.std():.4f}")
            # relative advantage of RGA over the better of {EL2N,random}
            rga = results[str(b)][mode]['RGA_KD']['acc_mean']
            base = max(results[str(b)][mode]['EL2N']['acc_mean'], results[str(b)][mode]['random']['acc_mean'])
            results[str(b)][mode]['RGA_advantage_over_best_baseline'] = rga - base
            print(f"  [b{b}][{mode}] RGA advantage over best baseline = {rga-base:+.4f}")

    out = {'setup': {'dataset': 'CIFAR-100', 'pair': 'ResNet-56->ResNet-20', 'N': n,
                     'seeds': args.seeds, 'budgets': budgets, 'teacher_test_acc': te_acc},
           'E3': results, 'wall_clock_sec': time.time() - t0}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
