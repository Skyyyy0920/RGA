#!/usr/bin/env python
"""
Phase C E6 (MANDATORY): teacher-noise falsification of the correctness gate.
ResNet-56 -> ResNet-20 / CIFAR-100.

Phase A could not test the gate (teachers too accurate on the pool). Here we inject a known
fraction of CONFIDENT-WRONG teacher labels (RAD trap) and ask:
  (1) detection: do noised samples get high r AND low p_T(y_true)?  (AUC of gate score)
  (2) purity:    does the soft gate (×p_T(y)) keep noised samples OUT of the RGA coreset?
  (3) efficacy:  is downstream student acc higher with gate-on vs gate-off under teacher noise?

A working gate => noised samples are high-r (so naive high-r selection grabs them) but the
p_T(y) gate filters them, so gate-on coreset is cleaner and trains a better student.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_rga_diag_real import (
    resnet20, CountSketch, relational_residual, effective_rank, select_coverage,
    load_cifar100, train_teacher, get_feat_logits, kd_train_student, gpu_augment,
)


def auc(score, label):  # label: 1 = noised; higher score => more likely noised
    s = score.detach().cpu().numpy(); y = label.detach().cpu().numpy()
    order = np.argsort(-s); y = y[order]
    P = y.sum(); Nn = len(y) - P
    if P == 0 or Nn == 0:
        return 0.5
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    tpr = tps / P; fpr = fps / Nn
    return float(np.trapz(tpr, fpr))


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
    ap.add_argument('--noise_frac', type=float, default=0.3)
    ap.add_argument('--budget', type=int, default=2000)
    ap.add_argument('--out', default='pilot/E6_GATE_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time(); device = torch.device('cuda')
    print(f"[env] {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0); np.random.seed(0)

    Xtr, ytr, Xte, yte = load_cifar100(device)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    rng = np.random.RandomState(0)
    pool_idx = rng.choice(Xtr.shape[0], args.N, replace=False)
    Xpool, ypool = Xtr[pool_idx], ytr[pool_idx]; n = args.N
    C = 100
    teacher, te_acc = train_teacher(Xtr, ytr, Xte, yte, device, args.teacher_epochs, args.bs,
                                    ckpt=f'pilot/teacher_resnet56_c100_e{args.teacher_epochs}.pt')
    fT_pool, zT_pool = get_feat_logits(teacher, Xpool, device)

    # ---- inject confident-wrong teacher labels on a known subset ----
    noised = torch.zeros(n, dtype=torch.bool, device=device)
    nidx = rng.choice(n, int(args.noise_frac * n), replace=False)
    noised[nidx] = True
    zT_corrupt = zT_pool.clone()
    wrong = (ypool + torch.randint(1, C, (n,), device=device)) % C  # a wrong class
    big = zT_pool.abs().max() * 3 + 10.0
    for i in nidx:  # set a confident peak on the wrong class
        zT_corrupt[i] = -big
        zT_corrupt[i, wrong[i]] = big
    PT_pool = F.softmax(zT_corrupt / args.temp, dim=-1)
    pT_y = PT_pool[torch.arange(n, device=device), ypool]  # teacher prob on TRUE label (gate signal)
    teacher_correct = (zT_corrupt.argmax(1) == ypool).float()
    print(f"[noise] frac={args.noise_frac} noised={noised.sum().item()} "
          f"pool_correct={teacher_correct.mean().item():.3f}")

    # ---- warm student, signatures w.r.t. corrupted teacher ----
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
    proj = lambda f, s: CountSketch(f.shape[1], args.proj_dim, s, device)(f)
    GS = proj(gS, 101); GT = proj(fT_pool, 202)
    anchor_idx = torch.tensor(rng.choice(n, args.anchors, replace=False), device=device)
    r, _, _, cka = relational_residual(GS, GT, anchor_idx)
    rel_resid_feat = GT - GS
    idx_all = torch.arange(n, device=device)

    # ---- (1) detection ----
    det = {
        'r_mean_noised': float(r[noised].mean()), 'r_mean_clean': float(r[~noised].mean()),
        'pTy_mean_noised': float(pT_y[noised].mean()), 'pTy_mean_clean': float(pT_y[~noised].mean()),
        'auc_r_detects_noise': auc(r, noised.float()),
        'auc_lowpTy_detects_noise': auc(-pT_y, noised.float()),
        'auc_gatescore_detects_noise': auc(r * (1 - pT_y), noised.float()),
    }
    print(f"[detect] r noised={det['r_mean_noised']:.3f} clean={det['r_mean_clean']:.3f} | "
          f"AUC(r)={det['auc_r_detects_noise']:.3f} AUC(1-pTy)={det['auc_lowpTy_detects_noise']:.3f}")

    # ---- selection arms: RGA gate-on (×pTy), gate-off (high-r only), random ----
    b = args.budget

    def select(mode, srng):
        if mode == 'random':
            return torch.tensor(srng.choice(n, b, replace=False), device=device)
        gate = teacher_correct > 0.5
        if mode == 'gate_on':
            score = r * pT_y                      # soft gate downweights low-pTy (noised)
            cand = idx_all[gate]; sc = score[gate]
        else:  # gate_off
            cand = idx_all; sc = r
        poolk = min(len(cand), max(b * 3, b + 1))
        top = cand[torch.argsort(sc, descending=True)[:poolk]]
        cov = select_coverage(rel_resid_feat[top], min(b, len(top)))
        return top[torch.tensor(cov, device=device)]

    results = {}
    for mode in ['gate_on', 'gate_off', 'random']:
        accs, purities = [], []
        for s in range(args.seeds):
            srng = np.random.RandomState(6000 + s)
            sel = select(mode, srng)
            frac_noised = float(noised[sel].float().mean())
            acc = kd_train_student(PT_pool, Xpool[sel], PT_pool[sel], Xte, yte,
                                   args.temp, 6000 + s, device, args.retrain_epochs, args.bs)
            accs.append(acc); purities.append(frac_noised)
        accs = np.array(accs); purities = np.array(purities)
        results[mode] = {'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
                         'frac_noised_in_coreset_mean': float(purities.mean()),
                         'n': args.seeds, 'budget': b}
        print(f"  [{mode:8s}] acc={accs.mean():.4f}±{accs.std():.4f}  "
              f"noised_in_coreset={purities.mean():.3f} (base rate {args.noise_frac})")

    out = {'setup': {'dataset': 'CIFAR-100', 'pair': 'ResNet-56->ResNet-20', 'N': n,
                     'noise_frac': args.noise_frac, 'budget': b, 'seeds': args.seeds,
                     'teacher_test_acc': te_acc},
           'detection': det, 'selection': results, 'wall_clock_sec': time.time() - t0}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
