#!/usr/bin/env python
"""
Phase C E5 (ablations) + E4 (efficiency): ResNet-56 -> ResNet-20 / CIFAR-100.
Budget fixed at the aggressive point where RGA shines (default b=4000).

Ablations:
  - criterion:  D-optimal coverage  vs  plain top-r        (does coverage matter?)
  - signature:  studentKD+teacherENTK  vs  both_entk  vs  both_KDfeature
  - projection: count-sketch d in {512, 2048, 8192}  vs  none(raw)
  - anchors:    m in {64, 256, 1024}
  - gate:       (covered decisively by E6; not repeated)
E4 efficiency:
  - grad locus: LAST-LAYER (closed form) vs FULL student KD gradient (autograd, timed)
    -> selection-quality parity + wall-clock break-even.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_rga_diag_real import (
    resnet20, CountSketch, relational_residual, effective_rank, select_coverage,
    load_cifar100, train_teacher, get_feat_logits, kd_train_student, gpu_augment,
)


def full_student_kd_grad(student, X, PT, T, device, proj_dim, seed, chunk=512):
    """Per-sample FULL student KD gradient, projected on the fly. Returns (projected (n,d), seconds)."""
    student.eval()
    params = {k: v.detach() for k, v in student.named_parameters()}
    buffers = {k: v.detach() for k, v in student.named_buffers()}
    P = sum(v.numel() for v in params.values())
    cs = CountSketch(P, proj_dim, seed, device)

    def kd_ce(params, x, pt):
        z = functional_call(student, (params, buffers), (x.unsqueeze(0),)).squeeze(0)
        logp = F.log_softmax(z / T, dim=-1)
        return -(pt * logp).sum()

    gfun = vmap(grad(kd_ce), in_dims=(None, 0, 0))
    t0 = time.time(); outs = []
    for i in range(0, X.shape[0], chunk):
        gd = gfun(params, X[i:i+chunk], PT[i:i+chunk])
        flat = torch.cat([v.reshape(v.shape[0], -1) for v in gd.values()], dim=1)
        outs.append(cs(flat).detach())
    return torch.cat(outs, 0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=10000)
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=4.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--teacher_epochs', type=int, default=50)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--retrain_epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--budget', type=int, default=4000)
    ap.add_argument('--out', default='pilot/E5_ABLATION_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time(); device = torch.device('cuda')
    print(f"[env] {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0); np.random.seed(0)
    Xtr, ytr, Xte, yte = load_cifar100(device)
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    rng = np.random.RandomState(0)
    pool_idx = rng.choice(Xtr.shape[0], args.N, replace=False)
    Xpool, ypool = Xtr[pool_idx], ytr[pool_idx]; n = args.N; b = args.budget
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
    gS_last = torch.einsum('nc,nd->ncd', rho, fS_pool).reshape(n, -1)  # last-layer student KD grad
    tkd = torch.einsum('nc,nd->ncd', rho, fT_pool).reshape(n, -1)      # teacher KD-feature
    gate = teacher_correct > 0.5; idx_all = torch.arange(n, device=device)
    sel_setup_last = time.time()  # marker

    def proj(feat, d, s):
        return CountSketch(feat.shape[1], d, s, device)(feat)

    def rga_select(Sfeat, Tfeat, d, m, criterion):
        t_sel = time.time()
        GS = proj(Sfeat, d, 101) if d > 0 else Sfeat
        GT = proj(Tfeat, d, 202) if d > 0 else Tfeat
        aidx = torch.tensor(rng.choice(n, min(m, n), replace=False), device=device)
        r, _, _, _ = relational_residual(GS, GT, aidx)
        cand = idx_all[gate]; rc = r[gate]
        if criterion == 'topr':
            sel = cand[torch.argsort(rc, descending=True)[:b]]
        else:  # coverage
            poolk = min(len(cand), max(b * 3, b + 1))
            top = cand[torch.argsort(rc, descending=True)[:poolk]]
            cov = select_coverage((GT - GS)[top], min(b, len(top)))
            sel = top[torch.tensor(cov, device=device)]
        return sel, time.time() - t_sel

    def run(sel):
        accs = []
        for s in range(args.seeds):
            acc = kd_train_student(PT_pool, Xpool[sel], PT_pool[sel], Xte, yte,
                                   args.temp, 5000 + s, device, args.retrain_epochs, args.bs)
            accs.append(acc)
        return float(np.mean(accs)), float(np.std(accs))

    configs = {
        'MAIN_studentKD_teacherENTK_cov_d2048_m256': (gS_last, fT_pool, 2048, 256, 'coverage'),
        'criterion_topr':                            (gS_last, fT_pool, 2048, 256, 'topr'),
        'sig_both_entk':                             (fS_pool, fT_pool, 2048, 256, 'coverage'),
        'sig_both_KDfeature':                        (gS_last, tkd,     2048, 256, 'coverage'),
        'proj_d512':                                 (gS_last, fT_pool, 512,  256, 'coverage'),
        'proj_d8192':                                (gS_last, fT_pool, 8192, 256, 'coverage'),
        'anchors_m64':                               (gS_last, fT_pool, 2048, 64,  'coverage'),
        'anchors_m1024':                             (gS_last, fT_pool, 2048, 1024,'coverage'),
    }
    results = {}
    for name, (Sf, Tf, d, m, crit) in configs.items():
        sel, sel_sec = rga_select(Sf, Tf, d, m, crit)
        acc_m, acc_s = run(sel)
        results[name] = {'acc_mean': acc_m, 'acc_std': acc_s, 'sel_sec': sel_sec, 'budget': b}
        print(f"  [{name:46s}] acc={acc_m:.4f}±{acc_s:.4f}  sel={sel_sec:.2f}s")

    # ---- E4: FULL student KD gradient vs last-layer (quality + cost) ----
    gS_full, full_sec = full_student_kd_grad(student, Xpool, PT_pool, args.temp, device, 2048, 404)
    # last-layer projection cost (for fair comparison): time the last-layer signature build (already cheap)
    ll_t = time.time(); _ = proj(gS_last, 2048, 101); ll_sec = time.time() - ll_t
    # RGA with full-grad student signature (teacher stays last-layer feature)
    t_sel = time.time()
    GSf = gS_full; GT = proj(fT_pool, 2048, 202)
    aidx = torch.tensor(rng.choice(n, 256, replace=False), device=device)
    rf, _, _, _ = relational_residual(GSf, GT, aidx)
    candf = idx_all[gate]; rcf = rf[gate]
    poolk = min(len(candf), max(b * 3, b + 1))
    topf = candf[torch.argsort(rcf, descending=True)[:poolk]]
    covf = select_coverage((GT - GSf)[topf], min(b, len(topf)))
    self_full = topf[torch.tensor(covf, device=device)]
    sel_full_sec = time.time() - t_sel
    acc_full_m, acc_full_s = run(self_full)
    e4 = {
        'full_grad_acc_mean': acc_full_m, 'full_grad_acc_std': acc_full_s,
        'full_grad_extract_sec': full_sec, 'lastlayer_proj_sec': ll_sec,
        'lastlayer_acc_mean': results['MAIN_studentKD_teacherENTK_cov_d2048_m256']['acc_mean'],
        'note': 'full vs last-layer student KD gradient; teacher = last-layer feature in both',
    }
    print(f"  [E4 full-grad] acc={acc_full_m:.4f}±{acc_full_s:.4f}  extract={full_sec:.1f}s "
          f"(vs last-layer proj {ll_sec:.2f}s, last-layer acc "
          f"{e4['lastlayer_acc_mean']:.4f})")

    out = {'setup': {'dataset': 'CIFAR-100', 'pair': 'ResNet-56->ResNet-20', 'N': n, 'budget': b,
                     'seeds': args.seeds, 'teacher_test_acc': te_acc},
           'E5_ablations': results, 'E4_efficiency': e4, 'wall_clock_sec': time.time() - t0}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
