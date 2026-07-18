#!/usr/bin/env python
"""
Fix + retest RGA on Qwen (user-approved). The sweep showed RGA_deep ~ random; suspected cause is
the coverage step selecting on (Gte - Gd), two DIFFERENTLY-sketched vectors (coordinate-misaligned
=> noise). Fix: run coverage on the ANCHOR-SPACE relational residual  R[i] = K'_T[i,:] - K'_S[i,:]
(both m-dim over the SAME 256 anchors => coordinate-aligned, meaningful).

Variants (reuse cached fingerprints Gd/Gte/tnll/idxs):
  RGA_fixcov        = gate + top-3b by r_deep + D-optimal coverage on residual rows R  (the fix)
  RGA_gatetopr      = gate + top-b by r_deep (no coverage)          -- isolates coverage's value
  RGA_fixcov_nogate = top-3b by r_deep + coverage on R (no gate)    -- isolates the gate
Compare vs old buggy RGA_deep (~random) and the sweep's TAGCOS/random.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import Subset

SAGD = "/data/tianhao/SaGD"
sys.path.insert(0, os.path.join(SAGD, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sagd.data import InstructionDataset
from sagd.models import load_teacher
from exp_retrain_qwen import kd_train, greedy_logdet


def double_center(K):
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()

def rel_matrices(G_S, G_T, anchor):
    """Return centered anchor-space relational matrices K'_S,K'_T (n,m) and per-sample r=1-corr."""
    KS = double_center(G_S @ G_S[anchor].T)
    KT = double_center(G_T @ G_T[anchor].T)
    A = KS - KS.mean(1, keepdim=True); B = KT - KT.mean(1, keepdim=True)
    corr = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-12)
    return KS, KT, (1.0 - corr)

def coverage(feats, k):
    Fn = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
    return greedy_logdet(Fn @ Fn.T, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='Qwen/Qwen3-8B')
    ap.add_argument('--student', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--teacher_ckpt', default=f'{SAGD}/data/teacher_sft_dolly_qwen.pt')
    ap.add_argument('--budgets', default='500,1000,2000')
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_eval', type=int, default=250)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache', default='pilot/_retrain_fp_cache.pt')
    ap.add_argument('--out', default='pilot/RGA_FIX_QWEN_SWEEP.json')
    args = ap.parse_args()

    t0 = time.time(); dev = args.device
    assert os.path.exists(args.cache)
    c = torch.load(args.cache)
    Gd, Gte, tnll, idxs = c['Gd'].to(dev), c['Gte'].to(dev), c['tnll'].to(dev), c['idxs'].cpu().numpy()
    n = Gd.shape[0]
    teacher, tok = load_teacher(args.teacher, device=dev, dtype=torch.float16,
                                ckpt_path=args.teacher_ckpt if os.path.exists(args.teacher_ckpt) else None)
    pool_ds = InstructionDataset(tok, max_seq_len=512, subset='train')
    test_ds = InstructionDataset(tok, max_seq_len=512, subset='test', max_samples=args.max_eval)

    rng = np.random.RandomState(0)
    anchor = torch.tensor(rng.choice(n, min(args.anchors, n), replace=False), device=dev)
    KS, KT, r_deep = rel_matrices(Gd, Gte, anchor)   # student=deep grad, teacher=feature
    resid = KT - KS                                  # (n,m) coordinate-aligned residual  <-- the fix
    gate = tnll <= torch.median(tnll)
    idx_all = torch.arange(n, device=dev)
    print(f"[setup] pool={n} anchors={anchor.numel()} r_deep={r_deep.mean():.3f}±{r_deep.std():.3f} "
          f"resid_dim={resid.shape[1]}")

    def sel(name, b):
        b = min(b, n - 1)
        if name == 'RGA_gatetopr':
            cand = idx_all[gate]; return cand[torch.argsort(r_deep[gate], descending=True)[:b]]
        if name == 'RGA_fixcov':
            cand = idx_all[gate]; rc = r_deep[gate]
            poolk = min(len(cand), max(b * 3, b + 1))
            top = cand[torch.argsort(rc, descending=True)[:poolk]]
            cov = coverage(resid[top], min(b, len(top)))
            return top[torch.tensor(cov, device=dev)]
        if name == 'RGA_fixcov_nogate':
            poolk = min(n, max(b * 3, b + 1))
            top = idx_all[torch.argsort(r_deep, descending=True)[:poolk]]
            cov = coverage(resid[top], min(b, len(top)))
            return top[torch.tensor(cov, device=dev)]

    budgets = [min(int(x), n - 1) for x in str(args.budgets).split(',')]
    variants = ['RGA_fixcov', 'RGA_gatetopr', 'RGA_fixcov_nogate']
    sweep = {}
    for b in budgets:
        sweep[str(b)] = {}
        print(f"===== budget b={b} =====")
        for name in variants:
            rl = []
            for s in range(args.seeds):
                local = sel(name, b).cpu().numpy()
                sub = Subset(pool_ds, [int(idxs[i]) for i in local])
                score = kd_train(teacher, args.student, sub, test_ds, tok, args.temp,
                                 1000 + s, dev, args.epochs, args.bs, args.lr, args.max_eval)
                rl.append(score)
                print(f"  [b{b} {name:18s} seed{s}] rougeL={score:.4f}  ({time.time()-t0:.0f}s)")
            rl = np.array(rl)
            sweep[str(b)][name] = {'rougeL_mean': float(rl.mean()), 'rougeL_std': float(rl.std()), 'n': args.seeds}
            print(f"  == b{b} {name:18s} rougeL={rl.mean():.4f}±{rl.std():.4f}")
            with open(args.out, 'w') as f:
                json.dump({'setup': {'teacher': args.teacher, 'student': args.student, 'N_pool': n,
                                     'budgets': budgets, 'anchors': int(anchor.numel()), 'seeds': args.seeds,
                                     'epochs': args.epochs, 'lr': args.lr, 'max_eval': args.max_eval},
                           'sweep': sweep, 'wall_clock_sec': time.time() - t0}, f, indent=2)
    print(json.dumps(sweep, indent=2)); print(f"[done] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
