#!/usr/bin/env python
"""
Fair head-to-head (removes the training-protocol confound): run the best corrected RGA variant
against the strong baselines ALL under the SAME micro-batch kd_train.

Arms: RGA_fixcov_nogate (corrected coverage on anchor-space residual, NO gate — best RGA variant),
      TAGCOS (kmeans on deep grad), deep_topr (top deep-grad r), GraNd (||last-layer grad||), random.
Budgets {500,1000,2000} × 3 seeds, ROUGE-L on Dolly test. Reuses cached fingerprints.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.cluster import MiniBatchKMeans

SAGD = "/data/tianhao/SaGD"
sys.path.insert(0, os.path.join(SAGD, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sagd.data import InstructionDataset
from sagd.models import load_teacher
from exp_retrain_qwen import kd_train, greedy_logdet


def double_center(K):
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()

def rel_matrices(G_S, G_T, anchor):
    KS = double_center(G_S @ G_S[anchor].T); KT = double_center(G_T @ G_T[anchor].T)
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
    ap.add_argument('--out', default='pilot/FINAL_COMPARE_QWEN.json')
    args = ap.parse_args()

    t0 = time.time(); dev = args.device
    c = torch.load(args.cache)
    Gh, Gd, Gte, idxs = c['Gh'].to(dev), c['Gd'].to(dev), c['Gte'].to(dev), c['idxs'].cpu().numpy()
    n = Gd.shape[0]; Gd_np = Gd.cpu().numpy()
    teacher, tok = load_teacher(args.teacher, device=dev, dtype=torch.float16,
                                ckpt_path=args.teacher_ckpt if os.path.exists(args.teacher_ckpt) else None)
    pool_ds = InstructionDataset(tok, max_seq_len=512, subset='train')
    test_ds = InstructionDataset(tok, max_seq_len=512, subset='test', max_samples=args.max_eval)

    rng = np.random.RandomState(0)
    anchor = torch.tensor(rng.choice(n, min(args.anchors, n), replace=False), device=dev)
    KS, KT, r_deep = rel_matrices(Gd, Gte, anchor)
    resid = KT - KS
    grand = Gh.norm(dim=1)
    idx_all = torch.arange(n, device=dev)

    def sel(name, b, seed):
        b = min(b, n - 1)
        if name == 'RGA_fixcov_nogate':
            poolk = min(n, max(b * 3, b + 1))
            top = idx_all[torch.argsort(r_deep, descending=True)[:poolk]]
            cov = coverage(resid[top], min(b, len(top)))
            return top[torch.tensor(cov, device=dev)]
        if name == 'deep_topr':
            return idx_all[torch.argsort(r_deep, descending=True)[:b]]
        if name == 'GraNd':
            return idx_all[torch.argsort(grand, descending=True)[:b]]
        if name == 'random':
            return torch.tensor(np.random.RandomState(seed).choice(n, b, replace=False), device=dev)
        if name == 'TAGCOS':
            km = MiniBatchKMeans(n_clusters=b, random_state=seed, batch_size=2048,
                                 n_init=3, max_iter=100).fit(Gd_np)
            cen = torch.tensor(km.cluster_centers_, device=dev, dtype=Gd.dtype)
            s = torch.cdist(cen, Gd).argmin(dim=1).cpu().numpy(); s = np.unique(s)
            if len(s) < b:
                rest = np.setdiff1d(np.arange(n), s); np.random.RandomState(seed).shuffle(rest)
                s = np.concatenate([s, rest[:b - len(s)]])
            return torch.tensor(s[:b], device=dev)

    budgets = [min(int(x), n - 1) for x in str(args.budgets).split(',')]
    arms = ['RGA_fixcov_nogate', 'TAGCOS', 'deep_topr', 'GraNd', 'random']
    sweep = {}
    for b in budgets:
        sweep[str(b)] = {}
        print(f"===== budget b={b} =====")
        for name in arms:
            rl = []
            for s in range(args.seeds):
                local = sel(name, b, 1000 + s).cpu().numpy()
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
                                     'budgets': budgets, 'seeds': args.seeds, 'epochs': args.epochs,
                                     'protocol': 'micro-batch grad-accum, effective bs=8, fp16 teacher'},
                           'sweep': sweep, 'wall_clock_sec': time.time() - t0}, f, indent=2)
    print(json.dumps(sweep, indent=2)); print(f"[done] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
