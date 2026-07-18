#!/usr/bin/env python
"""
External strong-baseline round on Qwen (fills the gap: RGA_deep vs native coreset methods on LLM).
Baselines: EL2N, GraNd, TAGCOS, GRAFT — same budgets/seeds/eval as the RGA sweep, so results
drop straight into the same table.

Reuses the CACHED fingerprints from exp_retrain_qwen (pool N=4000):
  GraNd  = top-k by ||last-layer KD gradient||  (Gh norm)              [Paul et al. 2021]
  TAGCOS = k-means on projected deep gradients, nearest-to-centroid medoids  [2024]
  GRAFT  = feature-space MaxVol = D-optimal coverage on teacher features (Gte)
  EL2N   = top-k by ||p_S - onehot(gold)|| over response tokens  (needs one student forward) [Paul 2021]
Then KD-train fresh Qwen3-0.6B on each subset, eval ROUGE-L (same harness as RGA sweep).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import MiniBatchKMeans

SAGD = "/data/tianhao/SaGD"
sys.path.insert(0, os.path.join(SAGD, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sagd.data import InstructionDataset, collate_fn
from sagd.models import load_teacher, load_student
# reuse the exact training + selection helpers from the RGA harness
from exp_retrain_qwen import kd_train, select_coverage


@torch.no_grad()
def compute_el2n(student, pool_ds, N, dev):
    """Per-sample EL2N = mean over response tokens of ||softmax(z_S,t) - onehot(gold_{t+1})||_2.
    Returns scores aligned to the SAME order/idxs as the cached extraction (batch=1, skip no-response)."""
    student.eval()
    loader = DataLoader(pool_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    scores, idxs, done = [], [], 0
    for batch in loader:
        if done >= N:
            break
        ids = batch['input_ids'].to(dev); am = batch['attention_mask'].to(dev); lm = batch['labels_mask'].to(dev)
        if lm[:, 1:].sum() < 1:
            continue
        z = student(input_ids=ids, attention_mask=am).logits.float()
        zt = z[:, :-1, :]; tgt = ids[:, 1:]; m = lm[:, 1:].float()  # (1,L-1)
        p = F.softmax(zt, dim=-1)                                   # (1,L-1,V)
        # ||p - onehot||^2 = sum p^2 - 2 p_gold + 1
        p_gold = p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)        # (1,L-1)
        el2n_t = torch.sqrt((p * p).sum(-1) - 2 * p_gold + 1.0).clamp_min(0)  # (1,L-1)
        s = (el2n_t * m).sum() / m.sum().clamp(min=1)
        scores.append(s.item()); idxs.append(int(batch['index'].item())); done += 1
    return np.array(scores), np.array(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='Qwen/Qwen3-8B')
    ap.add_argument('--student', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--teacher_ckpt', default=f'{SAGD}/data/teacher_sft_dolly_qwen.pt')
    ap.add_argument('--N_pool', type=int, default=4000)
    ap.add_argument('--budgets', default='500,1000,2000')
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_eval', type=int, default=250)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache', default='pilot/_retrain_fp_cache.pt')
    ap.add_argument('--out', default='pilot/BASELINES_QWEN_SWEEP.json')
    args = ap.parse_args()

    t0 = time.time(); dev = args.device
    assert os.path.exists(args.cache), f"need cached fingerprints {args.cache} (run exp_retrain_qwen first)"
    c = torch.load(args.cache)
    Gh, Gd, Gte, idxs = c['Gh'].to(dev), c['Gd'].to(dev), c['Gte'].to(dev), c['idxs'].cpu().numpy()
    n = Gh.shape[0]
    print(f"[cache] fingerprints pool={n}")

    teacher, tok = load_teacher(args.teacher, device=dev, dtype=torch.float16,
                                ckpt_path=args.teacher_ckpt if os.path.exists(args.teacher_ckpt) else None)
    pool_ds = InstructionDataset(tok, max_seq_len=512, subset='train')
    test_ds = InstructionDataset(tok, max_seq_len=512, subset='test', max_samples=args.max_eval)

    # ---- EL2N: one student forward over the pool, aligned to cache idxs ----
    student, _ = load_student(args.student, device=dev)
    el2n_raw, el2n_idx = compute_el2n(student, pool_ds, n, dev)
    del student; torch.cuda.empty_cache()
    # align el2n to cache idxs order
    pos = {int(i): k for k, i in enumerate(el2n_idx)}
    el2n = torch.tensor([el2n_raw[pos[int(i)]] for i in idxs], device=dev)
    print(f"[el2n] computed, mean={el2n.mean():.3f}  ({time.time()-t0:.0f}s)")

    grand = Gh.norm(dim=1)          # GraNd = last-layer KD gradient norm
    Gd_np = Gd.cpu().numpy()
    idx_all = torch.arange(n, device=dev)

    def select(name, b, seed):
        b = min(b, n - 1)
        if name == 'EL2N':
            return idx_all[torch.argsort(el2n, descending=True)[:b]]
        if name == 'GraNd':
            return idx_all[torch.argsort(grand, descending=True)[:b]]
        if name == 'GRAFT':          # feature-space MaxVol (D-optimal on teacher features)
            return idx_all[select_coverage(Gte, b)]
        if name == 'TAGCOS':         # k-means on deep gradients, nearest medoids
            km = MiniBatchKMeans(n_clusters=b, random_state=seed, batch_size=2048,
                                 n_init=3, max_iter=100).fit(Gd_np)
            cen = torch.tensor(km.cluster_centers_, device=dev, dtype=Gd.dtype)
            d = torch.cdist(cen, Gd); sel = d.argmin(dim=1).cpu().numpy()
            sel = np.unique(sel)
            if len(sel) < b:
                rest = np.setdiff1d(np.arange(n), sel)
                np.random.RandomState(seed).shuffle(rest); sel = np.concatenate([sel, rest[:b - len(sel)]])
            return torch.tensor(sel[:b], device=dev)

    budgets = [min(int(x), n - 1) for x in str(args.budgets).split(',')]
    baselines = ['EL2N', 'GraNd', 'TAGCOS', 'GRAFT']
    sweep = {}
    for b in budgets:
        sweep[str(b)] = {}
        print(f"===== budget b={b} =====")
        for name in baselines:
            rl = []
            for s in range(args.seeds):
                sel = select(name, b, 1000 + s).cpu().numpy()
                sub = Subset(pool_ds, [int(idxs[i]) for i in sel])
                score = kd_train(teacher, args.student, sub, test_ds, tok, args.temp,
                                 1000 + s, dev, args.epochs, args.bs, args.lr, args.max_eval)
                rl.append(score)
                print(f"  [b{b} {name:8s} seed{s}] rougeL={score:.4f}  ({time.time()-t0:.0f}s)")
            rl = np.array(rl)
            sweep[str(b)][name] = {'rougeL_mean': float(rl.mean()), 'rougeL_std': float(rl.std()), 'n': args.seeds}
            print(f"  == b{b} {name:8s} rougeL={rl.mean():.4f}±{rl.std():.4f}")
            with open(args.out, 'w') as f:
                json.dump({'setup': {'teacher': args.teacher, 'student': args.student, 'N_pool': n,
                                     'budgets': budgets, 'seeds': args.seeds, 'epochs': args.epochs,
                                     'lr': args.lr, 'max_eval': args.max_eval},
                           'sweep': sweep, 'wall_clock_sec': time.time() - t0}, f, indent=2)
    print(json.dumps(sweep, indent=2)); print(f"[done] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
