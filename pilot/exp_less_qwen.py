#!/usr/bin/env python
"""
LESS baseline on Qwen (Xia et al., ICML 2024 — "Selecting Influential Data for Targeted
Instruction Tuning"). Core of LESS: score each candidate training example by the cosine
similarity between its (random-projected) gradient and a HELD-OUT VALIDATION gradient
direction, then pick top-k. "Train on examples whose gradient points toward the target."

Faithful adaptation to our KD-on-Dolly setting:
  - candidate gradient = deep KD gradient (attn qkvo, all layers) — REUSE cached pool Gd
  - target direction   = mean deep KD gradient over a held-out validation set of Dolly train
                         sequences DISJOINT from both the candidate pool AND the eval/test set
                         (avoids leakage; the eval test set stays untouched)
  - score(i)           = cosine(Gd_pool[i], g_val)  ;  select top-b
  (Note: omits LESS's Adam preconditioning; keeps the random-projected gradient-cosine core.)

Same micro-batch kd_train + ROUGE-L eval as the fair head-to-head, so it drops into that table.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

SAGD = "/data/tianhao/SaGD"
sys.path.insert(0, os.path.join(SAGD, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sagd.data import InstructionDataset, collate_fn
from sagd.models import load_teacher, load_student
from exp_retrain_qwen import kd_train, extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='Qwen/Qwen3-8B')
    ap.add_argument('--student', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--teacher_ckpt', default=f'{SAGD}/data/teacher_sft_dolly_qwen.pt')
    ap.add_argument('--budgets', default='500,1000,2000')
    ap.add_argument('--proj_dim', type=int, default=2048)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--val_size', type=int, default=200)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_eval', type=int, default=250)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache', default='pilot/_retrain_fp_cache.pt')
    ap.add_argument('--out', default='pilot/LESS_QWEN.json')
    args = ap.parse_args()

    t0 = time.time(); dev = args.device
    c = torch.load(args.cache)
    Gd, idxs = c['Gd'].to(dev), c['idxs'].cpu().numpy()
    n = Gd.shape[0]
    teacher, tok = load_teacher(args.teacher, device=dev, dtype=torch.float16,
                               ckpt_path=args.teacher_ckpt if os.path.exists(args.teacher_ckpt) else None)
    pool_ds = InstructionDataset(tok, max_seq_len=512, subset='train')
    test_ds = InstructionDataset(tok, max_seq_len=512, subset='test', max_samples=args.max_eval)

    # ---- held-out validation set: pool_ds samples whose dataset index is NOT in the cached pool ----
    used = set(int(i) for i in idxs)
    val_positions = [p for p in range(len(pool_ds)) if p not in used][:args.val_size * 2]
    val_ds = Subset(pool_ds, val_positions)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # student (base, untie lm_head — matches how the pool cache was extracted, same projector seeds)
    student, _ = load_student(args.student, device=dev); student.train()
    if getattr(student.config, 'tie_word_embeddings', False) or \
       (student.lm_head.weight.data_ptr() == student.get_input_embeddings().weight.data_ptr()):
        w = student.get_input_embeddings().weight.detach().clone()
        student.lm_head.weight = nn.Parameter(w); student.config.tie_word_embeddings = False

    # extract() returns (Gt,Gh,Gd,Gte,tnll,idxs) with proj_deep seed=2 == the pool cache's Gd space
    _, _, Gd_val, _, _, _ = extract(teacher, student, val_loader, args.val_size, args.proj_dim, args.temp, dev)
    del student; torch.cuda.empty_cache()
    Gd_val = Gd_val.to(dev)
    g_val = Gd_val.mean(0)
    g_val = g_val / (g_val.norm() + 1e-12)
    print(f"[LESS] pool={n} val={Gd_val.shape[0]}  target-grad ready ({time.time()-t0:.0f}s)")

    # LESS score = cosine(pool gradient, validation gradient)
    Gd_n = Gd / (Gd.norm(dim=1, keepdim=True) + 1e-12)
    less_score = Gd_n @ g_val                       # (n,)
    idx_all = torch.arange(n, device=dev)
    print(f"[LESS] score mean={less_score.mean():.3f} std={less_score.std():.3f} "
          f"max={less_score.max():.3f} min={less_score.min():.3f}")

    budgets = [min(int(x), n - 1) for x in str(args.budgets).split(',')]
    results = {}
    for b in budgets:
        rl = []
        sel = idx_all[torch.argsort(less_score, descending=True)[:b]].cpu().numpy()
        for s in range(args.seeds):
            sub = Subset(pool_ds, [int(idxs[i]) for i in sel])
            score = kd_train(teacher, args.student, sub, test_ds, tok, args.temp,
                             1000 + s, dev, args.epochs, args.bs, args.lr, args.max_eval)
            rl.append(score)
            print(f"  [b{b} LESS seed{s}] rougeL={score:.4f}  ({time.time()-t0:.0f}s)")
        rl = np.array(rl)
        results[str(b)] = {'rougeL_mean': float(rl.mean()), 'rougeL_std': float(rl.std()), 'n': args.seeds}
        print(f"  == b{b} LESS rougeL={rl.mean():.4f}±{rl.std():.4f}")
        with open(args.out, 'w') as f:
            json.dump({'setup': {'method': 'LESS (grad cosine to held-out val grad, no Adam precond)',
                                 'teacher': args.teacher, 'student': args.student, 'N_pool': n,
                                 'val_size': int(Gd_val.shape[0]), 'budgets': budgets, 'seeds': args.seeds,
                                 'protocol': 'micro-batch grad-accum eff bs=8, fp16 teacher'},
                       'results': results, 'wall_clock_sec': time.time() - t0}, f, indent=2)
    print(json.dumps(results, indent=2)); print(f"[done] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
