#!/usr/bin/env python
"""
RETRAIN proof (pivot B): does selecting a Dolly coreset by the GENUINE deep-gradient
relational residual (r_deep) train a BETTER Qwen student than the last-layer proxy
(r_proxy), token-only (r_token), or random?  "Different" (shown by the decision expt)
is necessary but not sufficient — this tests "better".

Pipeline:
  1. Extract per-sample fingerprints over a pool of N_pool Dolly samples (cached):
     deep-grad (attn qkvo, all layers, real backward), last-layer proxy (lm_head grad),
     token-only ((p_S-p_T) aggregated); + teacher feature (mean hidden) + teacher NLL (gate).
  2. Selection arms at fixed budget b:
     RGA_deep   = teacher-competence gate × r_deep, then D-optimal coverage
     deep_topr  = top-r_deep (no gate/coverage)
     proxy_topr = top-r_proxy
     token_topr = top-r_token
     random
  3. For each arm × seed: fresh Qwen3-0.6B, KD-train on subset, eval ROUGE-L on Dolly test.
  Report mean±std. Decision: RGA_deep / deep_topr must beat proxy_topr & random.

Reuses SaGD: load_teacher/load_student, InstructionDataset, collate_fn, evaluate_rouge,
CountSketchProjector.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SAGD = "/data/tianhao/SaGD"
sys.path.insert(0, os.path.join(SAGD, "src"))
sys.path.insert(0, os.path.join(SAGD, "scripts"))
from sagd.data import InstructionDataset, collate_fn
from sagd.models import load_teacher, load_student
from sagd.evaluation import evaluate_rouge
from gradient_pca_selection import CountSketchProjector


# ---------- relational + coverage (ported from CIFAR RGA) ----------
def double_center(K):
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()

def relational_r(G_S, G_T, anchor_idx):
    KS = G_S @ G_S[anchor_idx].T; KT = G_T @ G_T[anchor_idx].T
    KSc, KTc = double_center(KS), double_center(KT)
    A = KSc - KSc.mean(1, keepdim=True); B = KTc - KTc.mean(1, keepdim=True)
    corr = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-12)
    return 1.0 - corr

def greedy_logdet(K, k, ridge=1e-3):
    n = K.shape[0]; K = K + ridge * torch.eye(n, device=K.device)
    d2 = K.diag().clone(); c = torch.zeros(n, k, device=K.device); sel = []
    for it in range(k):
        dm = d2.clone()
        if sel: dm[sel] = -float('inf')
        j = int(torch.argmax(dm).item()); sel.append(j)
        if it < k - 1:
            cj = (K[:, j] - c[:, :it] @ c[j, :it]) / torch.sqrt(d2[j].clamp_min(1e-12))
            c[:, it] = cj; d2 = (d2 - cj ** 2).clamp_min(1e-12)
    return sel

def select_coverage(feats, k):
    Fn = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
    return greedy_logdet(Fn @ Fn.T, k)


def kd_kl_loss(z_S, z_T, labels_mask, T):
    zs = z_S[:, :-1, :]; zt = z_T[:, :-1, :]; m = labels_mask[:, 1:].float()
    logp_s = F.log_softmax(zs / T, dim=-1); p_t = F.softmax(zt / T, dim=-1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - logp_s)).sum(-1)
    return (T * T) * (kl * m).sum() / m.sum().clamp(min=1)


def resp_nll(z, ids, labels_mask):
    zt = z[:, :-1, :]; tgt = ids[:, 1:]; m = labels_mask[:, 1:].float()
    lp = F.log_softmax(zt, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return -(lp * m).sum() / m.sum().clamp(min=1)


# ---------- fingerprint extraction over the pool ----------
def extract(teacher, student, loader, N, proj_dim, temp, dev):
    student.train()
    head_shapes = {n: p.numel() for n, p in student.named_parameters() if n == 'lm_head.weight'}
    attn_shapes = {n: p.numel() for n, p in student.named_parameters()
                   if any(k in n for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))}
    proj_head = CountSketchProjector(head_shapes, proj_dim, seed=1, device=dev)
    proj_deep = CountSketchProjector(attn_shapes, proj_dim, seed=2, device=dev)
    V = student.config.vocab_size
    g = torch.Generator().manual_seed(3)
    tok_hash = torch.randint(0, proj_dim, (V,), generator=g).to(dev)
    tok_sign = (torch.randint(0, 2, (V,), generator=g).float() * 2 - 1).to(dev)
    teach_hash = teach_sign = None
    Gt, Gh, Gd, Gte, tnll, idxs = [], [], [], [], [], []
    done = 0; t0 = time.time()
    for batch in loader:
        if done >= N: break
        ids = batch['input_ids'].to(dev); am = batch['attention_mask'].to(dev); lm = batch['labels_mask'].to(dev)
        if lm[:, 1:].sum() < 1: continue
        with torch.no_grad():
            t_out = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True)
            z_T = t_out.logits.float(); h_T = t_out.hidden_states[-1].float()
            resp = lm.float().unsqueeze(-1)
            feat_T = (h_T * resp).sum(1) / resp.sum().clamp(min=1)
            tnll.append(resp_nll(z_T, ids, lm).item())
        if teach_hash is None:
            d_T = feat_T.shape[1]; gt = torch.Generator().manual_seed(4)
            teach_hash = torch.randint(0, proj_dim, (d_T,), generator=gt).to(dev)
            teach_sign = (torch.randint(0, 2, (d_T,), generator=gt).float() * 2 - 1).to(dev)
        tf = torch.zeros(proj_dim, device=dev); tf.scatter_add_(0, teach_hash, feat_T.squeeze(0) * teach_sign)
        Gte.append(tf.cpu())
        student.zero_grad(set_to_none=True)
        z_S = student(input_ids=ids, attention_mask=am).logits.float()
        loss = kd_kl_loss(z_S, z_T, lm, temp); loss.backward()
        with torch.no_grad():
            m = lm[:, 1:].float().unsqueeze(-1)
            resid = ((F.softmax(z_S[:, :-1, :] / temp, -1) - F.softmax(z_T[:, :-1, :] / temp, -1)) * m).sum(1).squeeze(0)
            tv = torch.zeros(proj_dim, device=dev); tv.scatter_add_(0, tok_hash, resid * tok_sign)
        Gt.append(tv.cpu()); Gh.append(proj_head.project(student)); Gd.append(proj_deep.project(student))
        idxs.append(int(batch['index'].item())); done += 1
        if done % 200 == 0: print(f"  extract [{done}/{N}] ({time.time()-t0:.0f}s)")
    return (torch.stack(Gt), torch.stack(Gh), torch.stack(Gd), torch.stack(Gte),
            torch.tensor(tnll), torch.tensor(idxs))


def kd_train(teacher, student_name, subset_ds, test_ds, tok, temp, seed, dev,
             epochs, bs, lr, max_eval):
    torch.manual_seed(seed); np.random.seed(seed)
    student, _ = load_student(student_name, device=dev)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    # micro-batch + grad-accum: keeps effective batch = bs but forwards <=4 samples at a time,
    # halving the peak of the large (B, L, vocab~152k) logit/softmax tensors (contended GPUs).
    micro = min(4, bs); accum = max(1, bs // micro)
    loader = DataLoader(subset_ds, batch_size=micro, shuffle=True, collate_fn=collate_fn)
    student.train()
    for ep in range(epochs):
        opt.zero_grad()
        for i, batch in enumerate(loader):
            ids = batch['input_ids'].to(dev); am = batch['attention_mask'].to(dev); lm = batch['labels_mask'].to(dev)
            with torch.no_grad():
                z_T = teacher(input_ids=ids, attention_mask=am).logits.float()
            z_S = student(input_ids=ids, attention_mask=am).logits.float()
            loss = kd_kl_loss(z_S, z_T, lm, temp) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()  # flush any remaining partial accumulation
    # free optimizer/gradients before eval to cut peak memory (shared/contended GPUs)
    del opt
    for p in student.parameters():
        p.grad = None
    torch.cuda.empty_cache()
    # eval ROUGE-L on test (test_ds already size-limited via max_samples; has get_metadata)
    student.eval()
    scores = evaluate_rouge(student, tok, test_ds, max_new_tokens=256, batch_size=8, device=dev)
    del student; torch.cuda.empty_cache()
    return scores['rouge_l_f']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='Qwen/Qwen3-8B')
    ap.add_argument('--student', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--teacher_ckpt', default=f'{SAGD}/data/teacher_sft_dolly_qwen.pt')
    ap.add_argument('--N_pool', type=int, default=4000)
    ap.add_argument('--budgets', default='500,1000,2000', help='comma-sep aggressive budgets to sweep')
    ap.add_argument('--proj_dim', type=int, default=2048)
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_eval', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache', default='pilot/_retrain_fp_cache.pt')
    ap.add_argument('--out', default='pilot/RETRAIN_QWEN_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time(); dev = args.device
    teacher, tok = load_teacher(args.teacher, device=dev, dtype=torch.float16,
                                ckpt_path=args.teacher_ckpt if os.path.exists(args.teacher_ckpt) else None)
    pool_ds = InstructionDataset(tok, max_seq_len=512, subset='train')
    test_ds = InstructionDataset(tok, max_seq_len=512, subset='test', max_samples=args.max_eval)
    N = min(args.N_pool, len(pool_ds))

    if os.path.exists(args.cache):
        print(f"[cache] loading fingerprints {args.cache}")
        c = torch.load(args.cache)
        Gt, Gh, Gd, Gte, tnll, idxs = c['Gt'], c['Gh'], c['Gd'], c['Gte'], c['tnll'], c['idxs']
    else:
        student, _ = load_student(args.student, device=dev); student.train()
        if getattr(student.config, 'tie_word_embeddings', False) or \
           (student.lm_head.weight.data_ptr() == student.get_input_embeddings().weight.data_ptr()):
            w = student.get_input_embeddings().weight.detach().clone()
            student.lm_head.weight = nn.Parameter(w); student.config.tie_word_embeddings = False
        loader = DataLoader(pool_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        Gt, Gh, Gd, Gte, tnll, idxs = extract(teacher, student, loader, N, args.proj_dim, args.temp, dev)
        torch.save({'Gt': Gt, 'Gh': Gh, 'Gd': Gd, 'Gte': Gte, 'tnll': tnll, 'idxs': idxs}, args.cache)
        del student; torch.cuda.empty_cache()
    print(f"[fingerprints] pool={Gt.shape[0]} ({time.time()-t0:.0f}s)")

    Gt, Gh, Gd, Gte = (X.to(dev) for X in (Gt, Gh, Gd, Gte))
    n = Gt.shape[0]; tnll = tnll.to(dev); idxs = idxs.cpu().numpy()
    rng = np.random.RandomState(0)
    anchor = torch.tensor(rng.choice(n, min(args.anchors, n), replace=False), device=dev)
    r_deep = relational_r(Gd, Gte, anchor); r_proxy = relational_r(Gh, Gte, anchor); r_token = relational_r(Gt, Gte, anchor)
    # teacher-competence gate: keep samples where teacher NLL below median (confident on gold)
    gate = tnll <= torch.median(tnll)
    idx_all = torch.arange(n, device=dev)

    def sel_topr(r, b): return idx_all[torch.argsort(r, descending=True)[:b]]
    def sel_random(seed, b):
        return torch.tensor(np.random.RandomState(seed).choice(n, b, replace=False), device=dev)
    def sel_rga(b):
        cand = idx_all[gate]; rc = r_deep[gate]
        poolk = min(len(cand), max(b * 3, b + 1))
        top = cand[torch.argsort(rc, descending=True)[:poolk]]
        cov = select_coverage((Gte - Gd)[top], min(b, len(top)))
        return top[torch.tensor(cov, device=dev)]

    # arms: dropped token_topr (decisively established worst); deep>proxy story kept via deep vs proxy
    arms = {'RGA_deep': lambda s, b: sel_rga(b), 'deep_topr': lambda s, b: sel_topr(r_deep, b),
            'proxy_topr': lambda s, b: sel_topr(r_proxy, b), 'random': lambda s, b: sel_random(s, b)}

    budgets = [min(int(x), n - 1) for x in str(args.budgets).split(',')]
    sweep = {}
    for b in budgets:
        sweep[str(b)] = {}
        print(f"===== budget b={b} ({b/n:.0%} of pool) =====")
        for arm, selfn in arms.items():
            rl = []
            for s in range(args.seeds):
                local = selfn(1000 + s, b).cpu().numpy()
                sub = Subset(pool_ds, [int(idxs[i]) for i in local])
                score = kd_train(teacher, args.student, sub, test_ds, tok, args.temp,
                                 1000 + s, dev, args.epochs, args.bs, args.lr, args.max_eval)
                rl.append(score)
                print(f"  [b{b} {arm:11s} seed{s}] rougeL={score:.4f}  ({time.time()-t0:.0f}s)")
            rl = np.array(rl)
            sweep[str(b)][arm] = {'rougeL_mean': float(rl.mean()), 'rougeL_std': float(rl.std()), 'n': args.seeds}
            print(f"  == b{b} {arm:11s} rougeL={rl.mean():.4f}±{rl.std():.4f}")
            # write incrementally so partial results survive interruption
            with open(args.out, 'w') as f:
                json.dump({'setup': {'teacher': args.teacher, 'student': args.student, 'N_pool': n,
                                     'budgets': budgets, 'proj_dim': args.proj_dim, 'temp': args.temp,
                                     'seeds': args.seeds, 'epochs': args.epochs, 'lr': args.lr,
                                     'max_eval': args.max_eval, 'r_deep_std': float(r_deep.std()),
                                     'r_proxy_std': float(r_proxy.std())},
                           'sweep': sweep, 'wall_clock_sec': time.time() - t0}, f, indent=2)
    print(json.dumps(sweep, indent=2)); print(f"[done] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
