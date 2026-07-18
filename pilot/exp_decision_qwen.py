#!/usr/bin/env python
"""
DECISION experiment (advisor's challenge): on Qwen KD, is the relational residual r
*essentially token-level* (as the last-layer proxy (p_S-p_T)⊗h factorizes into
token-residual ⊙ feature), or does a GENUINE deep parameter gradient rank samples
differently?

For each Dolly sample we do ONE per-sample KD backward on the student and read three
fingerprints from different parameter groups of the SAME backward:
  token_only : aggregated (p_S - p_T) over response tokens        (dim V)   — pure token space
  proxy_last : grad of lm_head.weight  = (p_S-p_T) ⊗ h            (V×d)     — factorizing proxy
  deep_grad  : grad of attention q/k/v/o across ALL layers                   — genuine deep gradient
Teacher relational partner = mean last-hidden-state h_T (held FIXED across all three).

Then r_token, r_proxy, r_deep are computed identically (same teacher K_T, same anchors),
differing ONLY through the student fingerprint. We report:
  - Spearman(r_proxy, r_deep):  ~1.0 => advisor RIGHT (deep adds nothing over token×feature)
                                 moderate/low => deep gradient genuinely re-ranks => B justified
  - Spearman(r_token, r_proxy): does the feature even add anything over pure token?
  - CKA between the Gram matrices (matrix-level redundancy check).

Reuses SaGD infra: load_teacher/load_student, InstructionDataset, collate_fn, CountSketchProjector.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SAGD = "/data/tianhao/SaGD"
sys.path.insert(0, os.path.join(SAGD, "src"))
sys.path.insert(0, os.path.join(SAGD, "scripts"))
from sagd.data import InstructionDataset, collate_fn
from sagd.models import load_teacher, load_student
from gradient_pca_selection import CountSketchProjector


# ----------------- relational machinery (same as CIFAR RGA) -----------------
def double_center(K):
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-12)).item()


def relational_r(G_S, G_T, anchor_idx):
    """Per-sample relational residual r(i)=1-corr(K'_T[i],K'_S[i]). G_*: (n,d) fingerprints."""
    KS = G_S @ G_S[anchor_idx].T
    KT = G_T @ G_T[anchor_idx].T
    KSc, KTc = double_center(KS), double_center(KT)
    A = KSc - KSc.mean(1, keepdim=True); B = KTc - KTc.mean(1, keepdim=True)
    corr = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-12)
    return (1.0 - corr), KSc, KTc


def spearman(a, b):
    a = a.detach().cpu().numpy(); b = b.detach().cpu().numpy()
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a = a.detach().cpu().numpy(); b = b.detach().cpu().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def kd_kl_loss(z_S, z_T, labels_mask, T):
    """Standard KD forward-KL, shift-aligned, response-masked, ×T^2. z_*: (1,L,V)."""
    zs = z_S[:, :-1, :]; zt = z_T[:, :-1, :]
    m = labels_mask[:, 1:].float()
    logp_s = F.log_softmax(zs / T, dim=-1)
    p_t = F.softmax(zt / T, dim=-1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - logp_s)).sum(-1)  # (1,L-1)
    return (T * T) * (kl * m).sum() / m.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='Qwen/Qwen3-8B')
    ap.add_argument('--student', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--teacher_ckpt', default=f'{SAGD}/data/teacher_sft_dolly_qwen.pt')
    ap.add_argument('--N', type=int, default=1500)
    ap.add_argument('--proj_dim', type=int, default=2048)
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--max_seq_len', type=int, default=512)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='pilot/DECISION_QWEN_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time(); dev = args.device
    print(f"[env] torch={torch.__version__} dev={dev}")

    # ---- models ----
    print("[load] teacher (SFT Qwen3-8B) ...")
    teacher, tok = load_teacher(args.teacher, device=dev, dtype=torch.float16,
                                ckpt_path=args.teacher_ckpt if os.path.exists(args.teacher_ckpt) else None)
    print("[load] student (Qwen3-0.6B) ...")
    student, _ = load_student(args.student, device=dev)
    student.train()

    # untie lm_head so its .grad is purely the output-head (last-layer) gradient
    if getattr(student.config, 'tie_word_embeddings', False) or \
       (student.lm_head.weight.data_ptr() == student.get_input_embeddings().weight.data_ptr()):
        w = student.get_input_embeddings().weight.detach().clone()
        student.lm_head.weight = nn.Parameter(w)
        student.config.tie_word_embeddings = False
        print("[untie] lm_head untied from embeddings")

    # ---- data ----
    ds = InstructionDataset(tok, max_seq_len=args.max_seq_len, subset='train')
    N = min(args.N, len(ds))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"[data] dolly train={len(ds)} using N={N}")

    # ---- projectors (three parameter groups) ----
    head_shapes = {n: p.numel() for n, p in student.named_parameters() if n == 'lm_head.weight'}
    attn_shapes = {n: p.numel() for n, p in student.named_parameters()
                   if any(k in n for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))}
    print(f"[params] lm_head={sum(head_shapes.values())/1e6:.1f}M  "
          f"attn(qkvo across layers)={sum(attn_shapes.values())/1e6:.1f}M  "
          f"({len(attn_shapes)} tensors)")
    proj_head = CountSketchProjector(head_shapes, args.proj_dim, seed=1, device=dev)
    proj_deep = CountSketchProjector(attn_shapes, args.proj_dim, seed=2, device=dev)
    V = student.config.vocab_size
    # token-only projector: fixed random Rademacher-ish sketch V->proj_dim
    g = torch.Generator().manual_seed(3)
    tok_hash = torch.randint(0, args.proj_dim, (V,), generator=g).to(dev)
    tok_sign = (torch.randint(0, 2, (V,), generator=g).float() * 2 - 1).to(dev)

    G_tok, G_head, G_deep, G_teach = [], [], [], []
    kd_losses = []
    n_done = 0
    for batch in loader:
        if n_done >= N:
            break
        ids = batch['input_ids'].to(dev)
        am = batch['attention_mask'].to(dev)
        lm = batch['labels_mask'].to(dev)
        if lm[:, 1:].sum() < 1:
            continue  # no response tokens

        # ---- teacher forward (no grad): p_T + mean last hidden over response ----
        with torch.no_grad():
            t_out = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True)
            z_T = t_out.logits.float()
            h_T = t_out.hidden_states[-1].float()          # (1,L,d_T)
            resp = lm.float().unsqueeze(-1)                # (1,L,1)
            feat_T = (h_T * resp).sum(1) / resp.sum().clamp(min=1)  # (1,d_T) mean over response
        G_teach.append((tok_sign.new_zeros(0),))  # placeholder, filled below via proj
        # project teacher feature to proj_dim with its own sketch (dim d_T)
        # (build lazily once we know d_T)
        if len(G_teach) == 1:
            d_T = feat_T.shape[1]
            gt = torch.Generator().manual_seed(4)
            teach_hash = torch.randint(0, args.proj_dim, (d_T,), generator=gt).to(dev)
            teach_sign = (torch.randint(0, 2, (d_T,), generator=gt).float() * 2 - 1).to(dev)
        tf = torch.zeros(args.proj_dim, device=dev)
        tf.scatter_add_(0, teach_hash, feat_T.squeeze(0) * teach_sign)
        G_teach[-1] = tf.cpu()

        # ---- student forward+backward: KD loss ----
        student.zero_grad(set_to_none=True)
        z_S = student(input_ids=ids, attention_mask=am).logits.float()  # (1,L,V)
        loss = kd_kl_loss(z_S, z_T, lm, args.temp)
        loss.backward()
        kd_losses.append(loss.item())

        # token-only fingerprint: aggregated (p_S - p_T) over response tokens
        with torch.no_grad():
            m = lm[:, 1:].float().unsqueeze(-1)
            p_s = F.softmax(z_S[:, :-1, :] / args.temp, dim=-1)
            p_t = F.softmax(z_T[:, :-1, :] / args.temp, dim=-1)
            resid = ((p_s - p_t) * m).sum(1).squeeze(0)     # (V,)
            tv = torch.zeros(args.proj_dim, device=dev)
            tv.scatter_add_(0, tok_hash, resid * tok_sign)
        G_tok.append(tv.cpu())
        G_head.append(proj_head.project(student))   # lm_head.weight grad
        G_deep.append(proj_deep.project(student))    # attn qkvo grad

        n_done += 1
        if n_done % 100 == 0:
            print(f"  [{n_done}/{N}] loss={loss.item():.3f}  ({time.time()-t0:.0f}s)")

    # ---- stack ----
    G_tok = torch.stack(G_tok).to(dev)
    G_head = torch.stack(G_head).to(dev)
    G_deep = torch.stack(G_deep).to(dev)
    G_teach = torch.stack(G_teach).to(dev)
    n = G_tok.shape[0]
    print(f"[done extract] n={n}  ({time.time()-t0:.0f}s)")

    rng = np.random.RandomState(0)
    anchor_idx = torch.tensor(rng.choice(n, min(args.anchors, n), replace=False), device=dev)

    r_tok, KStok, KT = relational_r(G_tok, G_teach, anchor_idx)
    r_head, KShead, _ = relational_r(G_head, G_teach, anchor_idx)
    r_deep, KSdeep, _ = relational_r(G_deep, G_teach, anchor_idx)

    # fingerprint-level Gram similarity (does deep Gram ≈ proxy Gram?)
    def gram(G):
        Gn = G / (G.norm(dim=1, keepdim=True) + 1e-12)
        return Gn @ Gn.T
    cka_head_deep = linear_cka(gram(G_head), gram(G_deep))
    cka_tok_head = linear_cka(gram(G_tok), gram(G_head))
    cka_tok_deep = linear_cka(gram(G_tok), gram(G_deep))

    res = {
        'setup': {'teacher': args.teacher, 'student': args.student, 'N': n,
                  'proj_dim': args.proj_dim, 'anchors': int(anchor_idx.shape[0]),
                  'temp': args.temp, 'used_sft_teacher': os.path.exists(args.teacher_ckpt),
                  'lm_head_params_M': sum(head_shapes.values()) / 1e6,
                  'attn_params_M': sum(attn_shapes.values()) / 1e6},
        'r_stats': {
            'r_token':  {'mean': float(r_tok.mean()),  'std': float(r_tok.std())},
            'r_proxy':  {'mean': float(r_head.mean()), 'std': float(r_head.std())},
            'r_deep':   {'mean': float(r_deep.mean()), 'std': float(r_deep.std())},
        },
        'DECISION_spearman': {
            'r_proxy_vs_r_deep': spearman(r_head, r_deep),   # <== headline
            'r_token_vs_r_proxy': spearman(r_tok, r_head),
            'r_token_vs_r_deep': spearman(r_tok, r_deep),
        },
        'DECISION_pearson': {
            'r_proxy_vs_r_deep': pearson(r_head, r_deep),
            'r_token_vs_r_proxy': pearson(r_tok, r_head),
        },
        'gram_cka': {
            'proxy_vs_deep': cka_head_deep,   # 1.0 => deep gradient == last-layer proxy
            'token_vs_proxy': cka_tok_head,
            'token_vs_deep': cka_tok_deep,
        },
        'topk_overlap@10pct': {
            'proxy_vs_deep': topk_overlap(r_head, r_deep, 0.10),
            'token_vs_deep': topk_overlap(r_tok, r_deep, 0.10),
        },
        'wall_clock_sec': time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res['DECISION_spearman'], indent=2))
    print(json.dumps(res['gram_cka'], indent=2))
    print(f"[done] wrote {args.out} ({time.time()-t0:.0f}s)")


def topk_overlap(a, b, frac):
    k = max(1, int(len(a) * frac))
    ta = set(torch.topk(a, k).indices.cpu().numpy().tolist())
    tb = set(torch.topk(b, k).indices.cpu().numpy().tolist())
    return len(ta & tb) / k


if __name__ == '__main__':
    main()
