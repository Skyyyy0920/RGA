#!/usr/bin/env python
"""
RGA-KD Phase A diagnostic (pilot scale: sklearn-digits MLP).

Self-contained: rebuilds the infrastructure that RGA_KD_PROJECT.md attributes to
`pilot/pilot_gradgeom.py` (absent on disk) — teacher/student construction, per-sample
KD gradient extraction, count-sketch projection, effective rank, d_optimal coverage
selection, and a kd_train retrain harness.

Answers §5.1 of RGA_KD_PROJECT.md and writes pilot/RGA_DIAG_RESULTS.json.

Core question: does the student/teacher *relational* gradient residual r(i) carry a
non-trivial, non-(KD-loss-reskin) training-value signal?

  g_S(x) = grad_theta_S KL(p_T(x) || p_S(x))         (student KD gradient)
  g_T(x) = J_T(x)  = jacobian of teacher logits w.r.t. teacher params  (teacher eNTK)
  K_S[i,j] = <Pi g_S(x_i), Pi g_S(a_j)>  ;  K_T analogous       (a_j = anchors)
  K' = double-center(K)
  r(i) = 1 - corr(K'_T[i,:], K'_S[i,:])

Signature ablations: {studentKD + teacherENTK} vs {both eNTK} vs {both KD-grad-feature}.
"""
import os, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad, jacrev
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


# --------------------------- models ---------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def make_student(seed, d_in=64, d_out=10, d_hidden=32):
    g = torch.Generator().manual_seed(seed)
    m = MLP(d_in, d_hidden, d_out)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5), generator=g)
        else:
            nn.init.zeros_(p)
    return m


# --------------------------- count-sketch projection ---------------------------
class CountSketch:
    """Sparse JL projection: maps any D-dim vector to d-dim. Memory O(D), not O(D*d)."""
    def __init__(self, D, d, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.h = torch.randint(0, d, (D,), generator=g).to(device)
        self.s = (torch.randint(0, 2, (D,), generator=g).float() * 2 - 1).to(device)
        self.d = d
        self.device = device

    def __call__(self, X):  # X: (n, D) -> (n, d)
        n = X.shape[0]
        out = torch.zeros(n, self.d, device=X.device, dtype=X.dtype)
        out.index_add_(1, self.h, X * self.s)
        return out


# --------------------------- gradient / jacobian feature extraction ---------------------------
def flatten_dict(d):
    """dict of (n, *pshape) -> (n, P) by flattening per-sample."""
    parts = [v.reshape(v.shape[0], -1) for v in d.values()]
    return torch.cat(parts, dim=1)


def student_kd_grads(student, X, PT, T, device, chunk=256):
    """Per-sample g_S(x) = grad_theta_S KL(p_T || p_S).  Returns (n, P_S)."""
    params = {k: v.detach() for k, v in student.named_parameters()}

    def kd_ce(params, x, pt):  # x:(D,) pt:(C,) teacher probs at temperature T
        z = functional_call(student, params, (x.unsqueeze(0),)).squeeze(0)
        logp = F.log_softmax(z / T, dim=-1)
        return -(pt * logp).sum()  # grad equals grad of KL(p_T||p_S) w.r.t. theta_S

    gfun = vmap(grad(kd_ce), in_dims=(None, 0, 0))
    outs = []
    for i in range(0, X.shape[0], chunk):
        gd = gfun(params, X[i:i+chunk], PT[i:i+chunk])
        outs.append(flatten_dict(gd).detach())
    return torch.cat(outs, 0)


def entk_jac(model, X, device, chunk=128):
    """Per-sample full Jacobian J(x) of logits w.r.t. params (eNTK feature). Returns (n, C*P)."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def logits_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)  # (C,)

    jfun = vmap(jacrev(logits_single), in_dims=(None, 0))
    outs = []
    for i in range(0, X.shape[0], chunk):
        jd = jfun(params, X[i:i+chunk])  # dict each (n, C, *pshape)
        flat = torch.cat([v.reshape(v.shape[0], v.shape[1], -1) for v in jd.values()], dim=2)
        outs.append(flat.reshape(flat.shape[0], -1).detach())  # (n, C*P)
    return torch.cat(outs, 0)


def teacher_kd_feature(teacher, X, PS, PT, device, chunk=128):
    """Teacher 'KD-grad feature' = J_T(x)^T (p_S - p_T): teacher jacobian contracted with KD residual.
    Returns (n, P_T)."""
    params = {k: v.detach() for k, v in teacher.named_parameters()}

    def logits_single(params, x):
        return functional_call(teacher, params, (x.unsqueeze(0),)).squeeze(0)

    jfun = vmap(jacrev(logits_single), in_dims=(None, 0))
    rho = (PS - PT)  # (n, C)
    outs = []
    for i in range(0, X.shape[0], chunk):
        jd = jfun(params, X[i:i+chunk])
        flat = torch.cat([v.reshape(v.shape[0], v.shape[1], -1) for v in jd.values()], dim=2)  # (n,C,P)
        contracted = torch.einsum('ncp,nc->np', flat, rho[i:i+chunk])
        outs.append(contracted.detach())
    return torch.cat(outs, 0)


# --------------------------- relational machinery ---------------------------
def double_center(K):  # K: (n, m)
    rm = K.mean(dim=1, keepdim=True)
    cm = K.mean(dim=0, keepdim=True)
    gm = K.mean()
    return K - rm - cm + gm


def relational_residual(GS, GT, anchor_idx):
    """GS, GT: (n, d) projected signatures. Returns r(i) (n,), K'_S, K'_T (n,m), cka."""
    AS = GS[anchor_idx]            # (m, d)
    AT = GT[anchor_idx]
    KS = GS @ AS.T                 # (n, m)
    KT = GT @ AT.T
    KSc = double_center(KS)
    KTc = double_center(KT)
    # per-row correlation across anchors
    def rowcorr(A, B):
        A = A - A.mean(dim=1, keepdim=True)
        B = B - B.mean(dim=1, keepdim=True)
        num = (A * B).sum(dim=1)
        den = A.norm(dim=1) * B.norm(dim=1) + 1e-12
        return num / den
    corr = rowcorr(KSc, KTc)
    r = 1.0 - corr
    cka = linear_cka(KSc, KTc)
    return r, KSc, KTc, cka


def linear_cka(X, Y):
    """Linear CKA between two (n,p) centered-ish representations."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).norm() ** 2
    nx = (X.T @ X).norm()
    ny = (Y.T @ Y).norm()
    return (hsic / (nx * ny + 1e-12)).item()


def effective_rank(G):
    """Return (participation-ratio eff-rank, 90%-energy rank) of (n,d) matrix.
    Participation ratio = exp(entropy of normalized singular values).
    Energy rank = #singular values to reach 90% of squared-energy (matches frozen pilot's ~23)."""
    s = torch.linalg.svdvals(G.float())
    s = s[s > 1e-9]
    p = s / s.sum()
    H = -(p * (p + 1e-12).log()).sum()
    part = float(torch.exp(H).item())
    e = (s ** 2)
    ce = torch.cumsum(e, 0) / e.sum()
    energy90 = int((ce < 0.90).sum().item()) + 1
    return part, energy90


# --------------------------- selection methods ---------------------------
def greedy_logdet(K, k, ridge=1e-3):
    """Greedy D-optimal / DPP-MAP: pick k rows maximizing log det of selected Gram submatrix.
    K: (n, n) PSD kernel. Returns list of indices (coverage / max-volume selection)."""
    n = K.shape[0]
    K = K + ridge * torch.eye(n, device=K.device)
    diag = K.diag().clone()
    selected = []
    # incremental Cholesky-style gains (DPP greedy, O(n k^2))
    c = torch.zeros(n, k, device=K.device)
    d2 = diag.clone()
    for it in range(k):
        d2_masked = d2.clone()
        if selected:
            d2_masked[selected] = -float('inf')
        j = int(torch.argmax(d2_masked).item())
        selected.append(j)
        if it < k - 1:
            num = K[:, j] - c[:, :it] @ c[j, :it]
            cj = num / torch.sqrt(d2[j].clamp_min(1e-12))
            c[:, it] = cj
            d2 = d2 - cj ** 2
            d2 = d2.clamp_min(1e-12)
    return selected


def select_coverage(features, k, ridge=1e-3):
    """Coverage selection over feature rows via greedy log-det on linear kernel."""
    F_ = features / (features.norm(dim=1, keepdim=True) + 1e-12)
    K = F_ @ F_.T
    return greedy_logdet(K, k, ridge)


# --------------------------- retrain harness ---------------------------
def kd_train(teacher, coreset_X, coreset_PT, test_X, test_y, T, seed, device,
             epochs=120, lr=3e-3):
    """Train a fresh student from scratch on the coreset with pure KD loss; return test acc."""
    torch.manual_seed(seed)
    student = make_student(seed + 12345).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for ep in range(epochs):
        student.train()
        z = student(coreset_X)
        logp = F.log_softmax(z / T, dim=-1)
        loss = (T * T) * F.kl_div(logp, coreset_PT, reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
    student.eval()
    with torch.no_grad():
        pred = student(test_X).argmax(1)
        acc = (pred == test_y).float().mean().item()
    return acc


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=1200)
    ap.add_argument('--proj_dim', type=int, default=2048)
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=4.0)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--teacher_hidden', type=int, default=128)
    ap.add_argument('--student_hidden', type=int, default=32)
    ap.add_argument('--warmup_epochs', type=int, default=8)
    ap.add_argument('--out', default='pilot/RGA_DIAG_RESULTS.json')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    print(f"[env] device={device} ({gpu_name})  torch={torch.__version__}")

    torch.manual_seed(0); np.random.seed(0)

    # ----- data -----
    digits = load_digits()
    Xall = StandardScaler().fit_transform(digits.data).astype('float32')
    yall = digits.target.astype('int64')
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(Xall))
    n_test = 360
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    Xtr = torch.tensor(Xall[train_idx], device=device)
    ytr = torch.tensor(yall[train_idx], device=device)
    Xte = torch.tensor(Xall[test_idx], device=device)
    yte = torch.tensor(yall[test_idx], device=device)
    # candidate pool for selection = first N of train
    N = min(args.N, Xtr.shape[0])
    Xpool = Xtr[:N]; ypool = ytr[:N]
    print(f"[data] train={Xtr.shape[0]} pool={N} test={Xte.shape[0]}")

    # ----- train teacher -----
    teacher = make_student(7, d_hidden=args.teacher_hidden).to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=3e-3)
    for ep in range(200):
        teacher.train()
        loss = F.cross_entropy(teacher(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    teacher.eval()
    with torch.no_grad():
        tr_acc = (teacher(Xtr).argmax(1) == ytr).float().mean().item()
        te_acc = (teacher(Xte).argmax(1) == yte).float().mean().item()
        ZT_pool = teacher(Xpool)
        PT_pool = F.softmax(ZT_pool / args.temp, dim=-1).detach()
        PT_pool_hard = ZT_pool.argmax(1)
    teacher_correct = (PT_pool_hard == ypool).float()  # (N,)
    print(f"[teacher] train_acc={tr_acc:.3f} test_acc={te_acc:.3f} "
          f"pool_correct={teacher_correct.mean().item():.3f}")

    # teacher soft targets for the whole training set (for kd_train coresets)
    with torch.no_grad():
        PT_tr = F.softmax(teacher(Xtr) / args.temp, dim=-1).detach()

    # ----- warm up student (Step 0) -----
    student = make_student(1, d_hidden=args.student_hidden).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    for ep in range(args.warmup_epochs):
        student.train()
        logp = F.log_softmax(student(Xpool) / args.temp, dim=-1)
        loss = (args.temp ** 2) * F.kl_div(logp, PT_pool, reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
    student.eval()
    with torch.no_grad():
        ZS_pool = student(Xpool)
        PS_pool = F.softmax(ZS_pool / args.temp, dim=-1).detach()
        warm_acc = (ZS_pool.argmax(1) == ypool).float().mean().item()
        # per-sample KD loss at warmed student
        logp = F.log_softmax(ZS_pool / args.temp, dim=-1)
        kd_loss_i = (PT_pool * (PT_pool.clamp_min(1e-12).log() - logp)).sum(1)  # KL per sample
    print(f"[student warmup] {args.warmup_epochs} ep, pool_acc={warm_acc:.3f}")

    # ----- feature extraction -----
    print("[features] extracting student KD grad, student eNTK, teacher eNTK, teacher KD-feature ...")
    gS = student_kd_grads(student, Xpool, PT_pool, args.temp, device)      # (N, P_S)
    grad_norm_i = gS.norm(dim=1)                                           # true (pre-proj) KD grad norm
    JS = entk_jac(student, Xpool, device)                                  # (N, C*P_S)
    JT = entk_jac(teacher, Xpool, device)                                  # (N, C*P_T)
    Tkd = teacher_kd_feature(teacher, Xpool, PS_pool, PT_pool, device)     # (N, P_T)
    print(f"  dims: gS={tuple(gS.shape)} JS={tuple(JS.shape)} JT={tuple(JT.shape)} Tkd={tuple(Tkd.shape)}")

    # ----- signatures: project, build relational matrices, residual, correlations -----
    def proj(feat, seed):
        cs = CountSketch(feat.shape[1], args.proj_dim, seed, device)
        return cs(feat)

    anchor_idx = torch.tensor(rng.choice(N, size=min(args.anchors, N), replace=False), device=device)

    signatures = {
        'studentKD_teacherENTK': (gS, JT),
        'both_entk':             (JS, JT),
        'both_KDfeature':        (gS, Tkd),
    }

    def corr_np(a, b):
        a = a.detach().cpu().numpy(); b = b.detach().cpu().numpy()
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    noise = torch.tensor(rng.randn(N).astype('float32'), device=device)
    sig_results = {}
    GS_proj_primary = None
    r_primary = None
    for name, (Sf, Tf) in signatures.items():
        GS = proj(Sf, seed=101)
        GT = proj(Tf, seed=202)
        r, KSc, KTc, cka = relational_residual(GS, GT, anchor_idx)
        res = {
            'cka': cka,
            'r_mean': float(r.mean().item()),
            'r_std': float(r.std().item()),
            'r_min': float(r.min().item()),
            'r_max': float(r.max().item()),
            'corr_r_kdloss': corr_np(r, kd_loss_i),
            'corr_r_gradnorm': corr_np(r, grad_norm_i),
            'corr_r_teachercorrect': corr_np(r, teacher_correct),
            'corr_r_noise': corr_np(r, noise),  # structure sanity: should be ~0
        }
        sig_results[name] = res
        print(f"  [{name}] cka={cka:.3f} r={res['r_mean']:.3f}±{res['r_std']:.3f} "
              f"corr(r,kdloss)={res['corr_r_kdloss']:.3f} corr(r,tcorrect)={res['corr_r_teachercorrect']:.3f} "
              f"corr(r,noise)={res['corr_r_noise']:.3f}")
        if name == 'studentKD_teacherENTK':
            GS_proj_primary = GS
            r_primary = r

    # effective rank of projected student KD grads (budget calibration)
    R_part, R_energy = effective_rank(GS_proj_primary)
    R = R_energy  # use 90%-energy rank as the gradient-rank scale for aggressive budgets
    print(f"[eff-rank] participation={R_part:.1f}  energy90={R_energy}  (budget scale R={R})")

    # ----- retrain falsification (5 seeds) at aggressive budgets -----
    Rr = max(8, int(round(R)))
    cap = N // 4  # keep coresets a genuine aggressive sub-selection (<=25% of pool)
    budgets = {'1R': min(Rr, cap), '2R': min(2 * Rr, cap), '4R': min(4 * Rr, cap)}
    # drop degenerate duplicate budgets
    seen = set(); budgets = {k: v for k, v in budgets.items() if not (v in seen or seen.add(v))}
    print(f"[retrain] eff-rank scale R={Rr}, cap={cap}, budgets: {budgets}")

    gate = teacher_correct > 0.5  # teacher correctness gate
    r_np = r_primary
    rel_resid_feat = (proj(JT, 303) - GS_proj_primary)  # per-sample relational residual feature for coverage
    # NOTE: coverage feature for RGA = residual between teacher-relational and student-relational signatures

    def build_coreset(method, budget):
        idx_all = torch.arange(N, device=device)
        budget = min(budget, N - 1)
        if method == 'random':
            sel = rng.choice(N, size=budget, replace=False)
        elif method == 'hard_highloss':
            sel = torch.topk(kd_loss_i, budget).indices.cpu().numpy()
        elif method == 'd_optimal_grad':  # GradSpan-KD body: coverage over KD-grad subspace
            sel = select_coverage(GS_proj_primary, budget)
            sel = idx_all[sel].cpu().numpy() if isinstance(sel, list) else sel
        elif method == 'select_low_r':    # falsification arm: aligned samples (gated)
            cand = idx_all[gate]
            rc = r_np[gate]
            order = torch.argsort(rc)[:budget]  # lowest r
            sel = cand[order].cpu().numpy()
        elif method == 'select_high_r':   # RGA-KD: gate + high r, then coverage over residual feat
            cand = idx_all[gate]
            rc = r_np[gate]
            poolk = min(len(cand), max(budget * 3, budget + 1))
            top = torch.argsort(rc, descending=True)[:poolk]
            cand_top = cand[top]
            feats = rel_resid_feat[cand_top]
            cov = select_coverage(feats, min(budget, len(cand_top)))
            sel = cand_top[torch.tensor(cov, device=device)].cpu().numpy()
        else:
            raise ValueError(method)
        sel = np.array(sel).astype(int)[:budget]
        return torch.tensor(sel, device=device)

    methods = ['select_high_r', 'select_low_r', 'random', 'hard_highloss', 'd_optimal_grad']
    retrain = {}
    for bname, budget in budgets.items():
        retrain[bname] = {}
        for method in methods:
            accs = []
            for s in range(args.seeds):
                # rebuild coreset per seed only for stochastic methods
                sel = build_coreset(method, budget)
                cs_X = Xpool[sel]; cs_PT = PT_pool[sel]
                acc = kd_train(teacher, cs_X, cs_PT, Xte, yte, args.temp, seed=1000 + s, device=device)
                accs.append(acc)
            accs = np.array(accs)
            retrain[bname][method] = {'mean': float(accs.mean()), 'std': float(accs.std()),
                                      'n': int(args.seeds), 'budget': int(budget)}
            print(f"  [{bname}] {method:16s} acc={accs.mean():.4f}±{accs.std():.4f}")

    wall = time.time() - t0
    out = {
        'setup': {
            'dataset': 'sklearn-digits', 'N': N, 'proj_dim': args.proj_dim,
            'anchors': int(anchor_idx.shape[0]), 'temp': args.temp,
            'teacher_hidden': args.teacher_hidden, 'student_hidden': args.student_hidden,
            'warmup_epochs': args.warmup_epochs, 'seeds': args.seeds,
            'teacher_train_acc': tr_acc, 'teacher_test_acc': te_acc,
            'student_warm_acc': warm_acc, 'eff_rank_R': R,
            'device': str(device), 'gpu': gpu_name,
        },
        'signatures': sig_results,
        'retrain': retrain,
        'wall_clock_sec': wall,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out}  ({wall:.1f}s)")


if __name__ == '__main__':
    main()
