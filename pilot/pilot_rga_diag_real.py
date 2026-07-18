#!/usr/bin/env python
"""
RGA-KD Phase A real-pair recheck (§5.3): ResNet-56 -> ResNet-20 / CIFAR-100.

Purpose (decisive, see RGA_KD_PROJECT.md §5.2 gate verdict): the digits pilot left two
things unresolved that ONLY a real, imperfect-teacher CNN can settle:
  (5) gate necessity  -> digits teacher was 100% on the pool (gate inert). A ResNet-56
      teacher on CIFAR-100 (~70%) makes argmax p_T != y non-trivial, so corr(r,teacher_correct)
      becomes measurable.
  (4) RGA vs GradSpan-KD ordering on a real conv net (lazy/eNTK assumption risk).

Cheap-gradient proxy (sanctioned by §7.6/E5 "last-layer"): full per-sample eNTK Jacobians
for a ResNet are infeasible (C backward passes/sample). We use LAST-LAYER gradients in
closed form. With logits z = W f(x) + b:
  student KD grad (last layer)   g_S(x) = (p_S - p_T) ⊗ f_S(x)            dim C*d_S
  teacher last-layer eNTK        J_T(x) = e_c ⊗ f_T(x)  ->  relational inner product
       <J_T(x_i),J_T(a_j)>_F = C * <f_T(x_i), f_T(a_j)>, so signature := f_T(x)   dim d_T
  teacher KD-feature             (p_S - p_T) ⊗ f_T(x)                     dim C*d_T

Writes pilot/RGA_DIAG_REAL_RESULTS.json. Same relational machinery as pilot_rga_diag.py.
"""
import os, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TT


# --------------------------- CIFAR ResNet (He et al.) ---------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, n, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make(16, 16, n, 1)
        self.layer2 = self._make(16, 32, n, 2)
        self.layer3 = self._make(32, 64, n, 2)
        self.fc = nn.Linear(64, num_classes)
        self.feat_dim = 64

    def _make(self, cin, cout, n, stride):
        layers = [BasicBlock(cin, cout, stride)]
        for _ in range(n - 1):
            layers.append(BasicBlock(cout, cout))
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out  # (n, 64)

    def forward(self, x):
        return self.fc(self.features(x))


def resnet56(nc=100): return CifarResNet(9, nc)
def resnet20(nc=100): return CifarResNet(3, nc)


# --------------------------- count-sketch / relational (same as pilot) ---------------------------
class CountSketch:
    def __init__(self, D, d, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.h = torch.randint(0, d, (D,), generator=g).to(device)
        self.s = (torch.randint(0, 2, (D,), generator=g).float() * 2 - 1).to(device)
        self.d = d

    def __call__(self, X):
        out = torch.zeros(X.shape[0], self.d, device=X.device, dtype=X.dtype)
        out.index_add_(1, self.h, X * self.s)
        return out


def double_center(K):
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-12)).item()


def relational_residual(GS, GT, anchor_idx):
    KS = GS @ GS[anchor_idx].T
    KT = GT @ GT[anchor_idx].T
    KSc, KTc = double_center(KS), double_center(KT)
    A = KSc - KSc.mean(1, keepdim=True); B = KTc - KTc.mean(1, keepdim=True)
    corr = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-12)
    return 1.0 - corr, KSc, KTc, linear_cka(KSc, KTc)


def effective_rank(G):
    s = torch.linalg.svdvals(G.float()); s = s[s > 1e-9]
    p = s / s.sum(); H = -(p * (p + 1e-12).log()).sum()
    e = s ** 2; ce = torch.cumsum(e, 0) / e.sum()
    return float(torch.exp(H)), int((ce < 0.90).sum().item()) + 1


def greedy_logdet(K, k, ridge=1e-3):
    n = K.shape[0]; K = K + ridge * torch.eye(n, device=K.device)
    d2 = K.diag().clone(); c = torch.zeros(n, k, device=K.device); selected = []
    for it in range(k):
        dm = d2.clone()
        if selected: dm[selected] = -float('inf')
        j = int(torch.argmax(dm).item()); selected.append(j)
        if it < k - 1:
            cj = (K[:, j] - c[:, :it] @ c[j, :it]) / torch.sqrt(d2[j].clamp_min(1e-12))
            c[:, it] = cj; d2 = (d2 - cj ** 2).clamp_min(1e-12)
    return selected


def select_coverage(features, k, ridge=1e-3):
    Fn = features / (features.norm(dim=1, keepdim=True) + 1e-12)
    return greedy_logdet(Fn @ Fn.T, k, ridge)


# --------------------------- data ---------------------------
MEAN = (0.5071, 0.4865, 0.4409); STD = (0.2673, 0.2564, 0.2762)


def load_cifar100(device):
    norm = TT.Normalize(MEAN, STD)
    tf = TT.Compose([TT.ToTensor(), norm])
    tr = torchvision.datasets.CIFAR100('data', train=True, download=False)
    te = torchvision.datasets.CIFAR100('data', train=False, download=False)
    def to_tensor(ds):
        X = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 255.0
        for c in range(3): X[:, c] = (X[:, c] - MEAN[c]) / STD[c]
        y = torch.tensor(ds.targets)
        return X, y
    Xtr, ytr = to_tensor(tr); Xte, yte = to_tensor(te)
    return Xtr, ytr, Xte, yte


def gpu_augment(x):  # random crop (pad4) + horizontal flip, on normalized tensors
    n = x.shape[0]
    if torch.rand(1).item() < 0.5:
        pass
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    pad = F.pad(x, (4, 4, 4, 4), mode='reflect')
    i = torch.randint(0, 9, (1,)).item(); j = torch.randint(0, 9, (1,)).item()
    return pad[:, :, i:i + 32, j:j + 32]


# --------------------------- train / eval ---------------------------
def train_teacher(Xtr, ytr, Xte, yte, device, epochs, bs, ckpt):
    if os.path.exists(ckpt):
        m = resnet56().to(device); m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
        with torch.no_grad():
            acc = evaluate(m, Xte, yte, device)
        print(f"[teacher] loaded {ckpt}  test_acc={acc:.3f}")
        return m, acc
    m = resnet56().to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Xtr.shape[0]
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = gpu_augment(Xtr[idx]); yb = ytr[idx]
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = evaluate(m, Xte, yte, device)
            print(f"  [teacher ep{ep+1}] test_acc={acc:.3f}")
    torch.save(m.state_dict(), ckpt); m.eval()
    return m, evaluate(m, Xte, yte, device)


@torch.no_grad()
def evaluate(m, X, y, device, bs=1000):
    m.eval(); correct = 0
    for i in range(0, X.shape[0], bs):
        pred = m(X[i:i + bs].to(device)).argmax(1)
        correct += (pred == y[i:i + bs].to(device)).sum().item()
    return correct / X.shape[0]


@torch.no_grad()
def get_feat_logits(m, X, device, bs=1000):
    m.eval(); fs, zs = [], []
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        f = m.features(xb); z = m.fc(f)
        fs.append(f); zs.append(z)
    return torch.cat(fs), torch.cat(zs)


def kd_train_student(teacher_PT, coreset_X, coreset_PT, Xte, yte, T, seed, device,
                     epochs, bs, nc=100):
    torch.manual_seed(seed); np.random.seed(seed)
    s = resnet20(nc).to(device)
    opt = torch.optim.SGD(s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = coreset_X.shape[0]
    bs = min(bs, n)
    for ep in range(epochs):
        s.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = gpu_augment(coreset_X[idx]); pt = coreset_PT[idx]
            logp = F.log_softmax(s(xb) / T, dim=-1)
            loss = (T * T) * F.kl_div(logp, pt, reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return evaluate(s, Xte, yte, device)


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=2000)
    ap.add_argument('--n_test', type=int, default=5000)
    ap.add_argument('--proj_dim', type=int, default=2048)
    ap.add_argument('--anchors', type=int, default=256)
    ap.add_argument('--temp', type=float, default=4.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--teacher_epochs', type=int, default=50)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--retrain_epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--budgets', default='', help='comma-sep absolute budgets, overrides eff-rank-derived')
    ap.add_argument('--out', default='pilot/RGA_DIAG_REAL_RESULTS.json')
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device('cuda')
    print(f"[env] {torch.cuda.get_device_name(0)} torch={torch.__version__}")
    torch.manual_seed(0); np.random.seed(0)

    Xtr, ytr, Xte, yte = load_cifar100(device)
    # keep all data GPU-resident (kills per-batch H2D transfer; ~600MB train, fine on 80GB)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)
    rng = np.random.RandomState(0)
    te_idx = rng.choice(Xte.shape[0], min(args.n_test, Xte.shape[0]), replace=False)
    Xte, yte = Xte[te_idx], yte[te_idx]
    pool_idx = rng.choice(Xtr.shape[0], args.N, replace=False)
    Xpool = Xtr[pool_idx].to(device); ypool = ytr[pool_idx].to(device)
    print(f"[data] train={Xtr.shape[0]} pool={args.N} test={Xte.shape[0]}")

    teacher, te_acc = train_teacher(Xtr, ytr, Xte, yte, device, args.teacher_epochs, args.bs,
                                    ckpt=f'pilot/teacher_resnet56_c100_e{args.teacher_epochs}.pt')
    print(f"[teacher] test_acc={te_acc:.3f}")

    fT_pool, zT_pool = get_feat_logits(teacher, Xpool, device)
    PT_pool = F.softmax(zT_pool / args.temp, dim=-1)
    teacher_correct = (zT_pool.argmax(1) == ypool).float()
    print(f"[teacher] pool_correct={teacher_correct.mean().item():.3f}  (gate now testable)")
    # soft labels for whole pool already = PT_pool

    # warm up student
    student = resnet20().to(device)
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    n = args.N
    for ep in range(args.warmup_epochs):
        student.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = gpu_augment(Xpool[idx])
            logp = F.log_softmax(student(xb) / args.temp, dim=-1)
            loss = (args.temp ** 2) * F.kl_div(logp, PT_pool[idx], reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
    student.eval()
    fS_pool, zS_pool = get_feat_logits(student, Xpool, device)
    PS_pool = F.softmax(zS_pool / args.temp, dim=-1)
    warm_acc = (zS_pool.argmax(1) == ypool).float().mean().item()
    kd_loss_i = (PT_pool * (PT_pool.clamp_min(1e-12).log() - F.log_softmax(zS_pool / args.temp, -1))).sum(1)
    print(f"[student warmup] {args.warmup_epochs}ep pool_acc={warm_acc:.3f}")

    # last-layer closed-form signatures
    rho = (PS_pool - PT_pool)                                   # (N, C)
    gS = torch.einsum('nc,nd->ncd', rho, fS_pool).reshape(n, -1)   # (N, C*d_S) student KD grad (last layer)
    grad_norm_i = gS.norm(dim=1)
    Tkd = torch.einsum('nc,nd->ncd', rho, fT_pool).reshape(n, -1)  # (N, C*d_T) teacher KD-feature
    print(f"[features] gS={tuple(gS.shape)} fS={tuple(fS_pool.shape)} fT={tuple(fT_pool.shape)} Tkd={tuple(Tkd.shape)}")

    def proj(feat, seed):
        return CountSketch(feat.shape[1], args.proj_dim, seed, device)(feat)

    anchor_idx = torch.tensor(rng.choice(n, min(args.anchors, n), replace=False), device=device)
    signatures = {
        'studentKD_teacherENTK': (gS, fT_pool),
        'both_entk':             (fS_pool, fT_pool),   # last-layer eNTK == penultimate features
        'both_KDfeature':        (gS, Tkd),
    }

    def corr_np(a, b):
        a = a.detach().cpu().numpy(); b = b.detach().cpu().numpy()
        if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    noise = torch.tensor(rng.randn(n).astype('float32'), device=device)
    sig_results = {}; GS_primary = None; r_primary = None
    for name, (Sf, Tf) in signatures.items():
        GS = proj(Sf, 101); GT = proj(Tf, 202)
        r, KSc, KTc, cka = relational_residual(GS, GT, anchor_idx)
        sig_results[name] = {
            'cka': cka, 'r_mean': float(r.mean()), 'r_std': float(r.std()),
            'r_min': float(r.min()), 'r_max': float(r.max()),
            'corr_r_kdloss': corr_np(r, kd_loss_i),
            'corr_r_gradnorm': corr_np(r, grad_norm_i),
            'corr_r_teachercorrect': corr_np(r, teacher_correct),
            'corr_r_noise': corr_np(r, noise),
        }
        print(f"  [{name}] cka={cka:.3f} r={float(r.mean()):.3f}±{float(r.std()):.3f} "
              f"corr(r,kdloss)={sig_results[name]['corr_r_kdloss']:.3f} "
              f"corr(r,tcorrect)={sig_results[name]['corr_r_teachercorrect']:.3f}")
        if name == 'studentKD_teacherENTK':
            GS_primary = GS; r_primary = r

    R_part, R_energy = effective_rank(GS_primary)
    R = R_energy
    print(f"[eff-rank] participation={R_part:.1f} energy90={R_energy} (scale R={R})")

    Rr = max(50, int(round(R))); cap = n // 4
    if args.budgets:
        bl = [int(b) for b in args.budgets.split(',')]
        budgets = {f'b{b}': min(b, n - 1) for b in bl}
    else:
        budgets = {'1R': min(Rr, cap), '2R': min(2 * Rr, cap), '4R': min(4 * Rr, cap)}
    seen = set(); budgets = {k: v for k, v in budgets.items() if not (v in seen or seen.add(v))}
    print(f"[retrain] R={Rr} cap={cap} budgets={budgets}")

    gate = teacher_correct > 0.5
    rel_resid_feat = proj(fT_pool, 303) - GS_primary  # coverage feature for RGA = relational residual
    idx_all = torch.arange(n, device=device)

    def build_coreset(method, budget):
        budget = min(budget, n - 1)
        if method == 'random':
            sel = rng.choice(n, budget, replace=False)
        elif method == 'hard_highloss':
            sel = torch.topk(kd_loss_i, budget).indices.cpu().numpy()
        elif method == 'd_optimal_grad':
            sel = idx_all[select_coverage(GS_primary, budget)].cpu().numpy()
        elif method == 'select_low_r':
            cand = idx_all[gate]; rc = r_primary[gate]
            sel = cand[torch.argsort(rc)[:budget]].cpu().numpy()
        elif method == 'select_high_r':
            cand = idx_all[gate]; rc = r_primary[gate]
            poolk = min(len(cand), max(budget * 3, budget + 1))
            top = cand[torch.argsort(rc, descending=True)[:poolk]]
            cov = select_coverage(rel_resid_feat[top], min(budget, len(top)))
            sel = top[torch.tensor(cov, device=device)].cpu().numpy()
        return torch.tensor(np.array(sel).astype(int)[:budget], device=device)

    methods = ['select_high_r', 'select_low_r', 'random', 'hard_highloss', 'd_optimal_grad']
    retrain = {}
    for bname, budget in budgets.items():
        retrain[bname] = {}
        for method in methods:
            accs = []
            for s in range(args.seeds):
                sel = build_coreset(method, budget)
                acc = kd_train_student(PT_pool, Xpool[sel], PT_pool[sel], Xte, yte,
                                       args.temp, 1000 + s, device, args.retrain_epochs, args.bs)
                accs.append(acc)
            accs = np.array(accs)
            retrain[bname][method] = {'mean': float(accs.mean()), 'std': float(accs.std()),
                                      'n': args.seeds, 'budget': int(budget)}
            print(f"  [{bname}] {method:16s} acc={accs.mean():.4f}±{accs.std():.4f}")

    wall = time.time() - t0
    out = {
        'setup': {'dataset': 'CIFAR-100', 'pair': 'ResNet-56->ResNet-20', 'grad_proxy': 'last-layer',
                  'N': args.N, 'n_test': int(Xte.shape[0]), 'proj_dim': args.proj_dim,
                  'anchors': int(anchor_idx.shape[0]), 'temp': args.temp, 'seeds': args.seeds,
                  'teacher_test_acc': te_acc, 'teacher_pool_correct': float(teacher_correct.mean()),
                  'student_warm_acc': warm_acc, 'eff_rank_R': R,
                  'retrain_epochs': args.retrain_epochs, 'warmup_epochs': args.warmup_epochs},
        'signatures': sig_results, 'retrain': retrain, 'wall_clock_sec': wall,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({wall:.1f}s)")


if __name__ == '__main__':
    main()
