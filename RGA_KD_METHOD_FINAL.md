# RGA-KD — Finalized Method (Phase B)

> Data-driven finalization per `RGA_KD_PROJECT.md` §6, locked from Phase A evidence
> (`RGA_KD_PHASE_A_SUMMARY.md`). Every choice below cites the diagnostic that decided it.
> Positioning: **finding paper** — "the right signal for KD data selection is teacher–student
> *relational* gradient alignment", RGA-KD is its instance. Not pitched as a dominate-everything algorithm.

## 1. Locked design decisions

| Knob | Decision | Evidence (Phase A) |
|------|----------|--------------------|
| **Gradient signature** | student **KD gradient** `g_S(x)=∇_{θ_S}KL(p_T‖p_S)` + teacher **eNTK** `g_T(x)=J_T(x)` | Primary signature gave the cleanest KD-*distinct* `r` (corr(r,KD-loss)=−0.35 digits / −0.19 real). `both_entk` is loss-like (+0.33); `both_KDfeature` `r` is tiny/unstructured. |
| **Gradient locus** | **last-layer** by default; full-gradient as ablation | Last-layer closed form `g_S=(p_S−p_T)⊗f_S`, teacher relational ⇒ `f_T` inner products. Reproduced the full-gradient signal on the real CNN at ~0 extra cost → default for efficiency (E4). Full-vs-last-layer is an E5 ablation. |
| **Relational matrix** | random `m=256` anchors, `K[i,j]=⟨Πg(x_i),Πg(a_j)⟩`, double-centered `K'=HKH` | Nyström anchor reduction kept `r` structured (std 0.11–0.14, decorrelated from noise) at N up to 5000. |
| **Projection** | Count-Sketch, `d=2048` | Memory-O(D) sparse JL; preserved relational structure; pilot `d=512` already sufficient → 2048 is comfortable. Count-Sketch vs Rademacher vs none is an E5 ablation. |
| **Per-sample score** | `r(i)=1−corr(K'_T[i,:],K'_S[i,:])` | Robust ranking: `low_r` reliably the **worst** retrain arm across digits + 2 real-net regimes. |
| **Teacher gate** | **soft** ×`p_T(y)` default; **hard** `argmax p_T≠y` cutoff as ablation; necessity decided by **E6** | Could NOT be tested in Phase A (teachers 100%/92% on pool → corr(r,teacher-correct)≈0). Soft gate is the less brittle default; E6 (noised teacher) is **mandatory**, not optional. |
| **Selection** | **D-optimal coverage** over the relational-residual feature among gated high-`r` (NOT pure top-`r`) | The tested `high_r` arm = gate→top-3×budget by `r`→greedy log-det coverage; it was competitive-to-best and beat GradSpan-KD on the real net. Coverage-vs-top-`r` is an E5 ablation. |
| **Difficulty schedule** | **static**, re-estimate every N epochs only if drift shown | Per CLPD/DMC prior + project default; no Phase A evidence of large `r` drift warranting dynamic. |
| **Budget** | tie to **90%-energy rank** `R` of projected `g_S`; sweep `{1R,2R,4R,…}`; **report full curve** | RGA's advantage is **budget-dependent** (random strong at extreme-low budgets; RGA best at ≈20/class). Single-point claims are disallowed. |

## 2. Final algorithm (frozen)

```
Input: teacher T (frozen), student S, candidate pool X, temperature τ, budget b
0. Warm up S on the KD task a few epochs (calibrate output head).
1. Features (last-layer, one forward pass over X ∪ anchors):
     f_S(x), f_T(x) = penultimate features;  p_S, p_T = softmax(z/τ)
     g_S(x) = (p_S(x) − p_T(x)) ⊗ f_S(x)        # student KD gradient (last layer)
     g_T(x) = f_T(x)                            # teacher last-layer eNTK signature
2. Project: Π = Count-Sketch to d=2048;  G_S = Π g_S,  G_T = Π g_T
3. Anchors a_1..a_m (m=256, random);  K_S = G_S G_S[A]^T,  K_T = G_T G_T[A]^T
   Double-center: K' = K − rowmean − colmean + grandmean
4. Residual:  r(i) = 1 − corr(K'_T[i,:], K'_S[i,:])
5. Gate (soft default):  s(i) = r(i) · p_T(x_i)[y_i]      # E6 decides hard vs soft vs none
6. Coverage select: among top-(3b) by s, greedy log-det (D-optimal) over the relational-residual
   feature  (G_T − G_S)[i]  → pick b diverse samples
7. (optional) GraB-style staleness-balanced ordering;  static difficulty, re-score every N epochs.
Output: coreset of size b → distill S with KD loss (τ²·KL).
```

## 3. What the paper can and cannot claim (result-to-claim discipline)

**Can claim (Phase A-supported):**
- Teacher–student relational gradient alignment is a **non-trivial, KD-loss-distinct** per-sample signal.
- It is **correctly signed**: training on the most-aligned (low-`r`) samples is reliably the worst choice
  (robust across MLP/digits and ResNet/CIFAR-100, multiple budgets).
- As a selection rule it is **competitive-to-better than gradient-magnitude/subspace coverage (GradSpan-KD)
  and loss-based selection on a real CNN**, with advantage concentrated at meaningful aggressive budgets.

**Cannot (yet) claim — needs Phase C:**
- "RGA beats all baselines at all budgets" — FALSE at extreme-low budgets (random is strong). Report curves.
- "Teacher gate prevents RAD trap" — UNTESTED until E6.
- "Full-gradient eNTK needed" — last-layer proxy already carries the signal; full-grad is an ablation, not a requirement.
- KD-specificity — E3 must show the relational advantage is larger under KD than plain CE.

## 4. Hand-off to Phase C

Use `refine-logs/EXPERIMENT_PLAN.md` skeleton (E0–E6). RGA-KD enters as a new selection arm in E1/E2.
Phase A promotes two items to **mandatory**: full budget curves (E1) and the noised-teacher gate test (E6).
Note: `refine-logs/EXPERIMENT_PLAN.md` was absent on arrival and must be reconstructed before E0 (the
`pilot/pilot_rga_diag*.py` harness already implements the reusable primitives: count-sketch, relational
residual, D-optimal coverage, eff-rank, kd_train).
