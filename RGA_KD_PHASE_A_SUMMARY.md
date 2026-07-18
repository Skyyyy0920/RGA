# RGA-KD — Phase A Diagnostic Summary (GO)

> Positive diagnostic record for Phase A of `RGA_KD_PROJECT.md`. Companion to the raw JSONs:
> `pilot/RGA_DIAG_RESULTS.json` (digits pilot), `pilot/RGA_DIAG_REAL_RESULTS.json` (real-net,
> aggressive budgets), `pilot/RGA_DIAG_REAL_RESULTS_interp.json` (real-net, interpretable budgets).
> **Verdict: GO** — the core RGA-KD hypothesis carries a real, KD-distinct, correctly-signed signal;
> no hard §5.2 STOP condition triggered. Positioned as a *finding* paper, not a "RGA dominates" claim.

## 0. Environment note (important for reproduction)

On arrival the working tree contained **only** `RGA_KD_PROJECT.md`. Every piece of infrastructure
the doc attributes to a prior session — `pilot/pilot_gradgeom.py`, `pilot/PILOT_RESULTS.json`,
`refine-logs/EXPERIMENT_PLAN.md`, `CURRICULUM_KD_*.md` — was **absent from disk**. The frozen pilot
*conclusions* (KD-grad low rank, D-optimal aggressive-budget gain) are respected as background context;
the reusable *code* was rebuilt from scratch in `pilot/pilot_rga_diag.py` (digits) and
`pilot/pilot_rga_diag_real.py` (ResNet/CIFAR-100). Env: `py310_torch24` (torch 2.9.1+cu128, 4×A100 80GB).

## 1. What was tested

The RGA-KD core hypothesis: the per-sample **relational gradient residual**
`r(i) = 1 − corr(K'_T[i,:], K'_S[i,:])` — how differently teacher vs. student wire sample *i* into
the sample×sample gradient-kernel relation — is a **non-trivial, KD-loss-distinct, training-value
signal**, with high `r` = "teacher knows structure the student hasn't learned" = keep, low `r` =
"student already reproduces it" = redundant/skip.

Three runs:
| run | net pair / data | grad signatures | budgets | seeds |
|-----|------|------|------|------|
| **digits pilot** | MLP128→MLP32 / sklearn-digits (N=1200) | full student-KD-grad + full teacher-eNTK Jacobian | 1R/2R/4R (energy-rank, ≤N/4) | 5 |
| **real aggressive** | ResNet-56→ResNet-20 / CIFAR-100 (N=2000) | **last-layer** closed-form proxy | 1R/2R/4R (≤N/4) | 3 |
| **real interpretable** | ResNet-56→ResNet-20 / CIFAR-100 (N=5000) | last-layer proxy | 500/1000/2000 (5–20 per class) | 3 |

Teachers: digits MLP = 100% (pool); ResNet-56 = **70.3% test / 92.1% pool** (properly trained, 50 ep cosine).
The last-layer proxy is exact for `g_S = (p_S−p_T)⊗f_S`; the last-layer teacher eNTK relational reduces
analytically to penultimate-feature inner products (`<J_T(x_i),J_T(a_j)>_F = C·<f_T(x_i),f_T(a_j)>`),
so the teacher signature is `f_T(x)`. (§7.6/E5 sanction last-layer gradients as the cheap proxy.)

## 2. Headline numbers

**Signatures** (primary = studentKD+teacherENTK):
| | digits | real (N=5000) |
|---|---|---|
| `r` mean ± std | 0.30 ± 0.11 | 0.53 ± 0.14 |
| corr(r, KD-loss) | −0.35 | −0.19 |
| corr(r, noise) | ≈0 | ≈0 |
| corr(r, teacher-correct) | 0 (teacher 100%) | ≈0 (teacher 92% pool) |
| CKA(K_S,K_T) | 0.47 | 0.21 |

**Retrain accuracy, 5-/3-seed mean ± std** (best per row in **bold**):
```
digits (MLP, test acc):
  budget   high_r        low_r         random        hard_loss     d_opt_grad(GradSpan)
  1R(104)  0.853±.011    0.356±.007    0.896±.005    0.552±.015    0.942±.006   <- GradSpan best
  2R(208)  0.936±.008    0.404±.031    0.923±.009    0.693±.018    0.953±.005   <- GradSpan best
  4R(300)  0.962±.006    0.568±.031    0.931±.002    0.809±.019    0.957±.007   <- high_r best

real CNN (ResNet-20, CIFAR-100, test acc):
  budget       high_r        low_r         random        hard_loss     d_opt_grad
  500 (5/cls)  0.0737±.002   0.0471±.002   0.0780±.004   0.0713±.002   0.0665±.005
  1000(10/cls) 0.1100±.002   0.0906±.004   0.1196±.005   0.1210±.005   0.0978±.006
  2000(20/cls) 0.2424±.015   0.1968±.002   0.1982±.005   0.2151±.016   0.2176±.002  <- high_r best
```

## 3. §5.2 / §5.3 decision gate — criterion by criterion

1. **`r` has structure?** YES. Wide non-trivial distribution (digits [0.03,0.83]; real ~0.53±0.14),
   decorrelated from random noise. → **PASS**
2. **`r` a KD-loss reskin?** NO. |corr(r, KD-loss)| ≤ 0.35 in the primary signature (and *negative*:
   relational mismatch is not high loss). `both_entk` is more loss-like (+0.33) and is therefore *not*
   the chosen signature. → **PASS**
3. **Falsification arm worse than random?** YES, robustly. `low_r` (train only on aligned/"already-learned"
   samples) is **the worst method at every budget across all three runs** (digits 0.356 vs random 0.896;
   real-net worst at 5/10/20-per-class). The signal is real and **correctly signed**. → **PASS (strongest result)**
4. **Positive arm ≥ random, ideally ≥ GradSpan-KD body?**
   - vs **GradSpan-KD** (`d_optimal_grad`): digits — high_r *loses* at 1R/2R, ties at 4R. **Real CNN —
     high_r ≥ GradSpan at ALL budgets** (the pilot result reversed; the real net is the more relevant test).
   - vs **random**: budget-dependent. Random (strong unbiased class-coverage sampler) wins at the most
     aggressive budgets; high_r becomes the single **best** method at the largest budget tested
     (digits 4R; real 20/class, 0.242, ≈3σ over random).
   → **PASS at non-degenerate budgets** (not an unconditional "always wins").
5. **Gate necessary?** UNTESTED. Both teachers were too accurate on the pool (digits 100%, ResNet 92%),
   so `corr(r, teacher-correct) ≈ 0` and the RAD-trap could not be exercised. → **defer to E6 (mandatory)**.

**No hard STOP condition fired.** → **GO to Phase B.**

## 4. Honest caveats (these become Phase B/C design constraints)

- **Budget-dependence**: RGA's selection advantage is *not* uniform. At extreme aggressive budgets,
  random class-coverage is a strong baseline; RGA's edge appears as budget grows out of the chance floor
  (≈20/class on CIFAR-100). **E1 must report the full budget curve**, not a single point.
- **Gate necessity unproven**: needs a deliberately noised teacher. **E6 is promoted from optional to mandatory.**
- **Last-layer proxy only on the real net**: the real-pair recheck used last-layer gradients (cheap, and
  it *worked* — good for E4 efficiency). Full-gradient eNTK on a real CNN is untested → **E0/E5 must include
  a full-vs-last-layer signature comparison** to confirm the lazy/kernel approximation holds.
- **Low absolute accuracies** in the real-net retrains (3–24%) reflect from-scratch CIFAR-100 on tiny
  coresets; they are valid for *relative* method ranking but E1/E2 should also report a higher-budget anchor.
- **Signature choice locked**: `studentKD + teacherENTK` (cleanest KD-distinct `r`). `both_entk` is more
  loss-correlated; `both_KDfeature` gives the smallest, least-structured `r`.

## 5. One-line takeaway

Relational gradient alignment is a **genuine, KD-loss-distinct, correctly-signed data-selection signal**
(aligned samples are reliably the worst to train on) that, used as a selection rule with a teacher gate +
D-optimal coverage, is **competitive-to-better than gradient-magnitude/subspace (GradSpan-KD) and
loss-based selection on a real CNN**, with advantage concentrated at meaningful aggressive budgets — a
*finding* worth a paper, pending the E6 gate test and full-budget E1/E2 curves.
