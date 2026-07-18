# RGA-KD Experiment Plan (E0–E6) — reconstructed

> The original `refine-logs/EXPERIMENT_PLAN.md` was **absent on arrival** (see EXPERIMENT_TRACKER.md
> env note). Reconstructed here from `RGA_KD_PROJECT.md` §7 + Phase A findings so Phase C has its
> skeleton. GPU coordination rules live in `RGA_KD_PROJECT.md` §9 (authoritative — not duplicated).
> Method spec frozen in `RGA_KD_METHOD_FINAL.md`. Reusable primitives in `pilot/pilot_rga_diag*.py`.

## Fixed settings
- **Vision pair (primary)**: ResNet-56 → ResNet-20 / CIFAR-100. (Secondary: WRN-40-2 → WRN-16-2.)
- **Text pair**: BERT-base → BERT-small / SST-2 (or MNLI). *Heaviest infra; scheduled after vision E1 passes.*
- **Budget axis**: measured 90%-energy rank `R` of projected student KD-grad (E0), sweep
  `m ∈ {0.5R, 1R, 2R, 4R, 8R, 0.5N, N}`; **always report the full curve** (Phase A: advantage is budget-dependent).
- **Seeds**: ≥3 (5 where cheap). **Seed variance (std) is a reported metric, not an error-bar afterthought.**
- **Metrics**: student test acc; its std; **end-to-end wall-clock = selection cost (incl. eNTK/relational
  matrix + teacher soft-label inference) + training**.省算力主张由 wall-clock 决定, 不是 training-only.

## E-list
| id | purpose | est | decision gate |
|----|---------|-----|---------------|
| **E0** | per-sample KD-grad + teacher eNTK; Count-Sketch; SVD → `R`, relational CKA, `r` dist | 0.5 GPU-d | calibration only |
| **E1** ★GATE★ | space/criterion shoot-out @ {1R,2R,4R}: RGA-KD vs GradSpan-KD vs feature-space vs input-saliency vs random vs KD-loss-topk vs no-gate | 2 GPU-d | RGA ≥ feature-space w/ clear margin on ≥1 modality AND ≥ GradSpan or complementary; **else STOP** |
| **E2** | main anchor: full-budget sweep + native baselines (EL2N, GraNd, CRAIG, GRAFT, TAGCOS, LESS, DPP) | 5 GPU-d | RGA > GRAFT & TAGCOS at aggressive budget, else demote to E1 note |
| **E3** | KD vs CE: is RGA's relative advantage larger under KD? | 2 GPU-d | if KD≈CE, drop "KD-special" claim |
| **E4** | efficiency/amortization: full vs last-layer vs LoRA vs one-step grad; break-even | 1.5 GPU-d | report honest break-even incl selection cost |
| **E5** | ablations: projection, criterion (D-opt vs leverage vs top-r), signature (3 pairings), m/k/τ, **gate on/off** | 2 GPU-d | — |
| **E6** | teacher-noise falsification (gate necessity) — **mandatory** (Phase A could not test gate) | 1 GPU-d | gate must catch noised-teacher samples |

## Run order
```
E0 → E1(GATE) → {E2 ∥ E3 ∥ E4} → E5 → E6
            └─ fail → negative/diagnostic note or kill
```

## Resource discipline (from §9)
- nvidia-smi snapshot + `CUDA_VISIBLE_DEVICES` pin + record `gpu_uuid` before every CUDA launch.
- Any single run > 3× its estimate → kill. Total > 25 GPU-day → halt suite + report.
- Checkpoint per epoch; one process per card; log every >10min wait / OOM / kill in EXPERIMENT_TRACKER.md.

## Phase A → Phase C carry-overs (mandatory)
1. Report full budget curves (not single points).
2. E6 promoted optional → mandatory (gate untested on clean teachers).
3. E0/E5 must compare full-gradient vs last-layer eNTK (real-net recheck only used last-layer).
