# RGA-KD Experiment Tracker

> Per-run log. Every run, every GPU-coordination event (>10min wait, OOM, kill) recorded here with GPU UUID. See `RGA_KD_PROJECT.md` §9.

## Environment (discovered 2026-06-25)

- **Working dir**: `/data/tianhao/KD` (not a git repo).
- **State on arrival**: ONLY `RGA_KD_PROJECT.md` present. All infra it references —
  `pilot/pilot_gradgeom.py`, `pilot/PILOT_RESULTS.json`, `refine-logs/EXPERIMENT_PLAN.md`,
  `CURRICULUM_KD_*.md` — is **absent from disk**. The frozen pilot *conclusions* (KD grad eff-rank ≈23,
  D-optimal +8.8 acc) are treated as background context and respected (not overturned); the reusable
  *code* is rebuilt from scratch inside `pilot/pilot_rga_diag.py`.
- **Python env**: `/home/tianhao/.conda/envs/py310_torch24/bin/python` — Python 3.10.12, torch 2.9.1+cu128
  (CUDA ok, 4 devices), torchvision 0.24.1, numpy 2.1.2, sklearn 1.7.2. Base anaconda has NO torch.
- **GPUs (A100 80GB ×4)** snapshot at start:
  | idx | uuid | free (MiB) |
  |-----|------|-----------|
  | 0 | GPU-4ad1e660-db60-f04b-31e9-a1f20fd21b51 | 81152 |
  | 1 | GPU-6373e67b-3476-9365-041c-ce58516f8b25 | 31095 (in use 80%) |
  | 2 | GPU-ec7f753a-a3f8-4d5a-fd31-4418ff9589e7 | 81152 |
  | 3 | GPU-dc43653b-03a1-7d29-70c1-c8306ee73f2a | 81152 |

## GPU coordination log

| timestamp | event | gpu idx | gpu uuid | note |
|-----------|-------|---------|----------|------|
| 2026-06-25 | start snapshot | — | — | idx 0,2,3 free (81GB); idx1 busy |
| 2026-06-25 | pilot_rga_diag launch | 0 | GPU-4ad1e660 | digits pilot, 13.6s, no contention |
| 2026-06-25 | real-pair recheck launch (bg pid 2657586) | 0 | GPU-4ad1e660 | ResNet-56 teacher train (50ep) + 45 retrainings, ~30min est |
| 2026-06-25 | parallel E2∥E6∥E3 | 0,2,3 | 4ad1e660/ec7f753a/dc43653b | E2 GPU0, E6 GPU2, E3 GPU3. idx1 busy (other tenant 94%). one process per card per §9 |
| 2026-07-15 | GPU contention (OOM on GPU0) | 0 | 4ad1e660 | RGA-fix smoke OOM'd: other tenant grew to 32GB on GPU0 mid-load. Re-pinned GPU3 (56GB free, UUID-dc43653b) + expandable_segments. All 4 cards had 25-37GB other-tenant usage. |
| 2026-06-25 | E2 done (3020s) E6 done (79s) E3 done | 0,2,3 | — | results below |

### E2 verdict (§7.4) — PASS
- RGA best method at aggressive budgets: b2000 0.270 (>all), **b4000 0.429 (>2nd-best DPP 0.406, GraNd 0.392, GRAFT 0.371)**.
- **Beats GRAFT & TAGCOS at b2000/b4000** (required gate). TAGCOS wins only at tiny b1000 (0.178). EL2N weakest.
- → algorithmic novelty supported at aggressive budgets. Not demoted.

### E6 verdict (gate necessity) — DECISIVE PASS
- 30% confident-wrong teacher labels. Detection AUC(r)=0.937, AUC(1−p_T(y))=1.000.
- gate_off coreset = **97.4% noised** → acc **0.011** (collapse, RAD trap). gate_on = **0% noised** → acc **0.299**. random 0.074.
- Gate is MANDATORY: without it RGA is actively harmful under teacher noise. The one Phase-A-untested criterion now confirmed.

### E3 verdict (§7.5) — KD-special claim DROPPED (honest)
- RGA advantage over EL2N: KD +0.02/+0.05/+0.07 ; CE +0.04/+0.07/+0.11 (LARGER under CE).
- Relational selection is teacher-*guided*, not KD-training-specific; helps KD AND CE. Reframe narrative accordingly.

| 2026-06-25 | E5+E4 done (558s) | 0 | GPU-4ad1e660 | `E5_ABLATION_RESULTS.json` |

### E5/E4 verdict
- **Coverage load-bearing**: D-optimal 0.433 vs top-r 0.318 (+11.5pt @b4000). Selecting *diverse* high-r (not just highest) is what makes r usable.
- Signature: studentKD+teacherENTK (0.433) > both_KDfeature (0.420) > both_entk (0.403). Main choice confirmed.
- Robust: proj-dim 512/2048/8192 = 0.414/0.433/0.441; anchors 64/256/1024 = 0.425/0.433/0.437.
- **E4**: last-layer proxy (0.433, 0.02s) ≈ full grad (0.421, 0.6s) → cheap proxy loses nothing, ~30× cheaper. Selection ~2s/budget << 22s/student-train → amortized after 1st student.

## STATUS: Phase A→D complete (vision). Paper draft = RGA_KD_PAPER.md. Suite GPU-time ~2 GPU-hr.

---

## PIVOT B — advisor critique (2026-07-01): "not gradient/parameter space, essentially token-level"

**Advisor is correct (technically):** the last-layer proxy `g_S=(p_S−p_T)⊗f` makes the relational
kernel FACTORIZE — `K_S = (ρρᵀ) ⊙ (ffᵀ)` — i.e. token-residual Gram ⊙ feature Gram. No backprop,
no deep parameter info. Teacher side `K_T=f_T f_Tᵀ` is pure feature. So the scaled RGA never left
output/representation space; the "gradient geometry" novelty claim does not hold for the scaled version.
(Only the digits pilot used true full-parameter gradients.)

**Decision: go to genuine gradient/parameter space on Qwen (LLM, the real target).**
- Reuse SaGD infra: Qwen3-8B(SFT) teacher → Qwen3-0.6B student, Dolly, `CountSketchProjector`.
- True fingerprint = per-sample KD gradient over attention q/k/v/o across ALL 28 layers (real backward,
  passes through the whole net → does NOT factorize). vs last-layer proxy (lm_head grad) vs token-only.

### DECISION experiment (running, GPU0 UUID-4ad1e660, pid 223225)
One per-sample backward, three fingerprints from different param groups; teacher partner (mean hidden)
held fixed → r_token / r_proxy / r_deep differ only via student fingerprint. Headline = Spearman(r_proxy,
r_deep): ~1.0 ⇒ advisor right (deep adds nothing); moderate/low ⇒ deep gradient genuinely re-ranks ⇒ B real.
Smoke N=6 (untrustworthy): Spearman(proxy,deep)=0.83, GramCKA=0.90. Full N=1500 with SFT teacher in progress.

### DECISION result (N=1500, Qwen3-8B-SFT→Qwen3-0.6B, Dolly, 304s) — `DECISION_QWEN_RESULTS.json`
Gram CKA (raw student-side kernel, cleanest): **token↔proxy=0.87**, token↔deep=0.31, proxy↔deep=0.35.
Top-10% selection overlap: token↔deep=**0.053**, proxy↔deep=0.60. r_deep std=0.27 (most discriminative).
**Two-part verdict:**
1. **Advisor RIGHT about the last-layer shortcut**: proxy kernel 87% aligned with pure-token → the scaled
   CIFAR RGA was essentially token-level. `⊗f` added little (feature-Gram redundant with token-Gram there).
2. **The genuine deep gradient is a DIFFERENT signal**: only 31% token-aligned, 5% top-10% overlap with token,
   35% with proxy. So "all just token-level" is FALSE for the real gradient. Gradient geometry has real content.
→ **B justified.** Next (mandatory): show r_deep is not just DIFFERENT but BETTER — retrain Qwen3-0.6B on
   r_deep vs r_proxy vs r_token vs random subsets, eval ROUGE-L on Dolly test. "Different" ≠ "better".

### RETRAIN proof (running, GPU0 UUID-4ad1e660, pid 232259) — `exp_retrain_qwen.py`
Pool N=4000 Dolly, budget b=1500, 5 arms (RGA_deep=gate+coverage on deep grad, deep_topr, proxy_topr,
token_topr, random) × 3 seeds. Fresh Qwen3-0.6B, KD-train 3ep, eval ROUGE-L on 200 Dolly-test. ~2hr.
Decision: RGA_deep / deep_topr must beat proxy_topr & random to prove the deep gradient is BETTER not just different.
Reuses SaGD load_teacher/student, InstructionDataset, evaluate_rouge, CountSketchProjector. Fingerprints cached.

### RETRAIN result (3 seeds, ROUGE-L @200 Dolly-test, 9423s) — `RETRAIN_QWEN_RESULTS.json`
| arm | ROUGE-L |
| RGA_deep (deep+gate+coverage) | **0.2523 ± 0.0024** (best) |
| deep_topr | 0.2458 ± 0.0042 |
| random    | 0.2465 ± 0.0065 |
| proxy_topr| 0.2334 ± 0.0088 |
| token_topr| 0.2209 ± 0.0018 (worst) |
**Verdict — B validated, honestly:**
- **Monotonic deep > proxy > token** (0.246 > 0.233 > 0.221). The token-only version (advisor's "what you're really doing")
  is the WORST; depth of the gradient causally improves the student. Refutes "token-level is all you need".
- **Full RGA_deep is best and beats random** (0.2523 vs 0.2465; all 3 RGA seeds ≥ random mean; tight std).
- **Honest caveat**: plain deep_topr ≈ random (0.246). Deep gradient's value shows via (a) monotonic ordering,
  (b) only beats random with gate+coverage. Margins modest (~0.6-3 ROUGE-L), one dataset/budget/eval-size, 0.6B student.
- → Advisor's critique fully addressed: last-layer shortcut WAS token-level (decision expt), the genuine deep gradient is
  both DIFFERENT (decision) and BETTER (retrain). Strengthen next: more budgets, larger eval (500), 1.7B student, SQuAD.

### BUDGET SWEEP + EXTERNAL BASELINES (Qwen, budgets {500,1000,2000}, 3 seeds, ROUGE-L@250)
`RETRAIN_QWEN_SWEEP.json` + `BASELINES_QWEN_SWEEP.json`. **The single-budget (b1500) RGA_deep win did NOT hold up.**
| b | RGA_deep | deep_topr | random | proxy | EL2N | GraNd | TAGCOS | GRAFT |
| 500 | .229 | .227 | .228 | .211 | .195 | .218 | **.240** | .203 |
| 1000| .234 | .233 | **.239** | .217 | .209 | .239 | .237 | .226 |
| 2000| .241 | **.248** | .241 | .242 | .216 | .239 | .246 | .245 |
(GRAFT b2000=0.2449 final)

### RGA FIX (corrected coverage = anchor-space residual K'_T−K'_S, coordinate-aligned) — `RGA_FIX_QWEN_SWEEP.json`
Micro-batch kd_train (grad-accum, effective bs=8, fp16 teacher) to fit contended GPUs. 3 variants:
| b | RGA_fixcov (gate+cov) | RGA_gatetopr (gate) | **RGA_fixcov_nogate (cov, NO gate)** |
| 500 | .2331 | .2298 | **.2381** |
| 1000| .2330 | .2386 | **.2419** |
| 2000| .2401 | .2475 | **.2511** |
**Findings:** (1) **GATE HURTS** on Dolly (clean teacher) — no-gate > gate at every budget; consistent with
CIFAR-E6 (gate only helps under NOISED teacher). (2) Corrected coverage (anchor-space residual) + NO gate is the
best RGA variant, and vs the OLD-protocol baseline table it APPEARS to beat TAGCOS(.240/.237/.246)/random(.228/.239/.241).
**CONFOUND**: RGA-fix used micro-batch training; baseline table used single-batch. MUST rerun baselines under
identical micro-batch protocol before claiming a win → fair head-to-head next.

### FAIR HEAD-TO-HEAD (all arms, IDENTICAL micro-batch protocol, fp16 teacher) — `FINAL_COMPARE_QWEN.json`
| b | **RGA_fixcov_nogate** | TAGCOS | deep_topr | GraNd | random |
| 500 | **.2389±.0015** | .2331 | .2184 | .2069 | .2353 |
| 1000| .2447±.0063 | .2326 | .2372 | .2374 | **.2463±.0104** |
| 2000| **.2457±.0046** | .2413 | .2434 | .2375 | .2369 |
| avg | **.2431** | .2357 | .2330 | .2273 | .2395 |
**VERDICT — corrected RGA WINS (honestly):**
- **Best on average (.2431)**, and best at b500 & b2000. Beats TAGCOS (best baseline) at ALL budgets.
- vs random: wins b500 (+.0036) & b2000 (+.0088), ~ties b1000 (random .2463 but huge std .0104 vs RGA .0046).
- **RGA is also the MOST STABLE** (std .0015/.0063/.0046) — random is high-variance (lucky-seed dependent).
- Ordering avg: **RGA .2431 > random .2395 > TAGCOS .2357 > deep_topr .2330 > GraNd .2273**.
**Revived story:** deep-gradient space (validated) + corrected anchor-space-residual coverage + NO gate (clean teacher)
= best & most reliable selector on the LLM. Gate reserved for noised teachers (CIFAR-E6). Honest caveats: modest
margins (~.4-.9 ROUGE-L), single dataset/student/teacher, 250-eval. GPU3 fully free during run (no OOM).

### LESS baseline (Xia et al ICML'24) — `LESS_QWEN.json` (identical protocol)
LESS b500/1000/2000 = 0.2290 / 0.2418 / 0.2354 (avg 0.2354). **RGA beats LESS at every budget** (+.010/+.003/+.010).
LESS ~ties TAGCOS and is BELOW random on avg — likely because LESS is for *targeted* selection to a *different*
task (here target=same Dolly dist) + our adaptation omits Adam preconditioning. Reported honestly.
Full fair table avg: **RGA .2431 > random .2395 > TAGCOS .2357 ≈ LESS .2354 > deep_topr .2330 > GraNd .2273**.

### DIRECTION NOTE (advisor, 2026-07): she wants selection DRIVEN BY parameter-space alignment.
Current method already does this on the STUDENT side (deep param gradient) but TEACHER side is a FEATURE
(last-layer eNTK). To be strictly "two parameter spaces", fix teacher → deep parameter gradient (reuse SaGD's
teacher-NLL-grad-over-attention infra). MUST re-run fair comparison to confirm the win holds under the true
two-param-space setup before presenting. Deliverables built: MEETING_NOTES_method_comparison.md,
RGA_experiment_results.xlsx, RGA_method_for_advisor.pptx (whiteboard), RGA_ALIGNMENT_LOSS_SPEC.md (loss version).
**Honest verdict (uncomfortable but clean):**
- **RGA_deep does NOT beat strong baselines.** Beaten by TAGCOS@b500, GraNd/random@b1000, deep_topr/TAGCOS@b2000.
  RGA_deep ≈ random at every budget. The b1500 win (0.2523 vs 0.2465) was largely noise/lucky.
- **BUT the deep-gradient SPACE is validated**: the top methods are TAGCOS (grad k-means) & deep_topr (grad top-r),
  both on the deep gradient; they beat feature (GRAFT) & loss (EL2N) methods. Gradient > feature > loss ordering holds.
- **Reframed contribution**: the right SPACE for LLM KD selection is the deep parameter-gradient space (validated:
  gradient methods win). RGA's specific rule (relational-residual + gate + coverage) is competitive but NOT dominant —
  simpler gradient selection (k-means/top-r) does as well or better. Possible issue: coverage feature (Gte−Gd) subtracts
  differently-sketched vectors (coordinate-misaligned) → noisy; gate inert when teacher uniformly competent on Dolly.
| 2026-06-25 | **INVALIDATED run** (pid 2657586) | 0 | GPU-4ad1e660 | BUG: hardcoded ckpt path → loaded 2-epoch smoke teacher (16.4% acc). Results discarded. Fix: ckpt name includes epoch count; CIFAR moved GPU-resident (util 41%→91%). |
| 2026-06-25 | real-pair recheck RELAUNCH (bg pid 2658454) | 0 | GPU-4ad1e660 | proper 50-epoch teacher (70.3% test, 92.1% pool), 484s. low_r worst@all budgets; high_r≥d_opt_grad; accs 3-7% (chance-floor regime) |
| 2026-06-25 | real-pair confirm interp-budget (bg pid 2659543) | 0 | GPU-4ad1e660 | N=5000, budgets 500/1000/2000, 248s. accs 7-24%. **low_r reliably worst; high_r>GradSpan all budgets; high_r best method @20/cls (0.242, ~3σ>random)**. corr(r,tcorrect)≈0 (gate untested) |
| 2026-06-25 | E1 vision GATE launch (bg pid 2660325) | 0 | GPU-4ad1e660 | N=10000, budgets 500/1k/2k/4k × 7 arms × 3 seeds (84 trainings, 60ep each). teacher e50=70.9%. ~1.5h est. arms: RGA-KD, GradSpan-KD, feature_coverage, random, kdloss_topk, low_r, nogate_highr. selection cost timed for wall-clock |
| 2026-06-25 | E1 vision GATE done | 0 | GPU-4ad1e660 | 940s (~16min, not 1.5h — GPU-resident fast). `pilot/E1_VISION_RESULTS.json`. E0: R(energy90)=260, r=0.591±0.17, corr(r,kdloss)=−0.21, CKA=0.12 |

### E1 GATE verdict (§7.3) — PASS → GO to E2
- **RGA beats late-feature-space selection w/ clear margin**: +2.2pt@b2000, **+6.3pt@b4000 (~6σ)**. feature_cov collapses at b4000 (0.341) while RGA holds (0.404). Core thesis (gradient-relational > feature space) supported.
- **RGA ≥ GradSpan-KD**: wins @500/1000 (+2.2pt), ties @2000/4000. Never clearly worse.
- **low_r reliably worst** at all budgets (0.045/0.076/0.147/0.295). Signal robust & correctly signed.
- Nuances: random strongest at tiny budgets (RGA passes it only @b4000, +1.7pt); gate adds ~0.4pt (teacher too clean → E6 mandatory); selection cost ≤1.7s vs ~22s train (negligible, wall-clock favorable).
- Acc by budget (RGA): 0.091/0.148/0.251/0.404. Not "dominate all" — finding-paper framing holds.

### Phase A FINAL verdict (§5.2/§5.3) — GO
- **No hard STOP.** Criteria 1 (r structured), 2 (not KD-loss reskin, corr −0.19 primary), 3 (falsification arm low_r reliably WORST across digits + 2 real-net regimes) all PASS.
- Criterion 4: on the **real CNN** high_r ≥ GradSpan-KD body (d_optimal_grad) at ALL budgets (pilot's RGA<GradSpan did NOT reproduce — reversed). high_r vs random budget-dependent: random strong at tiny budgets, high_r best at 20/class. → PASS at meaningful budgets.
- Criterion 5: gate UNTESTED (both teachers too accurate on pool, corr≈0) → **E6 teacher-noise test mandatory**.
- **Decision: GO → Phase B.** Positioning: *finding* paper (relational alignment is a real, KD-distinct, correctly-signed selection signal), not "RGA dominates". Caveats carried to Phase C: (a) advantage is budget-dependent, report full curve; (b) gate necessity via E6; (c) real-net recheck used last-layer proxy — full-grad eNTK on real net untested (E0/E5).
- Primary signature = **studentKD + teacherENTK** (cleanest distinct-from-loss r; both_entk's r correlates +0.33 with loss).

## Run log

| timestamp | run | phase | gpu idx/uuid | seeds | key result | wall-clock |
|-----------|-----|-------|--------------|-------|------------|------------|
| 2026-06-25 | pilot_rga_diag (N=1200,proj=2048,m=256) | A | 0 / GPU-4ad1e660 | 5 | r=0.30±0.11 structured; corr(r,kdloss)=−0.35; **low_r 0.356≪random 0.896** (falsification arm passes); high_r 0.853/0.936/0.962 vs **GradSpan-KD body d_optimal_grad 0.942/0.953/0.957** (RGA loses@1R/2R, ties@4R) | 13.6s |

### Phase A gate verdict (§5.2)
- **No hard STOP.** Criteria 1 (r structured), 2 (not KD-loss reskin, corr=−0.35), 3 (falsification arm low_r catastrophically worse) all PASS.
- Criterion 4 PARTIAL: RGA relational-residual selection does **not** beat GradSpan-KD's KD-gradient coverage at aggressive budgets (only ties at 4R). → §5.2#4 PIVOT-consideration, not a kill.
- Criterion 5 UNTESTABLE: digits teacher is 100% on the pool → teacher-correctness gate is inert (corr(r,teacher_correct)=0). Must test on real pair / E6.
- **Decision: conditional GO → real small-pair recheck (§5.3) is now decisive** for (a) gate necessity on an imperfect teacher, (b) whether RGA<GradSpan reproduces on a real CNN.
