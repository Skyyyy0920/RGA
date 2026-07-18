# RGA — Full Experiment Plan for the Paper (execution-ready, ICLR-Oral target)

> **This is the authoritative plan to run all paper experiments, then write the paper.** Method is
> FROZEN (below). Execute E0 → E11 in order; each experiment lists purpose, setup, arms, budgets, seeds,
> metric, decision gate, est. cost, and priority (P0 = must-have for submission, P1 = strongly wanted,
> P2 = nice-to-have). Report **seed variance** and **end-to-end wall-clock incl. selection cost** for
> every result. Reuse existing harnesses in `pilot/` and SaGD infra; new code noted per experiment.
>
> **ORAL BAR (read first).** An ICLR Oral needs ALL of: (1) a decisive/surprising empirical result,
> (2) a principled *why* (theory or deep mechanism), (3) generality across scale & tasks. This plan is
> built to that bar — but **Oral status depends on results delivering**, not coverage alone. Three
> things are load-bearing and must be actively pursued, not assumed:
>   - **Decisive win**: current margins are ~0.4–0.9 ROUGE-L. That is NOT Oral-level. E1/E9/E10 must
>     hunt the regime where the advantage is LARGE (extreme budgets 1–2%, hard/reasoning tasks, scale).
>     If margins stay ~1 pt with no strong theory-backed insight → honest target is a strong poster, not Oral.
>   - **Theory (E8)**: a formal reason why relational parameter-gradient misalignment ⇒ high KD value,
>     with experiments validating its predictions. Without this it is not an Oral.
>   - **Scale/generality (E9/E10)**: ≥1 larger student (7B) and hard tasks, or it reads as small-scale.

---

## 0. Finalized method (FROZEN) — RGA: Relational Gradient Alignment for KD data valuation

> **Naming note (fix #1).** The method produces a per-sample value/importance signal `r`; the *form*
> in which `r` is used (SELECT a subset / REWEIGHT all data / CURRICULUM-order all data) is decided by
> results in E1. Until then the paper positioning is neutral: **"parameter-gradient relational alignment
> is the right data-valuation signal for KD"** — not committed to "selection".

**One line.** Score each candidate sample by **pairwise relational alignment of two parameter spaces**:
compare how the teacher and student relate it to a set of anchor samples in their respective
**parameter-gradient (empirical-NTK) spaces**; a sample whose relational pattern differs most means the
teacher encodes structure the student lacks (high value). The score `r` is then used to select / reweight
/ order training (form = E1).

**Signatures — SAME method both sides (symmetric).** For each model `M ∈ {T, S}` and sample `x`:
```
g_M(x) = ∇_{θ_M} ( − log p_M(y | x) )          # per-sample response-NLL gradient
θ_M = attention params (q/k/v/o) of the LAST K transformer blocks (default K=4)
G_M(x) = Π g_M(x)                               # Count-Sketch projection to d = 2048
```
(This is each model's empirical NTK signature restricted to `θ_M`. Symmetric → easy to describe;
last-K blocks → cheap AND non-token-level, i.e. does not collapse to the last-linear-layer `(p−onehot)⊗f`.)

**Relational matrices & residual.** Anchors `a_1..a_m` (m=256):
```
K_M[i,j] = ⟨ G_M(x_i), G_M(a_j) ⟩ ;   K'_M = H K_M H  (double-center, H = I − 11ᵀ/m)
r(i) = 1 − corr( K'_T[i,:], K'_S[i,:] )
```
**`r` is a per-sample IMPORTANCE signal — three use-modes (compared in E1b; final use decided by results):**
1. **SUBSET (coreset)**: (optional gate) → top-3b by `r` → D-optimal coverage on residual rows
   `K'_T[i,:]−K'_S[i,:]` → train on b samples. Gate `×p_T(y)`: OFF for a clean teacher, ON under noise.
2. **REWEIGHT**: train on ALL data; weight each sample's KD loss by `w(i) = softmax(r/τ)·N` (static;
   periodic re-scoring = P1). (This is the advisor-preferred "alignment as a training signal" framing.)
3. **CURRICULUM**: train on ALL data, ordered/paced by `r` (test hard-first AND easy-first — `r` is a
   *mis*alignment signal, so easy-first ≠ classic curriculum). NOTE: pure difficulty-ordering is a known
   weak lever (When-Do-Curricula-Work) — included for completeness, not assumed to win.

**Warm-up.** Student warmed **~200 KD steps (≈0.5 epoch of the pool)** before signatures (head
calibration; pretrained LLM needs little). Warm-up length sensitivity is checked in E4.

---

## 1. Fixed experimental settings

- **Model pairs** (teachers are the cached SFT checkpoints in `SaGD/data/teacher_sft_*.pt`):
  - **Primary**: Qwen3-8B → **Qwen3-0.6B**.
  - **Scale**: Qwen3-8B → **Qwen3-1.7B**.
  - **Cross-arch**: LLaMA-3.1-8B → **LLaMA-3.2-1B**. **Verify first** that `teacher_sft_*_llama.pt`
    is compatible with the LLaMA-3.2-1B student (tokenizer/arch) before relying on this pair.
- **Datasets**: **Dolly** (instruction), **SQuAD** (QA) [both have cached SFT teachers]; SAMSum (P2);
  **GSM8K / MATH / code (P0-Oral, see E10)**.
- **Budget axis**: candidate pool N (default 4000–8000 from train); budget fractions
  `b ∈ {1%, 2%, 5%, 10%, 25%, 50%}` of N + **full-data (100%) reference** (report the **full curve**;
  1–2% is the decisive-win regime). For REWEIGHT/CURRICULUM the "budget" is the training-compute match.
- **Seeds**: **3** per cell (5 where cheap). **Report std AND a paired significance test** (bootstrap or
  paired t-test over seeds) for every RGA-vs-baseline margin — margins are ~1 pt, so overlap must be tested.
- **KD training**: standard forward-KL, T=2.0, response-masked, micro-batch grad-accum (effective bs=8),
  3 epochs, lr 2e-5 (SaGD defaults). Fresh student per seed.
- **Metrics**:
  - **Instruction (Dolly/SAMSum)**: **win-rate via GPT-judge / AlpacaEval** (primary — reuse SaGD
    `gpt_judge.py`) + ROUGE-L (secondary). *(ROUGE-L alone is too weak for an Oral.)*
  - **SQuAD**: EM / token-F1 / ROUGE-L. **GSM8K/MATH**: accuracy. **Code**: pass@1. + perplexity throughout.
  - **End-to-end wall-clock = selection cost (both models' grad extraction + relational + coverage) + training.**
- **Env**: `/home/tianhao/.conda/envs/py310_torch24/bin/python`; GPUs shared/contended → per-§ GPU rules.

---

## 2. Experiment suite

### E0 — Signatures & diagnostics (build the pipeline)  · P0 · ~0.5 GPU-day
**New code**: symmetric NLL-gradient extraction over last-K blocks for BOTH teacher and student
(adapt `pilot/exp_retrain_qwen.py::extract` + SaGD `gradient_pca_selection` teacher-grad; cache both).
- Compute `G_T`, `G_S`, relational matrices, `r`. Report: `r` distribution; effective rank of each
  kernel; **CKA(K_S, K_T)**; corr(`r`, KD-loss) (must be low → not a loss reskin);
  corr(`r`, noise) ≈ 0. Confirm the last-K gradient is **non-token-level** (CKA vs pure-token and vs
  last-linear-layer proxy — reuse `pilot/exp_decision_qwen.py`).
- **Gate**: `r` structured, not a loss reskin, deep≠token. Else debug before proceeding.

### E1 — Method form + headline result (primary pair)  ★HEADLINE / GATE★ · P0 · ~5 GPU-days
Two phases (form MUST be decided before the headline table — fix #2). Pair Qwen3-8B→0.6B, Dolly, 3 seeds.

**Phase A — decide the use-mode of `r` (SUBSET vs REWEIGHT vs CURRICULUM).** Same `r` (frozen symmetric
signatures), three uses, head-to-head at **matched total training compute**:
- **SUBSET**: top-b by `r` + coverage. **CURRICULUM**: all data ordered/paced by `r` (hard-first AND
  easy-first + pacing). **REWEIGHT**: all data, KD loss × `softmax(r/τ)` (static; periodic re-scoring = P1).
- Shared reference for all three: **full-data (100%) uniform KD** (upper/reference bound — fix #4) and
  random-of-same-form. → pick the form with the best accuracy-vs-compute tradeoff = **"RGA form"**;
  keep the other two as reported comparisons (they show `r` is a strong signal regardless of use).

**Phase B — headline table: RGA (winning form) vs all baselines, matched to that form (fix #3):**
- If **SUBSET** → baselines = random, EL2N, GraNd, GRAFT, TAGCOS, LESS, token-level-RGA (all selection).
- If **REWEIGHT** → baselines = uniform KD, loss-weighting, SaGD reweight, + the selection baselines
  applied as soft weights, token-level-RGA.
- If **CURRICULUM** → baselines = random order, loss-ordered, uniform KD, token-level-RGA.
- Full budget curve `b ∈ {1,2,5,10,25,50%}` + full-data reference. Report the **primary metric
  (win-rate) + ROUGE-L**, mean±std, **paired significance**, **+ end-to-end wall-clock per method**.
- **GATE**: RGA best-or-tied on average AND ≥ every strong baseline (esp. TAGCOS, LESS) at aggressive
  budgets, **with significance**. Reuse `exp_final_compare_qwen.py`+`exp_baselines_qwen.py`+`exp_less_qwen.py`,
  re-run under the **symmetric last-K signature** and the winning form.

### E2 — Generalization: datasets × pairs  · P0 (SQuAD, 1.7B) / P1 (cross-arch) · ~5 GPU-days
- Repeat E1 on: **(a) SQuAD** (EM/F1/ROUGE-L); **(b) Qwen3-8B→1.7B**; **(c) LLaMA-3.1-8B→3.2-1B** (P1).
  (SAMSum / GSM8K = P2.) At least 2–3 aggressive budgets each (not the full sweep).
- **Gate**: the E1 win must hold on ≥1 more dataset AND ≥1 more student. Else scope claims down honestly.

### E3 — Isolate the value of the alignment  ★MECHANISM★ · P0 · ~1.5 GPU-days
The two comparisons that make the novelty precise (same deep gradient, different use):
- **vs TAGCOS** (same gradient, *clustered*) and **vs deep-topr** (same gradient, *top-r no alignment*):
  isolates that the **teacher–student relational alignment** (not merely deep gradients) is the value-add.
- **vs feature-space relational selection (RKD-style)**: same relational machinery but on **features**
  instead of parameter-gradients → isolates that **parameter/gradient space** beats feature space.
  **New code**: RKD-feature relational baseline.
- **Gate**: RGA > TAGCOS/deep-topr and > feature-relational, on the primary pair.

### E4 — Ablations  · P0 (core) / P1 (rest) · ~2.5 GPU-days
On Qwen3-8B→0.6B / Dolly, fixed aggressive budget:
- **Signature symmetry (P0)**: symmetric NLL-grad (ours) vs asymmetric (student KD-grad + teacher eNTK/feature).
  Justifies the symmetric choice to reviewers/advisor.
- **Parameter subset (P0)**: last-K (K∈{1,2,4,8}) vs all-attention-layers vs LoRA vs **last-linear-layer
  (token-level control)**. Shows last-K is enough and the token-level control is worst.
- **Signature type (P1)**: NLL-gradient vs eNTK output-Jacobian.
- **Criterion (P0)**: D-optimal coverage vs plain top-r (coverage's contribution).
- **Projection d ∈ {512, 2048, 8192} (P1)**; **anchors m ∈ {64, 256, 1024} (P1)**.
- **Gate on/off on a clean teacher (P0)**: confirm gate is off-by-default here.

### E5 — Efficiency / wall-clock  ★ANSWERS "too slow?"★ · P0 · ~1 GPU-day
- Break selection cost into: teacher-grad extraction, student-grad extraction, projection, relational,
  coverage. Report each. **Teacher side is one-time & cacheable → report amortized cost across K students.**
- **End-to-end wall-clock vs full-data training** → break-even (how many distilled students / how much
  saved training before selection cost is amortized). Compare selection cost head-to-head with LESS/TAGCOS.
- last-K vs all-layers extraction time (justify last-K).
- **Gate**: selection cost is a small, one-time, amortizable fraction of training. State the numbers plainly.

### E6 — Robustness: teacher-noise & the gate  · P1 · ~1 GPU-day
- Inject a known fraction of confident-wrong teacher labels (replicate vision E6 on the LLM).
- Show: high-`r` alone concentrates the noised samples (RAD trap); the `×p_T(y)` gate removes them and
  recovers accuracy; gate-off collapses. **New code**: LLM noised-teacher variant (adapt `exp_e6_gate.py`).
- **Gate**: demonstrates when/why the gate matters (justifies the on/off rule).

### E7 — Parameter-space alignment validation (conceptual)  · P1 · ~0.5 GPU-day
- The "is it really parameter space, not token" figure: CKA of the last-K gradient kernel vs
  (i) pure-token, (ii) last-linear-layer proxy, (iii) feature. Show the last-K kernel is distinct from
  token/feature. (Extends `exp_decision_qwen.py` to the symmetric last-K signature.)
- **Gate**: last-K gradient kernel is demonstrably not token-level → supports the framing.

---
## ORAL-LEVEL ADDITIONS (E8–E11) — required to clear the Oral bar

### E8 — Theory & its empirical validation  ★WHY★ · P0-for-Oral · ~1 GPU-day (analysis-heavy)
The paper needs a *principled reason* selecting high-`r` samples maximizes KD value. Develop ONE of:
- **NTK/kernel view**: the teacher's gradient-kernel `K_T` is the target kernel; the student's `K_S`
  is its current kernel. Show (bound/argument) that training on high-`r` (max `K_T`–`K_S` relational
  discrepancy) samples reduces the student↔teacher **functional/kernel discrepancy** fastest.
- **Influence view**: relate `r(i)` to the influence of `x_i` on closing the teacher–student gap
  (connect to LESS's influence framing but in the *relational* setting).
- **Validation experiments** (this is what makes it an experiment, not just prose):
  1. measure student↔teacher kernel-CKA / KL over training when selecting by high-`r` vs low-`r` vs
     random — the theory predicts high-`r` closes it fastest;
  2. correlate per-sample `r` with a ground-truth leave-one-out / influence proxy on a small setup.
- **Gate**: theory makes a falsifiable prediction AND the prediction holds empirically. If it fails,
  weaken to a "mechanistic analysis" framing (still P0, but not a theorem).

### E9 — Scale  ★GENERALITY★ · P0-for-Oral · ~6–8 GPU-days (larger models are expensive)
- Real compression at scale: **larger TEACHER → mid student**, e.g. **Qwen 14B/32B → 1.7B/4B** (keeps a
  meaningful teacher≫student gap; "8B→7B" is NOT a compression and is excluded — fix #7). Add a **4B
  student** point so the RGA-vs-baseline margin can be plotted against student size.
- **Cost caveat**: per-sample gradients on a 14–32B teacher × N samples are expensive; use last-K + a
  smaller pool N, and the teacher signature is one-time cacheable. Budget this within the §5 cap.
- **Gate**: advantage does not vanish (ideally grows) with the teacher/student gap. Scale plot = core Oral figure.

### E10 — Hard tasks & extreme budgets  ★DECISIVE-WIN HUNT★ · P0-for-Oral · ~3 GPU-days
- **Reasoning/code**: GSM8K, MATH (or a code task) — where data quality dominates and margins should be largest.
- **Extreme budgets**: 1%, 2% of the pool — the aggressive regime where selection matters most.
- **Multi-source / mixed-data selection** (P1): select from a heterogeneous pool (Dolly+SQuAD+…) — a
  realistic, high-impact setting for selection.
- **Gate**: find at least one regime with a **large, clear** RGA advantage (this is what carries an Oral).

### E11 — Deep analysis: what RGA selects & why  · P1 · ~1 GPU-day
- Selection-overlap heatmap RGA vs each baseline; how disjoint are the picks.
- What characterizes high-`r` samples (length, difficulty, category, teacher confidence) — qualitative + quantitative.
- Sample-value attribution: retrain leaving out RGA-picked vs baseline-picked, measure marginal value.
- Why the **symmetric NLL-gradient** signature (vs asymmetric) — analysis, not just the E4 number.
- **Gate**: a coherent story of *what* the parameter-gradient relational space captures that others miss.

---

## 3. Run order & dependencies
```
E0 (build+diagnose) ─▶ E1 (Phase A: pick form → Phase B: HEADLINE gate) ─┬─▶ E2 (generalize)  [P0]
                                           ├─▶ E3 (mechanism)                   [P0]
                                           ├─▶ E4 (ablations)                   [P0]
                                           ├─▶ E5 (efficiency/wall-clock)       [P0]
                                           ├─▶ E8 (theory + validation)         [P0-Oral]
                                           ├─▶ E9 (scale: 7B student)           [P0-Oral]
                                           ├─▶ E10 (hard tasks, extreme budgets)[P0-Oral]
                                           ├─▶ E6 (gate/noise)                  [P1]
                                           ├─▶ E7 (concept fig)                 [P1]
                                           └─▶ E11 (deep analysis)              [P1]
   E1 fail (RGA not ≥ strong baselines) ─▶ STOP, write honest negative/diagnostic note.
```
**Hard gates:**
- **E1**: if RGA not ≥ TAGCOS & LESS at aggressive budgets on the primary pair, do NOT burn downstream.
  Re-check the symmetric-signature change first (it may have shifted results), then GO / re-tune / negative note.
- **ORAL decision point (after E1+E8+E10 first pass)**: is there a **decisive-win regime** AND a
  **theory prediction that holds**? If YES → full Oral push (E9 scale, E11 analysis). If NO → honestly
  retarget to a strong poster; do not manufacture an Oral narrative the results don't support.

## 4. Baselines (final list)
- **Baselines MUST match RGA's winning form (fix #3)**: SUBSET→selection baselines; REWEIGHT→uniform KD /
  loss-weighting / SaGD reweight / selection-as-weights; CURRICULUM→random-order / loss-order / uniform KD.
  A **full-data (100%) uniform KD** reference is reported in every table.
- **Must (P0)**: random, EL2N, GraNd, GRAFT (feature-MaxVol), **TAGCOS**, **LESS**, feature-relational (RKD-style).
- **Token-level RGA (P0, named baseline)**: the pre-pivot method — relational alignment using the
  **per-token / last-linear-layer signature** `(p_S−p_T)⊗f` (which factorizes to token-residual ⊙ feature).
  Same relational + selection machinery, only the signature is token-level. Directly demonstrates
  **parameter-gradient space > token space** (the whole point of the pivot). Run it through all three
  use-modes in E1b too.
- **Quality-based SOTA (P0 for Oral)**: **DEITA**, **IFD / Cherry** — reviewers expect the current
  instruction-selection line, not only gradient/coreset methods.
- **Compute-matched (P0 for Oral)**: report accuracy at a **fixed total budget incl. selection FLOPs**,
  so "we win but pay more to select" cannot be raised.
- **Internal ablation arms**: deep-topr (grad top-r, no alignment), asymmetric-signature RGA.
- **Verify** GRAFT's exact reference name in the lit check; keep the generic "feature-space MaxVol" label if unclear.
- **Lit check (P0, BEFORE writing — do early)**: confirm no closer recent baseline (quality-based
  IFD/Cherry/DEITA/AlpaGasus; **NTK/kernel distillation**, **RKD & relational-KD variants**; recent
  2025–2026 selection). This also de-risks the **novelty** claim — critical for an Oral.

## 5. Resource discipline (non-negotiable)
1. Before every CUDA launch: `nvidia-smi` snapshot → pin the emptiest GPU via `CUDA_VISIBLE_DEVICES` →
   record `gpu_uuid`. Shared/contended box: use `PYTORCH_ALLOC_CONF=expandable_segments:True`, micro-batch
   training, free optimizer/grads before eval. Log every OOM / >10-min wait in `refine-logs/EXPERIMENT_TRACKER.md`.
2. **Cache** per-sample gradient fingerprints (teacher side especially — one-time, student-independent).
3. Kill any run >3× its estimate. **Total budget cap ≈ 40 GPU-days** (Oral scope incl. E9 large-model
   scale) — if exceeded, halt & report. Stage spend: run E0 + E1 (both phases) + E8/E10 first pass
   (~10 GPU-d) → hit the Oral decision point → only then commit the remaining ~30 GPU-d (E9 scale, E2
   full, E11) if the decisive-win + theory signals are there.
4. Every result reports **seed std** and **selection-inclusive wall-clock**.

## 6. Deliverables
- Result JSONs per experiment in `pilot/`; a rebuilt `RGA_experiment_results.xlsx`.
- Tables: E1 headline (methods × budgets), E2 (datasets × pairs), E3 (mechanism), E4 (ablations),
  E5 (wall-clock/break-even), E6 (gate), E7 (CKA).
- Figures: budget curve (RGA vs baselines); information-ladder / CKA concept fig; wall-clock break-even.
- Paper: positioning = **"teacher–student *parameter-gradient relational alignment* is the right
  data-valuation signal for KD"** (form = select/reweight/curriculum, per E1); honest limitations
  (margins, scope); claims via result-to-claim.
- Novelty note vs RKD (feature-relational), NTK distillation, LESS, TAGCOS.
- **Released code + configs** (reproducibility — Oral expectation): the RGA pipeline, all baselines, seeds.

## 7. Progress log (update after every step)

| date | experiment | status | key result / decision | next |
|------|-----------|--------|----------------------|------|
| 2026-07-18 | plan finalized (Oral target) | done | method frozen (symmetric NLL-grad, last-K); suite E0–E11 set; Oral gaps = margins/theory/scale | write E0 code |
| | lit-check (novelty + baselines) | todo | do EARLY (P0) | |
| | E0 signatures+diagnostics | todo | | |
| | E1 Phase A: use-mode (subset/reweight/curriculum) | todo | P0 — decides method form | |
| | E1 Phase B: headline vs baselines (GATE) | todo | P0 — form-matched baselines + full-data ref + significance | |
| | E8 theory + validation | todo | P0-Oral | |
| | E10 hard tasks / extreme budgets | todo | P0-Oral (decisive-win hunt) | |
| | **ORAL decision point** | todo | decisive win + theory holds? GO-Oral / retarget-poster | |
| | E2 generalize | todo | | |
| | E3 mechanism | todo | | |
| | E4 ablations | todo | | |
| | E5 efficiency | todo | | |
| | E9 scale (7B student) | todo | P0-Oral | |
| | E6 gate/noise | todo | | |
| | E7 concept fig | todo | | |
| | E11 deep analysis | todo | | |
| | paper | todo | | |

## 8. Existing assets to reuse
- `pilot/exp_retrain_qwen.py` — extraction + micro-batch `kd_train` + ROUGE eval (adapt signature to symmetric last-K).
- `pilot/exp_final_compare_qwen.py` — fair head-to-head harness (relational + coverage + selection).
- `pilot/exp_baselines_qwen.py` — EL2N/GraNd/TAGCOS/GRAFT. `pilot/exp_less_qwen.py` — LESS.
- `pilot/exp_decision_qwen.py` — CKA / token-vs-deep analysis (for E0/E7).
- `pilot/exp_e6_gate.py` — teacher-noise gate (vision; adapt to LLM for E6).
- SaGD: `load_teacher/load_student`, `InstructionDataset/SquadDataset`, `evaluate_rouge`/`evaluate_all`,
  `CountSketchProjector`, `gradient_pca_selection` (teacher NLL gradient over attention params).

## 9. One-line to execute (paste into /goal)
```
按 RGA_PAPER_EXPERIMENT_PLAN.md 跑完 E0–E11 全部实验并写 ICLR-Oral 目标论文。方法已冻结(§0 对称 NLL
梯度 + 最后K层,产出逐样本价值信号 r),不要改方法。先做 lit-check(novelty + baselines,尽早)。
按 §3 run order:E0 建管线并确认 r 有结构且非 token 级;E1 先 Phase A 在算力对齐下决定 r 的用法
(SUBSET / REWEIGHT / CURRICULUM,含 full-data 参照),再 Phase B 用胜出形态跑主表——baseline 必须与
该形态对齐(§4),含 token 级 RGA 对照,报 win-rate(GPT-judge,主) + ROUGE-L + 种子方差 + 配对显著性
+ 含选择成本的端到端 wall-clock。E1 是硬门(RGA 须显著 ≥ TAGCOS & LESS,否则 STOP 写负面 note)。
跑完 E1 + E8(理论及其验证)+ E10(难任务/极端预算)首轮到 ORAL 决策点:有决定性优势 regime 且理论预测成立
→ 全力冲 Oral(E9 大教师→中小学生的规模、E2 泛化、E11 分析);否则诚实退回 poster,不硬编 Oral 叙事。
每次 CUDA 启动按 §5 做 nvidia-smi 检查 + pin 最空卡 + 记 gpu_uuid,用 expandable_segments + micro-batch
防 OOM,缓存师生梯度指纹(教师端一次性)。每步更新 §7 进度日志和 refine-logs/EXPERIMENT_TRACKER.md。
复用 §8 已有 harness。总预算上限 ~40 GPU-天,分阶段花(先 ~10 GPU-天到决策点)。从第一个未完成步骤继续。
```
