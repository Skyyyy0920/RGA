# The Right Space for Teacher-Guided Data Selection is Gradient-Relational Alignment

*Draft — RGA-KD finding paper (Phase D). All numbers from `pilot/*_RESULTS.json`; 3 seeds, mean±std.*

## Abstract

Knowledge-distillation (KD) data selection has, almost universally, scored samples in the
output/loss scalar space, even though theory and recent empirics locate the useful KD signal in
**parameter-gradient geometry**. We ask a sharper question: *given a teacher, which samples carry
training value?* and answer it in a space no prior selector occupies — the **teacher–student
gradient-relational** space. For each sample we compare how the teacher and the (warmed) student
wire it into the sample×sample gradient-kernel relation; the per-sample **relational residual**
`r(i)=1−corr(K'_T[i,:],K'_S[i,:])` is large when the teacher encodes relational structure the
student has not reproduced. Three findings, each adversarially tested:
1. **`r` is a real, correctly-signed, loss-distinct selection signal**: training only on the most
   *aligned* (low-`r`) samples is reliably the worst choice across an MLP/digits pilot and a
   ResNet-56→ResNet-20/CIFAR-100 pair; `r` correlates only −0.2 with KD loss.
2. **The relational space beats the spaces selectors actually use**: relational coverage beats
   late-feature-space coverage by ~6σ at aggressive budgets, and beats gradient-magnitude/subspace
   (GradSpan-KD/DPP), gradient clustering (TAGCOS), feature MaxVol (GRAFT), EL2N and GraNd.
3. **A teacher-correctness gate is mandatory, not cosmetic**: under a noised teacher, ungated
   high-`r` selection fills 97% of the coreset with confident-wrong samples and collapses to chance;
   the gate `×p_T(y)` removes them (0% contamination) and recovers full accuracy.
We position this as a *finding*: the advantage is concentrated at non-degenerate aggressive budgets
(random is a strong baseline at extreme-low budgets), and it is **not specific to KD training** — the
teacher-guided coreset trains as well, or better, under plain cross-entropy.

## 1. Introduction

Curriculum/“easy-to-hard’’ ordering is a largely falsified lever for selection (When-Do-Curricula-Work;
Rethinking-Easy-to-Hard). Yet nearly every KD selector still ranks samples by an output-space scalar
(loss, margin, logit geometry). Theory (Panigrahi, ICLR’25) and empirics (PACED, 2026) instead place
the useful KD signal in **parameter-gradient geometry** — a space selectors have not occupied because
teacher and student have different parameter spaces, so their gradients cannot be compared directly.

We sidestep the dimension mismatch with a **relational** comparison: Gram/kernel matrices `K=GG^T` are
invariant to the orthogonal/basis ambiguity, so we never align parameters — we compare *how each model
relates a sample to other samples*. The per-sample relational residual `r(i)` then ranks samples by
"structure the teacher has that the student lacks." This is the first selector to operate in the
teacher–student gradient-relational space, combined with a teacher-correctness gate and D-optimal
coverage.

## 2. Related work (delta)

- **Relational distillation** (RKD CVPR’19; CKA/SVCCA): relations in *representation* space, used to
  *distill* or *measure* — never to *select/order* KD training data in gradient-kernel space.
- **Output-space KD selectors**: Selective-KD (ACL’21, per-token logit reweighting), TGeo-KD (ICLR’24,
  output triangle geometry) — scalar/output space, not gradient-relational.
- **Gradient-space coresets**: LESS (ICML’24, supervised cos-top-k, no coverage), TAGCOS (2024,
  gradient-mean matching), GradSpan-KD (KD-gradient magnitude/subspace). **Delta of RGA-KD**:
  teacher–student *relational alignment* + teacher gate + D-optimal coverage — the combination is new,
  and §5 shows the relational space beats the magnitude/subspace and feature spaces these methods use.

## 3. Method: RGA-KD

Notation: teacher `T` (frozen), student `S`; per-sample student KD gradient
`g_S(x)=∇_{θ_S}KL(p_T‖p_S)`; teacher eNTK signature `g_T(x)=J_T(x)`. Pipeline (frozen spec in
`RGA_KD_METHOD_FINAL.md`):

0. **Warm up** the student a few epochs on the KD task (calibrate the head; otherwise per-sample
   gradients are dominated by a common transient).
1. **Signatures** (last-layer, closed form — validated against full gradients in §5.4):
   `g_S(x)=(p_S−p_T)⊗f_S(x)`, teacher last-layer eNTK relational ⇒ `f_T(x)`.
2. **Project** with Count-Sketch to `d=2048`; **anchors** `m=256`; relational matrices
   `K=GG[A]^T`, double-centered `K'`.
3. **Residual** `r(i)=1−corr(K'_T[i,:],K'_S[i,:])`.
4. **Gate** (soft) `s(i)=r(i)·p_T(x_i)[y_i]` — essential (§5.3).
5. **Coverage select**: among top-`3b` by `s`, greedy log-det (D-optimal) over the relational-residual
   feature → `b` diverse samples.

## 4. Experimental setup

- Pair: ResNet-56 (teacher, 70.9% CIFAR-100 test) → ResNet-20 (student); pool N=10,000 from the
  train set; full 10k test set. Pilot: MLP→MLP / sklearn-digits (Phase A).
- Budget axis tied to the 90%-energy rank `R≈260` of the projected student KD gradient; we sweep
  500–4000 (5–40 samples/class) and **report the full curve** (the advantage is budget-dependent).
- **3 seeds; std is a reported metric. End-to-end wall-clock includes selection cost** (signature
  build + projection + relational matrices + coverage + teacher soft-label inference).
- Infrastructure self-contained in `pilot/pilot_rga_diag*.py`, `pilot/exp_e{1,2,3,5,6}_*.py`.

## 5. Results

### 5.1 The relational signal is real, correctly signed, loss-distinct (E0/E1)
`r=0.59±0.17`, corr(`r`,KD-loss)=−0.21, corr(`r`,noise)≈0. The falsification arm — training on the
most-aligned (low-`r`) samples — is **the worst method at every budget** (e.g. 0.045/0.076/0.147/0.295
at b=500/1k/2k/4k vs random 0.099/0.159/0.251/0.386). High-`r` ≠ high-loss: loss-top-k is far worse
than relational selection.

### 5.2 Relational space > feature / magnitude / clustering spaces (E1, E2)
Test accuracy (mean±std, 3 seeds):

| budget | **RGA-KD** | GradSpan/DPP | feature-cov (GRAFT) | TAGCOS | EL2N | GraNd | random |
|--------|-----------|--------------|---------------------|--------|------|-------|--------|
| 1000 | 0.155±.011 | 0.120±.015 | 0.163±.006 | 0.178±.007 | 0.098±.003 | 0.158±.008 | 0.157±.018 |
| 2000 | **0.270±.011** | 0.266±.032 | 0.254±.010 | 0.248±.006 | 0.207±.005 | 0.248±.008 | 0.242±.001 |
| 4000 | **0.429±.007** | 0.406±.014 | 0.371±.010 | 0.375±.015 | 0.337±.008 | 0.392±.007 | 0.392±.005 |

At aggressive budgets RGA-KD is the best method, beating feature-space coverage by **+5.8pt (~6σ)** at
b=4000 — feature-space coverage collapses (0.371) while relational coverage holds (0.429). It beats the
required native baselines GRAFT and TAGCOS at b≥2000. At the smallest budget, unbiased random / clustering
are strong and RGA-KD does not lead — an honest budget-dependence we report rather than hide.

### 5.3 The teacher gate is mandatory (E6)
Injecting 30% confident-wrong teacher labels: high-`r` detects them (AUC 0.94) but *because they look
like unlearned structure*, ungated selection fills **97.4%** of the coreset with them and collapses to
**0.011** (chance). The gate `×p_T(y)` (AUC 1.00 at flagging them) yields **0%** contamination and
**0.299** accuracy. The gate is the difference between a working method and an actively harmful one.

### 5.4 Ablations and efficiency (E5/E4)
At b=4000 (3 seeds, test acc):

| ablation | acc | note |
|----------|-----|------|
| **RGA-KD (main: studentKD+teacherENTK, coverage, d=2048, m=256)** | **0.433±.011** | reference |
| criterion: top-`r` (no coverage) | 0.318±.008 | **D-optimal coverage is essential: +11.5pt** |
| signature: both-eNTK (feature) | 0.403±.010 | main signature is best |
| signature: both-KD-feature | 0.420±.005 | — |
| projection d=512 / d=8192 | 0.414 / 0.441 | robust to projection dim |
| anchors m=64 / m=1024 | 0.425 / 0.437 | robust to anchor count |
| **E4: full student KD gradient** | 0.421±.013 | extract **0.6s** |
| **E4: last-layer proxy (used everywhere)** | 0.433±.011 | proj **0.02s** — equal quality, ~30× cheaper |

Two takeaways: (i) the **D-optimal coverage step is load-bearing** — plain top-`r` selection drops
11.5pt, i.e. selecting *diverse* high-residual samples (not just the highest) is what makes the
relational signal usable; (ii) the **last-layer gradient proxy matches the full gradient** in selection
quality at ~30× lower cost, so RGA-KD's selection overhead is negligible — ≈2s of selection per budget
plus a one-time ~6s signature/teacher-inference setup, versus ~22s to train one student on the coreset
and far more to train on the full pool. Selection cost is amortized after the first distilled student.

### 5.5 Not KD-training-specific (E3)
RGA-KD's advantage over loss-based (EL2N) selection is present under both KD and CE training, and is in
fact **larger under CE** (+0.11 vs +0.07 at b=4000). The relational signal is teacher-*guided* (it needs
a teacher to compute), but the selected coreset is broadly valuable — so we drop any "KD-special" claim.

## 6. Limitations (honest)

- **Budget-dependence**: at extreme-low budgets random class-coverage is competitive; RGA-KD's edge
  appears at non-degenerate aggressive budgets (≳20 samples/class here).
- **Single pair / modality so far**: results are CIFAR-100 ResNet-56→20 (+ digits pilot). WRN and a text
  (BERT) pair remain; the relational space is most motivated where teacher–student geometry differs most.
- **Last-layer gradient proxy**: used throughout for cost; §5.4 checks parity with full gradients but a
  fuller study across architectures is future work.
- **Teacher quality**: a 70.9% ResNet-56 teacher; a stronger teacher may shift absolute numbers.

## 7. Conclusion
The signal KD data selection has been missing is **teacher–student gradient-relational alignment**:
a real, correctly-signed, loss-distinct, gateable signal that selects better coresets than the feature,
gradient-magnitude, and clustering spaces prior selectors use — at negligible selection cost.
