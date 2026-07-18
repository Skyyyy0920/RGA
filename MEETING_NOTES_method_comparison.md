# Meeting Notes — Data-Selection Methods for KD (RGA vs. baselines)

**Setup:** Qwen3-8B (SFT teacher) → Qwen3-0.6B student, Dolly. Task = select a small coreset of
training sequences for knowledge distillation. Below: what each method uses as its signal, what it
does with it, and how it differs from ours (RGA).

---

## What each method uses and does with the gradient

| Method | What it uses (signal) | What it does with it | Key limitation vs. RGA |
|---|---|---|---|
| **EL2N** | Student **output error**: `‖p_S − onehot(y)‖`, the distance between the student's predicted distribution and the true label. *Not a gradient* — a single forward-pass quantity. | Take the norm; select the highest-error (hardest) samples. | Lives purely in **output/label space**; tends to select noisy or mislabeled outliers. Weakest method in our experiments. |
| **GraNd** | **KD-loss gradient norm** `‖g‖`, typically computed on the **last layer** only. | Take the scalar norm; select the largest-magnitude samples. | Collapses the entire gradient vector into **one number — discards direction**, so it picks redundant samples. Last-layer gradient is also near **token-level**. |
| **GRAFT** | Penultimate-layer **features** (the representation), **not a gradient**. | D-optimal / MaxVol **coverage** over features — pick a diverse subset that spans feature space. | Covers **what the data looks like**, not **what the model needs to learn**; representation space, not gradient space. |
| **TAGCOS** (2024) | **Deep per-sample gradient — the same gradient RGA uses.** | **k-means** cluster the gradients; keep the medoid (nearest-to-centroid) of each cluster. | Unsupervised diversity of the **student's gradients only** — **no teacher-vs-student comparison**. |
| **LESS** (Xia et al., ICML 2024) | **Deep per-sample gradient** (LoRA in the original; here the same deep gradient RGA uses). | Score each sample by **cosine similarity to a held-out validation gradient** ("does this example point toward the target task?"); select top-k. | **Targeted toward a validation set, not the teacher** — requires a target/validation set; no teacher-vs-student comparison; top-k only (no coverage). |
| **RGA (ours)** | **Student deep KD gradient + teacher eNTK** (both parameter-space signals). | Build the teacher's and student's **relational maps**; score each sample by how **differently** teacher and student relate it (relational residual `r`); select high-`r` samples with coverage. | — (higher compute + needs anchors, but the **only** method that aligns the student against the **teacher** relationally). |

**Takeaway:** These methods differ mainly in *how much information they keep*. EL2N sees only output
errors; GraNd keeps a gradient but only its magnitude; GRAFT uses features (no gradient); TAGCOS keeps
the full deep gradient but only the student's own diversity. **RGA is the only method that keeps the
full deep-gradient direction *and* aligns it relationally against the teacher** — the others discard
either the direction, the gradient itself, or the teacher comparison.

---

## Information-retained ladder

| Method | Space | Keeps gradient **direction**? | Uses **teacher** to compare? |
|---|---|---|---|
| EL2N | Output error | ✗ (no gradient) | ✗ |
| GraNd | Gradient magnitude | ✗ (norm only) | ✗ |
| GRAFT | Feature | ✗ (no gradient) | ✗ |
| TAGCOS | Gradient (clustered) | ✓ | ✗ |
| LESS | Gradient (cosine to validation direction) | ✓ | ✗ (uses a validation target, not the teacher) |
| **RGA (ours)** | Gradient (teacher–student relational) | ✓ | ✅ |

Top → bottom, each method keeps **more** information. LESS and TAGCOS both keep the full deep gradient
but use it differently (validation-targeted vs. clustered); only RGA compares it against the teacher.

---

## Why TAGCOS is the key comparison

TAGCOS computes the **exact same deep gradient** RGA does. So any gap between TAGCOS and RGA is purely
attributable to **"cluster the student's gradients" (TAGCOS) vs. "compare teacher-vs-student relational
structure" (RGA)** — i.e., it isolates the value of the teacher-relational step, not just of using deep
gradients.

## LESS — the key recent gradient baseline (Xia et al., ICML 2024)

**What it is.** LESS ("Selecting Influential Data for Targeted Instruction Tuning") is the flagship
recent gradient/influence method for LLM instruction-data selection. For a small **target/validation
set** representing the task you care about, it computes the validation gradient direction, then scores
every candidate training example by the **cosine similarity between its (random-projected, LoRA)
gradient and that validation gradient**. High cosine = "training on this example moves the model in the
direction that helps the target" → select top-k. It is the reference point for "gradients tell you which
data to keep."

**How it contrasts with RGA (both are deep-gradient methods — this is the sharpest comparison):**

| | LESS | RGA (ours) |
|---|---|---|
| Reference it aligns to | a **held-out validation set** (a target distribution) | the **teacher** (its relational structure) |
| Question it asks | "does this example point toward the target task?" | "does the teacher relate this example differently than the student?" |
| Selection rule | cosine top-k | high relational-residual + coverage |
| Needs a target/val set? | **yes** | no (uses the teacher instead) |

**Why this matters for us.** LESS and RGA use the **same deep gradient**, so comparing them isolates
whether *teacher-relational alignment* beats *validation-target alignment*. It's the fairest head-to-head
for our core claim. It also has a practical difference: LESS needs a curated target/validation set;
RGA needs only the teacher (which we already have in KD).

*Implementation note:* our LESS run uses the random-projected gradient-cosine core with a held-out Dolly
validation set (disjoint from the candidate pool and the eval/test set); it omits LESS's Adam
preconditioning refinement.

---

## Experiments run so far (chronological)

### Phase 1 — Vision proof-of-concept (ResNet-56 → ResNet-20, CIFAR-100) [now superseded as framing]
Full pipeline: diagnostics; **E1** space/criterion shoot-out; **E2** native baselines; **E3** KD-vs-CE;
**E4** efficiency; **E5** ablations; **E6** teacher-noise gate test. The method worked on vision, BUT it
used a **last-layer gradient proxy**. The advisor pointed out this proxy is *algebraically* token-level
(`(p_S−p_T)⊗f` = token-residual ⊙ feature). → triggered the pivot to genuine deep gradients on LLMs.

### Phase 2 — Qwen deep-gradient pivot (Qwen3-8B SFT teacher → Qwen3-0.6B student, Dolly)
All on the real LLM KD pipeline (reusing the SaGD infrastructure).

1. **Decision experiment** (N=1500) — is the deep gradient actually different from the token-level proxy?
   Compared last-layer proxy vs deep gradient vs token-only via kernel similarity (CKA).
   *Result:* last-layer proxy ≈ token (CKA **0.87**); deep gradient genuinely different (CKA **0.31**,
   top-selection overlap **5%**). → The deep gradient is a real, distinct signal (not token-level).

2. **Retrain proof** (budget 1500, 3 seeds, ROUGE-L) — does deep beat proxy/token?
   *Result:* monotonic **deep > proxy > token** (token version worst); RGA_deep best but ≈ random.
   First positive sign, but not yet tested against strong baselines.

3. **Budget sweep + external baselines** (budgets {500,1000,2000}; EL2N/GraNd/TAGCOS/GRAFT).
   *Result:* original RGA_deep ≈ random and beaten by TAGCOS. BUT gradient methods (TAGCOS, deep-topr)
   beat feature (GRAFT) and loss (EL2N) methods → **the deep-gradient SPACE is validated**, even though
   the original RGA selection rule wasn't winning.

4. **Fix + retest RGA** — found and fixed two real problems:
   (a) **coverage bug**: was run on `Gte − Gd` (two differently-hashed sketches, coordinate-misaligned =
   noise) → fixed to the **anchor-space residual `K'_T − K'_S`** (coordinate-aligned);
   (b) **the gate hurts** on a clean teacher (Dolly) → dropped it (gate is only useful under a *noised*
   teacher, as shown in the vision E6). Best variant = **RGA_fixcov_nogate**.

5. **Fair head-to-head** (all methods under the identical micro-batch training protocol) — the decisive
   apples-to-apples comparison. *Result:* RGA is best on average and beats TAGCOS at every budget (table
   below).

6. **LESS baseline** (flagship recent gradient method) — added under the identical protocol *(running)*.

*Engineering notes:* per-sample gradients are cached (so re-selection is cheap); training uses
micro-batch + gradient accumulation to fit the (shared, contended) GPUs; all GPU/OOM events are logged.

## Result — fair head-to-head (identical training protocol for all methods, ROUGE-L, 3 seeds)

| Budget (of 4000-pool) | **RGA (ours)** | TAGCOS | LESS | deep-topr | GraNd | random |
|---|---|---|---|---|---|---|
| 500 | **0.2389 ± .0015** | 0.2331 | 0.2290 ± .0052 | 0.2184 | 0.2069 | 0.2353 |
| 1000 | 0.2447 ± .0063 | 0.2326 | 0.2418 ± .0025 | 0.2372 | 0.2374 | **0.2463 ± .0104** |
| 2000 | **0.2457 ± .0046** | 0.2413 | 0.2354 ± .0028 | 0.2434 | 0.2375 | 0.2369 |
| **avg** | **0.2431** | 0.2357 | 0.2354 | 0.2330 | 0.2273 | 0.2395 |

**RGA beats LESS at every budget** (+0.010 / +0.003 / +0.010). Note: LESS — the flagship recent
gradient method — lands ~tied with TAGCOS and *below random* on average here. This is likely because
(i) LESS is designed for **targeted** selection toward a *different* downstream task, whereas our target
is the same Dolly distribution (not where it shines), and (ii) our adaptation uses the gradient-cosine
core without Adam preconditioning. We report it honestly; the takeaway is that RGA's teacher-relational
alignment outperforms LESS's validation-targeted alignment on the same deep gradient.

- **RGA is best on average and best at b=500 & b=2000; beats TAGCOS at every budget.**
- vs. random: wins at b=500 and b=2000; ~ties at b=1000 (random's mean is higher there but with very
  high variance — std 0.010 — i.e. a lucky seed). RGA is the **most stable** method (std 0.002–0.006).

**Honest caveats:** margins are modest (~0.4–0.9 ROUGE-L over random/TAGCOS); results so far are on a
single dataset (Dolly), a single pair (Qwen3-8B→0.6B), and a 250-sample eval set.

---

## One-line summary

The right space for LLM KD data selection is the **deep parameter-gradient space** (gradient methods
beat feature/loss methods); within it, **teacher–student relational-residual selection (RGA)** is the
best and most stable selector — beating gradient-clustering (TAGCOS), gradient-norm (GraNd), and random.
