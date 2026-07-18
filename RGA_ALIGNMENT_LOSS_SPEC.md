# RGA as a Training Loss — Relational Gradient Alignment in Parameter Space (spec v0)

> **Direction change (2026-07).** Per the advisor: the goal is **parameter-space alignment**, realized as
> **pairwise / relative** alignment (because the teacher and student parameter spaces have different
> dimensions). "Ensure a group of samples follows the **same pattern** in the two spaces." This is a
> **training objective (loss)**, NOT data selection. This doc specifies that method so it can be
> confirmed with the advisor before implementation.

## 1. Intuition

Teacher and student have different-dimensional parameter spaces → can't align them directly. But we can
compare **how a group of samples relate to each other** in each space (the pairwise Gram / kernel), and
train the student so that its pattern **matches** the teacher's. This is relational distillation, but in
**parameter-gradient (NTK) space** instead of feature space.

## 2. Definitions

- Student `S` (params `θ_S`, **trainable**), teacher `T` (frozen). Mini-batch `B = {x_1,…,x_n}`.
- **Per-sample parameter signatures** (over a chosen parameter subset `P` — e.g. last-K transformer
  blocks or LoRA, for tractability):
  - Student: `g_S(x_i) = ∇_{θ_S∈P} ℓ(x_i)` — gradient of a per-sample loss (KD or CE) w.r.t. student
    params. **Must stay differentiable w.r.t. θ_S** (it enters a loss we backprop through).
  - Teacher: `g_T(x_i)` = teacher **eNTK** signature `∇_{θ_T} f_T(x_i)` (or teacher loss-gradient),
    **frozen → precomputed, no grad.**
- **Pairwise relational matrices (n×n)**, e.g. cosine Gram:
  - `K_S[i,j] = ⟨g_S(x_i), g_S(x_j)⟩ / (‖g_S(x_i)‖‖g_S(x_j)‖)`
  - `K_T[i,j] = ⟨g_T(x_i), g_T(x_j)⟩ / (…)`

## 3. The alignment loss

Relational distance that is invariant to the dimension mismatch and to orthogonal/scaling changes
(so "same pattern" is well-defined):

```
L_align(B) = 1 − CKA(K_S, K_T)          # CKA = linear centered kernel alignment
             (alternative: ‖ H K_S H − H K_T H ‖_F^2 , H = centering)
```

Total training objective:
```
L = L_KD  +  λ · L_align
```
The student is trained (in addition to standard KD) so its **parameter-space pairwise pattern follows
the teacher's** — exactly "一组样本在两个 space follow 同一个 pattern."

## 4. The technical crux (be honest with the advisor about this)

`g_S(x_i)` is itself a **gradient** of θ_S, and it appears inside `L_align`, so `∇_{θ_S} L_align`
requires **second-order** differentiation (Hessian-vector products, `create_graph=True`). This is the
main cost/feasibility question. Mitigations:
- Restrict `P` to **last-K blocks or LoRA** → low-dim per-sample gradients → cheaper second order.
- Small **alignment batch** `n` (e.g. 8–16) for the Gram.
- **Random-project** the gradients (Count-Sketch, reuse our infra) before the Gram.
- Consider using the **eNTK output-Jacobian** (first-order in the loss) rather than a loss-gradient, so
  the student signature needs only first derivatives of `f_S` (still second-order overall, but cleaner).
- Reusable machinery: SaGD's `saliency.compute_differentiable` already does differentiable second-order
  (input-Jacobian) alignment with `create_graph=True` + flash-attn disabled — same pattern applies.

## 5. Novelty / related work (must check)

- **RKD** (Park et al., CVPR'19): relational distillation in **feature** space. Ours = relational in
  **parameter-gradient / NTK** space. Key delta.
- **NTK distillation / kernel matching**: check what exists (e.g. "distilling the NTK"), position against it.
- SaGD (this lab): saliency (input-gradient) **per-sample** alignment; ours is **relational (pairwise)**
  in **parameter** space — different object.
- → run a real novelty check before committing.

## 6. What carries over from the (now-secondary) selection work

The selection experiments were not wasted — they are the **diagnostic/motivation**: they showed the
**deep-gradient relational structure carries genuine teacher–student information** (deep vs token CKA
0.31; gradient methods beat feature/loss methods). That is exactly the evidence that *aligning* this
structure should help. The deep-gradient + Count-Sketch + relational infrastructure is directly reused.

## 7. Proposed plan

1. **Confirm this spec with the advisor** ("is this what you mean by pairwise/relative parameter-space
   alignment as a loss?"). ← do this first.
2. **Novelty check** vs RKD / NTK-distillation.
3. **Minimal tractable prototype** on Qwen: `P` = last-2 blocks or LoRA, alignment batch n=8, verify
   `L_align` is differentiable, trains stably, and adding it (vs plain KD) improves the student.
4. **Scale + baselines**: vs plain KD, vs RKD (feature-relational), vs SaGD, across Dolly (+SQuAD).
5. Ablations: which params `P`, batch `n`, λ, CKA vs Frobenius, eNTK vs loss-gradient signature.

## 8. Open questions for the advisor

- **Signature**: student **loss-gradient** vs **eNTK output-Jacobian**? (affects cost + meaning)
- **Which parameters** `P` should the alignment cover — last-K, LoRA, all attention?
- **Alignment target**: match teacher's Gram exactly, or only its *pattern* (CKA, scale-free)?
- Should KD loss stay, or is the alignment loss meant to (partly) replace it?
