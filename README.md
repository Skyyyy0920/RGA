# RGA — Relational Gradient Alignment for KD Data Valuation

Score each candidate KD training sample by **pairwise relational alignment of two parameter spaces**:
for a sample, compare how the teacher and student relate it to a set of anchor samples in their
respective **parameter-gradient (empirical-NTK) spaces**. A large relational residual `r` means the
teacher encodes structure the student lacks (high value). `r` is then used to **select / reweight /
curriculum-order** KD training data (final form decided empirically).

## Start here
- **`RGA_PAPER_EXPERIMENT_PLAN.md`** — authoritative, execution-ready plan (method spec + experiment
  suite E0–E11 + settings + run order + baselines). This is the entry point.
- `RGA_ALIGNMENT_LOSS_SPEC.md` — the loss-form variant (alignment as a training objective).
- `MEETING_NOTES_method_comparison.md` — method-vs-baselines comparison (for advisor discussion).
- `refine-logs/EXPERIMENT_TRACKER.md` — per-run log incl. GPU-coordination notes.
- `RGA_KD_PROJECT.md`, `RGA_KD_PHASE_A_SUMMARY.md`, `RGA_KD_METHOD_FINAL.md`, `RGA_KD_PAPER.md` —
  earlier (vision-scale) diagnostics and the superseded proof-of-concept.

## Code (`pilot/`)
- `exp_retrain_qwen.py` — per-sample gradient extraction + micro-batch KD training + ROUGE eval.
- `exp_final_compare_qwen.py` — fair head-to-head (relational + coverage + selection).
- `exp_baselines_qwen.py` (EL2N/GraNd/TAGCOS/GRAFT), `exp_less_qwen.py` (LESS).
- `exp_decision_qwen.py` — CKA / deep-vs-token analysis. `pilot_rga_diag*.py` — vision-scale diagnostics.
- `build_excel.py`, `build_ppt.py` — regenerate the results workbook / slide deck from the result JSONs.

Result JSONs for completed runs are committed under `pilot/*.json`.

## Dependencies / environment
- Python env: `py310_torch24` (torch 2.9, transformers 4.57, peft, sklearn). 4× A100 80GB.
- **The LLM harnesses import from the separate SaGD project** (`/data/tianhao/SaGD/src`) for model/data/
  eval utilities (`load_teacher`, `InstructionDataset`, `evaluate_rouge`, `CountSketchProjector`, ...).
  That path is required to run the Qwen experiments and is **not** vendored here.
- Not committed (see `.gitignore`): datasets (`data/`), gradient-fingerprint caches and checkpoints (`*.pt`),
  run logs, `__pycache__`.
