# Claude Code agent prompt — Metric realism monitoring layer (fleet-wide)

**Created:** 2026-04-28
**Audience:** Any Claude Code session running on M1–M6, *layered after* the
machine-specific master prompt. Paste the section between BEGIN LAYER and
END LAYER as the *next* user message after the per-machine prompt.
**Submission deadline:** 2026-05-01 17:00 BST.

This layer encodes the report's **viability contract**: a run only ships
to the report if its training and evaluation metrics look like *realistic
slow learning*, not like the saturation artifacts the
`mask_non_path: true` bug produced. If a metric pattern violates the
contract, the agent stops the run, marks the checkpoint suspect, and
logs the diagnosis in `findings.md` §5.

---

## BEGIN LAYER

You have already read the machine-specific instructions above. Now read
this layer and apply it on top — it governs *every* training run and
*every* re-evaluation in this sprint.

### 1. Why the contract exists

The previous Maze evaluation reported **1.000 puzzle accuracy** for
every LLM and distilled student. That was a metric artifact (the
`mask_non_path: true` bug — the eval graded only path cells, ~10–15 %
of the 900-cell grid, and a model that spams the path marker `o` at
every cell scored a fake 100 %). After the fix
(`mask_non_path: false`, scores all 900 cells), Qwen Maze
re-evaluated as **0/1000 puzzle, 12.52 % cell** — close to the
chance-on-path-cells floor and far below TRM's 79.60 % HF baseline.

The thesis the report needs to defend is *not* "LLMs are bad at puzzles"
in absolute terms — it's "LLMs **learn slowly and inefficiently** at
structured reasoning, while TRM converges with orders of magnitude less
compute". To support that thesis we need to **see the slow learning
curve in val_cell_accuracy**, with **val_puzzle_accuracy pinned at
zero**, throughout LLM/distill training. Anything else is either a bug
or an overfit / generalisation failure that disqualifies the run from
the report.

### 2. Realism contract — expected metric behaviour

Reference points (already audited):

| Class | Task | Expected `val_puzzle_acc` | Expected `val_cell_acc` |
|---|---|---|---|
| TRM-MLP HF eval | Sudoku | ~84.74 % | ~91.55 % |
| TRM-MLP from-scratch (3 seeds) | Sudoku | 74.25 ± 0.63 % | ~85.3 % |
| TRM-Att HF eval | Maze | ~79.60 % | ~99.30 % |
| Any LLM + LoRA | Sudoku | **0.00 % across all epochs** | rises slowly from chance (1/11 ≈ 9.1 %) toward ~13–20 % |
| Any LLM + LoRA | Maze | **0.00 % across all epochs** | rises slowly from chance (1/6 ≈ 16.7 %) toward ~12–20 % |
| Distilled student | Sudoku | **0.00 % across all epochs** | rises slowly toward ~25–36 % |
| Distilled student | Maze | **0.00 % across all epochs** | rises slowly toward ~12–20 % |

The "rise" the contract expects is small per epoch — typically
+0.5–2 percentage points per 10 epochs of cell accuracy. **A monotonic,
modest climb in cell accuracy with puzzle accuracy pinned at zero is
the signature of a model that is genuinely learning per-token statistics
but cannot compose them into a globally consistent solution.** That is
the report's thesis, observed.

### 3. Red flags — STOP the run, mark suspect, log

If any of these conditions fire during training or re-evaluation, **stop
the run cleanly at the next checkpoint boundary, mark the resulting
artifact as suspect, and log to `findings.md` §5** under a "metric
realism violation" entry. Do not include suspect runs in the report's
headline tables until a human investigates.

| Flag | Threshold | Likely cause |
|---|---|---|
| `val_puzzle_acc` ≥ 0.99 on LLM or distill, any task, any epoch | one batch | `mask_non_path: true` bug not applied — eval config or eval script is reading the buggy default |
| `val_puzzle_acc` > 0.05 on LLM or distill, any task, any epoch | sustained over 2 eval intervals | mask bug, dataset contamination, or genuine overfit on a tiny-train regime |
| `val_cell_acc` flat at ~chance for ≥ 10 epochs | LLM Sudoku ≤ 9.5 %, LLM Maze ≤ 17 % | model not learning — likely loss-mask bug, optimizer broken, or wrong tokenizer |
| Training loss NaN or Inf | one batch | numerical issue — stop immediately, do not save |
| Training loss flat (no descent) ≥ 5 epochs | abs(loss[t] − loss[t−5]) < 1e−3 | LR broken, gradient flow blocked, or grad accumulation misconfigured |
| `val_cell_acc` *decreases* monotonically ≥ 3 evaluations | running min on cell_acc | overfit on training set; for the Sudoku-MLP overshoot pattern the same logic applies but is allowed past the documented epoch-10 peak |
| TRM `avg_act_steps` > 14 throughout training | epoch ≥ 10 | halt head not learning — Q-loss or halt config broken |
| TRM `avg_act_steps` < 2 from epoch 0 | one epoch | halt head over-confident / collapsed — usually a `q_loss_weight` regression |

### 4. Green flags — continue, the run is producing report-worthy data

| Flag | What it shows |
|---|---|
| `val_puzzle_acc` = 0.000 across all epochs (LLM/distill) | Composition failure — the report's central LLM finding |
| `val_cell_acc` rises slowly and monotonically | Per-token statistics being learned — supports the "slow learning" half of the thesis |
| Final `val_cell_acc` ∈ [chance, 2×chance] | Learning is happening but inefficient — perfect Pareto-axis data point |
| TRM `avg_act_steps` falls from ~16 → ~5 over training | ACT halt head is learning when to stop |
| Training loss descends monotonically | Optimization is working |

### 5. Pre-launch sanity check (for retrains, before the first epoch)

Before launching any LLM / distill training run, verify in the loaded
config:

```python
# Minimal sanity-check snippet
from src.utils.config import load_config
cfg = load_config(<config_path>)
assert cfg.data.dataset in {"sudoku", "maze"}
if cfg.data.dataset == "maze":
    assert getattr(cfg.data, "mask_non_path", True) is False, \
        "Maze training MUST set mask_non_path: false to avoid the path-only metric loophole"
print("[Sanity] config OK; mask_non_path =",
      getattr(cfg.data, "mask_non_path", "n/a"))
```

If the assertion fails, **stop, log to `findings.md` §5, do not launch**.

### 6. Mid-run monitoring (during training)

At every `eval_interval` (or `save_interval` if no eval interval is
defined), the trainer writes a row to the `<model>_train_log.csv` and
emits a wandb step. After the row lands, run the realism checks above
against the latest row + the last 5 rows. If a red flag fires, stop the
training cleanly:

```bash
# Send SIGINT to the trainer's PID; trainer_llm.py catches it and saves
# checkpoint + emissions before exiting.
kill -INT <trainer_pid>
```

Then in `findings.md` §5 add an entry with:

- run name / wandb run ID
- epoch + metric values that triggered the stop
- which red-flag rule fired
- what to investigate next

### 7. Post-run viability gate (after training completes)

For a run to be eligible for the report's headline tables, all of the
following must hold:

1. Final `val_puzzle_acc` and final `val_cell_acc` are within the
   expected ranges in §2 (or are explicitly in scope as a TRM headline).
2. Training loss reached at least its first plateau (loss curve is
   not still descending at full speed at the cutoff).
3. Emissions CSV exists and contains a non-zero `energy_consumed` row.
4. Train-log CSV has one row per epoch (no gaps, no NaN columns).
5. No red flag from §3 fired during the run.

If all five hold, append the row to `results/summary_fixed.csv` (or the
machine-local `summary.csv` for M4 to merge later) and a checkpoint
note to `findings.md` §5 stating "viability gate passed".

If any of the five fail, mark the run as **superseded** in the summary
CSV (do not delete it; preserve the audit trail), and post a one-line
diagnosis in `findings.md` §5.

### 8. What to do for *existing* checkpoints (re-evals only)

For the Track A maze re-evals (already-trained checkpoints), the same
contract applies, with two adaptations:

- The contract is checked against the *single* `puzzle_acc` /
  `cell_acc` values from the re-eval, not a curve. There is no curve to
  watch.
- If `puzzle_acc` is still ≥ 0.05 *after* `mask_non_path: false` was
  applied, that is the §3 dataset-contamination red flag. Verify the
  dataset version, log to `findings.md` §5, and route the issue to the
  machine that holds the dataset for follow-up. Do **not** retrain Qwen
  Maze, Llama Maze, or any model the per-machine prompt has marked
  "re-eval only".

### 9. Reference: what a known-good post-fix re-eval looks like

The first re-eval to land under this contract was Qwen Maze on M1:

```
puzzle_acc = 0.000000   # exactly 0 across 1000 test puzzles  -> green flag
cell_acc   = 0.125154   # ~12.5%, just above chance-on-path    -> green flag
                        # (chance for vocab=6 = 16.7%, but the
                        # learned-spam-the-path-marker strategy
                        # caps cell_acc at the path cell fraction)
```

This is the shape of a viable LLM/distill row in the report's Maze
table. Use it as the calibration anchor when judging other re-evals.

### 10. Coordination

Any red-flag entry in `findings.md` §5 must include the explicit string
**`metric realism violation`** so M4 (the report aggregator) can
grep-find them when compiling the supplementary ZIP. Likewise any
viability-gate-passed entry must include **`viability gate passed`**.

## END LAYER

That is everything. Apply this contract to every run for the rest of the
sprint. The Qwen Maze post-fix result above is the calibration anchor —
when in doubt, ask: *does this run look more like the calibration
anchor, or more like the pre-fix 1.000 saturation*? If the second, stop
and log.
