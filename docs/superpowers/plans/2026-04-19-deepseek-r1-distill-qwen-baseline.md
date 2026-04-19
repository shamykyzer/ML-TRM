# DeepSeek-R1-Distill-Qwen-1.5B Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DeepSeek-R1-Distill-Qwen-1.5B LLM baseline to the ML-TRM fleet on both sudoku and maze at 30 epochs, with three shared trainer diagnostics (epoch-0 baseline eval, `lm/loss` alias, optional per-step logging) and a `loss_delta_pct` column in `summary.csv` that produce measurable non-learning evidence.

**Architecture:** Purely additive. Two new YAML configs. Three additive diffs to the shared `trainer_llm.py` (inherited by GPT-2, SmolLM, Qwen, Llama baselines — all get the diagnostics for free). One new optional field in `TrainingConfig`. One new computed column in `parse_train_log`'s return dict. Existing `baseline_llm.py` already has a DeepSeek dispatch branch (lines 51-54), so no model-code change.

**Tech Stack:** Python 3.10-3.12, PyTorch, HuggingFace Transformers + PEFT (LoRA) + BitsAndBytes (QLoRA for maze), Pydantic for config validation, Wandb for logging, PyYAML, CodeCarbon, pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-19-deepseek-r1-distill-qwen-baseline-design.md`

---

## File Structure

**Files to create (3):**
- `configs/llm_deepseek.yaml` — sudoku run config
- `configs/llm_deepseek_maze.yaml` — maze run config
- (implicit) new synthetic CSV fixtures inside `tests/test_aggregate.py`

**Files to modify (4):**
- `src/utils/config.py` — add `log_per_step: bool = False` field to `TrainingConfig`
- `src/training/trainer_llm.py` — three additive changes (epoch-0 eval, `lm/loss` aliases, per-step logging)
- `src/evaluation/aggregate.py` — add `initial_val_loss`, `final_val_loss`, `loss_delta_pct` to `parse_train_log` return dict
- `tests/test_aggregate.py` — 3 new test cases + LLM-format CSV fixtures

**Files not touched (by design):**
- `src/models/baseline_llm.py` — already routes `deepseek` → `["q_proj", "k_proj", "v_proj"]` on lines 53-54
- `main.py`, `src/data/*`, `src/evaluation/evaluate.py`, `scripts/*` — unchanged

---

## Task 1: Add `log_per_step` field to `TrainingConfig`

**Files:**
- Modify: `src/utils/config.py` (insert after line 100, in the Logging block)

- [ ] **Step 1: Edit `src/utils/config.py` — add the new field**

Use Edit to insert the new field right after the `eval_interval` field. The `old_string` should be the existing `eval_interval` comment + field; the `new_string` prepends that with the new `log_per_step` field.

```python
# OLD (exact match, currently around lines 95-100):
    log_interval: int = 50
    save_interval: int = 500
    # How often (epochs) to run the full validation pass. 0 = fall back to
    # log_interval (legacy behavior where logging and eval were fused). Split
    # them when eval is expensive — e.g. log_interval=5 for cheap train-metric
    # printing, eval_interval=50 to run the full test split only 10 times per
    # 500-epoch run instead of 100 times.
    eval_interval: int = 0
```

```python
# NEW (adds log_per_step between save_interval and eval_interval block):
    log_interval: int = 50
    save_interval: int = 500
    # Per-step LM loss logging. When True, every gradient-accumulation
    # micro-batch emits `lm/step_loss` to wandb. Off by default — enabling
    # produces 1000-4000 points per run which clutters the default dashboard.
    # Turn on for runs where the plateau claim needs within-epoch visibility
    # (e.g. DeepSeek-R1-Distill-Qwen-1.5B baseline runs).
    log_per_step: bool = False
    # How often (epochs) to run the full validation pass. 0 = fall back to
    # log_interval (legacy behavior where logging and eval were fused). Split
    # them when eval is expensive — e.g. log_interval=5 for cheap train-metric
    # printing, eval_interval=50 to run the full test split only 10 times per
    # 500-epoch run instead of 100 times.
    eval_interval: int = 0
```

- [ ] **Step 2: Verify the field is accessible via Pydantic**

Run:
```bash
python -c "from src.utils.config import TrainingConfig; c = TrainingConfig(); assert c.log_per_step is False; print('OK')"
```
Expected output: `OK` (exit code 0)

- [ ] **Step 3: Verify no existing YAML breaks**

Run:
```bash
python -c "from src.utils.config import load_config; [load_config(f'configs/{y}') for y in ['llm_qwen.yaml','llm_qwen_maze.yaml','llm_llama.yaml','llm_llama_maze.yaml']]; print('all 4 load')"
```
Expected output: `all 4 load`

- [ ] **Step 4: Commit**

```bash
git add src/utils/config.py
git commit -m "feat(config): add log_per_step flag for per-step LM loss logging"
```

---

## Task 2: Aggregator TDD — happy-path `loss_delta_pct`

**Files:**
- Modify: `tests/test_aggregate.py` (add LLM CSV fixtures near top + new test case)
- Modify: `src/evaluation/aggregate.py` (modify `parse_train_log`)

- [ ] **Step 1: Add LLM-format CSV fixtures near the other fixtures in `tests/test_aggregate.py`**

Use Edit. Insert these three constants right after the existing `OUT_OF_ORDER_TRAIN_LOG` constant (search for `OUT_OF_ORDER_TRAIN_LOG = """` and its closing `"""`, then insert after).

```python
# LLM trainer CSV schema (trainer_llm.py) — columns differ from the
# official/legacy TRM schemas above. Used for the non-learning diagnostic
# tests added with loss_delta_pct support.
LLM_TRAIN_LOG_WITH_BASELINE = """epoch,loss,val_loss,val_puzzle_acc,val_cell_acc,elapsed_min
0,,2.5000,0.0000,0.1900,0.0
10,2.4800,2.4900,0.0000,0.1895,15.3
20,2.4700,2.4700,0.0000,0.1896,30.5
30,2.4600,2.4500,0.0000,0.1905,45.8
"""

LLM_TRAIN_LOG_WITHOUT_BASELINE = """epoch,loss,val_loss,val_puzzle_acc,val_cell_acc,elapsed_min
10,2.4800,2.4900,0.0000,0.1895,15.3
20,2.4700,2.4700,0.0000,0.1896,30.5
30,2.4600,2.4500,0.0000,0.1905,45.8
"""

LLM_TRAIN_LOG_NEGATIVE_DELTA = """epoch,loss,val_loss,val_puzzle_acc,val_cell_acc,elapsed_min
0,,2.0000,0.0000,0.1900,0.0
10,2.1000,2.1000,0.0000,0.1895,15.3
20,2.1500,2.1500,0.0000,0.1896,30.5
30,2.2000,2.2000,0.0000,0.1905,45.8
"""
```

- [ ] **Step 2: Append the happy-path test at the bottom of `tests/test_aggregate.py`**

```python
# ---------------------------------------------------------------------------
# loss_delta_pct — non-learning diagnostic column (LLM trainer only)
# ---------------------------------------------------------------------------

def test_loss_delta_pct_computed_when_epoch_zero_row_present(tmp_path):
    """A train log containing an epoch=0 baseline row yields a numeric
    loss_delta_pct = (initial - final) / initial * 100."""
    log_path = tmp_path / "test_train_log.csv"
    log_path.write_text(LLM_TRAIN_LOG_WITH_BASELINE)

    result = parse_train_log(str(log_path))

    assert result is not None
    assert result["initial_val_loss"] == 2.5
    assert result["final_val_loss"] == 2.45
    # (2.5 - 2.45) / 2.5 * 100 = 2.0
    assert abs(result["loss_delta_pct"] - 2.0) < 1e-6
```

- [ ] **Step 3: Run the new test; confirm it fails**

Run:
```bash
pytest tests/test_aggregate.py::test_loss_delta_pct_computed_when_epoch_zero_row_present -v
```
Expected: FAIL — `KeyError: 'initial_val_loss'` or `KeyError: 'loss_delta_pct'` (the key isn't in the returned dict yet).

- [ ] **Step 4: Edit `src/evaluation/aggregate.py` — declare the two trackers**

In `parse_train_log`, right after the existing tracker declarations (search for `best_val_puzzle = 0.0` and locate the block declaring `best_val_cell`, `peak_epoch`, `train_time_min_at_peak`, `max_epoch`, `final_loss`, `max_elapsed`, `steps_sum`, `steps_count`), insert two new lines at the end of that block:

```python
# OLD (exact match, currently around lines 135-143):
    best_val_puzzle = 0.0
    best_val_cell = 0.0
    peak_epoch = -1
    train_time_min_at_peak = 0.0
    max_epoch = -1
    final_loss: float | None = None
    max_elapsed = 0.0
    steps_sum = 0.0
    steps_count = 0
```

```python
# NEW (adds two trackers at the end):
    best_val_puzzle = 0.0
    best_val_cell = 0.0
    peak_epoch = -1
    train_time_min_at_peak = 0.0
    max_epoch = -1
    final_loss: float | None = None
    max_elapsed = 0.0
    steps_sum = 0.0
    steps_count = 0
    # Non-learning diagnostics — require epoch-0 baseline row from the
    # LLM trainer's pre-training evaluation hook. Blank when absent.
    initial_val_loss: float | None = None
    final_val_loss: float | None = None
```

- [ ] **Step 5: Edit `src/evaluation/aggregate.py` — capture initial/final val_loss inside the loop**

Find the existing row-iteration body (search for `for row in rows:` inside `parse_train_log`). Insert a new block immediately after the `elapsed = _to_float(row.get("elapsed_min"))` line:

```python
# OLD (exact match, currently around lines 145-149):
    for row in rows:
        epoch = _to_int(row.get("epoch"))
        if epoch is None:
            continue
        elapsed = _to_float(row.get("elapsed_min"))
```

```python
# NEW (adds val_loss capture right after elapsed_min parse):
    for row in rows:
        epoch = _to_int(row.get("epoch"))
        if epoch is None:
            continue
        elapsed = _to_float(row.get("elapsed_min"))

        # Capture val_loss at epoch 0 (baseline) and epoch == max (final) for
        # the loss_delta_pct diagnostic. LLM trainer writes val_loss in column
        # 3; official/legacy TRM logs don't have this column, so the `.get`
        # returns None and both trackers stay blank — which is correct.
        vl = _to_float(row.get("val_loss"))
        if epoch == 0 and vl is not None:
            initial_val_loss = vl
        if vl is not None and epoch >= max_epoch:
            # max_epoch is updated further down in the loop; for out-of-order
            # rows the final pass overrides earlier captures. Using >= (not ==)
            # lets the first row seen at the new max-epoch win even before
            # max_epoch state has been updated.
            final_val_loss = vl
```

- [ ] **Step 6: Edit `src/evaluation/aggregate.py` — add computed delta to the return dict**

Find the return dict at the bottom of `parse_train_log` (search for `return {\n        "best_val_puzzle_acc"`). Replace the return block:

```python
# OLD (exact match, currently around lines 184-193):
    return {
        "best_val_puzzle_acc": best_val_puzzle,
        "best_val_cell_acc": best_val_cell,
        "peak_epoch": peak_epoch if peak_epoch >= 0 else max_epoch,
        "train_time_min_at_peak": train_time_min_at_peak,
        "final_train_loss": final_loss if final_loss is not None else 0.0,
        "final_epoch": max_epoch,
        "train_time_min": max_elapsed,
        "avg_act_steps": (steps_sum / steps_count) if steps_count else 0.0,
    }
```

```python
# NEW (adds initial_val_loss, final_val_loss, loss_delta_pct):
    if (
        initial_val_loss is not None
        and final_val_loss is not None
        and initial_val_loss > 0
    ):
        loss_delta_pct: float | str = (
            (initial_val_loss - final_val_loss) / initial_val_loss * 100.0
        )
    else:
        # Blank cell for runs without the epoch-0 baseline row (TRM runs,
        # pre-diagnostics LLM runs). Matches the ``attach_efficiency_metrics``
        # blank-cell convention for zero-correct runs.
        loss_delta_pct = ""

    return {
        "best_val_puzzle_acc": best_val_puzzle,
        "best_val_cell_acc": best_val_cell,
        "peak_epoch": peak_epoch if peak_epoch >= 0 else max_epoch,
        "train_time_min_at_peak": train_time_min_at_peak,
        "final_train_loss": final_loss if final_loss is not None else 0.0,
        "final_epoch": max_epoch,
        "train_time_min": max_elapsed,
        "avg_act_steps": (steps_sum / steps_count) if steps_count else 0.0,
        "initial_val_loss": initial_val_loss if initial_val_loss is not None else "",
        "final_val_loss": final_val_loss if final_val_loss is not None else "",
        "loss_delta_pct": loss_delta_pct,
    }
```

- [ ] **Step 7: Run the happy-path test; confirm it passes**

Run:
```bash
pytest tests/test_aggregate.py::test_loss_delta_pct_computed_when_epoch_zero_row_present -v
```
Expected: PASS (1 passed).

- [ ] **Step 8: Run the entire `test_aggregate.py` suite; confirm no regression**

Run:
```bash
pytest tests/test_aggregate.py -v
```
Expected: all 11 pre-existing tests plus the 1 new test pass — "12 passed".

- [ ] **Step 9: Commit**

```bash
git add tests/test_aggregate.py src/evaluation/aggregate.py
git commit -m "feat(aggregate): add loss_delta_pct non-learning diagnostic column"
```

---

## Task 3: Aggregator edge-case — blank when no epoch-0 row

**Files:**
- Modify: `tests/test_aggregate.py` (append one test)

- [ ] **Step 1: Append the blank-case test at the bottom of `tests/test_aggregate.py`**

```python
def test_loss_delta_pct_blank_when_no_epoch_zero_row(tmp_path):
    """A train log without an epoch=0 baseline row yields loss_delta_pct=""
    and both initial/final val_loss fields blank — matches pre-diagnostics
    LLM runs and all TRM runs."""
    log_path = tmp_path / "test_train_log.csv"
    log_path.write_text(LLM_TRAIN_LOG_WITHOUT_BASELINE)

    result = parse_train_log(str(log_path))

    assert result is not None
    assert result["initial_val_loss"] == ""
    # final_val_loss IS populated (the epoch=30 row has val_loss=2.45)
    # but loss_delta_pct is blank because initial is missing.
    assert result["final_val_loss"] == 2.45
    assert result["loss_delta_pct"] == ""
```

- [ ] **Step 2: Run the test; confirm it passes**

Run:
```bash
pytest tests/test_aggregate.py::test_loss_delta_pct_blank_when_no_epoch_zero_row -v
```
Expected: PASS. The code from Task 2 already handles this case correctly via the `if initial_val_loss is not None` guard.

- [ ] **Step 3: Run the entire `test_aggregate.py` suite**

Run:
```bash
pytest tests/test_aggregate.py -v
```
Expected: 13 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_aggregate.py
git commit -m "test(aggregate): cover loss_delta_pct blank case (no epoch-0 row)"
```

---

## Task 4: Aggregator edge-case — negative delta when val_loss increased

**Files:**
- Modify: `tests/test_aggregate.py` (append one test)

- [ ] **Step 1: Append the negative-delta test at the bottom of `tests/test_aggregate.py`**

```python
def test_loss_delta_pct_negative_when_val_loss_increased(tmp_path):
    """val_loss getting WORSE over training yields a negative delta —
    valid semantics, not a bug. The plateau figure can distinguish
    'flat' (delta ≈ 0) from 'degraded' (delta < 0)."""
    log_path = tmp_path / "test_train_log.csv"
    log_path.write_text(LLM_TRAIN_LOG_NEGATIVE_DELTA)

    result = parse_train_log(str(log_path))

    assert result is not None
    assert result["initial_val_loss"] == 2.0
    assert result["final_val_loss"] == 2.2
    # (2.0 - 2.2) / 2.0 * 100 = -10.0
    assert abs(result["loss_delta_pct"] - (-10.0)) < 1e-6
```

- [ ] **Step 2: Run the test; confirm it passes**

Run:
```bash
pytest tests/test_aggregate.py::test_loss_delta_pct_negative_when_val_loss_increased -v
```
Expected: PASS. The formula is sign-agnostic by construction.

- [ ] **Step 3: Run the entire `test_aggregate.py` suite**

Run:
```bash
pytest tests/test_aggregate.py -v
```
Expected: 14 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_aggregate.py
git commit -m "test(aggregate): cover loss_delta_pct negative case (val_loss rose)"
```

---

## Task 5: Trainer Change A — pre-training eval at step 0

**Files:**
- Modify: `src/training/trainer_llm.py` (insert block in `train()` after `self.carbon.start()`)

*TDD note:* `trainer_llm.py` is a training-loop integration point that is verified by the smoke test (Task 10) rather than unit-tested. The existing codebase does not unit-test trainers — it relies on integration (smoke) verification. This task does not add a unit test; the pass criteria are exercised in Task 10.

- [ ] **Step 1: Edit `src/training/trainer_llm.py` — insert the epoch-0 eval block**

Find the existing `train()` method (search for `def train(self) -> None:`). The current body starts with `self._init_log(); self.carbon.start(); t_start = time.time()` followed by early-stopping setup and the `for epoch in range(...)` loop. Insert the baseline eval block between the early-stopping setup and the epoch loop.

```python
# OLD (exact match, currently lines 96-112):
    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        # Early stopping state: a non-zero patience arms it. `best` starts at the
        # worst possible value for the chosen mode so the first eval always wins.
        es_patience = int(self.tc.early_stop_patience or 0)
        es_mode = self.tc.early_stop_mode
        es_metric = self.tc.early_stop_metric
        es_min_delta = float(self.tc.early_stop_min_delta)
        best_value = float("-inf") if es_mode == "max" else float("inf")
        best_epoch = 0
        stopped_early = False
        last_epoch = self.tc.epochs - 1

        for epoch in range(self.tc.epochs):
```

```python
# NEW (inserts baseline eval block before the epoch loop):
    def train(self) -> None:
        self._init_log()
        self.carbon.start()
        t_start = time.time()

        # Early stopping state: a non-zero patience arms it. `best` starts at the
        # worst possible value for the chosen mode so the first eval always wins.
        es_patience = int(self.tc.early_stop_patience or 0)
        es_mode = self.tc.early_stop_mode
        es_metric = self.tc.early_stop_metric
        es_min_delta = float(self.tc.early_stop_min_delta)
        best_value = float("-inf") if es_mode == "max" else float("inf")
        best_epoch = 0
        stopped_early = False
        last_epoch = self.tc.epochs - 1

        # Pre-training eval at step 0, before any gradient update. Anchors the
        # plateau plot: if val metrics at step 0 ≈ step N, the model never moved.
        # Without this baseline, the first data point is at step=log_interval
        # (e.g. 10) and reviewers can ask "how do we know it didn't briefly
        # learn and then stall?" — this answers that definitively.
        val_metrics_initial = self.evaluate()
        tqdm.write(
            f"[baseline] Epoch 0/{self.tc.epochs} | "
            f"ValLoss: {val_metrics_initial['loss']:.4f} | "
            f"Puzzle: {val_metrics_initial['puzzle_acc']:.4f} | "
            f"Cell: {val_metrics_initial['cell_acc']:.4f}"
        )
        self._append_log([
            0, "",  # no train_loss at epoch 0 — no gradient steps taken yet
            f"{val_metrics_initial['loss']:.4f}",
            f"{val_metrics_initial['puzzle_acc']:.4f}",
            f"{val_metrics_initial['cell_acc']:.4f}",
            "0.0",
        ])
        if self.use_wandb:
            wandb.log(
                {
                    "val/loss": val_metrics_initial["loss"],
                    "val/lm_loss": val_metrics_initial["loss"],
                    "val/puzzle_acc": val_metrics_initial["puzzle_acc"],
                    "val/cell_acc": val_metrics_initial["cell_acc"],
                    "val/accuracy": val_metrics_initial["cell_acc"],
                    "val/exact_accuracy": val_metrics_initial["puzzle_acc"],
                },
                step=0,
            )

        for epoch in range(self.tc.epochs):
```

- [ ] **Step 2: Syntax check — ensure file still parses**

Run:
```bash
python -c "import src.training.trainer_llm; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer_llm.py
git commit -m "feat(trainer_llm): add pre-training evaluation at step 0"
```

---

## Task 6: Trainer Change B — `lm/loss` and `val/lm_loss` aliases

**Files:**
- Modify: `src/training/trainer_llm.py` (extend the periodic `wandb.log` call inside the epoch loop)

- [ ] **Step 1: Edit `src/training/trainer_llm.py` — add the two new keys to the wandb.log dict**

Find the existing `wandb.log` call inside the epoch loop (search for `wandb.log(` with the dict containing `"train/loss": metrics["loss"]`). Currently around lines 138-149.

```python
# OLD (exact match, currently lines 138-149):
                if self.use_wandb:
                    # Primary names + symmetric aliases matching trainer_official:
                    # val/accuracy mirrors train/accuracy (cell-level),
                    # val/exact_accuracy mirrors train/exact_accuracy (puzzle-level).
                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "val/loss": val_metrics["loss"],
                            "val/puzzle_acc": val_metrics["puzzle_acc"],
                            "val/cell_acc": val_metrics["cell_acc"],
                            "val/accuracy": val_metrics["cell_acc"],
                            "val/exact_accuracy": val_metrics["puzzle_acc"],
                            "train/elapsed_min": elapsed,
                        },
                        step=epoch + 1,
                    )
```

```python
# NEW (adds lm/loss and val/lm_loss aliases):
                if self.use_wandb:
                    # Primary names + symmetric aliases matching trainer_official:
                    # val/accuracy mirrors train/accuracy (cell-level),
                    # val/exact_accuracy mirrors train/exact_accuracy (puzzle-level).
                    # lm/loss and val/lm_loss are explicit semantic names for the
                    # LM cross-entropy — same scalars as train/loss, val/loss;
                    # duplicate keys so existing dashboards keep working.
                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "lm/loss": metrics["loss"],
                            "val/loss": val_metrics["loss"],
                            "val/lm_loss": val_metrics["loss"],
                            "val/puzzle_acc": val_metrics["puzzle_acc"],
                            "val/cell_acc": val_metrics["cell_acc"],
                            "val/accuracy": val_metrics["cell_acc"],
                            "val/exact_accuracy": val_metrics["puzzle_acc"],
                            "train/elapsed_min": elapsed,
                        },
                        step=epoch + 1,
                    )
```

- [ ] **Step 2: Syntax check**

Run:
```bash
python -c "import src.training.trainer_llm; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer_llm.py
git commit -m "feat(trainer_llm): add lm/loss and val/lm_loss wandb aliases"
```

---

## Task 7: Trainer Change C — optional per-step LM loss logging

**Files:**
- Modify: `src/training/trainer_llm.py` (add a conditional per-step wandb.log inside `_train_epoch`)

- [ ] **Step 1: Edit `src/training/trainer_llm.py` — conditionally log per-step loss**

Find `_train_epoch()`'s for-step loop (search for `for step, (inputs, labels) in enumerate(pbar):`). Insert the per-step wandb.log right after `pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")`.

```python
# OLD (exact match, currently around lines 196-214):
        pbar = tqdm(self.train_loader, desc=f"LLM Epoch {epoch + 1}", leave=False)
        for step, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Remap dataset's ignore sentinel (0) to HF's ignore_index (-100)
            # so the LoRA only learns to predict target positions, not pad/clues.
            labels_for_loss = labels.masked_fill(labels == 0, HF_IGNORE_INDEX)

            outputs = self.model(input_ids=inputs, labels=labels_for_loss)
            loss = outputs.loss / accum
            loss.backward()

            if (step + 1) % accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")
```

```python
# NEW (adds conditional per-step wandb.log at end of loop body):
        pbar = tqdm(self.train_loader, desc=f"LLM Epoch {epoch + 1}", leave=False)
        for step, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Remap dataset's ignore sentinel (0) to HF's ignore_index (-100)
            # so the LoRA only learns to predict target positions, not pad/clues.
            labels_for_loss = labels.masked_fill(labels == 0, HF_IGNORE_INDEX)

            outputs = self.model(input_ids=inputs, labels=labels_for_loss)
            loss = outputs.loss / accum
            loss.backward()

            if (step + 1) % accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

            # Per-step LM loss logging — only enabled for runs that set
            # training.log_per_step: true in their YAML (e.g. DeepSeek
            # plateau runs). Off by default to avoid dashboard clutter.
            if self.use_wandb and self.tc.log_per_step:
                wandb.log({"lm/step_loss": outputs.loss.item()})
```

- [ ] **Step 2: Syntax check**

Run:
```bash
python -c "import src.training.trainer_llm; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Verify the gate defaults to off — existing LLM configs unchanged**

Run:
```bash
python -c "from src.utils.config import load_config; c = load_config('configs/llm_qwen.yaml'); assert c.training.log_per_step is False; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/training/trainer_llm.py
git commit -m "feat(trainer_llm): add optional per-step lm/step_loss logging"
```

---

## Task 8: Create `configs/llm_deepseek.yaml` (sudoku)

**Files:**
- Create: `configs/llm_deepseek.yaml`

- [ ] **Step 1: Create the sudoku config file**

Write the full file content with the locked hyperparameters from the spec:

```yaml
model:
  model_type: llm_finetune
  vocab_size: 11
  seq_len: 81
  num_classes: 11
  llm_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  lora_r: 8
  lora_alpha: 16
  use_qlora: false                  # 1.5B fp16 + seq=81 fits 12GB easily

training:
  lr: 0.00005
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 500
  batch_size: 8                     # 3x Qwen-0.5B params, halved batch to fit
  grad_accum_steps: 2               # effective batch 16 (fleet parity)
  epochs: 30
  early_stop_patience: 0            # disabled: plateau IS the result
  log_per_step: true                # emit lm/step_loss for plateau figure
  use_wandb: true
  wandb_project: TRM-LLM
  wandb_entity: ""                  # set via TRM_WANDB_ENTITY in .env
  log_interval: 10
  save_interval: 50

data:
  dataset: sudoku
  data_dir: data/sudoku-extreme-full
  num_workers: 4

seed: 42
device: cuda
checkpoint_dir: models/llm
experiment_dir: experiments/llm
```

- [ ] **Step 2: Verify the YAML loads cleanly**

Run:
```bash
python -c "from src.utils.config import load_config; c = load_config('configs/llm_deepseek.yaml'); assert c.training.epochs == 30; assert c.training.log_per_step is True; assert c.training.early_stop_patience == 0; assert c.model.llm_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add configs/llm_deepseek.yaml
git commit -m "feat(configs): add DeepSeek-R1-Distill-Qwen-1.5B sudoku config"
```

---

## Task 9: Create `configs/llm_deepseek_maze.yaml` (maze)

**Files:**
- Create: `configs/llm_deepseek_maze.yaml`

- [ ] **Step 1: Create the maze config file**

```yaml
model:
  model_type: llm_finetune
  vocab_size: 6
  seq_len: 900
  num_classes: 6
  llm_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  lora_r: 8
  lora_alpha: 16
  use_qlora: true                   # 1.5B x 900 tokens needs 4-bit on 12GB
  use_gradient_checkpointing: true  # ~3-4x activation memory reduction

training:
  lr: 0.00005
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 500
  batch_size: 1                     # mirrors llm_llama_maze (1.2B is smaller)
  grad_accum_steps: 16              # effective batch 16
  epochs: 30
  early_stop_patience: 0            # disabled: plateau IS the result
  log_per_step: true                # emit lm/step_loss for plateau figure
  use_wandb: true
  wandb_project: TRM-LLM
  wandb_entity: ""
  log_interval: 10
  save_interval: 50

data:
  dataset: maze
  data_dir: data/maze-30x30-hard-1k-aug
  num_workers: 4

seed: 42
device: cuda
checkpoint_dir: models/llm
experiment_dir: experiments/llm
```

- [ ] **Step 2: Verify the YAML loads cleanly**

Run:
```bash
python -c "from src.utils.config import load_config; c = load_config('configs/llm_deepseek_maze.yaml'); assert c.data.dataset == 'maze'; assert c.model.use_qlora is True; assert c.model.use_gradient_checkpointing is True; assert c.training.grad_accum_steps == 16; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add configs/llm_deepseek_maze.yaml
git commit -m "feat(configs): add DeepSeek-R1-Distill-Qwen-1.5B maze config"
```

---

## Task 10: Smoke test — 3 epochs, ~5 minutes

**Files:** (no files created — manual verification)

This is the cheapest possible end-to-end check before burning hours on the full runs. Launches a 3-epoch sudoku training and verifies all four pipeline components emit the expected artifacts.

- [ ] **Step 1: Activate the venv (Windows)**

Run:
```powershell
C:\Users\amm-alshamy\.venvs\ml-trm\Scripts\Activate.ps1
```
Expected: prompt shows `(ml-trm)` prefix.

- [ ] **Step 2: Launch the smoke run**

Run (bash syntax on Windows — use `bash` terminal if PowerShell chokes on backslash continuations):
```bash
python main.py --mode train --config configs/llm_deepseek.yaml \
    training.epochs=3 \
    training.log_interval=1 \
    training.save_interval=100
```

Expected runtime: ~5 minutes. Wait for completion or for the first full epoch to print before moving to Step 3.

- [ ] **Step 3: Verify stdout — baseline line present**

Look for the `[baseline]` line at the start of stdout. It should appear BEFORE any `LLM Epoch 1` progress bar. Expected line pattern:

```
[baseline] Epoch 0/3 | ValLoss: <some-float> | Puzzle: <some-float> | Cell: <some-float>
```

If this line does NOT appear: Task 5's diff was not applied correctly — revisit Task 5.

- [ ] **Step 4: Verify wandb dashboard — step-0 data point present**

Open the wandb run URL printed at the start of training. Check the `val/loss` panel. There should be a data point at `step=0` (the baseline) in addition to `step=1, 2, 3` from the log-every-epoch cadence.

If no `step=0` point: Task 5's `wandb.log(..., step=0)` block did not run — check `self.use_wandb` is True (i.e., wandb auth is working).

- [ ] **Step 5: Verify wandb dashboard — new alias keys present**

In the run's "Charts" tab, search for `lm/loss`, `val/lm_loss`, `lm/step_loss`. All three should appear with non-empty data.

- `lm/loss` — one data point per epoch (matches `train/loss`)
- `val/lm_loss` — one data point per epoch (matches `val/loss`)
- `lm/step_loss` — many data points (one per gradient-accumulation micro-batch)

If any missing: revisit Tasks 6 (aliases) or 7 (per-step).

- [ ] **Step 6: Verify aggregator produces `loss_delta_pct` for the smoke run**

Run:
```bash
TRM_EXPERIMENT_DIR=experiments/llm python scripts/aggregate_metrics.py
```

Then inspect the new row:
```bash
python -c "import csv; rows=list(csv.DictReader(open('results/summary.csv'))); r=[r for r in rows if 'deepseek' in r.get('task','')][0]; print('loss_delta_pct =', r['loss_delta_pct'])"
```
Expected: prints `loss_delta_pct = <some-numeric-value-or-empty>` — either a number or "" (for a 3-epoch run the value may be noisy). Crucially: the column exists and the script didn't crash.

- [ ] **Step 7: Decide whether to proceed**

All four pass criteria:
1. `[baseline]` line in stdout — ✅
2. `val/loss` at `step=0` in wandb — ✅
3. `lm/loss`, `val/lm_loss`, `lm/step_loss` keys all present — ✅
4. `loss_delta_pct` column populated (or blank, but exists) in summary.csv — ✅

If all four pass: proceed to Task 11. If any fail: fix the relevant earlier task, delete the smoke run's experiment directory (`experiments/llm/deepseek_r1_distill_qwen_1_5b_sudoku/`), re-run this task.

- [ ] **Step 8: Clean up the smoke run artifacts**

To avoid the 3-epoch smoke run polluting the real run's metrics, remove its train log + checkpoint:

```bash
rm -f experiments/llm/deepseek_r1_distill_qwen_1_5b_sudoku_train_log.csv
rm -f experiments/llm/deepseek_r1_distill_qwen_1_5b_sudoku_training_results.json
rm -rf experiments/llm/deepseek_r1_distill_qwen_1_5b_sudoku_train/
rm -f models/llm/deepseek_r1_distill_qwen_1_5b_sudoku_latest.pt
```

(No commit — these are runtime artifacts, typically gitignored.)

---

## Task 11: Full sudoku run — 30 epochs, ~4 hours

**Files:** (no files created — produces experiment artifacts)

- [ ] **Step 1: Launch the full sudoku training**

Run (in an activated venv, from the repo root):
```bash
python main.py --mode train --config configs/llm_deepseek.yaml
```

Expected runtime: ~4 hours on RTX 5070. Training will log every 10 epochs per the config.

Starting `bash scripts/auto_push.sh` in a separate terminal (per README line 325-327) will hourly-commit the training CSV + emissions logs for remote progress monitoring.

- [ ] **Step 2: Monitor wandb for expected behavior**

During the run, watch the wandb dashboard. Key checks (no action required during run, but catch obvious failures fast):

- `val/loss` at step 0 is ~2-3 nats (typical range for 1.5B LM on a small-vocab grid task)
- `train/loss` at epoch 10 is approximately equal to step-0 `val/loss` (confirming non-learning)
- No OOM crashes (run keeps progressing)

If OOM: reduce `batch_size` to 4 in `configs/llm_deepseek.yaml` and resume:
```bash
python main.py --mode train --config configs/llm_deepseek.yaml --resume models/llm/deepseek_r1_distill_qwen_1_5b_sudoku_latest.pt
```

- [ ] **Step 3: Wait for completion (~4 hours wall clock)**

Run is complete when stdout shows the final `Epoch 30/30` line and `wandb.finish()` is called. The experiment directory will contain:
- `deepseek_r1_distill_qwen_1_5b_sudoku_train_log.csv` — 4 rows (epoch 0, 10, 20, 30)
- `emissions.csv` — CodeCarbon output
- `deepseek_r1_distill_qwen_1_5b_sudoku_training_results.json` — final emissions summary

- [ ] **Step 4: Confirm acceptance criteria from the spec**

Verify:
1. Training reached epoch 30 without OOM — ✅
2. `val/loss` has data points at step 0 AND step 30 in wandb — ✅
3. `loss_delta_pct` for the sudoku row will be computed by Task 13 — defer.

No commit — experiment artifacts are not committed (their paths may be gitignored).

---

## Task 12: Full maze run — 30 epochs, ~12 hours

**Files:** (no files created — produces experiment artifacts)

- [ ] **Step 1: Launch the full maze training**

Run (in an activated venv):
```bash
python main.py --mode train --config configs/llm_deepseek_maze.yaml
```

Expected runtime: ~12 hours on RTX 5070. Schedule to start before a stretch away from the machine (overnight is ideal).

- [ ] **Step 2: Monitor wandb dashboard for first 30 minutes**

After 30 minutes of runtime, verify:
- Training progressed past the first full epoch (not stuck at 0%)
- No CUDA OOM in stderr
- `val/loss` at step 0 logged

If OOM: maze at seq_len=900 is tight on 12GB. Options:
- Verify `use_qlora: true` and `use_gradient_checkpointing: true` are set (re-read YAML if unsure)
- If already set and still OOM, lower `batch_size` to — wait, it's already 1 and cannot go lower. If still OOM, the 5070 may be too small; document the failure in `log.md` and mark Task 12 as blocked.

- [ ] **Step 3: Wait for completion (~12 hours wall clock)**

Run is complete when stdout shows `Epoch 30/30` and `wandb.finish()` is called.

- [ ] **Step 4: Confirm acceptance criteria from the spec**

Verify:
1. Training reached epoch 30 without OOM — ✅
2. `val/loss` has data points at step 0 AND step 30 in wandb — ✅

No commit — same as Task 11.

---

## Task 13: Post-run housekeeping — `summary.csv`, glossary, README, wandb Report

**Files:**
- Modify: `docs/wandb_metrics_glossary.md` (document new keys)
- Modify: `README.md` (update Models Compared table + Quick Start + Recent updates)
- Regenerate (not committed): `results/summary.csv`

- [ ] **Step 1: Regenerate `results/summary.csv`**

Run:
```bash
TRM_EXPERIMENT_DIR=experiments/llm python scripts/aggregate_metrics.py
```

Verify the two new rows exist and have populated `loss_delta_pct`:
```bash
python -c "import csv; rows=[r for r in csv.DictReader(open('results/summary.csv')) if 'deepseek' in r.get('task','')]; [print(r['task'], '| loss_delta_pct =', r.get('loss_delta_pct','MISSING')) for r in rows]"
```

Expected output: two lines, one for `deepseek_r1_distill_qwen_1_5b_sudoku` and one for `deepseek_r1_distill_qwen_1_5b_maze`, both with numeric `loss_delta_pct` values.

Record the two values — you will cite them below.

- [ ] **Step 2: Update `docs/wandb_metrics_glossary.md` — document new metric keys**

Use Edit. Find the existing "Train metrics" section (search for "train/loss") and append four new entries. Exact edits depend on the glossary's current structure — read the file first, then add entries like:

```markdown
### `lm/loss`

Identical scalar to `train/loss` but named for its semantic role — the language-model cross-entropy on the HF causal-LM forward pass (shifted internally by HF so `logits[i]` predicts `input_ids[i+1]`). Dual-logged as an alias so "LM loss" dashboards match the metric's name.

Emitted every `log_interval` epochs by `src/training/trainer_llm.py`.

### `lm/step_loss`

Per-micro-batch LM loss, emitted only when `training.log_per_step: true` in the run's YAML. Produces 1000-4000 points per 30-epoch run — intended for plateau-visualisation runs (e.g. DeepSeek baselines) where within-epoch noise floor matters. Off by default.

### `val/lm_loss`

Alias of `val/loss` — LM cross-entropy on the validation split. Same value, semantic name for plateau panels.

### Baseline evaluation at `step=0`

LLM trainer runs `evaluate()` once before any gradient update and logs val metrics at `step=0`. Anchors plateau plots so "nothing changed" is visible from the very start, not inferred from the first post-training data point. Present in all LLM runs after 2026-04-19.
```

- [ ] **Step 3: Update `README.md` — add two rows to the "Models Compared" table**

Use Edit on `README.md`. Find the existing table (search for "Qwen2.5-0.5B + LoRA"). Insert two new rows before the "Distilled LLM" row (or at the end of the LLM section):

Replace `<sudoku-delta>` and `<maze-delta>` with the actual values you recorded in Step 1.

```markdown
| DeepSeek-R1-Distill-Qwen-1.5B + LoRA | Fine-tuned LLM | 1.5B | Sudoku | ~0% (non-learning) | **loss_delta_pct = <sudoku-delta>%** | 30 ep, log_per_step logged |
| DeepSeek-R1-Distill-Qwen-1.5B + LoRA | Fine-tuned LLM | 1.5B | Maze | ~0% (non-learning) | **loss_delta_pct = <maze-delta>%** | 30 ep, QLoRA + grad-ckpt |
```

- [ ] **Step 4: Update `README.md` — add configs to the Quick Start command block**

Find the existing Quick Start commands (search for `python main.py --mode train --config configs/llm_llama.yaml`). Append two new lines:

```markdown
python main.py --mode train --config configs/llm_deepseek.yaml       # DeepSeek-R1-Distill-Qwen-1.5B sudoku
python main.py --mode train --config configs/llm_deepseek_maze.yaml  # DeepSeek-R1-Distill-Qwen-1.5B maze
```

- [ ] **Step 5: Update `README.md` — add Recent updates bullet**

Find the "Recent updates (April 2026)" section at the top. Append a new bullet (keep the existing bullets intact):

```markdown
- **Added: DeepSeek-R1-Distill-Qwen-1.5B LLM baseline.** `configs/llm_deepseek.yaml` + `configs/llm_deepseek_maze.yaml` run DeepSeek-R1's Qwen-1.5B distill on sudoku and maze. Three shared diagnostics added to `src/training/trainer_llm.py`: (1) pre-training evaluation at step 0 so plateau plots are anchored at the initial loss; (2) `lm/loss` and `val/lm_loss` semantic aliases alongside `train/loss` / `val/loss`; (3) optional `training.log_per_step: true` flag emitting `lm/step_loss` every micro-batch. `src/evaluation/aggregate.py` now computes `loss_delta_pct = (val_loss[0] - val_loss[N]) / val_loss[0] * 100` per run, included in `results/summary.csv`. Design spec: `docs/superpowers/specs/2026-04-19-deepseek-r1-distill-qwen-baseline-design.md`.
```

- [ ] **Step 6: Publish the updated wandb metrics report**

Run:
```bash
python scripts/publish_wandb_metrics_report.py
```

Expected: script prints a wandb Report URL. If the existing report was for `TRM-LLM` project, the URL should match the one in README line 221. If script errors on a missing wandb entity, set `TRM_WANDB_ENTITY` in `.env` first.

- [ ] **Step 7: Commit the docs + README changes**

```bash
git add docs/wandb_metrics_glossary.md README.md
git commit -m "docs: document DeepSeek baseline + lm/loss alias + loss_delta_pct"
```

- [ ] **Step 8: Notify teammates (manual)**

Post a message to the team (Slack/Discord/whatever you use) with:
- The spec path: `docs/superpowers/specs/2026-04-19-deepseek-r1-distill-qwen-baseline-design.md`
- The plan path: `docs/superpowers/plans/2026-04-19-deepseek-r1-distill-qwen-baseline.md`
- Summary: *"Merged DeepSeek-1.5B baseline. `trainer_llm.py` now runs an epoch-0 eval and logs `lm/loss` aliases for every LLM run. No regression; the new keys are additive. If you have an in-flight LLM run, it'll pick up the new behavior on its next trainer instantiation."*

No code action — just a heads-up to Armin and Nickolas so they aren't surprised by an extra minute of pre-training evaluation on their next run.

---

## Self-Review Checklist

(For the author; not a task step.)

**Spec coverage:** Does every requirement in the spec have a task?

| Spec requirement | Covered by |
|---|---|
| §"Files added": `llm_deepseek.yaml` | Task 8 |
| §"Files added": `llm_deepseek_maze.yaml` | Task 9 |
| §"Files modified": `config.py` (`log_per_step` field) | Task 1 |
| §"Files modified": `trainer_llm.py` Change A | Task 5 |
| §"Files modified": `trainer_llm.py` Change B | Task 6 |
| §"Files modified": `trainer_llm.py` Change C | Task 7 |
| §"Files modified": `aggregate.py` (`loss_delta_pct`) | Task 2 (implementation) |
| §"Tests to add": 3 test cases | Tasks 2, 3, 4 |
| §"Implementation order" #6: Smoke test | Task 10 |
| §"Implementation order" #7: Full sudoku run | Task 11 |
| §"Implementation order" #8: Full maze run | Task 12 |
| §"Implementation order" #9: Post-run housekeeping | Task 13 |
| §"Open questions": teammate coordination | Task 13 Step 8 |

All spec requirements have tasks. ✅

**Placeholder scan:** Searched for "TBD", "TODO", "fill in", "similar to Task N", "appropriate error handling". None found. ✅

**Type consistency:** Method names used across tasks:
- `parse_train_log` (used in Tasks 2, 3, 4) — matches `src/evaluation/aggregate.py` signature
- `self.evaluate()` (used in Task 5) — exists in `trainer_llm.py` already
- `wandb.log(..., step=N)` — matches existing usage pattern
- `tc.log_per_step` (Task 7) — matches field defined in Task 1

Consistent. ✅

---

## References

- **Spec:** `docs/superpowers/specs/2026-04-19-deepseek-r1-distill-qwen-baseline-design.md`
- **Closest analog configs:** `configs/llm_qwen.yaml` (Qwen2 family — architectural twin), `configs/llm_llama_maze.yaml` (VRAM + seq=900 twin)
- **Trainer:** `src/training/trainer_llm.py`
- **Aggregator:** `src/evaluation/aggregate.py`
- **Aggregator tests:** `tests/test_aggregate.py`
- **DeepSeek model card:** https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
