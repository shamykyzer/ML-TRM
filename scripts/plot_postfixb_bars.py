"""Figure 1 (puzzle_acc) + Figure 3 (CO2-per-correct, log) — post-Fix-B Sudoku.

Source priority for each model row:
  1. results/summary_fixed.csv     — post-Fix-B retrains (M1/M3/M5/M6)
  2. results/summary.csv           — existing rows (fallback)
  3. results/trm_official_sudoku_eval.json — HF eval-only TRM-MLP row
  4. Hardcoded constants from findings.md §5.12 — TRM-MLP 3-seed mean

The 8 canonical models are matched against the CSV `task` column by
substring. Missing rows are reported on stderr but don't kill the figure
— the plot still renders for whichever models *are* present, so M2 can
preview Figure 1/3 ahead of the full Track B fleet completion.

Trigger threshold (per the M2 brief): ≥ 3 of the 4 Track B retrains
({SmolLM, Llama, Qwen-FixB, Distill-Qwen}) must be present in
summary_fixed.csv before this is the "official" Figure 1/3. Below that,
the script still runs but exits with code 2 and a stderr warning so an
automated caller can detect the partial state.

For Figure 3, models that score 0% puzzle accuracy have CO2/correct =
inf — log scale can't render that, so they're plotted at a sentinel
1.5x the maximum finite value with an "∞" annotation.

Outputs (always overwritten):
  results/figures/sudoku_acc_postfixb.png
  results/figures/co2_per_correct_postfixb.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402
    sns.set_theme(context="paper", style="whitegrid")
    _HAVE_SNS = True
except ImportError:
    _HAVE_SNS = False


SUMMARY_FIXED = REPO_ROOT / "results" / "summary_fixed.csv"
SUMMARY_OLD = REPO_ROOT / "results" / "summary.csv"
HF_TRM_MLP_EVAL_JSON = REPO_ROOT / "results" / "trm_official_sudoku_eval.json"
KVOTE_DIR = REPO_ROOT / "results" / "novelty"
FIG1_OUT = REPO_ROOT / "results" / "figures" / "sudoku_acc_postfixb.png"
FIG3_OUT = REPO_ROOT / "results" / "figures" / "co2_per_correct_postfixb.png"

# Sudoku-Extreme test split size — used to back-derive correct_puzzles when a
# CSV row only has puzzle_acc. From data/sudoku-extreme-full/test/.
TEST_SET_SIZE = 422_786


@dataclass
class ModelEntry:
    display: str
    track: str  # "eval-only", "trm-from-scratch", "trackB"
    label_match: tuple[str, ...]  # substrings tried in order against `task`
    family: str = "trm"  # "trm" | "llm" | "distill" — drives Contract B band
    puzzle_acc: Optional[float] = None
    puzzle_std: Optional[float] = None
    cell_acc: Optional[float] = None
    co2_kg: Optional[float] = None  # train CO2 (or inference for eval-only)
    correct_puzzles: Optional[float] = None
    co2_per_correct: Optional[float] = None  # if present, used directly
    source: str = ""  # which file populated this row
    notes: list[str] = field(default_factory=list)
    contract_b: str = ""  # "pass" | "violation:<reason>" | "" (not checked)


# Contract B (findings.md §5.13.4 + §B.2/§B.3) bands for Sudoku rows.
# puzzle_acc must be ≤ this; cell_acc must be in [lo, hi].
CONTRACT_B_SUDOKU_BAND = {
    "trm":     {"puzzle_max": 1.00, "cell_lo": 0.0, "cell_hi": 1.0},  # eval-only / from-scratch — no gate
    "llm":     {"puzzle_max": 0.05, "cell_lo": 0.091, "cell_hi": 0.30},
    "distill": {"puzzle_max": 0.05, "cell_lo": 0.10,  "cell_hi": 0.40},
}


def _check_contract_b(e: ModelEntry) -> None:
    """Apply Contract B §B.3 red-flag checks to a single row.

    LLM/distill Sudoku rows that violate the puzzle/cell bands get
    `contract_b = "violation:<reason>"` and a stderr-bound note. TRM rows
    are exempt (the contract scopes them as headline-eligible regardless).
    """
    band = CONTRACT_B_SUDOKU_BAND.get(e.family)
    if band is None:
        return
    if e.puzzle_acc is None:
        return
    reasons: list[str] = []
    if e.puzzle_acc > band["puzzle_max"]:
        reasons.append(
            f"puzzle_acc={e.puzzle_acc:.4f} > {band['puzzle_max']:.4f} "
            f"(LLM/distill should pin at 0)"
        )
    if e.cell_acc is not None:
        if e.cell_acc < band["cell_lo"]:
            reasons.append(
                f"cell_acc={e.cell_acc:.4f} < {band['cell_lo']:.4f} "
                f"(below chance — model not learning)"
            )
        if e.cell_acc > band["cell_hi"]:
            reasons.append(
                f"cell_acc={e.cell_acc:.4f} > {band['cell_hi']:.4f} "
                f"(above expected band — possible mask/contamination bug)"
            )
    if reasons:
        e.contract_b = "violation: " + "; ".join(reasons)
        e.notes.append("[contract B] metric realism violation — EXCLUDED from figure")
    else:
        e.contract_b = "pass"


def _build_canonical_entries() -> list[ModelEntry]:
    """The 8 models in the canonical Figure 1/3 order (top → bottom in bar
    charts; barh inverts to bottom → top, so first entry is at the top).
    `family` drives the Contract B viability band applied per row.
    """
    return [
        ModelEntry(
            display="TRM-MLP\n(HF eval, no train)",
            track="eval-only", family="trm",
            label_match=(
                "trm-mlp-hf-eval", "trm-mlp-eval-only",
                "sudoku-mlp-hf", "trm-mlp-eval",
            ),
        ),
        ModelEntry(
            display="TRM-MLP\n(from-scratch, 3 seeds)",
            track="trm-from-scratch", family="trm",
            label_match=(
                "trm-mlp-3seed", "sudoku-mlp-3seed", "sudoku-mlp-mean",
                "trm-mlp-from-scratch",
            ),
        ),
        ModelEntry(
            display="GPT-2 + LoRA",
            track="trackB", family="llm",
            label_match=("gpt2-sudoku-fixb", "llm-gpt2-sudoku", "gpt2-sudoku"),
        ),
        ModelEntry(
            display="SmolLM-360M + LoRA",
            track="trackB", family="llm",
            label_match=(
                "smollm-sudoku-fixb", "llm-smollm-sudoku",
                "smollm2-sudoku", "smollm-sudoku",
            ),
        ),
        ModelEntry(
            display="Qwen-0.5B + LoRA",
            track="trackB", family="llm",
            label_match=(
                "qwen-sudoku-fixb", "llm-qwen-sudoku-fixb",
                "llm-qwen-sudoku", "qwen2.5-sudoku", "qwen-sudoku",
            ),
        ),
        ModelEntry(
            display="Llama-3.2-1B + LoRA",
            track="trackB", family="llm",
            label_match=(
                "llama-sudoku-fixb", "llm-llama-sudoku",
                "llama-3.2-sudoku", "llama-sudoku",
            ),
        ),
        ModelEntry(
            display="Distill-GPT-2",
            track="trackB", family="distill",
            label_match=("distill-gpt2-sudoku-fixb", "distill-gpt2-sudoku"),
        ),
        ModelEntry(
            display="Distill-Qwen",
            track="trackB", family="distill",
            label_match=("distill-qwen-sudoku-fixb", "distill-qwen-sudoku"),
        ),
    ]


def _to_float(s) -> Optional[float]:
    if s in (None, ""):
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _match_row(rows: list[dict], substrs: tuple[str, ...]) -> Optional[dict]:
    """Return the first row whose `task` (case-insensitive) contains the
    earliest substring in `substrs`. Earlier substrings win — they're
    listed in priority order (most specific first).
    """
    tasks = [(r, str(r.get("task", "")).lower()) for r in rows]
    for s in substrs:
        s_lc = s.lower()
        for r, t in tasks:
            if s_lc in t:
                return r
    return None


def _populate_from_csv_row(e: ModelEntry, row: dict, source_label: str) -> None:
    e.puzzle_acc = _to_float(row.get("best_val_puzzle_acc"))
    e.cell_acc = _to_float(row.get("best_val_cell_acc"))
    e.co2_kg = _to_float(row.get("train_co2_kg"))
    e.correct_puzzles = _to_float(row.get("correct_puzzles"))
    e.co2_per_correct = _to_float(row.get("co2_per_correct_puzzle"))
    e.source = source_label
    # Allow optional std column (e.g., for 3-seed-aggregated rows).
    std = _to_float(row.get("best_val_puzzle_acc_std"))
    if std is not None:
        e.puzzle_std = std


def _back_derive_co2_per_correct(e: ModelEntry) -> None:
    """If summary row has co2_kg + puzzle_acc but no co2_per_correct, back
    it out from puzzle_acc * TEST_SET_SIZE. Honest: this assumes the
    eval was run on the full test split (the convention in this repo).

    Special case: if puzzle_acc = 0 and we have ANY puzzle data, mark
    co2_per_correct = inf so Fig 3 still plots the bar (with ∞ annotation
    via log-scale handling). This covers K-vote-only entries where we
    have inference puzzle/cell numbers but no train CO2.
    """
    if e.co2_per_correct is not None:
        return
    if e.puzzle_acc is None:
        return
    if e.puzzle_acc == 0:
        # 0 correct puzzles -> any non-zero CO2 is infinite cost-per-correct.
        # Even if co2_kg is missing, the right Fig 3 bar is "∞".
        e.co2_per_correct = math.inf
        if e.correct_puzzles is None:
            e.correct_puzzles = 0
        return
    if e.co2_kg is None:
        return
    if e.correct_puzzles is None:
        e.correct_puzzles = e.puzzle_acc * TEST_SET_SIZE
    if e.correct_puzzles and e.correct_puzzles > 0:
        e.co2_per_correct = e.co2_kg / e.correct_puzzles


def _read_kvote_csv_for(label_substrs: tuple[str, ...]) -> Optional[dict]:
    """Search results/novelty/k_vote_results-*.csv for a per-label file
    matching any substring. Return the K=1 row (the 1-sample eval
    baseline) as a dict with `puzzle_acc` and `cell_acc` keys.

    Match precedence (per substring, in label_match order, most-specific
    first): exact stem-suffix match > startswith. This stops "gpt2-sudoku"
    from incorrectly grabbing "k_vote_results-distill-gpt2-sudoku.csv".
    """
    if not KVOTE_DIR.is_dir():
        return None
    candidates: list[tuple[str, Path]] = []
    for csv_path in sorted(KVOTE_DIR.glob("k_vote_results-*.csv")):
        stem = csv_path.stem.lower()
        # Strip the canonical prefix to get the per-run label part.
        label_part = stem[len("k_vote_results-"):]
        candidates.append((label_part, csv_path))

    def _k1_row(p: Path) -> Optional[dict]:
        rows = _read_csv(p)
        for r in rows:
            if int(_to_float_safe(r.get("k", 0))) != 1:
                continue
            return {
                "best_val_puzzle_acc": r.get("puzzle_acc", ""),
                "best_val_cell_acc": r.get("cell_acc", ""),
                "task": r.get("label", ""),
                "_kvote_source": p.name,
            }
        return None

    for s in label_substrs:
        s_lc = s.lower()
        # Pass 1: exact label_part match — "gpt2-sudoku" only finds
        # k_vote_results-gpt2-sudoku.csv, not the distill variant.
        for lp, p in candidates:
            if lp == s_lc:
                return _k1_row(p)
        # Pass 2: startswith — "qwen-sudoku" finds "qwen-sudoku-fixb".
        for lp, p in candidates:
            if lp.startswith(s_lc):
                return _k1_row(p)
    return None


def _to_float_safe(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _populate_eval_only(e: ModelEntry) -> None:
    """TRM-MLP HF eval — read trm_official_sudoku_eval.json if present."""
    if not HF_TRM_MLP_EVAL_JSON.is_file():
        e.notes.append(f"missing {HF_TRM_MLP_EVAL_JSON.name}")
        return
    try:
        data = json.loads(HF_TRM_MLP_EVAL_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        e.notes.append(f"failed to read HF eval JSON: {exc}")
        return
    e.puzzle_acc = data.get("puzzle_accuracy")
    e.cell_acc = data.get("cell_accuracy")
    inf_em = data.get("inference_emissions") or {}
    e.co2_kg = inf_em.get("emissions_kg")
    if e.puzzle_acc is not None:
        e.correct_puzzles = e.puzzle_acc * TEST_SET_SIZE
        if e.co2_kg is not None and e.correct_puzzles > 0:
            e.co2_per_correct = e.co2_kg / e.correct_puzzles
    e.source = HF_TRM_MLP_EVAL_JSON.name
    e.notes.append("inference-only emissions (no training run)")


def _populate_trm_from_scratch(e: ModelEntry, summary_old: list[dict]) -> None:
    """TRM-MLP 3-seed from-scratch mean. Per findings.md §5.12: peaks
    s0=0.7456 / s1=0.7420 / s2=0.7486, mean 0.7425 ± 0.0063 across 4 runs
    (s0 has 2 runs). CO2 from summary.csv sudoku-mlp-seed0 row × 3.
    """
    e.puzzle_acc = 0.7425
    e.puzzle_std = 0.0063
    e.cell_acc = 0.8584  # seed-0 peak cell_acc; the 3-seed cell-mean is ~0.86
    # Try to derive CO2 from summary.csv if available (multiplied across 3 seeds);
    # otherwise fall back to the documented 13.77 kg total from §5.12 narrative.
    seed0 = _match_row(summary_old, ("sudoku-mlp-seed0",))
    if seed0:
        co2_one = _to_float(seed0.get("train_co2_kg"))
        if co2_one is not None:
            e.co2_kg = co2_one * 3.0  # 3 seeds, per-seed CO2 ~= seed0
            e.notes.append(f"CO2 = 3 x summary.csv:sudoku-mlp-seed0 ({co2_one:.3f} kg)")
        e.source = "summary.csv:sudoku-mlp-seed0 + findings.md §5.12"
    else:
        # Fall back to documented per-seed numbers from §1: 5.23, 4.15, 4.39 kg.
        e.co2_kg = 13.77
        e.source = "findings.md §1 + §5.12 (no summary.csv row)"
    e.correct_puzzles = e.puzzle_acc * TEST_SET_SIZE
    if e.co2_kg and e.correct_puzzles:
        e.co2_per_correct = e.co2_kg / e.correct_puzzles


def _populate_trackB(e: ModelEntry, fixed: list[dict], old: list[dict]) -> None:
    row = _match_row(fixed, e.label_match)
    if row is not None:
        _populate_from_csv_row(e, row, "summary_fixed.csv")
        _back_derive_co2_per_correct(e)
        return
    row = _match_row(old, e.label_match)
    if row is not None:
        _populate_from_csv_row(e, row, "summary.csv (PRE-FixB fallback)")
        _back_derive_co2_per_correct(e)
        e.notes.append("PRE-Fix-B numbers - overwrite when retrain lands")
        return
    # Fallback: per-label K-vote CSV produced by run_kvote_llm_single.py.
    # K=1 row gives the inference-time puzzle/cell numbers — same metric
    # universe as Fix-B re-eval, just inference-only. Useful for M4-owned
    # GPT-2 + Distill-GPT-2 which were already K-voted pre-sprint.
    kvote_row = _read_kvote_csv_for(e.label_match)
    if kvote_row is not None:
        _populate_from_csv_row(e, kvote_row, kvote_row.get("_kvote_source", "kvote csv"))
        _back_derive_co2_per_correct(e)
        e.notes.append("derived from K-vote K=1 row (no Fix-B re-eval); train CO2 unknown")
        return
    e.notes.append("no matching row in summary CSVs or K-vote CSVs")


def _resolve_entries(args) -> tuple[list[ModelEntry], list[dict], list[dict]]:
    fixed = _read_csv(SUMMARY_FIXED)
    old = _read_csv(SUMMARY_OLD)
    entries = _build_canonical_entries()
    for e in entries:
        if e.track == "eval-only":
            _populate_eval_only(e)
        elif e.track == "trm-from-scratch":
            _populate_trm_from_scratch(e, old)
        else:
            _populate_trackB(e, fixed, old)
        # Contract B viability gate — only Track B (LLM/distill) rows are
        # actually checked; TRM rows pass trivially via the wide band.
        _check_contract_b(e)
    return entries, fixed, old


def _excluded_for_contract_b(e: ModelEntry) -> bool:
    return e.contract_b.startswith("violation")


def _track_b_landed(entries: list[ModelEntry], require_fixb: bool) -> int:
    n = 0
    for e in entries:
        if e.track != "trackB":
            continue
        if e.puzzle_acc is None:
            continue
        if require_fixb and "summary_fixed.csv" not in e.source:
            continue
        n += 1
    return n


def _viridis_palette(n: int) -> list:
    if _HAVE_SNS and n > 0:
        return list(sns.color_palette("viridis", n))
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _rocket_palette(n: int) -> list:
    if _HAVE_SNS and n > 0:
        return list(sns.color_palette("rocket_r", n))
    cmap = plt.get_cmap("rocket_r") if "rocket_r" in plt.colormaps() else plt.get_cmap("magma_r")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _plot_fig1(entries: list[ModelEntry]) -> None:
    plotted = [e for e in entries
               if e.puzzle_acc is not None and not _excluded_for_contract_b(e)]
    if not plotted:
        print("[fig1] no entries to plot — skipping", file=sys.stderr)
        return

    labels = [e.display for e in plotted]
    values = [e.puzzle_acc for e in plotted]
    errs = [
        e.puzzle_std if e.puzzle_std is not None else 0.0
        for e in plotted
    ]
    colors = _viridis_palette(len(plotted))

    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(plotted) + 1.6), dpi=150)
    y = list(range(len(plotted)))
    ax.barh(y, values, color=colors,
            xerr=[errs, errs], capsize=4, error_kw={"linewidth": 1.0})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # first entry on top
    ax.set_xlabel("Sudoku puzzle accuracy (post-Fix-B eval)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Figure 1 — Sudoku puzzle accuracy by model")
    for yi, v, std in zip(y, values, errs):
        suffix = f" ± {std:.3f}" if std > 0 else ""
        ax.text(v + 0.015, yi, f"{v:.3f}{suffix}", va="center", fontsize=8)
    fig.tight_layout()
    FIG1_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG1_OUT), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] wrote {FIG1_OUT} ({len(plotted)}/8 entries)")


def _plot_fig3(entries: list[ModelEntry]) -> None:
    """Log-y bar of CO2 per correct puzzle. Entries with co2_per_correct =
    inf (i.e., 0% puzzle accuracy) are plotted at a sentinel above the
    finite max with an ∞ annotation. Entries with no CO2 number at all
    are skipped.
    """
    plotted = [e for e in entries
               if e.co2_per_correct is not None and not _excluded_for_contract_b(e)]
    if not plotted:
        print("[fig3] no entries to plot — skipping", file=sys.stderr)
        return

    finite_vals = [e.co2_per_correct for e in plotted
                   if math.isfinite(e.co2_per_correct) and e.co2_per_correct > 0]
    if not finite_vals:
        print("[fig3] no finite CO2/correct values — skipping", file=sys.stderr)
        return
    sentinel = max(finite_vals) * 5.0  # 5x above max so the bar is clearly off-scale

    labels = [e.display for e in plotted]
    raw_values = [e.co2_per_correct for e in plotted]
    plot_values = [
        sentinel if (not math.isfinite(v) or v <= 0) else v
        for v in raw_values
    ]
    is_inf = [not math.isfinite(v) or v <= 0 for v in raw_values]
    colors = _rocket_palette(len(plotted))

    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(plotted) + 1.6), dpi=150)
    y = list(range(len(plotted)))
    bars = ax.barh(y, plot_values, color=colors)
    # Hatch the inf bars so they read as "off scale" not "very large".
    for bar, inf in zip(bars, is_inf):
        if inf:
            bar.set_hatch("//")
            bar.set_edgecolor("#2c3e50")
            bar.set_alpha(0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(r"CO$_2$ per correct puzzle (kg, log scale)")
    ax.set_title("Figure 3 — Carbon cost per correct Sudoku puzzle")
    for yi, v, raw in zip(y, plot_values, raw_values):
        if math.isfinite(raw) and raw > 0:
            ax.text(v * 1.15, yi, f"{raw:.2e} kg", va="center", fontsize=8)
        else:
            ax.text(v * 1.15, yi, "∞ (0% puzzle acc)",
                    va="center", fontsize=8, color="#7f0e1a")
    fig.tight_layout()
    FIG3_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG3_OUT), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] wrote {FIG3_OUT} ({len(plotted)}/8 entries)")


def _print_audit(entries: list[ModelEntry], fixed: list[dict], old: list[dict]) -> None:
    print(f"\n[audit] summary_fixed.csv exists: {SUMMARY_FIXED.is_file()} "
          f"(rows={len(fixed)})")
    print(f"[audit] summary.csv exists: {SUMMARY_OLD.is_file()} "
          f"(rows={len(old)})")
    print(f"[audit] HF eval JSON exists: {HF_TRM_MLP_EVAL_JSON.is_file()}")
    print()
    print(f"{'#':<2} {'model':<35} {'puzzle':>8} {'co2/correct (kg)':>20} {'B-gate':<14} {'source'}")
    for i, e in enumerate(entries, 1):
        p = f"{e.puzzle_acc:.4f}" if e.puzzle_acc is not None else "  -    "
        cpc = "-"
        if e.co2_per_correct is not None:
            cpc = "inf" if not math.isfinite(e.co2_per_correct) else f"{e.co2_per_correct:.3e}"
        flat_display = e.display.replace("\n", " ")
        bgate = e.contract_b[:14] if e.contract_b else ""
        print(f"{i:<2} {flat_display:<35} {p:>8} {cpc:>20} {bgate:<14} {e.source}")
        for note in e.notes:
            print(f"     note: {note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--require-trackb", type=int, default=3,
                    help="Min Track B retrains in summary_fixed.csv to "
                         "consider Figs 1+3 'official' (default 3 of 4). "
                         "Below threshold, exits 2 with a warning but still "
                         "writes the figure.")
    args = ap.parse_args(argv)

    entries, fixed, old = _resolve_entries(args)
    _print_audit(entries, fixed, old)

    n_fixb = _track_b_landed(entries, require_fixb=True)
    n_any = _track_b_landed(entries, require_fixb=False)
    print(f"\n[trigger] {n_fixb}/4 Track B retrains in summary_fixed.csv "
          f"({n_any}/4 in any source)")

    _plot_fig1(entries)
    _plot_fig3(entries)

    if n_fixb < args.require_trackb:
        print(f"\n[warn] only {n_fixb}/{args.require_trackb} Track B retrains "
              f"present in summary_fixed.csv - figures are PROVISIONAL",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
