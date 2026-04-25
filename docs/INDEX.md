# ML-TRM Documentation Index

Quick reference to every doc in the repo so you can find what you need
without opening everything. Lives under `docs/` for the detail; the
top-level Markdown files are architectural + status-oriented.

## Top-level documents (repo root)

- [README.md](../README.md) — Architecture overview, setup (bootstrap.ps1 +
  start.py), invocation cheat-sheet, GPU batch-size reference. Start here
  if you're new to the repo.
- [findings.md](../findings.md) — Empirical results + fleet recommendations.
  Active decisions live here (e.g. "do NOT retrain sudoku-att"). Updated
  after every run.
- [plan.md](../plan.md) — Course-level plan + hard constraints + submission
  checklist. Not a daily doc; touch when scope changes.
- [log.md](../log.md) — Checkpoint strategy + multi-device sync notes
  (which seed is running on which machine, what got Ctrl+C'd, what got
  auto-pushed to GitHub, etc.).
- [ARC-AGI3.md](../ARC-AGI3.md) — Historical proposal draft kept for
  provenance. Not load-bearing for current work.

## Under `docs/`

- [report.md](report.md) — Distinction-tier coursework report draft.
  Rubric-aligned sections; target for the final submission.
- [setup-guide.txt](setup-guide.txt) — Windows + Unix venv walkthrough
  with fallback steps when the one-liner bootstrap hits issues.
- [training-notes.md](training-notes.md) — Per-task hyperparameter notes:
  what worked, what didn't, GPU batch sizes, typical epoch counts.
- [wandb_metrics_glossary.md](wandb_metrics_glossary.md) — Definitions of
  every wandb key we log (puzzle_acc, cell_acc, avg_act_steps,
  halt_exploration_prob, CO2/correct, etc.).
- [weave_setup.md](weave_setup.md) — Operational guide for the Weave layer:
  per-puzzle eval traces, mid-run regression alerts, `weave.Model`
  wrappers, cross-checkpoint `weave.Evaluation`, and the
  auto-rebuilding runs Report publisher. Read this when you want to
  diagnose *which* puzzles a checkpoint fails on, not just how often.
- [architecture.puml](architecture.puml) — PlantUML source for the
  architecture diagram in the report.
- [2510.04871v1.pdf](2510.04871v1.pdf) — Less-is-More TRM paper (the
  reference implementation this repo reproduces).
- [Project Proposal.pdf](Project%20Proposal.pdf) — Original coursework proposal.
- [Group Project Specification 2025-26-v4.pdf](Group%20Project%20Specification%202025-26-v4.pdf)
  — Assessment specification.

## Under `docs/superpowers/specs/`

Design specs for recent structural changes (Windows bootstrap, official
TRM port, distinction-tier submission). Read these when you want the
"why" behind an architectural decision.

## Runtime dashboard

There is NO static project status page — `python start.py dashboard`
renders a live view from `results/summary.csv`,
`results/trm_runs_overview.csv`, and `findings.md`.
