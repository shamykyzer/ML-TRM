# Handoff

## State
- **M6 sprint contribution COMPLETE.** Both Qwen-0.5B Sudoku Fix-B seeds passed Contract B.7 viability gates with rc=0:
  - seed 1 (wandb `w079srmr`): 0% puzzle / **21.21%** cell, 540 min, finished 2026-04-29T04:01:59+01:00
  - seed 2 (wandb `m6ybb9gy`): 0% puzzle / **21.52%** cell, 537 min, finished 2026-04-29T13:00:32+01:00
  - Cross-seed: Cell mean **0.2137** / std 0.0022 (tight). Headline: **0% puzzle / 21.4% cell**.
- All 8 M4-format canonical files in `C:/ml-trm-work/checkpoints to use/machine 6/` (`04a_*` seed 1, `04b_*` seed 2, four artifacts each: `-ep30.pt`, `-train_log.csv`, `-emissions.csv`, `-training_results.json`).
- `findings.md §§5.17-5.19` carries full audit trail (per-seed viability gates + cross-seed table + exit checklist).
- Branch `MACHINE-6` will be ahead of origin after the commit being made now.
- Drive sync watchdog (pid 2220) still in **patient mode** — `rclone listremotes` returns "Config file not found" → user has not completed `rclone config` OAuth yet. Non-blocking; watchdog will auto-engage on completion. M4 files are already local regardless.
- `AutoLock0830` daily Windows scheduled task active.

## Next
1. **Optional**: open PR for `MACHINE-6` if M4 expects one. User instruction was push-only; PR not opened.
2. **User action**: complete `rclone config` to finish Drive OAuth, then drive sync watchdog auto-engages and pushes the `04a_*`/`04b_*` files to the team Drive folder `18EXQL5h6MF5i8RbB4Zb97oU7wO9LXlbP`. Until then, M4 must `git pull` MACHINE-6 to get the canonical files via the repo (or accept that drive sync is offline until OAuth).
3. **Maze track is degenerate** (separate finding from this session — `mask_non_path=true` default makes a constant-`o` predictor score 100% puzzle_acc). User dropped that thread. If they raise it again: 5-line config fix in all 7 maze configs + smoke test, then full re-run is the realistic path. Otherwise let Sudoku carry the headline.
4. Launch script bug (`scripts/launch_m6_qwen_fixb.sh` post-trainer M4 block silently no-op'd on both seeds) — diagnosed but not patched. Fallback `scripts/m4_canonical_copy_seed.sh` covers the gap. Debug + fix is a post-deadline task (low priority since the fallback script worked twice).

## Context
- `.venv/` is on a OneDrive-synced path; if `OSError: [Errno 22]` recurs on import, force-materialize via PowerShell `Get-ChildItem ... | ForEach-Object { [System.IO.File]::ReadAllBytes($_.FullName) }` (recipe in §5.15).
- M3's Llama Sudoku Fix-B 4-hour status was never posted; M6 defaulted to Option B per spec. Don't switch to Option A retroactively.
- M4 greps `metric realism violation` / `viability gate passed` / `redundancy snapshot machine6` in `findings.md` to find each rig's contribution.
- Coursework deadline: 2026-05-01 17:00 BST.
