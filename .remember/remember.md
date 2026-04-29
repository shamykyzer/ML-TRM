# Handoff

## State
- M6 Qwen Sudoku Fix-B **seed 1 done rc=0** at 2026-04-29T04:01:59+01:00 (wandb `w079srmr`, 540 min). Final epoch-30: Loss 1.8535 / ValLoss 1.8518 / Puzzle 0.0000 / **Cell 0.2121**. Contract B.7 **viability gate passed** (slow rise from chance 6.87% → 21.21%, no realism violation). Logged to `findings.md §5.17` (commit `a9ffe43` on `MACHINE-6`).
- M6 **seed 2 in flight** (wandb `m6ybb9gy`, https://wandb.ai/shamykyzer/TRM/runs/m6ybb9gy) since 04:02:07 BST — auto-chained from seed 1 by `launch_m6_qwen_fixb.sh`. Watchdog pid 5806. ETA ~12:00 BST 2026-04-29.
- **Launch script bug**: post-trainer block in `scripts/launch_m6_qwen_fixb.sh` silently no-op'd for seed 1 — neither `__2026-04-29T0402__*` nor `04a_*` files appeared, neither M4 success nor skip echo landed in `/tmp/m6-launcher.log`. Source files in run_dir intact; manual `cp` produced `04a_*` files. New fallback script `scripts/m4_canonical_copy_seed.sh {1|2}` staged for seed 2.
- Drive sync watchdog (pid 2220) **still patient** — `rclone listremotes` returns "Config file not found" → user has not completed `rclone config` OAuth yet.
- `AutoLock0830` daily Windows scheduled task active (08:30 BST daily, `rundll32 user32.dll,LockWorkStation`).
- Monitors active: `bn248759b` (Contract B.6 + watchdog), `bfaqm4qv7` (drive sync engagement).

## Next
1. On seed 2 trainer exit (Monitor `bn248759b` will fire `[seed2] trainer exit code=`): apply Contract B.7 → append §5.18 to `findings.md` (task #10). **Preemptively run `bash scripts/m4_canonical_copy_seed.sh 2`** since the launch script's M4 block will likely no-op again.
2. Verify `04b_Qwen-0.5B_Sudoku_LoRA-FixB-seed2-{ep30,train_log,emissions,training_results}.{pt,csv,csv,json}` landed in `C:/ml-trm-work/checkpoints to use/machine 6/`. Together with the existing `04a_*` files this completes the M4 headline-table contribution.
3. Final exit-checklist entry to `findings.md §5` (task #4) once both seeds are ranked, then push to `MACHINE-6` and (only if user asks) open PR.
4. **Maze track is degenerate** — investigation in this session found all 7 maze configs use `mask_non_path=true` default → constant-`o` predictor scores 100% puzzle_acc. Apr-22 Qwen-maze and Apr-23 distill runs both saturated trivially. Fix is `mask_non_path: false` in all 7 configs. User has not asked me to act on this; flag it again if they raise the maze track.

## Context
- `.venv/` is on a OneDrive-synced path; if `OSError: [Errno 22]` recurs on import, force-materialize via PowerShell `Get-ChildItem ... | ForEach-Object { [System.IO.File]::ReadAllBytes($_.FullName) }` (recipe in §5.15).
- M3's Llama Sudoku Fix-B 4-hour status was never posted; M6 defaulted to Option B per spec. Don't switch to Option A retroactively.
- Findings §§5.13-5.17 carry full audit trail; M4 greps `metric realism violation` / `viability gate passed` / `redundancy snapshot machine6`.
- `MACHINE-6` is the active branch (HEAD `a9ffe43`, in sync with origin). PR not opened.
