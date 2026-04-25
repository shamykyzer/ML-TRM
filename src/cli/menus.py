"""Interactive menus + prompts + copy-paste command printing for ML-TRM.

Combines what the original start.py called the "interactive launcher",
the "resume picker", the "fresh-start launcher", and the "copy-paste
commands" dump. Kept in one module because they share many prompt
helpers and call each other freely — splitting them forces late imports
that obscure the control flow.

All TASK_DISPATCH / FLEET_PLAN references are late-imported from
bootstrap inside the functions that use them, because bootstrap imports
this module at its top level.
"""
import os
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple

from src.cli.checkpoints import (
    _config_for_run_dir,
    _scan_for_checkpoints,
    _seed_for_run_dir,
)
from src.cli.console import BOLD, CYAN, DIM, GREEN, RESET, YELLOW
from src.cli.launchers import _dispatch_training, _run, _run_training_subprocess
from src.cli.paths import PYTHON, ROOT, WANDB
from src.cli.wandb_bootstrap import _wandb_ready
from src.cli.workdir import _default_work_dir, _resolve_work_dir


def _prompt(msg: str, default: str = "") -> str:
    """Prompt user on stdin, return default on empty input or EOF."""
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"{CYAN}{msg}{suffix}: {RESET}").strip()
    except EOFError:
        reply = ""
    return reply or default


def _prompt_task_and_seed() -> Tuple[str, int]:
    """Interactive picker for task label and seed int.

    Shows every task in TASK_DISPATCH as a numbered menu, grouped TRM-first
    then LLM, so the operator can scan by family. Tasks whose HF init file is
    missing get a yellow flag (the run still works — it just starts from
    random init, exploratory rather than paper-faithful). Defaults to seed 0
    (first row of FLEET_PLAN, a safe starter on any machine).
    """
    from src.cli.bootstrap import TASK_DISPATCH

    tasks = list(TASK_DISPATCH.keys())
    trm_tasks = [t for t in tasks if not t.startswith("llm-")]
    llm_tasks = [t for t in tasks if t.startswith("llm-")]

    print(f"\n{BOLD}Which task?{RESET}")
    print(f"  {DIM}-- TRM (paper architectures) --{RESET}")
    for t in trm_tasks:
        i = tasks.index(t) + 1
        _, init, desc = TASK_DISPATCH[t]
        suffix = (
            f"  {YELLOW}(HF init missing \u2014 will use random init){RESET}"
            if init and not os.path.exists(init) else ""
        )
        print(f"  {CYAN}{i:>2}{RESET}) {t:<20s}  {DIM}{desc}{RESET}{suffix}")
    print(f"  {DIM}-- LLM baselines (LoRA fine-tune) --{RESET}")
    for t in llm_tasks:
        i = tasks.index(t) + 1
        _, _, desc = TASK_DISPATCH[t]
        print(f"  {CYAN}{i:>2}{RESET}) {t:<20s}  {DIM}{desc}{RESET}")

    choice = _prompt(f"Pick 1-{len(tasks)}", default="1")
    try:
        task = tasks[int(choice) - 1]
        if int(choice) < 1:
            raise IndexError
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid task choice '{choice}'.{RESET}")
        sys.exit(2)

    seed_str = _prompt("Seed (non-negative int)", default="0")
    try:
        seed = int(seed_str)
        if seed < 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Seed must be a non-negative integer, got '{seed_str}'.{RESET}")
        sys.exit(2)

    return task, seed


def _resume_training_picker() -> None:
    """Interactive: pick a finished/interrupted run and extend it by N epochs.

    Discovery scans both:
      - $TRM_WORK_DIR (fleet runs from `start.py` dispatch / run_seed.sh)
      - <repo>/models/  (legacy simple-TRM runs that wrote inside the repo)
    Each candidate run is shown with its highest epoch_<N>.pt checkpoint —
    NOT `latest.pt`, because latest.pt is only written when a run completes
    cleanly, but Ctrl+C recovery has to start from the most recent
    epoch_N.pt that the save_interval cadence wrote to disk.

    The user picks a run, the picker resolves the right config via the dir
    name, asks how many ADDITIONAL epochs to run, then dispatches main.py
    with --resume <ckpt> --epochs <max_ep + extra>. main.py's _run_train
    treats --epochs as the new total target — the trainer loop runs from
    the resumed start_epoch through to the new total.
    """
    work_dir = os.environ.get("TRM_WORK_DIR") or _default_work_dir()
    candidates = (
        _scan_for_checkpoints(work_dir)
        + _scan_for_checkpoints(os.path.join(ROOT, "models"))
    )

    if not candidates:
        print(f"\n{YELLOW}!!! No resumable runs found.{RESET}")
        print(f"{DIM}    Looked in:{RESET}")
        print(f"{DIM}      \u2022 {work_dir}{RESET}")
        print(f"{DIM}      \u2022 {os.path.join(ROOT, 'models')}{RESET}")
        print(f"{DIM}    A run is resumable when its directory contains at least one{RESET}")
        print(f"{DIM}    epoch_<N>.pt file (written every save_interval epochs).{RESET}\n")
        return

    print(f"\n{BOLD}Resumable runs:{RESET}  {DIM}(newest first){RESET}")
    for i, (run_dir, max_ep, ckpt_path) in enumerate(candidates, 1):
        cfg = _config_for_run_dir(run_dir) or f"{YELLOW}<unknown \u2014 will prompt>{RESET}"
        seed = _seed_for_run_dir(run_dir)
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(
            f"  {CYAN}{i:>2}{RESET}) {os.path.basename(run_dir):<28s}"
            f"  epoch={CYAN}{max_ep:>4}{RESET}"
            f"  seed={seed}"
            f"  ({size_mb:.0f} MB)"
        )
        print(f"      {DIM}{ckpt_path}{RESET}")
        print(f"      {DIM}config: {cfg}{RESET}")

    raw = _prompt(f"\nPick run 1-{len(candidates)}", default="1")
    try:
        idx = int(raw) - 1
        run_dir, max_ep, ckpt_path = candidates[idx]
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid choice '{raw}'.{RESET}")
        return

    config = _config_for_run_dir(run_dir)
    if not config:
        config = _prompt(
            f"Config YAML for this run (relative to repo root)",
            default="configs/trm_sudoku.yaml",
        )
        if not os.path.exists(os.path.join(ROOT, config)):
            print(f"{YELLOW}!!! Config not found: {config}{RESET}")
            return

    # The "how many MORE epochs?" framing is the natural one for the user
    # ("I just hit 500, give me 200 more"); we convert to the absolute
    # total that main.py / the trainer loop expect via --epochs.
    extra_str = _prompt(
        f"Run for how many MORE epochs? (current = {max_ep})",
        default="100",
    )
    try:
        extra = int(extra_str)
        if extra <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{extra_str}'.{RESET}")
        return
    new_total = max_ep + extra
    seed = _seed_for_run_dir(run_dir)

    # Pin checkpoint/experiment dirs back to the SAME run dir so the new
    # epoch_<N>.pt files, train_log.csv, and emissions.csv append to the
    # existing artifacts instead of starting a fresh dir somewhere else.
    env = os.environ.copy()
    env["TRM_CHECKPOINT_DIR"] = run_dir
    env["TRM_EXPERIMENT_DIR"] = run_dir

    args = [
        PYTHON, "main.py",
        "--mode", "train",
        "--config", config,
        "--resume", ckpt_path,
        "--epochs", str(new_total),
        "--seed", str(seed),
    ]

    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  resume from        : {CYAN}epoch_{max_ep}.pt{RESET}  {DIM}({ckpt_path}){RESET}")
    print(f"  config             : {config}")
    print(f"  current epoch      : {max_ep}")
    print(f"  extra epochs       : {CYAN}{extra}{RESET}")
    print(f"  new total epochs   : {CYAN}{new_total}{RESET}")
    print(f"  seed               : {seed}")
    print(f"  TRM_CHECKPOINT_DIR : {run_dir}")
    print(f"  wandb              : "
          f"{GREEN}\u2713 ready{RESET}" if _wandb_ready() else f"{YELLOW}not configured (will train without){RESET}")
    print(f"{BOLD}{bar}{RESET}\n")

    confirm = _prompt(f"Launch? [y/N]", default="N").lower()
    if confirm not in ("y", "yes"):
        print(f"{DIM}Aborted \u2014 nothing launched.{RESET}\n")
        return

    result = subprocess.run(args, env=env, cwd=ROOT)
    sys.exit(result.returncode)


# Fresh-start menu collapses the 8 per-task LLM entries into 4 family combos.
# Picking a family runs sudoku FIRST then maze, each as its own subprocess
# (and thus its own wandb run with the enriched name in wandb_utils.py).
# Tuple shape: (family_label, description, sudoku_task_key, maze_task_key).
# The two task keys must exist in TASK_DISPATCH or the picker errors loudly
# at launch — single source of truth for "which configs go with which family".
LLM_FAMILIES: List[Tuple[str, str, str, str]] = [
    ("llm-gpt2",     "GPT-2 (124M) LoRA",               "llm-gpt2-sudoku",     "llm-gpt2-maze"),
    ("llm-smollm",   "SmolLM2-360M LoRA",               "llm-smollm-sudoku",   "llm-smollm-maze"),
    ("llm-qwen",     "Qwen2.5-0.5B LoRA",               "llm-qwen-sudoku",     "llm-qwen-maze"),
    ("llm-llama",    "Llama-3.2-1B LoRA",               "llm-llama-sudoku",    "llm-llama-maze"),
    ("llm-deepseek", "DeepSeek-R1-Distill-Qwen-1.5B",   "llm-deepseek-sudoku", "llm-deepseek-maze"),
]


def _prompt_fresh_target_and_seed() -> Tuple[str, List[str], int]:
    """Show TRM tasks + 5 LLM families + an all-families sweep + a per-rig fleet option; return (label, [task, ...], seed).

    TRM picks return a single-element task list; LLM family picks return
    [sudoku_task, maze_task]; the all-families pick returns all 8 LLM task
    keys in ascending-model-size order (gpt2 -> smollm -> qwen -> llama),
    each family still interleaved sudoku-then-maze so wandb runs cluster
    sensibly. Caller wipes + runs them all back-to-back via the same loop
    the 2-task family case uses; a crash mid-sweep doesn't abort the rest.
    """
    from src.cli.bootstrap import TASK_DISPATCH

    trm_tasks = [t for t in TASK_DISPATCH if not t.startswith("llm-")]
    entries: List[Tuple[str, str, List[str]]] = []
    for t in trm_tasks:
        _, init, desc = TASK_DISPATCH[t]
        suffix = (
            "  (HF init missing \u2014 random init)"
            if init and not os.path.exists(init) else ""
        )
        entries.append((t, desc + suffix, [t]))
    for family, desc, sudoku_task, maze_task in LLM_FAMILIES:
        entries.append((family, f"{desc}  (sudoku \u2192 maze)", [sudoku_task, maze_task]))

    # Flatten every family's (sudoku, maze) into a single 8-task sweep.
    # Order follows LLM_FAMILIES ascending by base-model size so the
    # scaling-comparison narrative reads left-to-right. Fits in ~10h on
    # an RTX 5070 at 30 epochs per task (proposal-committed cadence).
    all_llm_tasks: List[str] = []
    for _family, _desc, sudoku_task, maze_task in LLM_FAMILIES:
        all_llm_tasks.extend([sudoku_task, maze_task])
    entries.append((
        "llm-all",
        f"All {len(LLM_FAMILIES)} LLM families back-to-back "
        f"({len(all_llm_tasks)} runs, sudoku \u2192 maze per family)",
        all_llm_tasks,
    ))
    # Placeholder for "this rig's fleet queue" — rig identity is resolved
    # AFTER the user picks this option (so we don't prompt for TRM_RIG if
    # they pick anything else). See the post-selection branch below.
    entries.append((
        "llm-fleet-rig",
        "This rig's Option-B fleet queue (TRM_RIG-scoped, longest-first)",
        [],
    ))

    print(f"\n{BOLD}Which target?{RESET}")
    print(f"  {DIM}-- TRM (paper architectures) --{RESET}")
    n_trm = len(trm_tasks)
    for i, (label, desc, _) in enumerate(entries[:n_trm], 1):
        print(f"  {CYAN}{i:>2}{RESET}) {label:<13s}  {DIM}{desc}{RESET}")
    print(f"  {DIM}-- LLM families (each runs sudoku then maze) --{RESET}")
    # Show individual families first, then the all-families sweep last
    # (so the menu numbering puts the bigger option at the bottom where
    # the eye naturally scans for "run everything").
    for i, (label, desc, _) in enumerate(entries[n_trm:-2], n_trm + 1):
        print(f"  {CYAN}{i:>2}{RESET}) {label:<13s}  {DIM}{desc}{RESET}")
    print(f"  {DIM}-- Sweep --{RESET}")
    # The two sweep entries are the last two appended to `entries`.
    for offset, entry in enumerate(entries[-2:], start=len(entries) - 1):
        label, desc, _ = entry
        print(f"  {CYAN}{offset:>2}{RESET}) {label:<15s}  {DIM}{desc}{RESET}")

    choice = _prompt(f"Pick 1-{len(entries)}", default="1")
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(entries):
            raise IndexError
    except (ValueError, IndexError):
        print(f"{YELLOW}!!! Invalid target choice '{choice}'.{RESET}")
        sys.exit(2)
    label, _, tasks = entries[idx]

    # Late-resolve the rig queue. Only prompts for TRM_RIG when this
    # entry is actually selected — if the operator picks anything else,
    # TRM_RIG is never asked about.
    if label == "llm-fleet-rig":
        from src.cli.bootstrap import RIG_FLEET_PLAN, _resolve_rig
        rig = _resolve_rig()
        tasks = list(RIG_FLEET_PLAN[rig])
        label = f"llm-fleet-rig{rig}"

    seed_str = _prompt("Seed (non-negative int)", default="0")
    try:
        seed = int(seed_str)
        if seed < 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Seed must be a non-negative integer, got '{seed_str}'.{RESET}")
        sys.exit(2)

    return label, tasks, seed


def _confirm_wipe_run_dir(task_dir: str) -> bool:
    """List the contents of task_dir, prompt y/N to delete, then rmtree.

    Returns True if the dir was wiped (or was already empty/absent), False if
    the user declined or the rmtree failed. Bails after printing instead of
    raising — callers usually want to abort the whole launcher cleanly.
    """
    if not os.path.isdir(task_dir):
        print(f"{DIM}[fresh-start] no existing dir at {task_dir} \u2014 clean launch.{RESET}")
        return True
    try:
        existing = sorted(os.listdir(task_dir))
    except OSError:
        existing = []
    if not existing:
        return True

    print(f"\n{YELLOW}\u26a0  Existing run directory will be WIPED:{RESET}")
    print(f"   {task_dir}")
    print(f"{DIM}   Contains {len(existing)} item(s):{RESET}")
    for name in existing[:10]:
        full = os.path.join(task_dir, name)
        if os.path.isfile(full):
            size_mb = os.path.getsize(full) / 1e6
            size_label = f"  ({size_mb:.1f} MB)" if size_mb >= 0.1 else ""
            print(f"{DIM}     \u2022 {name}{size_label}{RESET}")
        else:
            print(f"{DIM}     \u2022 {name}/{RESET}")
    if len(existing) > 10:
        print(f"{DIM}     ... and {len(existing) - 10} more{RESET}")

    confirm = _prompt(
        f"\n{BOLD}Delete these and start fresh?{RESET} [y/N]",
        default="N",
    ).lower()
    if confirm not in ("y", "yes"):
        return False
    try:
        shutil.rmtree(task_dir)
    except OSError as exc:
        print(f"{YELLOW}!!! Could not remove {task_dir}: {exc}{RESET}")
        return False
    print(f"{DIM}[fresh-start] removed {task_dir}{RESET}")
    return True


def _fresh_start_launcher() -> None:
    """Launch NEW runs for a chosen target+seed, wiping existing run dir(s).

    Counterpart to the resume picker: instead of continuing from the latest
    epoch_<N>.pt, this path guarantees epoch 0 by deleting everything under
    <TRM_WORK_DIR>/<task>-seed<N>/ first.

    Two modes, both driven by `_prompt_fresh_target_and_seed`:
      - TRM pick -> wipe + run one task with the chosen epochs (default 2000).
      - LLM family pick -> wipe BOTH the sudoku and maze seed dirs for that
        family, then run sudoku then maze sequentially as two distinct wandb
        runs, each capped at the chosen epochs (default 30 — the scaling-
        comparison cadence).

    A failure in run 1 (sudoku) of an LLM family does NOT abort run 2 (maze):
    we collect the exit codes and print a summary so the operator can retry
    only what failed.
    """
    label, tasks, seed = _prompt_fresh_target_and_seed()
    is_family = len(tasks) > 1

    # 30 epochs is the LLM scaling default the user is comparing across the 4
    # families on both datasets. TRM tasks still default to the long 2000-epoch
    # cadence the paper-faithful runs need to converge.
    default_epochs = "30" if is_family else "2000"
    epochs_str = _prompt("Epochs per task (overrides YAML)", default=default_epochs)
    try:
        epochs = int(epochs_str)
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need a positive integer, got '{epochs_str}'.{RESET}")
        return

    work_dir = _resolve_work_dir()

    # Per-task wipe pass — confirm once per dir, abort the whole launcher if
    # the user declines any single confirmation (safer than a partial sweep
    # against half-stale dirs that would silently resume).
    for task in tasks:
        task_dir = os.path.join(work_dir, f"{task}-seed{seed}")
        if not _confirm_wipe_run_dir(task_dir):
            print(f"{DIM}Aborted \u2014 existing run preserved, nothing launched.{RESET}\n")
            return

    if not is_family:
        _dispatch_training(tasks[0], seed, epochs=epochs)
        return  # unreachable — _dispatch_training calls sys.exit

    bar = "=" * 64
    run_kind = "LLM fleet sweep" if len(tasks) > 2 else "LLM family run"
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  {BOLD}{run_kind} \u2014 {label}{RESET}")
    print(f"  seed   : {CYAN}{seed}{RESET}")
    print(f"  epochs : {CYAN}{epochs}{RESET} per task")
    # For short sweeps the arrow chain is readable inline; for the full
    # 8-task sweep it wraps past terminal width, so print one task per line.
    if len(tasks) > 4:
        print(f"  order  :")
        for i, t in enumerate(tasks, 1):
            print(f"    {DIM}{i:>2}. {t}{RESET}")
    else:
        print(f"  order  : {DIM}{' \u2192 '.join(tasks)}{RESET}")
    print(f"{BOLD}{bar}{RESET}\n")

    results: List[Tuple[str, int]] = []
    for i, task in enumerate(tasks, 1):
        print(f"\n{BOLD}>>> [{i}/{len(tasks)}] {task}{RESET}")
        rc = _run_training_subprocess(task, seed, dry_run=False, epochs=epochs)
        results.append((task, rc))
        if rc != 0:
            print(f"{YELLOW}!!! [{task}] exited {rc} \u2014 continuing.{RESET}")

    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  {BOLD}{label} sweep complete:{RESET}")
    n_ok = 0
    for task, rc in results:
        if rc == 0:
            mark, color = "\u2713", GREEN
            n_ok += 1
        else:
            mark, color = "\u2717", YELLOW
        print(f"  [{color}{mark}{RESET}] {task:<22s} rc={rc}")
    print(f"{BOLD}{bar}{RESET}")
    print(f"  {n_ok}/{len(results)} tasks succeeded.\n")


def _interactive_launcher() -> None:
    """Prompt the user to pick an action after all setup stages are ready.

    Main entry for the 6-box workflow: re-run `python start.py` per seed,
    pick a task + seed from the menu, train. Option 3 is the Regime A
    one-shot (no training); option 4 dumps the full copy-paste command
    list for scripted workflows. Option 5 is the resume/extend path —
    finds the latest epoch_<N>.pt for any prior run (including ones the
    user Ctrl+C'd) and extends training by a chosen number of epochs.
    Option 6 is the fresh-start path; for LLM family picks it sequences
    sudoku -> maze as two distinct wandb runs.
    """
    print(f"\n{BOLD}What do you want to run?{RESET}")
    print(f"  {CYAN}1{RESET}) Dry run         {DIM}(5-epoch pipeline smoke test \u2014 always do this first){RESET}")
    print(f"  {CYAN}2{RESET}) Seed-variance   {DIM}(full fine-tune from HF init \u2014 the 6-machine plan){RESET}")
    print(f"  {CYAN}3{RESET}) Evaluate HF     {DIM}(Regime A \u2014 all 3 paper checkpoints, no training){RESET}")
    print(f"  {CYAN}4{RESET}) Show commands   {DIM}(print copy-paste commands and exit){RESET}")
    print(f"  {CYAN}5{RESET}) Resume/extend   {DIM}(continue a finished or Ctrl+C'd run for N more epochs){RESET}")
    print(f"  {CYAN}6{RESET}) Fresh start     {DIM}(new run; LLM family picks run sudoku \u2192 maze back-to-back){RESET}")
    print(f"  {CYAN}7{RESET}) Novelty         {DIM}(iso-time + K-vote experiments for the report){RESET}")
    print(f"  {DIM}-- Group Project sprint (Apr-26 plan) ---------------------------{RESET}")
    print(f"  {CYAN}8{RESET}) Maze 50-ep      {DIM}(TRM-Att Maze HF-init fine-tune, 50 ep, 3 seeds: M1/M2/M3){RESET}")
    print(f"  {CYAN}Q{RESET}) Quit")

    choice = _prompt("Pick", default="Q").upper()

    if choice in ("Q", ""):
        print(f"\n{DIM}Nothing launched. Re-run `python start.py` when ready.{RESET}\n")
        return

    if choice == "3":
        _run([PYTHON, os.path.join("scripts", "eval_hf_checkpoints.py")])
        return

    if choice == "4":
        _print_copy_paste_commands()
        return

    if choice == "5":
        _resume_training_picker()
        return

    if choice == "6":
        _fresh_start_launcher()
        return

    if choice == "7":
        from src.cli.novelty import novelty_launcher
        novelty_launcher()
        return

    if choice == "8":
        _group_project_maze_sprint()
        return

    if choice in ("1", "2"):
        task, seed = _prompt_task_and_seed()
        _dispatch_training(task, seed, dry_run=(choice == "1"))
        return

    print(f"{YELLOW}!!! Unknown choice '{choice}'.{RESET}")


def _group_project_maze_sprint() -> None:
    """Group Project sprint launcher — TRM-Att Maze fine-tune, 50 epochs.

    Per the Apr-26 plan: 3 maze seeds across M1/M2/M3, each running a 50-epoch
    HF-init fine-tune with the corrected hparams from
    ``configs/trm_official_maze_finetune.yaml`` (q_loss_weight=0.0,
    halt_exploration_prob=0.0, lr=1e-5, weight_decay=0.1).

    This is a **separate** menu entry (option 8) so it doesn't get confused
    with the from-scratch options 2/6 — both the wall-clock budget (~6h
    per seed) and the kill-rule (epoch-1 val_exact ≥ 0.78 or kill) are
    documented here, not buried in the generic launcher.

    Does three things in order:
      1. Run scripts/bootstrap_hf_maze.py to verify
         hf_checkpoints/Maze-Hard/remapped_for_local.pt exists locally
         (idempotent — no-op if already bootstrapped).
      2. Prompt for seed (0/1/2 — operator picks the row matching their box).
      3. Dispatch ``main.py --mode train --config <maze_finetune> --seed N
         --init-weights <hf-maze.pt>`` via the standard launcher so
         TRM_CHECKPOINT_DIR / TRM_EXPERIMENT_DIR / wandb logging all flow
         through the same plumbing as options 2/6.
    """
    print(f"\n{BOLD}Group Project sprint — TRM-Att Maze 50-epoch fine-tune{RESET}")
    print(f"  {DIM}per docs/trm_3day_todo_v4.pdf + configs/trm_official_maze_finetune.yaml{RESET}\n")

    print(f"{BOLD}3-seed assignment (re-run this menu on each box):{RESET}")
    print(f"  {DIM}machine   seed   ETA{RESET}")
    print(f"  {CYAN}M1{RESET}        {CYAN}0{RESET}      ~6h on RTX 5070")
    print(f"  {CYAN}M2{RESET}        {CYAN}1{RESET}      ~6h")
    print(f"  {CYAN}M3{RESET}        {CYAN}2{RESET}      ~6h")
    print()
    print(f"{BOLD}Kill rule (watch ~30 min in, after epoch-1 eval):{RESET}")
    print(f"  val_exact ≥ 0.78  {GREEN}✓ let it run to epoch 50{RESET}")
    print(f"  0.5 ≤ val < 0.78  {YELLOW}⚠ let it finish, modest gain expected{RESET}")
    print(f"  val_exact < 0.5   {YELLOW}✗ Ctrl+C, q_loss_weight=0.0 fix didn't save it{RESET}\n")

    config = "configs/trm_official_maze_finetune.yaml"
    init_weights = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"

    # 1. Bootstrap HF checkpoint if missing — idempotent.
    bootstrap = os.path.join(ROOT, "scripts", "bootstrap_hf_maze.py")
    if os.path.exists(bootstrap):
        print(f"{BOLD}Step 1: ensure HF Maze-Hard checkpoint is present{RESET}")
        rc = subprocess.run([PYTHON, bootstrap], cwd=ROOT).returncode
        if rc != 0:
            print(f"{YELLOW}!!! bootstrap_hf_maze.py exited {rc}. Fix that, then re-run.{RESET}")
            return
        print()
    else:
        print(f"{YELLOW}!!! {bootstrap} not found — skipping bootstrap. Make sure{RESET}")
        print(f"{YELLOW}    {init_weights} exists locally before launching.{RESET}\n")

    # 2. Prompt for seed.
    seed_str = _prompt("Seed for this machine (0 / 1 / 2)", default="0")
    try:
        seed = int(seed_str)
        if seed not in (0, 1, 2):
            raise ValueError
    except ValueError:
        print(f"{YELLOW}!!! Need 0, 1, or 2 (got '{seed_str}').{RESET}")
        return

    # 3. Verify the init-weights file actually landed.
    init_path = os.path.join(ROOT, init_weights)
    if not os.path.exists(init_path):
        print(f"{YELLOW}!!! Missing {init_weights} — cannot launch without HF init.{RESET}")
        print(f"{DIM}    Re-run scripts/bootstrap_hf_maze.py manually to fix.{RESET}")
        return

    # 4. Print the launch summary so the operator can sanity-check before training.
    bar = "=" * 64
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  config        : {CYAN}{config}{RESET}")
    print(f"  seed          : {CYAN}{seed}{RESET}")
    print(f"  --init-weights: {CYAN}{init_weights}{RESET}")
    print(f"  epochs        : {CYAN}50{RESET}  {DIM}(from config){RESET}")
    print(f"  eval_interval : {CYAN}1{RESET}  {DIM}(catches dz3tkge9-style cliff at epoch 1){RESET}")
    print(f"{BOLD}{bar}{RESET}\n")

    confirm = _prompt("Launch? (y/n)", default="y").lower()
    if confirm not in ("y", "yes"):
        print(f"{DIM}Nothing launched.{RESET}")
        return

    # 5. Dispatch via the standard subprocess launcher so TRM_CHECKPOINT_DIR /
    #    TRM_EXPERIMENT_DIR / wandb all set up identically to options 2/6.
    args = [
        PYTHON, "main.py",
        "--mode", "train",
        "--config", config,
        "--seed", str(seed),
        "--init-weights", init_weights,
    ]
    work_dir = os.environ.get("TRM_WORK_DIR") or _resolve_work_dir()
    run_dir = os.path.join(work_dir, f"trm-att-maze-50ep-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    env = os.environ.copy()
    env["TRM_CHECKPOINT_DIR"] = run_dir
    env["TRM_EXPERIMENT_DIR"] = run_dir

    print(f"{DIM}Launching: {' '.join(args)}{RESET}")
    print(f"{DIM}Run dir   : {run_dir}{RESET}\n")
    sys.exit(_run_training_subprocess(args, env=env))


def _print_copy_paste_commands() -> None:
    """Print all manual training commands. Fallback for scripted workflows."""
    from src.cli.bootstrap import TASK_DISPATCH

    py_quoted = f'"{PYTHON}"'
    is_powershell = (
        platform.system() == "Windows"
        and os.environ.get("PSModulePath") is not None
    )
    py = f"& {py_quoted}" if is_powershell else py_quoted

    print(f"\n{BOLD}Regime A \u2014 Evaluate paper checkpoints (no training):{RESET}")
    print(f"  {CYAN}{py} scripts/eval_hf_checkpoints.py{RESET}\n")

    print(f"{BOLD}Regime B \u2014 Seed-variance fine-tune:{RESET} {DIM}(direct main.py){RESET}")
    for task, (config, init, desc) in TASK_DISPATCH.items():
        init_arg = f" --init-weights {init}" if init and os.path.exists(init) else ""
        print(f"  {CYAN}{py} main.py --mode train --config {config}{init_arg} --seed 0{RESET} {DIM}# {task}{RESET}")
    print()

    print(f"{BOLD}...or the shell launchers (they auto-set TRM_CHECKPOINT_DIR per seed):{RESET}")
    if platform.system() == "Windows":
        print(f"  {CYAN}scripts/run_seed.ps1 -Task sudoku-mlp -Seed 0{RESET}")
        print(f"  {CYAN}scripts/run_seed.ps1 -Task sudoku-mlp -Seed 0 -DryRun{RESET}  {DIM}# 5-epoch smoke{RESET}")
    else:
        print(f"  {CYAN}scripts/run_seed.sh sudoku-mlp 0{RESET}")
        print(f"  {CYAN}scripts/run_seed.sh sudoku-mlp 0 --dry-run{RESET}  {DIM}# 5-epoch smoke{RESET}")
    print()

    print(f"{BOLD}Other (non-fleet) TRM configs:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_sudoku.yaml --seed 0{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/trm_maze.yaml --seed 0{RESET}\n")

    print(f"{BOLD}LLM baselines on Sudoku (LoRA fine-tune):{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_config.yaml --seed 0{RESET}  {DIM}# GPT-2 (124M){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen.yaml --seed 0{RESET}    {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}LLM baselines on Maze (LoRA fine-tune):{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_gpt2_maze.yaml --seed 0{RESET}    {DIM}# GPT-2 (124M){RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_smollm_maze.yaml --seed 0{RESET}  {DIM}# SmolLM2-360M{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_qwen_maze.yaml --seed 0{RESET}    {DIM}# Qwen2.5-0.5B{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/llm_llama_maze.yaml --seed 0{RESET}   {DIM}# Llama-3.2-1B{RESET}\n")

    print(f"{BOLD}Distill a fine-tuned LLM into a small student:{RESET}")
    print(f"  {CYAN}{py} main.py --mode distill --config configs/llm_qwen.yaml --checkpoint models/llm/qwen2.5_0.5b_sudoku_latest.pt{RESET}\n")

    print(f"{BOLD}Evaluate (after training):{RESET}")
    print(f"  {CYAN}{py} main.py --mode eval --config configs/<name>.yaml --checkpoint models/<name>/best.pt{RESET}\n")

    print(f"{BOLD}Resume from last checkpoint:{RESET}")
    print(f"  {CYAN}{py} main.py --mode train --config configs/<name>.yaml --resume models/<name>/latest.pt --seed 0{RESET}\n")

    print(f"{BOLD}Reproducibility \u2014 seed convention:{RESET}")
    print(f"  {DIM}Every command above passes {RESET}{CYAN}--seed N{RESET}{DIM} explicitly so the seed shows up in{RESET}")
    print(f"  {DIM}shell history AND the wandb run name. Omitting --seed inherits the{RESET}")
    print(f"  {DIM}{RESET}{CYAN}seed:{RESET}{DIM} field from the YAML (default 42 \u2014 reproducible). Set {RESET}{CYAN}seed: -1{RESET}{DIM}{RESET}")
    print(f"  {DIM}in the YAML for wall-clock seeding.{RESET}\n")

    print(f"{DIM}Tip: run `python start.py status` any time to re-check stages.{RESET}")
    print(f"{DIM}Tip: Weave traces appear at wandb.ai/<entity>/<project>/weave/monitors{RESET}\n")


def _print_training_menu() -> None:
    from src.cli.bootstrap import FLEET_PLAN

    print(f"\n{GREEN}{BOLD}\u2713 All setup complete!{RESET}\n")

    # 6-machine fleet assignment table. Each operator picks the row matching
    # their box — seeds 0..5 across tasks give the full seed-variance set the
    # coursework report needs for mean +/- std on the three-way model comparison.
    print(f"{BOLD}6-Machine Fleet Plan:{RESET} {DIM}(pick the row for this box){RESET}")
    print(f"  {DIM}machine   task          seed{RESET}")
    for idx, task, seed in FLEET_PLAN:
        print(f"  {CYAN}{idx:<9}{RESET} {task:<13s} {CYAN}{seed}{RESET}")
    print()

    # TRM_WORK_DIR status + reminder. The launcher will exit(3) later if this
    # resolves inside OneDrive; this printout is the friendly early warning.
    print(f"{BOLD}Work dir for training outputs:{RESET}")
    current_work = os.environ.get("TRM_WORK_DIR")
    if current_work:
        work_label = f"{CYAN}{current_work}{RESET}"
    else:
        work_label = f"{CYAN}{_default_work_dir()}{RESET}  {DIM}(auto-picked \u2014 TRM_WORK_DIR not set){RESET}"
    print(f"  TRM_WORK_DIR = {work_label}")
    if platform.system() == "Windows":
        print(f"  {DIM}To set for this shell: {RESET}{CYAN}$env:TRM_WORK_DIR = 'C:/ml-trm-work'{RESET}")
    else:
        print(f"  {DIM}To set for this shell: {RESET}{CYAN}export TRM_WORK_DIR=$HOME/ml-trm-work{RESET}")
    print(f"  {DIM}(MUST be a local non-OneDrive path \u2014 parallel runs on the shared{RESET}")
    print(f"  {DIM} OneDrive would corrupt each machine's checkpoints during upload.){RESET}\n")

    # wandb reminder — re-check here even though the setup stage covered it,
    # because it's common to lose the netrc entry when switching machines.
    if _wandb_ready():
        print(f"{BOLD}wandb:{RESET} {GREEN}\u2713 logged in{RESET}  {DIM}(runs will track to your wandb project){RESET}\n")
    else:
        print(f"{BOLD}wandb:{RESET} {YELLOW}not logged in{RESET}  {DIM}(runs still train, just without cloud tracking){RESET}")
        if os.path.exists(WANDB):
            print(f"  {CYAN}\"{WANDB}\" login{RESET}  {DIM}# paste your key from wandb.ai/authorize{RESET}\n")
        else:
            print(f"  {CYAN}wandb login{RESET}  {DIM}# paste your key from wandb.ai/authorize{RESET}\n")

    # Hand off to the interactive launcher. If the user picks a task, it
    # dispatches via subprocess.run + sys.exit — we never return here. If
    # they quit or pick "show commands", we fall through and exit naturally.
    _interactive_launcher()
