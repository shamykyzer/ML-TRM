<#
.SYNOPSIS
    Launch one seed of one task, with OneDrive-safe local paths.

.DESCRIPTION
    PowerShell twin of scripts/run_seed.sh. Same semantics, same env vars.
    Every machine in the 6-box fleet has the repo synced via OneDrive, so
    code + data + HF checkpoints are consistent across machines (good),
    but writing training outputs inside the sync folder would corrupt
    checkpoints during OneDrive uploads. Every run must route its outputs
    to a local path — which is exactly what this script enforces.

.PARAMETER Task
    sudoku-mlp | sudoku-att | maze | llm-sudoku

.PARAMETER Seed
    Non-negative integer. Machines 1..6 should use seeds 0..5.

.PARAMETER DryRun
    Shorten training to 5 epochs via main.py's --epochs override. Use this
    for a pipeline smoke test before committing to a multi-day seed run.

.EXAMPLE
    # machine 1, seed 0, sudoku-mlp fine-tune
    scripts\run_seed.ps1 -Task sudoku-mlp -Seed 0

.EXAMPLE
    # 5-epoch smoke test on any machine
    scripts\run_seed.ps1 -Task sudoku-mlp -Seed 0 -DryRun

.NOTES
    TRM_WORK_DIR env var controls the root of per-seed output dirs.
    Defaults to $env:USERPROFILE\ml-trm-work. MUST be a local path,
    never inside OneDrive.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("sudoku-mlp", "sudoku-att", "maze", "llm-sudoku")]
    [string]$Task,

    [Parameter(Mandatory = $true, Position = 1)]
    [ValidateRange(0, [int]::MaxValue)]
    [int]$Seed,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Repo root = parent of the scripts/ dir this file lives in.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

# Venv python resolution — mirrors start.py's VENV_DIR logic. $env:TRM_PYTHON
# wins if set, otherwise we walk the standard venv locations.
if ($env:TRM_PYTHON) {
    $Py = $env:TRM_PYTHON
}
else {
    $candidates = @(
        (Join-Path $env:USERPROFILE ".venvs\ml-trm\Scripts\python.exe"),
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root ".venv\bin\python")
    )
    $Py = $null
    foreach ($c in $candidates) {
        if (Test-Path $c) { $Py = $c; break }
    }
    if (-not $Py) {
        Write-Error "No venv python found. Run 'python start.py' first."
        exit 2
    }
}

# Per-machine local work dir. The hard constraint: it must NOT be inside
# OneDrive, or parallel runs across 6 machines will corrupt each other's
# checkpoints. Default to %USERPROFILE%\ml-trm-work (outside OneDrive on a
# standard Windows install).
if ($env:TRM_WORK_DIR) {
    $WorkDir = $env:TRM_WORK_DIR
}
else {
    $WorkDir = Join-Path $env:USERPROFILE "ml-trm-work"
}

if ($WorkDir.ToLower().Contains("onedrive")) {
    Write-Error @"
TRM_WORK_DIR='$WorkDir' looks like a OneDrive path.
Parallel training on shared OneDrive will corrupt checkpoints.
Pick a local path (e.g. C:\ml-trm-work) and re-run:
    `$env:TRM_WORK_DIR = 'C:\ml-trm-work'
    scripts\run_seed.ps1 -Task $Task -Seed $Seed
"@
    exit 3
}

$TaskDir = Join-Path $WorkDir "$Task-seed$Seed"
New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null
$env:TRM_CHECKPOINT_DIR = $TaskDir
$env:TRM_EXPERIMENT_DIR = $TaskDir

# Task dispatch — one (config, init_weights) pair per task label. Matches
# the bash version line-for-line so the two are swap-in replacements.
switch ($Task) {
    "sudoku-mlp" {
        $Config = "configs/trm_official_sudoku_mlp.yaml"
        $Init = "hf_checkpoints/Sudoku-Extreme-mlp/remapped_for_local.pt"
    }
    "sudoku-att" {
        $Config = "configs/trm_official_sudoku.yaml"
        $Init = "hf_checkpoints/Sudoku-Extreme-att/remapped_for_local.pt"
    }
    "maze" {
        $Config = "configs/trm_official_maze.yaml"
        $Init = "hf_checkpoints/Maze-Hard/remapped_for_local.pt"
    }
    "llm-sudoku" {
        # LLM dispatch in main.py currently only wires the sudoku data
        # loader, so llm-maze is not supported yet. Qwen2.5-0.5B as the
        # representative baseline — switch to llm_smollm / llm_llama /
        # llm_config for the other three LLMs in the proposal.
        $Config = "configs/llm_qwen.yaml"
        $Init = ""
    }
}

# Build argv for main.py. Seed is always explicit so it appears in wandb
# run names and in the CSV log filename for post-hoc aggregation.
$MainArgs = @(
    "main.py",
    "--mode", "train",
    "--config", $Config,
    "--seed", "$Seed"
)
if ($Init) {
    $MainArgs += @("--init-weights", $Init)
}
if ($DryRun) {
    # main.py --epochs 5 overrides training.epochs from the YAML.
    $MainArgs += @("--epochs", "5")
}

$dryLabel = if ($DryRun) { "YES (5 epochs)" } else { "no" }
$initLabel = if ($Init) { $Init } else { "<none, random init>" }

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  task           : $Task"
Write-Host "  seed           : $Seed"
Write-Host "  config         : $Config"
Write-Host "  init_weights   : $initLabel"
Write-Host "  TRM_CHECKPOINT_DIR : $env:TRM_CHECKPOINT_DIR"
Write-Host "  TRM_EXPERIMENT_DIR : $env:TRM_EXPERIMENT_DIR"
Write-Host "  python         : $Py"
Write-Host "  dry-run        : $dryLabel"
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

& $Py @MainArgs
exit $LASTEXITCODE
