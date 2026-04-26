# Upload this machine's training artifacts to the appropriate GitHub release.
#
# Run this on any lab machine (M1-M5) AFTER its sprint phases complete.
# Auto-discovers run folders under $TRM_WORK_DIR (or C:\ml-trm-work) and
# uploads them as zipped assets to the chosen release tag.
#
# Usage (PowerShell):
#   .\scripts\upload_to_release.ps1                                  # interactive — asks for machine + release
#   .\scripts\upload_to_release.ps1 -Machine M1 -Release v1.1-sprint # non-interactive
#   .\scripts\upload_to_release.ps1 -DryRun                          # show what would be zipped/uploaded
#
# Prerequisites:
#   1. gh CLI installed:    winget install --id GitHub.cli
#      (restart shell to pick up PATH after install)
#   2. gh authenticated:    gh auth login   (web browser flow, one-time)
#   3. Machine has produced its sprint outputs (best.pt files in run folders).

[CmdletBinding()]
param(
    [ValidateSet('M1','M2','M3','M4','M5','M6')]
    [string]$Machine,
    [string]$Release = 'v1.1-sprint',
    [string]$Repo = 'shamykyzer/ML-TRM',
    [string]$WorkDir,
    [switch]$DryRun
)

# Resolve gh path even if PATH wasn't refreshed after install.
$ghCandidates = @(
    "C:\Program Files\GitHub CLI\gh.exe",
    "$env:LOCALAPPDATA\GitHub CLI\gh.exe",
    'gh'
)
$gh = $ghCandidates | Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if (-not $gh) {
    Write-Error "gh CLI not found. Install with:  winget install --id GitHub.cli"
    exit 1
}

if (-not $WorkDir) {
    $WorkDir = $env:TRM_WORK_DIR
    if (-not $WorkDir) { $WorkDir = 'C:\ml-trm-work' }
}
if (-not (Test-Path $WorkDir)) {
    Write-Error "Work dir not found: $WorkDir"
    exit 1
}

if (-not $Machine) {
    $Machine = Read-Host "Which machine is this? (M1/M2/M3/M4/M5/M6)"
}

# Per-machine expected artifacts. Each entry is a list of folder NAMES under
# $WorkDir; the script picks the first one that exists.
$machineFolders = @{
    M1 = @(
        # Existing Qwen checkpoints (no retrain — uploaded for completeness)
        @{name='llm-qwen-sudoku-seed0';        candidates=@('llm-qwen-sudoku-seed0')},
        @{name='llm-qwen-maze-seed0';          candidates=@('llm-qwen-maze-seed0')},
        # Sprint outputs M1 produces
        @{name='trm-att-maze-seed0';           candidates=@('trm-att-maze-50ep-seed0','maze-seed0')},
        @{name='distill-qwen-sudoku-seed0';    candidates=@('distill-qwen-sudoku-seed0')},
        @{name='distill-qwen-maze-seed0';      candidates=@('distill-qwen-maze-seed0')}
    )
    M2 = @(
        @{name='trm-att-maze-seed1';           candidates=@('trm-att-maze-50ep-seed1','maze-seed1')}
    )
    M3 = @(
        @{name='trm-att-maze-seed2';           candidates=@('trm-att-maze-50ep-seed2','maze-seed2')}
    )
    M4 = @(
        @{name='gpt2-sudoku-seed0';            candidates=@('llm-gpt2-sudoku-seed0','gpt2-sudoku-seed0','llm-config-seed0')},
        @{name='distill-gpt2-sudoku-seed0';    candidates=@('distill-gpt2-sudoku-seed0')}
    )
    M5 = @(
        @{name='gpt2-maze-seed0';              candidates=@('llm-gpt2-maze-seed0','gpt2-maze-seed0')},
        @{name='distill-gpt2-maze-seed0';      candidates=@('distill-gpt2-maze-seed0')}
    )
    M6 = @()
}

if (-not $machineFolders.ContainsKey($Machine)) {
    Write-Error "Unknown machine: $Machine"
    exit 1
}

$plan = $machineFolders[$Machine]
if ($plan.Count -eq 0) {
    Write-Output "Machine $Machine has no artifacts to upload (paper-writing/backup)."
    exit 0
}

$staging = "C:\release-staging-$Machine"
if (-not (Test-Path $staging)) {
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
}

Write-Output ""
Write-Output "=================================================================="
Write-Output "  Machine: $Machine"
Write-Output "  Release: $Release  ($Repo)"
Write-Output "  Work dir: $WorkDir"
Write-Output "  Staging: $staging"
Write-Output "  DryRun: $DryRun"
Write-Output "=================================================================="
Write-Output ""

# Step 1: zip each expected folder if it exists.
$zipsToUpload = @()
foreach ($entry in $plan) {
    $found = $null
    foreach ($cand in $entry.candidates) {
        $tryPath = Join-Path $WorkDir $cand
        if (Test-Path $tryPath) {
            $found = $tryPath
            break
        }
    }
    if (-not $found) {
        Write-Output "SKIP  $($entry.name)  (no folder found in $WorkDir for candidates: $($entry.candidates -join ', '))"
        continue
    }

    $zipPath = Join-Path $staging "$($entry.name).zip"
    $size = (Get-ChildItem $found -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB
    Write-Output ("ZIP   {0,-32s}  source: {1}  ({2:N0} MB raw)" -f $entry.name, $found, $size)

    if (-not $DryRun) {
        if (Test-Path $zipPath) {
            Remove-Item $zipPath -Force
        }
        Compress-Archive -Path "$found\*" -DestinationPath $zipPath -CompressionLevel Optimal
        $zipSize = (Get-Item $zipPath).Length / 1MB
        Write-Output ("      -> {0}  ({1:N1} MB compressed)" -f $zipPath, $zipSize)
    }
    $zipsToUpload += $zipPath
}

if ($zipsToUpload.Count -eq 0) {
    Write-Output ""
    Write-Output "Nothing to upload. Exiting."
    exit 0
}

Write-Output ""
Write-Output "Files to upload to '$Release':"
foreach ($z in $zipsToUpload) { Write-Output "  - $z" }

if ($DryRun) {
    Write-Output ""
    Write-Output "DryRun: stopping before gh release upload."
    exit 0
}

# Step 2: ensure release exists. Create if missing.
& $gh release view $Release --repo $Repo 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output ""
    Write-Output "Release '$Release' does not exist on $Repo — creating it."
    & $gh release create $Release --repo $Repo --title "ML-TRM sprint outputs ($Release)" --notes "Sprint training outputs from M1-M5. See CHECKPOINTS.md in main for the manifest."
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create release $Release"
        exit 1
    }
}

# Step 3: upload all zips at once. --clobber overwrites if a previous run uploaded the same name.
Write-Output ""
Write-Output "Uploading $($zipsToUpload.Count) zip(s) to release $Release ..."
$start = Get-Date
& $gh release upload $Release @zipsToUpload --repo $Repo --clobber
if ($LASTEXITCODE -ne 0) {
    Write-Error "Upload failed (exit $LASTEXITCODE)"
    exit 1
}
$elapsed = (Get-Date) - $start
Write-Output ""
Write-Output "Done in $([math]::Round($elapsed.TotalMinutes, 1)) min."
Write-Output ""
Write-Output "Verify at:  https://github.com/$Repo/releases/tag/$Release"
