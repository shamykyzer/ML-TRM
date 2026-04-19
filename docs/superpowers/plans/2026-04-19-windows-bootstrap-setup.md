# Windows One-Liner Bootstrap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a fresh Windows machine reach "ready to train ML-TRM" from a single PowerShell one-liner, with no SSH-key setup.

**Architecture:** Two layers.
- **L1** — a new public repo `shamykyzer/ml-machine-setup` holding `setup.ps1`. Fetched anonymously via `irm | iex`. Installs toolchain, checks GPU, prompts for project, clones + dispatches.
- **L2** — `bootstrap.ps1` inside ML-TRM that creates `.venv`, installs GPU-aware PyTorch, installs requirements. Called by L1 after clone, or run manually.

**Tech Stack:** PowerShell 5.1+ (built into Windows), winget (built into Win10 21H2+/Win11), Python 3.12, PyTorch (CUDA 12.4 wheels), Git + Git Credential Manager (HTTPS auth only — no SSH keys).

**Spec reference:** `docs/superpowers/specs/2026-04-19-windows-bootstrap-setup-design.md`

---

## File Structure

**New repo — `ml-machine-setup/`** (lives at `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup/` locally, sibling to ML-TRM):
- `setup.ps1` — L1 script (~130 lines). Single responsibility: install toolchain + prompt + clone/dispatch.
- `README.md` — one-liner at top, manual fallback commands below.
- `.gitignore` — minimal PS/VS Code/Windows junk.

**ML-TRM modifications:**
- Create `bootstrap.ps1` — L2 script (~70 lines). Single responsibility: venv + PyTorch + pip requirements. Mirrors `run.sh setup-cuda` in native PowerShell.
- Modify `.gitignore` — add `papers/*.pdf` entry.
- Modify `README.md` — prepend "New Windows machine setup" section.
- Untrack `papers/01_TRM_Jolicoeur-Martineau_2025.pdf`, `papers/02_HRM_Wang_2025.pdf`, `papers/03_BDH_Pathway_2025.pdf` with `git rm --cached` (files remain on disk).

**Unchanged:** `run.sh`, `start.py`, all `src/`. `bootstrap.ps1` is purely install-time; nothing calls it at runtime.

---

## Testing Approach

PowerShell scripts are verified in three lightweight passes (no Pester for v1):
1. **Syntax check via AST parser** — catches typos without executing anything.
2. **Dry-run execution of safe sections** — checks winget is on PATH, nvidia-smi behaves, `git` works.
3. **Manual end-to-end** on a fresh environment (Windows Sandbox preferred; fresh Windows user account acceptable).

No CI. The script runs rarely; manual verification is proportionate.

---

## Task 1: Create local `ml-machine-setup` folder and skeleton files

**Files:**
- Create: `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup/.gitignore`
- Create: `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup/setup.ps1` (empty for now)
- Create: `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup/README.md` (empty for now)

- [ ] **Step 1.1: Create the directory and `git init` it**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents"
mkdir -p ml-machine-setup
cd ml-machine-setup
git init
git config user.email  # confirm global identity is picked up (should be shamyyk@gmail.com)
```

Expected: empty repo on branch `main` (run `git branch -M main` if it created `master`).

- [ ] **Step 1.2: Create `.gitignore`**

Write to `ml-machine-setup/.gitignore`:

```gitignore
# Windows
Thumbs.db
Desktop.ini
$RECYCLE.BIN/

# VS Code
.vscode/
*.code-workspace

# Test output from script dry-runs
*.log
```

- [ ] **Step 1.3: Create placeholder files for `setup.ps1` and `README.md`**

Just create them empty — content comes in Tasks 2 and 3.

```bash
touch setup.ps1 README.md
```

- [ ] **Step 1.4: First commit**

```bash
git add .gitignore setup.ps1 README.md
git commit -m "chore: scaffold ml-machine-setup repo"
```

**⚠️ Do NOT push.** User-memory rule: no pushing to any remote without explicit "push" word from the user. Tasks 5 and 10 later cover user-driven push.

---

## Task 2: Write `setup.ps1` (L1 one-liner script)

**Files:**
- Modify: `ml-machine-setup/setup.ps1` — full script contents.

This is the heart of the plan. The script has five phases: preflight, install loop, PATH refresh, GPU check, project menu + dispatch. Code below is complete and self-contained.

- [ ] **Step 2.1: Replace `setup.ps1` contents with the full script**

```powershell
# ml-machine-setup / setup.ps1
#
# One-liner installer for a new Windows ML dev machine.
# Fetched via:
#   irm https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | iex
#
# What it does:
#   1. Verifies winget is present
#   2. winget-installs Git, Python 3.12, VS Code, Windows Terminal, GitHub CLI, PowerShell 7
#   3. Refreshes PATH so newly-installed tools are usable in this session
#   4. Checks for an NVIDIA GPU via nvidia-smi (informational; driver install is manual)
#   5. Prompts for which project to clone, clones it, runs its bootstrap.ps1
#
# Design spec: ML-TRM/docs/superpowers/specs/2026-04-19-windows-bootstrap-setup-design.md

$ErrorActionPreference = 'Stop'

function Write-Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

# ---------- 1. Preflight ----------
Write-Section "Preflight"
try {
    $wingetVersion = (winget --version) 2>$null
    Write-Host "winget: $wingetVersion"
} catch {
    Write-Host "ERROR: winget not found." -ForegroundColor Red
    Write-Host "Install the App Installer from https://aka.ms/getwinget and re-run." -ForegroundColor Red
    exit 1
}

# ---------- 2. Install tools ----------
$tools = @(
    @{ Id = 'Git.Git';                    Name = 'Git (+ Credential Manager)' },
    @{ Id = 'Python.Python.3.12';         Name = 'Python 3.12' },
    @{ Id = 'Microsoft.VisualStudioCode'; Name = 'VS Code' },
    @{ Id = 'Microsoft.WindowsTerminal';  Name = 'Windows Terminal' },
    @{ Id = 'GitHub.cli';                 Name = 'GitHub CLI' },
    @{ Id = 'Microsoft.PowerShell';       Name = 'PowerShell 7' }
)

$failures = @()
Write-Section "Installing tools"
foreach ($tool in $tools) {
    Write-Host ""
    Write-Host "- $($tool.Name) ($($tool.Id))"
    winget install --id $tool.Id --silent --accept-source-agreements --accept-package-agreements 2>&1 | Out-Host
    $code = $LASTEXITCODE
    # 0 = installed; -1978335189 (0x8A15002B) = "no applicable upgrade"/"already installed" — treat as success.
    if ($code -ne 0 -and $code -ne -1978335189) {
        Write-Host "  [warn] winget exit code $code for $($tool.Name)" -ForegroundColor Yellow
        $failures += $tool.Name
    }
}
Refresh-Path

# ---------- 3. GPU check (informational) ----------
Write-Section "GPU check"
$gpuDetected = $false
try {
    $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $gpuName) {
        Write-Host "NVIDIA GPU detected: $gpuName" -ForegroundColor Green
        $gpuDetected = $true
    }
} catch { }
if (-not $gpuDetected) {
    Write-Host "No NVIDIA GPU / driver detected." -ForegroundColor Yellow
    Write-Host "ML-TRM's bootstrap.ps1 will install CPU-only PyTorch." -ForegroundColor Yellow
    Write-Host "If this box has a GPU, install drivers from https://www.nvidia.com/Download/index.aspx and re-run." -ForegroundColor Yellow
}

# ---------- 4. Project menu ----------
$projectsDir = Join-Path $env:USERPROFILE 'projects'
if (-not (Test-Path $projectsDir)) { New-Item -ItemType Directory -Path $projectsDir | Out-Null }

function Clone-AndBootstrap($url, $folderName) {
    $target = Join-Path $projectsDir $folderName
    if (Test-Path $target) {
        Write-Host "Folder already exists: $target — skipping clone, will run bootstrap if present." -ForegroundColor Yellow
    } else {
        Write-Host "Cloning $url into $target..."
        git clone $url $target
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: git clone failed (exit $LASTEXITCODE)." -ForegroundColor Red
            return
        }
    }
    Push-Location $target
    try {
        if (Test-Path ".\bootstrap.ps1") {
            Write-Host "Running bootstrap.ps1..."
            & ".\bootstrap.ps1"
        } else {
            Write-Host "No bootstrap.ps1 in the cloned repo — project may need manual setup." -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
}

Write-Section "Which project would you like to set up?"
Write-Host "  1) ML-TRM"
Write-Host "  2) Other (paste a GitHub HTTPS URL)"
Write-Host "  3) None (tools only, exit)"
$choice = Read-Host "Choice"

switch ($choice) {
    '1' {
        Clone-AndBootstrap 'https://github.com/shamykyzer/ML-TRM.git' 'ML-TRM'
    }
    '2' {
        $url = Read-Host "GitHub HTTPS URL (e.g. https://github.com/user/repo.git)"
        if (-not $url) {
            Write-Host "No URL given — skipping clone." -ForegroundColor Yellow
        } else {
            $folder = ($url.TrimEnd('/') -split '/')[-1] -replace '\.git$',''
            Clone-AndBootstrap $url $folder
        }
    }
    '3' {
        Write-Host "Tools installed; no project cloned."
    }
    default {
        Write-Host "Invalid choice — no project cloned." -ForegroundColor Yellow
    }
}

# ---------- 5. Summary ----------
Write-Section "Done"
if ($failures.Count -gt 0) {
    Write-Host "The following tools failed to install:" -ForegroundColor Yellow
    $failures | ForEach-Object { Write-Host "  - $_" }
    Write-Host "Re-run the one-liner to retry; winget will skip already-installed tools." -ForegroundColor Yellow
}
Write-Host "Open a new terminal so PATH changes are picked up everywhere."
```

- [ ] **Step 2.2: Syntax-check the script without running it**

```bash
pwsh -NoProfile -Command "$errs = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'setup.ps1').Path, [ref]$null, [ref]$errs) | Out-Null; if ($errs) { $errs | ForEach-Object { Write-Host $_ }; exit 1 } else { Write-Host 'SYNTAX OK' }"
```

Run this from the `ml-machine-setup/` folder. Expected output: `SYNTAX OK`. If PowerShell 7 (`pwsh`) is not installed yet, substitute `powershell` (Windows PowerShell 5.1 is always present and the AST parser works identically for our syntax).

- [ ] **Step 2.3: Sanity-check the menu rendering (no installs)**

Comment out the `winget install` loop temporarily and the `git clone` call; run the script; confirm the menu prompt appears and accepting `3` exits cleanly. Uncomment after.

Alternatively (no source edits): run only the trailing menu/summary portion by dot-sourcing functions and invoking `Write-Section`, `Refresh-Path`, `Clone-AndBootstrap`-free sections interactively. If uncertain, skip this step — Task 8's end-to-end run will catch menu issues.

- [ ] **Step 2.4: Commit `setup.ps1`**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup"
git add setup.ps1
git commit -m "feat: L1 setup.ps1 (winget install + GPU check + project menu)"
```

---

## Task 3: Write `README.md` for `ml-machine-setup`

**Files:**
- Modify: `ml-machine-setup/README.md`.

- [ ] **Step 3.1: Replace README with final content**

```markdown
# ml-machine-setup

One-liner bootstrap for a new Windows machine running ML experiments.

## Use

Open **Windows PowerShell** (not cmd), paste:

```powershell
irm https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | iex
```

That single line downloads and runs `setup.ps1`, which:

1. Verifies `winget` is available.
2. Installs: Git (+ Credential Manager), Python 3.12, VS Code, Windows Terminal, GitHub CLI, PowerShell 7.
3. Checks for an NVIDIA GPU via `nvidia-smi` (informational only — driver install is manual).
4. Asks which project to set up: `ML-TRM`, another repo by URL, or none.
5. Clones the chosen repo into `%USERPROFILE%\projects\<repo>` and runs its `bootstrap.ps1` if one exists.

No SSH keys are set up or used. Public repos clone anonymously; the first `git push` from the new machine triggers Git Credential Manager to open a browser once for GitHub login.

## Manual fallback

If you cannot run the one-liner (corporate locked-down PowerShell, etc.), the same effect can be reproduced by hand:

1. Install `winget` from <https://aka.ms/getwinget> if missing.
2. `winget install --id Git.Git`
3. `winget install --id Python.Python.3.12`
4. `winget install --id Microsoft.VisualStudioCode`
5. `winget install --id Microsoft.WindowsTerminal`
6. `winget install --id GitHub.cli`
7. `winget install --id Microsoft.PowerShell`
8. Open a **new** PowerShell window.
9. `git clone https://github.com/shamykyzer/ML-TRM.git`
10. `cd ML-TRM`
11. `./bootstrap.ps1`

## NVIDIA driver

This script does **not** install the NVIDIA driver — drivers are not available on winget for consumer GPUs. Install from <https://www.nvidia.com/Download/index.aspx> before running if the box has a GPU.

## Design notes

Full design rationale lives in ML-TRM: `docs/superpowers/specs/2026-04-19-windows-bootstrap-setup-design.md`.
```

- [ ] **Step 3.2: Commit README**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup"
git add README.md
git commit -m "docs: README with one-liner and manual fallback"
```

---

## Task 4: User action — create GitHub repo + push `ml-machine-setup`

This step requires the user's explicit action. The assistant prepares the command block; the user runs it.

- [ ] **Step 4.1: Ask the user whether to create the GitHub repo via `gh` CLI or via the GitHub website**

Present the user with two options:
- **`gh` CLI** (if GitHub CLI is already authenticated locally):
  ```bash
  cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup"
  gh repo create shamykyzer/ml-machine-setup --public --source=. --remote=origin --push
  ```
  This creates the GitHub repo AND pushes in one shot. The `--push` flag is the only acceptable push in this plan and is explicit user intent.
- **Web UI**: user visits <https://github.com/new>, creates `shamykyzer/ml-machine-setup` as **Public**, no README/license/gitignore (we already have them), then locally:
  ```bash
  cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ml-machine-setup"
  git remote add origin https://github.com/shamykyzer/ml-machine-setup.git
  git branch -M main
  git push -u origin main
  ```

- [ ] **Step 4.2: User confirms the repo is live**

Visit <https://github.com/shamykyzer/ml-machine-setup>. Raw URL must resolve:
```
https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1
```

Quick check from the terminal:
```bash
curl -sI https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | head -1
```
Expected: `HTTP/2 200`.

---

## Task 5: Write `bootstrap.ps1` in ML-TRM

**Files:**
- Create: `C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM/bootstrap.ps1`.

- [ ] **Step 5.1: Create `bootstrap.ps1` with the full script**

Working directory: ML-TRM repo root.

```powershell
# ML-TRM / bootstrap.ps1
#
# Windows-native venv + deps setup for ML-TRM.
# Run once after `git clone`. Mirrors `run.sh setup-cuda` in PowerShell.
# Auto-detects GPU via nvidia-smi; override with $env:FORCE_CPU=1 to force CPU wheels.

$ErrorActionPreference = 'Stop'

function Write-Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

# 1. Verify Python 3.12 is on PATH
Write-Section "Verify Python 3.12"
try {
    $pyVersion = py -3.12 --version 2>&1
    Write-Host $pyVersion
} catch {
    Write-Host "ERROR: Python 3.12 not found (`py -3.12`)." -ForegroundColor Red
    Write-Host "Run the machine setup first:" -ForegroundColor Red
    Write-Host "  irm https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | iex" -ForegroundColor Red
    exit 1
}

# 2. Refuse to overwrite an existing .venv (per user's 'no destructive actions' rule)
if (Test-Path .venv) {
    Write-Host "ERROR: .venv already exists in $(Get-Location)." -ForegroundColor Red
    Write-Host "Remove it manually before re-running:" -ForegroundColor Red
    Write-Host "  Remove-Item -Recurse -Force .venv" -ForegroundColor Red
    exit 2
}

# 3. Create venv
Write-Section "Create venv"
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
$py  = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$pip = Join-Path (Get-Location) ".venv\Scripts\pip.exe"

# 4. Upgrade pip
Write-Section "Upgrade pip"
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }

# 5. GPU-aware PyTorch install
Write-Section "Install PyTorch"
$useCuda = $false
if (-not $env:FORCE_CPU) {
    try {
        nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $useCuda = $true }
    } catch { }
}
if ($useCuda) {
    Write-Host "NVIDIA GPU detected — installing CUDA 12.4 wheels." -ForegroundColor Green
    & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "Installing CPU-only PyTorch wheels." -ForegroundColor Yellow
    & $pip install torch torchvision
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed (exit $LASTEXITCODE)" }

# 6. Install requirements.txt
Write-Section "Install requirements.txt"
& $pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed (exit $LASTEXITCODE)" }

# 7. Verify
Write-Section "Verify"
& $py -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# 8. Next steps
Write-Section "Next steps"
Write-Host "Activate the venv and try a quick eval:"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  python main.py --mode eval --config configs\trm_sudoku.yaml --checkpoint models\sudoku\best.pt"
Write-Host ""
Write-Host "Or use the existing task menu (requires Git Bash):"
Write-Host "  bash run.sh"
```

- [ ] **Step 5.2: Syntax-check `bootstrap.ps1`**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
powershell -NoProfile -Command "$errs = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'bootstrap.ps1').Path, [ref]$null, [ref]$errs) | Out-Null; if ($errs) { $errs | ForEach-Object { Write-Host $_ }; exit 1 } else { Write-Host 'SYNTAX OK' }"
```

Expected: `SYNTAX OK`.

- [ ] **Step 5.3: Do NOT run the script against the current ML-TRM checkout — `.venv` already exists**

The script is designed to refuse when `.venv` is present (matches §8 of the spec — non-destructive). Confirm by observing:
```bash
ls .venv | head -3
```
(You should see `Scripts/`, `Lib/`, etc.) This confirms Step 5.1's refuse-case is reachable. The end-to-end test in Task 9 runs the script on a fresh environment where `.venv` won't exist.

- [ ] **Step 5.4: Commit `bootstrap.ps1`**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
git add bootstrap.ps1
git commit -m "feat(setup): add Windows-native bootstrap.ps1 (venv + GPU-aware PyTorch + requirements)"
```

---

## Task 6: Untrack `papers/*.pdf` in ML-TRM

**Files:**
- Modify: `ML-TRM/.gitignore`.
- Untrack (git, not filesystem): `papers/01_TRM_Jolicoeur-Martineau_2025.pdf`, `papers/02_HRM_Wang_2025.pdf`, `papers/03_BDH_Pathway_2025.pdf`.

- [ ] **Step 6.1: Read the current `.gitignore` to find the best insertion point**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
cat -n .gitignore
```

- [ ] **Step 6.2: Append `papers/*.pdf` to `.gitignore`**

Append these three lines (after the existing content, no blank line deletions):

```gitignore

# Papers (arXiv PDFs — fetched on demand via papers/download.sh or download.ps1)
papers/*.pdf
```

- [ ] **Step 6.3: Untrack the three PDFs (files stay on disk)**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
git rm --cached papers/01_TRM_Jolicoeur-Martineau_2025.pdf
git rm --cached papers/02_HRM_Wang_2025.pdf
git rm --cached papers/03_BDH_Pathway_2025.pdf
```

- [ ] **Step 6.4: Verify with `git status`**

```bash
git status --short -- papers/ .gitignore
```

Expected output (order may vary):
```
 M .gitignore
 D papers/01_TRM_Jolicoeur-Martineau_2025.pdf
 D papers/02_HRM_Wang_2025.pdf
 D papers/03_BDH_Pathway_2025.pdf
```

And verify the files still exist on disk:
```bash
ls papers/*.pdf
```

Expected: the three PDFs are still listed.

- [ ] **Step 6.5: Commit**

```bash
git add .gitignore papers/01_TRM_Jolicoeur-Martineau_2025.pdf papers/02_HRM_Wang_2025.pdf papers/03_BDH_Pathway_2025.pdf
git commit -m "chore(papers): untrack arXiv PDFs (still fetchable via papers/download.sh)"
```

Note: `git add` on a deleted-from-index path stages the removal correctly — it's the standard way to confirm a `git rm --cached`.

---

## Task 7: Update ML-TRM's `README.md` with a "New Windows machine setup" section

**Files:**
- Modify: `ML-TRM/README.md` — insert new section near the top.

- [ ] **Step 7.1: Read the README's first ~30 lines to find the right insertion point**

```bash
head -40 "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM/README.md"
```

The current README starts with the title, a one-paragraph description, a "Module/Team" line, then "Recent updates (April 2026)". The new setup section belongs **between the team line and "Recent updates"**.

- [ ] **Step 7.2: Insert the new section**

Use Edit/Write to insert this block after the line starting with `**Module:** UFCFAS-15-2 ...` and before `## Recent updates`:

```markdown

## New Windows machine — one-liner setup

To bring a fresh Windows box from blank to "ready to train ML-TRM" in one paste:

```powershell
irm https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | iex
```

This installs Git, Python 3.12, VS Code, Windows Terminal, GitHub CLI, and PowerShell 7 via winget, then prompts you to clone ML-TRM and runs `bootstrap.ps1` to create the venv and install PyTorch (GPU-aware). No SSH keys needed.

If ML-TRM is already cloned on this machine, you can skip the one-liner and run just the project bootstrap:

```powershell
cd ML-TRM
./bootstrap.ps1
```

See `docs/superpowers/specs/2026-04-19-windows-bootstrap-setup-design.md` for the full design rationale.

```

- [ ] **Step 7.3: Commit**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
git add README.md
git commit -m "docs(readme): add Windows one-liner setup section"
```

---

## Task 8: User action — make ML-TRM public and push

This is **two distinct actions** that only the user should authorise:
1. Flipping the repo's visibility to Public on github.com.
2. Pushing the local commits (bootstrap, untracked PDFs, README).

- [ ] **Step 8.1: User flips ML-TRM to Public on GitHub**

The user visits <https://github.com/shamykyzer/ML-TRM/settings> → scroll to **Danger Zone** → **Change repository visibility** → **Make public**. GitHub asks to confirm by typing the repo name.

Before clicking: confirm the PDFs have been untracked (Task 6 done). `git log --stat` should show the untrack commit.

- [ ] **Step 8.2: User confirms public visibility**

```bash
curl -sI https://github.com/shamykyzer/ML-TRM | head -1
```
Expected: `HTTP/2 200` (the unauthenticated URL resolves for anyone).

Or visit <https://github.com/shamykyzer/ML-TRM> in a private/incognito window and confirm the repo contents render.

- [ ] **Step 8.3: User pushes the local commits**

**This is the only `git push` command that should be issued in this plan for ML-TRM, and it requires the user's explicit "push" instruction per their memory rule.**

Present the command to the user; do not run it unbidden:

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
git push origin main
```

If GCM is not configured yet on this machine, the push triggers a browser sign-in once. After that, all future pushes work without prompts.

---

## Task 9: Manual end-to-end verification

**Goal:** confirm the one-liner actually works on a fresh Windows environment.

Three options, in order of preference. Pick one.

- [ ] **Step 9.1 (option A, preferred): Windows Sandbox**

Requirement: Windows 10/11 Pro, Enterprise, or Education. Enable "Windows Sandbox" from "Turn Windows features on or off".

1. Start **Windows Sandbox** (fresh VM that resets on close).
2. Inside the sandbox, open PowerShell.
3. Paste the one-liner:
   ```powershell
   irm https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | iex
   ```
4. Observe each phase prints its section header.
5. When prompted, choose `1) ML-TRM`.
6. Expect: `git clone` of ML-TRM → `bootstrap.ps1` runs → CPU-only PyTorch installs (sandbox has no GPU) → `requirements.txt` installs → final line prints `torch X.Y.Z, CUDA available: False`.
7. Close the sandbox (state is discarded).

Success criteria:
- One-liner completes without an exception.
- `%USERPROFILE%\projects\ML-TRM\.venv\Scripts\python.exe -c "import torch"` succeeds inside the sandbox before closing.

- [ ] **Step 9.2 (option B, lower fidelity): Fresh local Windows user account**

If Windows Sandbox isn't available:
1. Create a new Windows user via Settings → Accounts → Other users → Add account.
2. Sign in as that user.
3. Repeat Steps 9.1.3–9.1.6 above.
4. After verifying, delete the user account.

Known quirk: tools winget has already installed at the machine level (Git, Python) may be reported as "already installed" to the new user; this is correct behaviour and the script treats it as success.

- [ ] **Step 9.3 (option C, highest fidelity, costs money): Cloud Windows VM**

Out of scope for v1. Document as future work if the script grows enough to justify real CI.

- [ ] **Step 9.4: Capture the verification outcome**

Append a dated note to `log.md` at the ML-TRM repo root:

```markdown
## YYYY-MM-DD — one-liner verified
- Environment: Windows Sandbox / fresh user account
- ml-machine-setup commit: <hash from git rev-parse HEAD>
- ML-TRM commit: <hash>
- Result: success / details
```

- [ ] **Step 9.5: Commit the log entry**

```bash
cd "C:/Users/amm-alshamy/OneDrive - UWE Bristol/Documents/ML-TRM"
git add log.md
git commit -m "log: one-liner verification passed"
```

(No push — user pushes when they choose to.)

---

## Self-Review Notes

**Spec coverage:**
- §2 "one-line bootstrap" → Task 2 (`setup.ps1`) + Task 3 (README) + Task 4 (push).
- §2 "no SSH keys" → baked into §4.2 of spec; no plan task needed (behaviour of HTTPS + GCM).
- §2 "no manual clone of the setup script repo" → Task 2, Step 2.1 (uses `irm | iex`).
- §2 "re-runnable" → Task 2, Step 2.1 (winget idempotent; exit code -1978335189 treated as success).
- §2 "auto-detect GPU" → Task 2, Step 2.1 (L1 info check) + Task 5, Step 5.1 (L2 wheel selection).
- §2 "generic machine + project split" → Task 2 (L1 generic) + Task 5 (L2 project-specific).
- §3 non-goals (cross-platform, CI, dotfiles, driver, CUDA toolkit) → not touched.
- §5.1 tools list → Task 2, Step 2.1 `$tools` array.
- §5.2 flow → Task 2, Step 2.1 sections 1–5.
- §5.3 edge cases → Task 2, Step 2.1 (winget missing = preflight exit; Python 3.13 alongside 3.12 = winget handles; metered = FORCE_CPU env var in L2).
- §6.1/6.2 L2 flow → Task 5, Step 5.1.
- §6.3 FORCE_CPU → Task 5, Step 5.1.
- §7 repo layout → Task 1 + Task 3 (ml-machine-setup); Task 5 + Task 7 (ML-TRM additions).
- §8 idempotency → Task 2 Step 2.1 winget codes; Task 5 Step 5.1 `.venv` refuse.
- §9 security: papers untrack → Task 6. History not rewritten → confirmed (no `filter-repo`/BFG in plan).
- §10 testing → Task 9.
- §11 open questions → accepted defaults (main branch, no menu memory); nothing to implement.

**Placeholder scan:** no TBD/TODO; every code step contains the complete code; every command has the expected outcome.

**Type/name consistency:** `Write-Section`, `Refresh-Path`, `Clone-AndBootstrap` defined once and reused; `$tools`, `$failures`, `$py`, `$pip` variables consistent across tasks.

**One gap fixed inline:** initial draft of Task 2 omitted the "Tools that failed to install" summary block — added to Step 2.1 as the final section.
