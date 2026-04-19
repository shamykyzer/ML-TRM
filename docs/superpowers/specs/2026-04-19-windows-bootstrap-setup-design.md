# Windows Bootstrap Setup — Design

**Date:** 2026-04-19
**Status:** Design approved; awaiting user review before implementation plan.
**Author:** shamykyzer (with Claude Code, brainstorming skill)

## 1. Problem

Setting up a fresh Windows machine to work on ML-TRM currently requires manual
steps: install Git, generate/register an SSH key, install Python, create venv,
install CUDA wheels, install requirements. The SSH-key dance is the most
friction-heavy step and has no functional payoff — GitHub's HTTPS + Git
Credential Manager (GCM) gives the same capabilities via a one-time browser
login per machine, and public repos need no login at all.

Goal: bring a new Windows box from blank to "ready to train ML-TRM" with a
single copy-paste command and no SSH-key work.

## 2. Goals

- **One-line bootstrap**: paste `irm <URL> | iex` in PowerShell → everything
  installs.
- **No SSH keys**, ever. All Git operations use HTTPS; GCM handles auth
  interactively on first push.
- **No manual clone of the setup script repo.** The one-liner fetches the
  script anonymously from a public raw URL.
- **Re-runnable (idempotent)**. Running the one-liner twice is safe; already-
  installed tools are detected and skipped.
- **Auto-detect GPU**. Install CUDA PyTorch wheels only if `nvidia-smi`
  confirms an NVIDIA GPU.
- **Generic machine setup + project-specific setup cleanly separated**, so a
  second ML repo added later reuses the machine layer unchanged.

## 3. Non-goals (v1)

- Cross-platform (macOS/Linux) bootstrap. Scope is Windows only.
- Automated CI for the setup scripts. Manual verification on a throwaway
  user/VM is proportionate given how rarely this runs.
- Dotfiles, PowerShell profile customisation, VS Code settings sync. Can be
  added to the setup repo later without breaking the one-liner.
- NVIDIA driver installation. Drivers must be installed from nvidia.com by the
  user; the script only detects their presence and warns if missing.
- CUDA Toolkit installation. PyTorch wheels bundle the CUDA runtime; no
  toolkit needed.

## 4. Architecture

Two layers in two repos:

| Layer | Repo | File(s) | Runs |
|---|---|---|---|
| **L1 — machine setup** | `shamykyzer/ml-machine-setup` (new, public) | `setup.ps1` | Once per new Windows machine |
| **L2 — project setup** | `shamykyzer/ML-TRM` (existing, made public) | `bootstrap.ps1` | Once per clone of ML-TRM |

### 4.1 The one-liner

```powershell
irm https://raw.githubusercontent.com/shamykyzer/ml-machine-setup/main/setup.ps1 | iex
```

`irm` = `Invoke-RestMethod` (fetches the raw script content anonymously over
HTTPS). `iex` = `Invoke-Expression` (runs the content as PowerShell). The
gist-vs-repo decision was resolved as **repo** for extensibility — a single-
file repo today, room to grow later (dotfiles, PS profile, a second script).

### 4.2 Auth model

- **L1 fetch (anonymous HTTPS)**: `ml-machine-setup` is public, so `irm`
  downloads `setup.ps1` with no credentials.
- **L2 clone (anonymous HTTPS)**: ML-TRM will be made public (see §9 Security),
  so `git clone https://github.com/shamykyzer/ML-TRM.git` runs with no
  credentials.
- **First `git push` from the new machine** (post-setup, whenever the user
  edits code): GCM opens a browser once, user signs into GitHub, credential
  stored in Windows Credential Manager. This is the single unavoidable browser
  login and is not part of the setup flow.
- **SSH is never used.** `~/.ssh` is not touched.

## 5. L1 — `setup.ps1` (ml-machine-setup repo)

### 5.1 Tools installed via winget (Tier 1 + 2 + 3)

| ID | Tool | Tier | Rationale |
|---|---|---|---|
| `Git.Git` | Git for Windows (incl. GCM) | 1 | Needed for clone/push; GCM provides browser-based HTTPS auth |
| `Python.Python.3.12` | CPython 3.12 | 1 | PyTorch CUDA wheels require 3.10–3.12; 3.12 is newest supported |
| `Microsoft.VisualStudioCode` | VS Code | 2 | Primary editor |
| `Microsoft.WindowsTerminal` | Windows Terminal | 2 | Better default terminal |
| `GitHub.cli` | `gh` | 3 | Second credential helper option; useful on Linux training boxes later |
| `Microsoft.PowerShell` | PowerShell 7 | 3 | Modern cross-platform PowerShell |

Install command per tool:
```
winget install --id <ID> --silent --accept-source-agreements --accept-package-agreements
```

### 5.2 Flow

1. **Preflight.**
   - Verify `winget --version` works. If not, print a clear error pointing to
     <https://aka.ms/getwinget> and exit with non-zero code.
2. **Install loop.** Iterate tool list, calling winget for each. Treat "already
   installed" exit codes as success. If any single tool fails, report it and
   continue.
3. **PATH refresh.** After installs, re-import system and user PATH from the
   registry so newly-installed `git` and `python` binaries are callable in the
   same session.
4. **GPU check.** Call `nvidia-smi`. On success, print the detected GPU name.
   On failure, print a warning ("no NVIDIA GPU detected; CPU-only PyTorch will
   be installed for ML-TRM") — not fatal.
5. **Project menu.** Prompt:
   ```
   Which project would you like to set up?
     1) ML-TRM
     2) Other (paste a GitHub HTTPS URL)
     3) None (tools only, exit)
   Choice:
   ```
6. **Dispatch on choice.**
   - `1` → `git clone https://github.com/shamykyzer/ML-TRM.git` into
     `%USERPROFILE%\projects\ML-TRM` (create parent if needed) → `cd` in →
     run `.\bootstrap.ps1`.
   - `2` → prompt for URL → derive folder name from URL (last path segment,
     strip `.git`) → clone into `%USERPROFILE%\projects\<name>` → `cd` in →
     if `bootstrap.ps1` exists at repo root, run it; otherwise print "no
     bootstrap.ps1 found — project may need manual setup" and exit.
   - `3` → print next-steps hint and exit 0.
7. **Exit codes.** 0 on success; non-zero only on preflight failure. Tool-
   install failures are reported but do not change exit code (user can re-run).

### 5.3 Edge cases

- **Old Windows without winget.** Detected in step 1; user gets the fix URL
  and exits before any state change.
- **User already has Python 3.13 but not 3.12.** Winget installs 3.12
  alongside; `py -3.12` selects it for venv creation in L2.
- **Metered connection.** Not handled in v1. User can set `$env:FORCE_CPU=1`
  before the one-liner to skip CUDA wheels in L2 (see §6.3).
- **No internet mid-run.** Winget will fail each package; script reports all
  failures at the end so the user knows exactly which tools to retry.

## 6. L2 — `bootstrap.ps1` (ML-TRM repo)

### 6.1 Responsibilities

Mirror `run.sh`'s `setup-cuda` target in native PowerShell, so a Windows user
does not need Git Bash to bootstrap the project.

### 6.2 Flow

1. **Verify Python 3.12.** `py -3.12 --version`. If missing, print an error
   referring the user to the L1 setup and exit with code 1.
2. **Create venv.** `py -3.12 -m venv .venv`.
3. **Upgrade pip.** `.venv\Scripts\python.exe -m pip install --upgrade pip`.
4. **GPU detection.** Run `nvidia-smi` quietly. On success AND
   `$env:FORCE_CPU` unset/empty → install CUDA wheels:
   `.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
   Otherwise install default (CPU) wheels: `pip install torch torchvision`.
5. **Install requirements.** `.venv\Scripts\pip.exe install -r requirements.txt`.
6. **Verify.** Run a one-line torch check:
   `.venv\Scripts\python.exe -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"`.
7. **Print next steps.** Suggest `python main.py --mode eval ...` or the run
   menu.

### 6.3 Environment overrides

- `$env:FORCE_CPU=1` before running `bootstrap.ps1` → skips CUDA wheel install
  even if `nvidia-smi` succeeds. For laptops where the user wants CPU torch
  to save 2.3 GB of download.

## 7. Repo naming & layout

**L1 repo name:** `shamykyzer/ml-machine-setup` (short, appears in the
one-liner URL; kebab-case for URL-friendliness; generic so a second ML repo
can reuse it unchanged).

**L1 repo layout:**
```
ml-machine-setup/
├── README.md        # the one-liner at the top; manual fallback below
└── setup.ps1        # the entire script (single file for v1)
```

**L2 additions to ML-TRM:**
```
ML-TRM/
├── bootstrap.ps1    # NEW — Windows-native venv+deps setup
└── README.md        # UPDATE — add "New Windows machine setup" section
```

`start.py` (existing) continues to serve as the post-setup runtime launcher.
`run.sh` (existing) is unchanged; it remains the primary task runner for
Git Bash / Linux users. `bootstrap.ps1` is explicitly install-time only.

## 8. Idempotency & error handling

- **Winget idempotency.** Already-installed packages return a non-zero exit
  code with a recognisable message; the script treats these as success.
  Nothing is ever uninstalled or overwritten.
- **Venv re-creation.** `bootstrap.ps1` does **not** delete an existing
  `.venv`. If the directory exists, the script prints a warning and exits
  with code 2 (distinct from success/pre-flight failure), asking the user to
  remove it manually. Preserves in-progress work and aligns with the user's
  "no destructive actions" rule.
- **Partial L1 failure.** If a winget install fails, the script records the
  failure, keeps going, and reports all failures at the end. Exit code stays
  0 (user can re-run the one-liner).
- **PATH stale within session.** After each install, the script re-reads PATH
  from the registry so subsequent steps see the new binary without needing a
  fresh terminal.

## 9. Security considerations

- **Making ML-TRM public.** Required for anonymous HTTPS clone. Verified
  before spec sign-off: `wandb_api.txt` is gitignored (line 13 of
  `.gitignore`) and has never been committed. No API keys, tokens, or
  credentials are in tracked files.
- **Papers untracked before going public.** Three arXiv preprints are
  currently committed under `papers/` (`01_TRM_...pdf`, `02_HRM_...pdf`,
  `03_BDH_...pdf`, ~15 MB total). The implementation plan will add
  `papers/*.pdf` to `.gitignore` and run `git rm --cached papers/*.pdf`
  (non-destructive — files remain on disk). The existing
  `papers/download.sh` / `download.ps1` fetch them from arXiv on demand, so
  anyone cloning the repo runs those once to recreate the folder. History is
  **not** rewritten; the 15 MB in past commits is harmless (public arXiv
  PDFs, not secrets) and rewriting history is destructive.
- **`irm | iex` trust model.** Running `irm <URL> | iex` executes whatever is
  at that URL at that moment. Mitigation: the repo is owned by the user. For
  stronger guarantees the one-liner could pin a tag
  (`.../releases/download/v1/setup.ps1`) later; not v1 scope.
- **GCM credential storage.** First `git push` prompts a browser; resulting
  token is stored in Windows Credential Manager (DPAPI-encrypted, per-user).
  No plaintext credentials on disk.

## 10. Testing

Manual verification on a fresh environment. Options, in order of preference:

1. **Windows Sandbox** (built into Win10/11 Pro) — throwaway VM that resets
   on close. Paste one-liner, confirm flow reaches menu, pick ML-TRM, confirm
   `.venv` is created and `torch.cuda.is_available()` prints.
2. **Fresh Windows user account** on the current machine — lower fidelity
   (winget cache, drivers shared) but catches user-PATH issues.
3. **Cloud Windows VM** (Azure, GCP) — highest fidelity, costs real money.

No automated tests for v1. A future CI could run the script in a GitHub
Actions Windows runner, but it's overkill until the script grows.

## 11. Open questions (non-blocking)

- Does the user want a "pin to tag" version of the one-liner documented in the
  L1 README (for security-conscious re-runs), or is `main` fine? — default to
  `main` for v1.
- Should the L1 menu remember the last choice (e.g. via a file in
  `$env:USERPROFILE`) for subsequent runs? — YAGNI until a second machine
  setup exists as evidence the choice matters.

## 12. Next step

Invoke `superpowers:writing-plans` skill to turn this design into a concrete
implementation plan (creating the repo, writing both scripts, updating the
ML-TRM README, verifying manually).
