# Reading list — ARC-AGI-3 strategy

20 papers backing the architecture proposal in `../ARC-AGI3.md`. Click links in §8 of that doc for context on why each one is here.

## Download

```bash
# Linux / macOS / git-bash
bash ./download.sh
```

```powershell
# Windows PowerShell
pwsh ./download.ps1
# or, on Windows PowerShell 5.1:
powershell -File ./download.ps1
```

Both scripts skip files already present, so re-running is safe.

## Suggested reading order (40 hours total)

1. **Foundations (10h)** — `16 → 18 → 20 → 19 → 17`. ARC canon first.
2. **Reasoning options (10h)** — `01 → 02 → 03 → 05 → 04`. TRM → HRM → BDH → ACT → Universal.
3. **Planning + world models (10h)** — `09 → 10 → 11 → 12`. MuZero → DreamerV3 → World Models → DT.
4. **Exploration (5h)** — `13 → 14 → 15`. RND → ICM → Go-Explore.
5. **Memory (5h)** — `06 → 07 → 08`. Engram → NTM → Transformer-XL.

## Note on paper #7

Graves' Differentiable Neural Computer (DNC) is paywalled in *Nature* (538, 471–476, 2016). We substitute its open-access predecessor, the **Neural Turing Machine** (arXiv:1410.5401), which introduces the same external-memory read/write primitives. If you have institutional access to *Nature*, swap in the DNC paper instead — it's the more polished version.
