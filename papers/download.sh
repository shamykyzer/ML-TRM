#!/usr/bin/env bash
# Downloads the 20 papers cited in ../ARC-AGI3.md.
# Skips papers already on disk so it's safe to re-run.
# Run from anywhere; it cd's into its own directory first.
set -euo pipefail
cd "$(dirname "$0")"

# Format: "<filename>|<url>"
PAPERS=(
  # A. Reasoning cores
  "01_TRM_Jolicoeur-Martineau_2025.pdf|https://arxiv.org/pdf/2510.04871.pdf"
  "02_HRM_Wang_2025.pdf|https://arxiv.org/pdf/2506.21734.pdf"
  "03_BDH_Pathway_2025.pdf|https://arxiv.org/pdf/2509.26507.pdf"
  "04_UniversalTransformer_Dehghani_2018.pdf|https://arxiv.org/pdf/1807.03819.pdf"
  "05_ACT_Graves_2016.pdf|https://arxiv.org/pdf/1603.08983.pdf"

  # B. Memory architectures
  "06_Engram_DeepSeek_2026.pdf|https://arxiv.org/pdf/2601.07372.pdf"
  # DNC (Graves 2016, Nature) is paywalled — we substitute the open-access
  # predecessor, Neural Turing Machine, which has the same memory primitives.
  "07_NeuralTuringMachine_Graves_2014.pdf|https://arxiv.org/pdf/1410.5401.pdf"
  "08_TransformerXL_Dai_2019.pdf|https://arxiv.org/pdf/1901.02860.pdf"

  # C. Model-based RL & planning
  "09_MuZero_Schrittwieser_2020.pdf|https://arxiv.org/pdf/1911.08265.pdf"
  "10_DreamerV3_Hafner_2023.pdf|https://arxiv.org/pdf/2301.04104.pdf"
  "11_WorldModels_HaSchmidhuber_2018.pdf|https://arxiv.org/pdf/1803.10122.pdf"
  "12_DecisionTransformer_Chen_2021.pdf|https://arxiv.org/pdf/2106.01345.pdf"

  # D. Exploration / sparse-reward
  "13_RND_Burda_2018.pdf|https://arxiv.org/pdf/1810.12894.pdf"
  "14_ICM_Pathak_2017.pdf|https://arxiv.org/pdf/1705.05363.pdf"
  "15_GoExplore_Ecoffet_2019.pdf|https://arxiv.org/pdf/1901.10995.pdf"

  # E. ARC-AGI canon
  "16_MeasureOfIntelligence_Chollet_2019.pdf|https://arxiv.org/pdf/1911.01547.pdf"
  "17_TTT_Akyurek_2024.pdf|https://arxiv.org/pdf/2411.07279.pdf"
  "18_ARC-AGI-3_TechnicalReport_2026.pdf|https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf"
  "19_ARCPrize2024_TechnicalReport.pdf|https://arxiv.org/pdf/2412.04604.pdf"
  "20_ARCPrize2025_TechnicalReport.pdf|https://arxiv.org/pdf/2601.10904.pdf"
)

ok=0
skip=0
fail=0

for entry in "${PAPERS[@]}"; do
  fname="${entry%%|*}"
  url="${entry##*|}"
  if [[ -s "$fname" ]]; then
    echo "[skip] $fname (already present)"
    skip=$((skip + 1))
    continue
  fi
  echo "[get ] $fname  <-  $url"
  if curl -fsSL --retry 3 --retry-delay 2 -o "$fname.tmp" "$url"; then
    mv "$fname.tmp" "$fname"
    ok=$((ok + 1))
  else
    rm -f "$fname.tmp"
    echo "[FAIL] $fname"
    fail=$((fail + 1))
  fi
done

echo ""
echo "Done. downloaded=$ok  skipped=$skip  failed=$fail"
[[ $fail -eq 0 ]]
