# Downloads the 20 papers cited in ../ARC-AGI3.md.
# Skips papers already on disk so it's safe to re-run.
# Usage from PowerShell (any directory):
#   pwsh ./papers/download.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$papers = @(
    # A. Reasoning cores
    @{ name = "01_TRM_Jolicoeur-Martineau_2025.pdf";       url = "https://arxiv.org/pdf/2510.04871.pdf" }
    @{ name = "02_HRM_Wang_2025.pdf";                       url = "https://arxiv.org/pdf/2506.21734.pdf" }
    @{ name = "03_BDH_Pathway_2025.pdf";                    url = "https://arxiv.org/pdf/2509.26507.pdf" }
    @{ name = "04_UniversalTransformer_Dehghani_2018.pdf";  url = "https://arxiv.org/pdf/1807.03819.pdf" }
    @{ name = "05_ACT_Graves_2016.pdf";                     url = "https://arxiv.org/pdf/1603.08983.pdf" }

    # B. Memory architectures
    @{ name = "06_Engram_DeepSeek_2026.pdf";                url = "https://arxiv.org/pdf/2601.07372.pdf" }
    # DNC (Graves 2016, Nature) is paywalled — substitute the open-access
    # predecessor, Neural Turing Machine, which has the same memory primitives.
    @{ name = "07_NeuralTuringMachine_Graves_2014.pdf";     url = "https://arxiv.org/pdf/1410.5401.pdf" }
    @{ name = "08_TransformerXL_Dai_2019.pdf";              url = "https://arxiv.org/pdf/1901.02860.pdf" }

    # C. Model-based RL & planning
    @{ name = "09_MuZero_Schrittwieser_2020.pdf";           url = "https://arxiv.org/pdf/1911.08265.pdf" }
    @{ name = "10_DreamerV3_Hafner_2023.pdf";               url = "https://arxiv.org/pdf/2301.04104.pdf" }
    @{ name = "11_WorldModels_HaSchmidhuber_2018.pdf";      url = "https://arxiv.org/pdf/1803.10122.pdf" }
    @{ name = "12_DecisionTransformer_Chen_2021.pdf";       url = "https://arxiv.org/pdf/2106.01345.pdf" }

    # D. Exploration / sparse-reward
    @{ name = "13_RND_Burda_2018.pdf";                      url = "https://arxiv.org/pdf/1810.12894.pdf" }
    @{ name = "14_ICM_Pathak_2017.pdf";                     url = "https://arxiv.org/pdf/1705.05363.pdf" }
    @{ name = "15_GoExplore_Ecoffet_2019.pdf";              url = "https://arxiv.org/pdf/1901.10995.pdf" }

    # E. ARC-AGI canon
    @{ name = "16_MeasureOfIntelligence_Chollet_2019.pdf";  url = "https://arxiv.org/pdf/1911.01547.pdf" }
    @{ name = "17_TTT_Akyurek_2024.pdf";                    url = "https://arxiv.org/pdf/2411.07279.pdf" }
    @{ name = "18_ARC-AGI-3_TechnicalReport_2026.pdf";      url = "https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf" }
    @{ name = "19_ARCPrize2024_TechnicalReport.pdf";        url = "https://arxiv.org/pdf/2412.04604.pdf" }
    @{ name = "20_ARCPrize2025_TechnicalReport.pdf";        url = "https://arxiv.org/pdf/2601.10904.pdf" }
)

$ok = 0; $skip = 0; $fail = 0
foreach ($p in $papers) {
    if ((Test-Path $p.name) -and ((Get-Item $p.name).Length -gt 0)) {
        Write-Host "[skip] $($p.name) (already present)"
        $skip++
        continue
    }
    Write-Host "[get ] $($p.name)  <-  $($p.url)"
    try {
        Invoke-WebRequest -Uri $p.url -OutFile "$($p.name).tmp" -UseBasicParsing -MaximumRetryCount 3 -RetryIntervalSec 2
        Move-Item -Force "$($p.name).tmp" $p.name
        $ok++
    } catch {
        Remove-Item -Force "$($p.name).tmp" -ErrorAction SilentlyContinue
        Write-Host "[FAIL] $($p.name) - $($_.Exception.Message)"
        $fail++
    }
}

Write-Host ""
Write-Host "Done. downloaded=$ok  skipped=$skip  failed=$fail"
if ($fail -gt 0) { exit 1 }
