[CmdletBinding()]
param(
    [string]$CleanupRoot = "",
    [string]$ValidationFreezeRoot = "",
    [switch]$SkipEnvironment
)

$ErrorActionPreference = "Stop"
$AnalysisRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $CleanupRoot) {
    $CleanupRoot = Join-Path (Split-Path $AnalysisRoot -Parent) "Paper_1_Final_Cleanup"
}
$CleanupRoot = (Resolve-Path -LiteralPath $CleanupRoot).Path

if (-not $ValidationFreezeRoot) {
    if ($env:PAPER1_VALIDATION_FREEZE_DIR) {
        $ValidationFreezeRoot = $env:PAPER1_VALIDATION_FREEZE_DIR
    }
    else {
        $LocalValidationFreeze = Join-Path $AnalysisRoot "data\raw\family_freezes"
        if (Test-Path -LiteralPath $LocalValidationFreeze) {
            $ValidationFreezeRoot = $LocalValidationFreeze
        }
        else {
            throw (
                "Notebook 01 requires the governed full family-validation freeze. " +
                "Pass -ValidationFreezeRoot <path> or set PAPER1_VALIDATION_FREEZE_DIR."
            )
        }
    }
}
$ValidationFreezeRoot = (Resolve-Path -LiteralPath $ValidationFreezeRoot).Path

$env:PAPER1_ANALYSIS_ROOT = $AnalysisRoot
$env:PAPER1_PIPELINE_ROOT = $CleanupRoot
$env:PAPER1_DATA_FREEZE_DIR = Join-Path $CleanupRoot "outputs\data"
$env:PAPER1_FAMILY_FREEZE_DIR = Join-Path $CleanupRoot "outputs\features"
$env:PAPER1_VALIDATION_FREEZE_DIR = $ValidationFreezeRoot
$NevenaRoot = Split-Path (Split-Path $CleanupRoot -Parent) -Parent
$env:PAPER1_HUMAN_QC_ROOT = Join-Path $NevenaRoot "Data_13072026\Bamboo_passage_HumanQC"

$Required = @(
    $env:PAPER1_DATA_FREEZE_DIR,
    $env:PAPER1_FAMILY_FREEZE_DIR,
    (Join-Path $env:PAPER1_FAMILY_FREEZE_DIR "families\01_QGAIN"),
    (Join-Path $env:PAPER1_FAMILY_FREEZE_DIR "families\02_QADD"),
    (Join-Path $env:PAPER1_FAMILY_FREEZE_DIR "families\03_QREV"),
    (Join-Path $env:PAPER1_FAMILY_FREEZE_DIR "families\04_QCHAN"),
    (Join-Path $env:PAPER1_FAMILY_FREEZE_DIR "families\05_QDIST"),
    (Join-Path $env:PAPER1_FAMILY_FREEZE_DIR "experimental\QTEMP_LEGACY_NO_RETAINED"),
    $env:PAPER1_VALIDATION_FREEZE_DIR,
    (Join-Path $env:PAPER1_VALIDATION_FREEZE_DIR "additive_interference"),
    (Join-Path $env:PAPER1_VALIDATION_FREEZE_DIR "gain_dynamics"),
    (Join-Path $env:PAPER1_VALIDATION_FREEZE_DIR "reverberation"),
    (Join-Path $env:PAPER1_VALIDATION_FREEZE_DIR "channel_device"),
    (Join-Path $env:PAPER1_VALIDATION_FREEZE_DIR "nonlinear_distortion"),
    (Join-Path $env:PAPER1_VALIDATION_FREEZE_DIR "temporal_discontinuity")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required frozen input is missing: $Path"
    }
}

$VenvPython = Join-Path $AnalysisRoot ".venv\Scripts\python.exe"
if (-not $SkipEnvironment) {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & py -3.12 -m venv (Join-Path $AnalysisRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Virtual-environment creation failed." }
    }
    & $VenvPython -m pip install -e "$AnalysisRoot[notebooks,dev]"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
elseif (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "SkipEnvironment was requested, but .venv does not exist."
}

& $VenvPython (Join-Path $AnalysisRoot "scripts\preflight.py")
if ($LASTEXITCODE -ne 0) { throw "Preflight failed." }

Write-Host "Setup complete. No patient data were copied into the analysis repository."
Write-Host "Run .\scripts\setup\open_jupyter_lab.ps1 to open the seven notebooks."
