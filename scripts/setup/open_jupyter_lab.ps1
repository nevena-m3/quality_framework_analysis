[CmdletBinding()]
param(
    [string]$Notebook = "notebooks\01_run_final_analysis.ipynb",
    [switch]$OpenProjectRoot
)

$ErrorActionPreference = "Stop"

$AnalysisRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = Join-Path $AnalysisRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "The analysis environment was not found at: $PythonExe. Create it and install -e '.[notebooks,dev]' first."
}

if ($OpenProjectRoot) {
    Write-Host "Opening JupyterLab at the analysis project root..."
    & $PythonExe -m jupyter lab --notebook-dir="$AnalysisRoot"
}
else {
    $NotebookPath = Join-Path $AnalysisRoot $Notebook

    if (-not (Test-Path -LiteralPath $NotebookPath -PathType Leaf)) {
        throw "Notebook was not found: $NotebookPath"
    }

    Write-Host "Opening JupyterLab with:"
    Write-Host "  $NotebookPath"
    & $PythonExe -m jupyter lab "$NotebookPath" --notebook-dir="$AnalysisRoot"
}
