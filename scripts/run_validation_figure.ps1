$ErrorActionPreference = "Stop"
$AnalysisRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $AnalysisRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Environment not found. Run scripts\setup\setup_project.ps1 first."
}
& $PythonExe (Join-Path $AnalysisRoot "scripts\01_build_input_registries.py")
& $PythonExe (Join-Path $AnalysisRoot "scripts\03_build_figure31_publication.py")
Write-Host "Registries and Figure 3.1 were rebuilt successfully."
