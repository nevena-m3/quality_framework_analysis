[CmdletBinding()]
param(
    [string]$FamilyFreezeFolder = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1_cleanup\MAIN outputs\02_FEATURE_REVIEWED\06_family_freezes",
    [switch]$ForceRelink,
    [switch]$SkipEnvironment
)

$ErrorActionPreference = "Stop"

$AnalysisRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$FamilyFreezeDir = Join-Path $AnalysisRoot "data\raw\family_freezes"
$VenvDir = Join-Path $AnalysisRoot ".venv"
$RequirementsFile = Join-Path $AnalysisRoot "requirements.txt"
$RegistryScript = Join-Path $AnalysisRoot "scripts\01_build_input_registries.py"
$FigureScript = Join-Path $AnalysisRoot "scripts\02_build_figure31.py"

Write-Host ""
Write-Host "Paper 1 recording-quality analysis setup"
Write-Host "Analysis root:       $AnalysisRoot"
Write-Host "Family freeze source: $FamilyFreezeFolder"
Write-Host ""

if (-not (Test-Path -LiteralPath $FamilyFreezeFolder -PathType Container)) {
    throw "Family freeze folder was not found: $FamilyFreezeFolder"
}

$ExpectedFamilies = @(
    "additive_interference",
    "gain_dynamics",
    "reverberation",
    "channel_device",
    "nonlinear_distortion",
    "temporal_discontinuity"
)

$MissingFamilies = @()
foreach ($Family in $ExpectedFamilies) {
    $ExpectedPath = Join-Path $FamilyFreezeFolder $Family
    if (-not (Test-Path -LiteralPath $ExpectedPath -PathType Container)) {
        $MissingFamilies += $Family
    }
}

if ($MissingFamilies.Count -gt 0) {
    throw "The supplied folder is missing expected family directories: $($MissingFamilies -join ', ')"
}

$RawDataDir = Split-Path $FamilyFreezeDir -Parent
New-Item -ItemType Directory -Force -Path $RawDataDir | Out-Null

$DestinationReady = $false
if (Test-Path -LiteralPath $FamilyFreezeDir -PathType Container) {
    $ExistingComplete = $true
    foreach ($Family in $ExpectedFamilies) {
        if (-not (Test-Path -LiteralPath (Join-Path $FamilyFreezeDir $Family) -PathType Container)) {
            $ExistingComplete = $false
            break
        }
    }

    if ($ExistingComplete -and -not $ForceRelink) {
        Write-Host "Family-freeze input is already available at:"
        Write-Host "  $FamilyFreezeDir"
        $DestinationReady = $true
    }
    else {
        Write-Host "Removing the incomplete or replaceable family-freeze input path..."
        Remove-Item -LiteralPath $FamilyFreezeDir -Recurse -Force
    }
}

if (-not $DestinationReady) {
    Write-Host "Creating a directory junction to the existing family freezes..."

    try {
        New-Item -ItemType Junction `
            -Path $FamilyFreezeDir `
            -Target $FamilyFreezeFolder `
            -ErrorAction Stop | Out-Null

        Write-Host "Created junction:"
        Write-Host "  $FamilyFreezeDir"
        Write-Host "  -> $FamilyFreezeFolder"
    }
    catch {
        Write-Warning "Directory junction creation failed. Falling back to a full copy."
        New-Item -ItemType Directory -Force -Path $FamilyFreezeDir | Out-Null
        Copy-Item -Path (Join-Path $FamilyFreezeFolder "*") `
            -Destination $FamilyFreezeDir `
            -Recurse `
            -Force
        Write-Host "Family freezes copied to:"
        Write-Host "  $FamilyFreezeDir"
    }
}

if (-not $SkipEnvironment) {
    if (-not (Test-Path -LiteralPath $RequirementsFile -PathType Leaf)) {
        throw "Requirements file was not found: $RequirementsFile"
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        Write-Host ""
        Write-Host "Creating the Python virtual environment..."

        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv $VenvDir

            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Python 3.12 was not available through py. Trying the default Python 3 installation."
                & py -3 -m venv $VenvDir
            }
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv $VenvDir
        }
        else {
            throw "Python was not found. Install Python 3.12, then run this script again."
        }

        if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
            throw "The virtual environment could not be created."
        }
    }
    else {
        Write-Host ""
        Write-Host "Using the existing virtual environment:"
        Write-Host "  $VenvDir"
    }

    $PythonExe = $VenvPython

    Write-Host ""
    Write-Host "Updating pip and installing the analysis packages..."
    & $PythonExe -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "pip setup failed."
    }

    & $PythonExe -m pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Package installation failed."
    }

    Write-Host ""
    Write-Host "Registering the Jupyter kernel..."
    & $PythonExe -m ipykernel install `
        --user `
        --name "paper1-recording-quality" `
        --display-name "Paper 1 - Recording Quality"

    if ($LASTEXITCODE -ne 0) {
        throw "Jupyter kernel registration failed."
    }
}
else {
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $PythonExe = $VenvPython
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = (Get-Command python).Source
    }
    else {
        throw "SkipEnvironment was requested, but no usable Python executable was found."
    }
}

if (-not (Test-Path -LiteralPath $RegistryScript -PathType Leaf)) {
    throw "Registry script was not found: $RegistryScript"
}

if (-not (Test-Path -LiteralPath $FigureScript -PathType Leaf)) {
    throw "Figure script was not found: $FigureScript"
}

Write-Host ""
Write-Host "Building the family and feature input registries..."
& $PythonExe $RegistryScript
if ($LASTEXITCODE -ne 0) {
    throw "Input-registry construction failed."
}

Write-Host ""
Write-Host "Building the Section 3.1 validation-summary figure..."
& $PythonExe $FigureScript
if ($LASTEXITCODE -ne 0) {
    throw "Figure 3.1 construction failed."
}

Write-Host ""
Write-Host "SETUP COMPLETE"
Write-Host "Jupyter kernel: Paper 1 - Recording Quality"
Write-Host "Notebook:"
Write-Host "  $AnalysisRoot\notebooks\01_validation_synthesis.ipynb"
Write-Host "Editable figure:"
Write-Host "  $AnalysisRoot\outputs\figures\main\Figure_3_1_validation_summary_draft.svg"
Write-Host ""
