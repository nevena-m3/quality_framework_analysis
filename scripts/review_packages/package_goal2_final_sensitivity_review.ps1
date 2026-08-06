[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Notebook = Join-Path $Root "notebooks\01_extent_distribution_persistence_FINAL.ipynb"
$Output = Join-Path $Root "outputs\extent_distribution_persistence"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Temp = Join-Path $env:TEMP ("Goal2_Final_Sensitivity_" + [guid]::NewGuid().ToString("N"))
$Zip = Join-Path $env:USERPROFILE "Downloads\Goal2_FINAL_SENSITIVITY_REVIEW_$Stamp.zip"

New-Item -ItemType Directory -Force -Path $Temp | Out-Null

try {
    if (-not (Test-Path -LiteralPath $Notebook)) {
        throw "Notebook not found: $Notebook"
    }

    if (-not (Test-Path -LiteralPath $Output)) {
        throw "Goal 2 output folder not found: $Output"
    }

    Copy-Item -LiteralPath $Notebook -Destination $Temp -Force

    Copy-Item -LiteralPath $Output `
        -Destination (Join-Path $Temp "extent_distribution_persistence") `
        -Recurse -Force

    Compress-Archive `
        -Path (Join-Path $Temp "*") `
        -DestinationPath $Zip `
        -CompressionLevel Optimal `
        -Force
}
finally {
    if (Test-Path -LiteralPath $Temp) {
        Remove-Item -LiteralPath $Temp -Recurse -Force
    }
}

Write-Host ""
Write-Host "Final Goal 2 sensitivity review ZIP created:"
Write-Host $Zip
