[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Notebook = Join-Path $Root "notebooks\02_multidimensional_structure_FINAL.ipynb"
$Outputs = Join-Path $Root "outputs\multidimensional_structure"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Temp = Join-Path $env:TEMP "Goal2_structure_review_$Stamp"
$Zip = Join-Path $env:USERPROFILE "Downloads\Goal2_STRUCTURE_REVIEW_$Stamp.zip"

New-Item -ItemType Directory -Force -Path $Temp | Out-Null
try {
    if (-not (Test-Path -LiteralPath $Notebook)) {
        throw "Notebook not found: $Notebook"
    }
    Copy-Item -LiteralPath $Notebook -Destination $Temp -Force
    if (Test-Path -LiteralPath $Outputs) {
        Copy-Item -LiteralPath $Outputs `
            -Destination (Join-Path $Temp "multidimensional_structure_outputs") `
            -Recurse -Force
    }
    Compress-Archive -Path "$Temp\*" -DestinationPath $Zip -CompressionLevel Optimal -Force
} finally {
    if (Test-Path -LiteralPath $Temp) {
        Remove-Item -LiteralPath $Temp -Recurse -Force
    }
}
Write-Host ""
Write-Host "Review ZIP created:"
Write-Host $Zip
