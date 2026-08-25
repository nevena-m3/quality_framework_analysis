[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Notebook = Join-Path $Root "notebooks\04_human_reference_reliability_stage1B.ipynb"
$Output = Join-Path $Root "outputs\goal3_human_correspondence\stage1b_human_reference_reliability"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Temp = Join-Path $env:TEMP ("Goal3_HumanQC_Stage1B_" + [guid]::NewGuid().ToString("N"))
$Zip = Join-Path $env:USERPROFILE "Downloads\Goal3_HUMANQC_STAGE1B_REVIEW_$Stamp.zip"

New-Item -ItemType Directory -Force -Path $Temp | Out-Null

try {
    if (-not (Test-Path -LiteralPath $Notebook)) {
        throw "Notebook not found: $Notebook"
    }

    Copy-Item -LiteralPath $Notebook -Destination $Temp -Force

    if (Test-Path -LiteralPath $Output) {
        Copy-Item -LiteralPath $Output `
            -Destination (Join-Path $Temp "stage1b_outputs") `
            -Recurse -Force
    }

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
Write-Host "Stage-1B review ZIP created:"
Write-Host $Zip
