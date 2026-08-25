[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Notebook = Join-Path $Root "notebooks\06_human_machine_item_mapping_atlas.ipynb"
$Output = Join-Path $Root "outputs\goal3_human_correspondence\stage3_item_mapping_atlas"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Temp = Join-Path $env:TEMP ("Goal4_Item_Atlas_" + [guid]::NewGuid().ToString("N"))
$Zip = Join-Path $env:USERPROFILE "Downloads\Goal4_ITEM_MAPPING_ATLAS_REVIEW_$Stamp.zip"

New-Item -ItemType Directory -Force -Path $Temp | Out-Null

try {
    if (-not (Test-Path -LiteralPath $Notebook)) {
        throw "Notebook not found: $Notebook"
    }

    Copy-Item -LiteralPath $Notebook -Destination $Temp -Force

    if (Test-Path -LiteralPath $Output) {
        Copy-Item -LiteralPath $Output `
            -Destination (Join-Path $Temp "stage3_outputs") `
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
Write-Host "Item-mapping atlas review ZIP created:"
Write-Host $Zip
