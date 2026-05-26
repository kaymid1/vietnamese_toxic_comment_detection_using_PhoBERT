param(
  [Parameter(Mandatory = $true)]
  [string]$Owner,

  [Parameter(Mandatory = $true)]
  [string]$Slug,

  [Parameter(Mandatory = $false)]
  [string]$Title = "VietComment Analyzer MLflow Retrain",

  [Parameter(Mandatory = $false)]
  [ValidateSet("none", "NvidiaTeslaT4", "NvidiaTeslaP100", "NvidiaTeslaA100", "NvidiaL4", "NvidiaH100")]
  [string]$Accelerator = "NvidiaTeslaT4",

  [Parameter(Mandatory = $false)]
  [switch]$Private
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceFile = Join-Path $repoRoot "kaggle\notebooks\mlflow_retrain\viettoxic_mlflow_retrain.py"
if (-not (Test-Path -LiteralPath $sourceFile)) {
  throw "Mirror notebook source not found: $sourceFile"
}

$kaggleCmd = Get-Command kaggle -ErrorAction SilentlyContinue
if ($null -eq $kaggleCmd) {
  throw "kaggle CLI is not installed or not found in PATH."
}

$tempDir = Join-Path $env:TEMP ("viettoxic-kaggle-push-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir | Out-Null

try {
  $codeFile = "viettoxic_mlflow_retrain.py"
  Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $tempDir $codeFile)

  $metadata = @{
    id               = "$Owner/$Slug"
    title            = $Title
    code_file        = $codeFile
    language         = "python"
    kernel_type      = "script"
    is_private       = if ($Private) { "true" } else { "false" }
    enable_gpu       = if ($Accelerator -eq "none") { "false" } else { "true" }
    enable_internet  = "true"
    dataset_sources  = @()
    competition_sources = @()
    kernel_sources   = @()
    model_sources    = @()
  }
  if ($Accelerator -ne "none") {
    $metadata["accelerator"] = $Accelerator
  }

  $metadataPath = Join-Path $tempDir "kernel-metadata.json"
  $metadata | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $metadataPath

  Write-Host "Publishing Kaggle kernel..."
  Write-Host "  id: $($metadata.id)"
  Write-Host "  source: $sourceFile"
  Write-Host "  temp: $tempDir"
  & kaggle kernels push -p $tempDir
}
finally {
  if (Test-Path -LiteralPath $tempDir) {
    Remove-Item -LiteralPath $tempDir -Recurse -Force
  }
}


