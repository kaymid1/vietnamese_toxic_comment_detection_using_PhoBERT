param(
    [string]$Owner = "",
    [string]$Slug = "viettoxic-victsd-gold",
    [string]$Title = "VietComment Analyzer VictSD Gold",
    [string]$SourceDir = "",
    [string]$VersionMessage = "Update victsd_gold splits",
    [switch]$Public
)

$ErrorActionPreference = "Stop"

function Import-EnvFileIfExists {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }
    Get-Content -Path $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            return
        }
        $name = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if (-not $name) {
            return
        }
        $existing = [Environment]::GetEnvironmentVariable($name, "Process")
        if ([string]::IsNullOrWhiteSpace($existing) -and -not [string]::IsNullOrWhiteSpace($value)) {
            Set-Item -Path ("Env:" + $name) -Value $value
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Import-EnvFileIfExists (Join-Path $repoRoot "backend\.env.local")
Import-EnvFileIfExists (Join-Path $repoRoot ".env.local")

if (-not $SourceDir) {
    $SourceDir = Join-Path $repoRoot "data\victsd"
}

if (-not $Owner) {
    $Owner = $env:KAGGLE_USERNAME
}
if (-not $Owner) {
    $Owner = $env:KAGGLE_KERNEL_OWNER
}
if (-not $Owner) {
    throw "Missing Kaggle owner. Pass -Owner or set KAGGLE_USERNAME/KAGGLE_KERNEL_OWNER."
}

$kaggleJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleJson) -and (-not $env:KAGGLE_USERNAME -or -not $env:KAGGLE_KEY)) {
    throw "Kaggle credentials not found. Create $kaggleJson or set KAGGLE_USERNAME + KAGGLE_KEY."
}

$required = @(
    "train_augmented.jsonl",
    "validation_augmented.jsonl",
    "test_augmented.jsonl"
)
foreach ($name in $required) {
    $path = Join-Path $SourceDir $name
    if (-not (Test-Path $path)) {
        throw "Missing source file: $path"
    }
}

$stageDir = Join-Path $repoRoot ".runtime\kaggle_dataset\victsd_gold"
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

Copy-Item (Join-Path $SourceDir "train_augmented.jsonl") (Join-Path $stageDir "train.jsonl") -Force
Copy-Item (Join-Path $SourceDir "validation_augmented.jsonl") (Join-Path $stageDir "validation.jsonl") -Force
Copy-Item (Join-Path $SourceDir "test_augmented.jsonl") (Join-Path $stageDir "test.jsonl") -Force

$datasetRef = "$Owner/$Slug"
$isPrivate = if ($Public) { $false } else { $true }
$metaPath = Join-Path $stageDir "dataset-metadata.json"

$metadata = @{
    title = $Title
    id = $datasetRef
    licenses = @(@{ name = "CC0-1.0" })
    subtitle = "VictSD gold-style JSONL splits for VietComment Analyzer retraining"
    description = "Contains train.jsonl, validation.jsonl, test.jsonl prepared from local augmented splits."
    isPrivate = $isPrivate
}
$metaJson = $metadata | ConvertTo-Json -Depth 8
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($metaPath, $metaJson, $utf8NoBom)

Write-Host "Staged dataset files:"
Get-ChildItem -Path $stageDir -File | Select-Object Name,Length | Format-Table -AutoSize

$status = $null
try {
    $status = kaggle datasets status -d $datasetRef 2>$null
} catch {
    $status = $null
}

if ($LASTEXITCODE -eq 0 -and $status) {
    Write-Host "Dataset exists -> create new version"
    kaggle datasets version -p $stageDir -m $VersionMessage
} else {
    Write-Host "Dataset not found -> create new dataset"
    kaggle datasets create -p $stageDir
}

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle CLI upload failed."
}

Write-Host "Done. Dataset ref: $datasetRef"

