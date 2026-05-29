[CmdletBinding()]
param(
    [string[]]$ChangedFiles = @(),
    [switch]$OnlyAffected,
    [switch]$SkipFrontend,
    [switch]$SkipBackendTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonExe = "python"
if (Test-Path ".venv\\Scripts\\python.exe") {
    $PythonExe = ".venv\\Scripts\\python.exe"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Command
    )
    Write-Host "==> $Name"
    $args = @()
    if ($Command.Length -gt 1) {
        $args = $Command[1..($Command.Length - 1)]
    }
    & $Command[0] @args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($Name) with exit code $LASTEXITCODE"
    }
}

function Get-ChangedFilesFromGit {
    $files = @()
    try {
        $files = @(git diff --name-only --diff-filter=ACMRTUXB HEAD~1 HEAD 2>$null)
    } catch {
        $files = @()
    }
    if (-not $files -or $files.Count -eq 0) {
        try {
            $statusLines = @(git status --porcelain)
            $files = @($statusLines | ForEach-Object {
                if ($_.Length -ge 4) { $_.Substring(3).Trim() } else { "" }
            } | Where-Object { $_ })
        } catch {
            $files = @()
        }
    }
    return $files
}

function Test-AnyMatch {
    param(
        [string[]]$Files,
        [string[]]$Patterns
    )
    foreach ($file in $Files) {
        foreach ($pattern in $Patterns) {
            if ($file -match $pattern) {
                return $true
            }
        }
    }
    return $false
}

if (-not $ChangedFiles -or $ChangedFiles.Count -eq 0) {
    $ChangedFiles = Get-ChangedFilesFromGit
}

$backendPatterns = @(
    '^backend/',
    '^infer_crawled_local\.py$',
    '^comment_crawl\.py$'
)
$frontendPatterns = @('^comprehensive_ui/src/')
$kagglePatterns = @(
    '^kaggle/',
    '^scripts/publish_kaggle_',
    '^backend/.*mlflow',
    '^backend/app\.py$'
)
$depsPatterns = @(
    '^docker-compose\.yml$',
    '^backend/Dockerfile$',
    '^comprehensive_ui/Dockerfile$',
    '^requirements.*\.txt$',
    '^requirements-base\.txt$',
    '^requirements-ml\.txt$'
)

$hasBackend = Test-AnyMatch -Files $ChangedFiles -Patterns $backendPatterns
$hasFrontend = Test-AnyMatch -Files $ChangedFiles -Patterns $frontendPatterns
$hasKaggle = Test-AnyMatch -Files $ChangedFiles -Patterns $kagglePatterns
$hasDeps = Test-AnyMatch -Files $ChangedFiles -Patterns $depsPatterns

$runBackendTests = -not $SkipBackendTests -and ((-not $OnlyAffected) -or $hasBackend -or $hasKaggle -or $hasDeps)
$runFrontendBuild = -not $SkipFrontend -and ((-not $OnlyAffected) -or $hasFrontend -or $hasDeps)

Invoke-Step -Name "Python compile sanity" -Command @($PythonExe, "-m", "py_compile", "backend/app.py")

if ($runBackendTests) {
    Invoke-Step -Name "Backend QA pytest" -Command @(
        $PythonExe,
        "-m",
        "pytest",
        "-q",
        "tests/test_backend_smoke.py",
        "tests/test_mlflow_kaggle.py",
        "tests/test_api_contract_mlflow_store.py",
        "tests/test_frontend_i18n_encoding.py"
    )
} else {
    Write-Host "==> Skip backend pytest (not affected)"
}

if ($runFrontendBuild) {
    Push-Location "comprehensive_ui"
    try {
        Invoke-Step -Name "Frontend production build" -Command @("npm", "run", "build")
    } finally {
        Pop-Location
    }
} else {
    Write-Host "==> Skip frontend build (not affected)"
}

Write-Host "QA checks completed."
