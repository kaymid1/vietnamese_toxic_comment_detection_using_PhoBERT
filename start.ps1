[CmdletBinding()]
param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [int]$WebhookPort = 9001,
  [string]$WebhookNgrokDomain = "living-rare-ram.ngrok-free.app",
  [switch]$StartFrontendNgrok,
  [switch]$SkipWebhookSettingsSync,
  [switch]$NoLogStream
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $RepoRoot ".runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Resolve-Python {
  $candidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot "venv\Scripts\python.exe")
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) { return $candidate }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw "Python not found. Create .venv or install python."
}

function Resolve-RequiredCommand([string]$Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $cmd) { throw "$Name not found in PATH." }
  return $cmd.Source
}

function Resolve-NpmCommand {
  $npmCmd = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
  if ($npmCmd -and (Test-Path $npmCmd.Source)) {
    return $npmCmd.Source
  }

  $npm = Get-Command "npm" -ErrorAction SilentlyContinue
  if ($npm -and $npm.Source -like "*.cmd") {
    return $npm.Source
  }

  throw "npm.cmd not found in PATH. Install Node.js and ensure npm.cmd is available."
}

function Get-ProcessRowsByCommandPattern {
  param([string]$Pattern)
  try {
    return @(Get-CimInstance Win32_Process |
      Where-Object { $_.CommandLine -and $_.CommandLine -match $Pattern })
  } catch {
    Write-Warning "Unable to inspect process command lines; skipping automatic stale-service cleanup. Run this script elevated to clean services from another Windows session."
    return @()
  }
}

function Stop-ProcessRows {
  param(
    [object[]]$Rows,
    [string]$Reason
  )
  foreach ($row in $Rows) {
    try {
      Write-Host ("[INFO] Stopping stale process PID={0} ({1}) [{2}]" -f $row.ProcessId, $row.Name, $Reason)
      Stop-Process -Id $row.ProcessId -Force -ErrorAction SilentlyContinue
    } catch {
    }
  }
}

function Assert-PortsAreFree {
  param(
    [int[]]$Ports,
    [int]$TimeoutSec = 10
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  do {
    $busy = @()
    foreach ($port in $Ports) {
      $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
      if ($listeners) {
        $pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
        $busy += ("{0} (PID {1})" -f $port, ($pids -join ", "))
      }
    }
    if (-not $busy) { return }
    Start-Sleep -Milliseconds 250
  } while ((Get-Date) -lt $deadline)

  $message = (
    "Cannot start because required ports are still in use: {0}. " +
    "Only recognized VietToxic services are stopped automatically; stop the remaining process or run start.ps1 from an elevated PowerShell if it belongs to another Windows session."
  ) -f ($busy -join "; ")
  throw $message
}

function Wait-HttpReady {
  param(
    [string]$Name,
    [string]$Url,
    [int]$TimeoutSec = 90,
    [scriptblock]$OnPoll
  )
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    if ($OnPoll) {
      & $OnPoll
    }
    try {
      $resp = Invoke-WebRequest -Uri $Url -Method Get -UseBasicParsing -TimeoutSec 5
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
        Write-Host ("[INFO] {0} online at {1} (HTTP {2})" -f $Name, $Url, $resp.StatusCode)
        return
      }
    } catch {
    }
    Start-Sleep -Seconds 2
  }
  throw ("Timed out waiting for {0} at {1}" -f $Name, $Url)
}

function Start-ServiceProcess {
  param(
    [string]$Name,
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$WorkingDirectory,
    [string]$OutLogFile,
    [string]$ErrLogFile
  )

  $proc = Start-Process `
    -FilePath $FilePath `
    -ArgumentList $Arguments `
    -WorkingDirectory $WorkingDirectory `
    -RedirectStandardOutput $OutLogFile `
    -RedirectStandardError $ErrLogFile `
    -WindowStyle Hidden `
    -PassThru

  $proc | Add-Member -NotePropertyName ServiceName -NotePropertyValue $Name -Force
  Write-Host ("[INFO] Started {0} (PID={1})" -f $Name, $proc.Id)
  return $proc
}

function Start-LogTailJobs {
  param([hashtable]$LogFiles)
  $jobs = @()
  foreach ($entry in $LogFiles.GetEnumerator()) {
    $logName = $entry.Key
    $logPath = $entry.Value
    if (-not (Test-Path $logPath)) {
      New-Item -ItemType File -Path $logPath -Force | Out-Null
    }
    $jobs += Start-Job -Name ("log-tail-{0}" -f $logName) -ScriptBlock {
      param($path, $name)
      Get-Content -Path $path -Tail 0 -Wait | ForEach-Object {
        "[{0}] {1}" -f $name, $_
      }
    } -ArgumentList $logPath, $logName
  }
  return $jobs
}

function Sync-LocalWebhookSettings {
  param(
    [string]$PythonExe,
    [string]$RepoRoot,
    [int]$WebhookPort,
    [string]$WebhookPublicBaseUrl
  )

  $triggerUrl = "http://127.0.0.1:$WebhookPort/kaggle/trigger"
  $statusUrl = "http://127.0.0.1:$WebhookPort/kaggle/status"
  $code = @"
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root))

from backend.system_settings import DEFAULT_SETTINGS_DB_PATH, update_system_settings

DEFAULT_SETTINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
update_system_settings(
    DEFAULT_SETTINGS_DB_PATH,
    {
        "KAGGLE_WEBHOOK_URL": sys.argv[2],
        "KAGGLE_STATUS_WEBHOOK_URL": sys.argv[3],
        "KAGGLE_BUNDLE_PUBLIC_BASE_URL": sys.argv[4],
    },
    updated_by="start.ps1",
)
"@

  $code | & $PythonExe - $RepoRoot $triggerUrl $statusUrl $WebhookPublicBaseUrl
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to sync Kaggle webhook settings."
  }
  Write-Host ("[INFO] Synced Kaggle webhook settings: {0}; bundle public base: {1}" -f $triggerUrl, $WebhookPublicBaseUrl)
}

function Flush-LogJobs {
  param([object[]]$Jobs)
  foreach ($job in $Jobs) {
    $lines = Receive-Job -Job $job -Keep -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
      Write-Host $line
    }
  }
}

$PythonExe = Resolve-Python
$NpmExe = Resolve-NpmCommand
$NgrokExe = Resolve-RequiredCommand "ngrok"

$UiDir = Join-Path $RepoRoot "comprehensive_ui"
if ($WebhookNgrokDomain -match "^https?://") {
  $WebhookNgrokUrl = $WebhookNgrokDomain
} else {
  $WebhookNgrokUrl = "https://$WebhookNgrokDomain"
}

if (-not $SkipWebhookSettingsSync) {
  Sync-LocalWebhookSettings -PythonExe $PythonExe -RepoRoot $RepoRoot -WebhookPort $WebhookPort -WebhookPublicBaseUrl $WebhookNgrokUrl
} else {
  Write-Host "[INFO] Skipping Kaggle webhook settings sync."
}

# Clean up stale VietToxic service parents from earlier runs. Uvicorn --reload
# creates a parent/worker pair that can both appear as listeners, so stop the
# recognized parent command rather than killing an arbitrary process by port.
Stop-ProcessRows -Rows (Get-ProcessRowsByCommandPattern "uvicorn\s+backend\.app:app") -Reason "old backend"
Stop-ProcessRows -Rows (Get-ProcessRowsByCommandPattern "uvicorn\s+backend\.kaggle_webhook_receiver:app") -Reason "old webhook-receiver"
Stop-ProcessRows -Rows (Get-ProcessRowsByCommandPattern "vite(?:\.js)?(?:\s|.*\s)--.*\bport\s+$FrontendPort\b") -Reason "old frontend"
Stop-ProcessRows -Rows (Get-ProcessRowsByCommandPattern "ngrok\s+http.*\b$WebhookPort\b") -Reason "old ngrok-webhook"
if ($StartFrontendNgrok) {
  Stop-ProcessRows -Rows (Get-ProcessRowsByCommandPattern "ngrok\s+http.*\b$FrontendPort\b") -Reason "old ngrok-frontend"
}
Assert-PortsAreFree -Ports @($BackendPort, $FrontendPort, $WebhookPort)

if (-not (Test-Path (Join-Path $UiDir "node_modules"))) {
  Write-Host "[INFO] Installing frontend dependencies..."
  & $NpmExe install --prefix $UiDir
}

$processes = @()
$logJobs = @()

try {
  $logFiles = [ordered]@{
    "backend.out"          = (Join-Path $RuntimeDir "backend.out.log")
    "backend.err"          = (Join-Path $RuntimeDir "backend.err.log")
    "webhook-receiver.out" = (Join-Path $RuntimeDir "webhook-receiver.out.log")
    "webhook-receiver.err" = (Join-Path $RuntimeDir "webhook-receiver.err.log")
    "frontend.out"         = (Join-Path $RuntimeDir "frontend.out.log")
    "frontend.err"         = (Join-Path $RuntimeDir "frontend.err.log")
    "ngrok-webhook.out"    = (Join-Path $RuntimeDir "ngrok-webhook.out.log")
    "ngrok-webhook.err"    = (Join-Path $RuntimeDir "ngrok-webhook.err.log")
  }
  if ($StartFrontendNgrok) {
    $logFiles["ngrok-frontend.out"] = (Join-Path $RuntimeDir "ngrok-frontend.out.log")
    $logFiles["ngrok-frontend.err"] = (Join-Path $RuntimeDir "ngrok-frontend.err.log")
  }

  $processes += Start-ServiceProcess `
    -Name "backend" `
    -FilePath $PythonExe `
    -Arguments @("-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "$BackendPort", "--reload") `
    -WorkingDirectory $RepoRoot `
    -OutLogFile (Join-Path $RuntimeDir "backend.out.log") `
    -ErrLogFile (Join-Path $RuntimeDir "backend.err.log")

  $processes += Start-ServiceProcess `
    -Name "webhook-receiver" `
    -FilePath $PythonExe `
    -Arguments @("-m", "uvicorn", "backend.kaggle_webhook_receiver:app", "--host", "0.0.0.0", "--port", "$WebhookPort", "--reload") `
    -WorkingDirectory $RepoRoot `
    -OutLogFile (Join-Path $RuntimeDir "webhook-receiver.out.log") `
    -ErrLogFile (Join-Path $RuntimeDir "webhook-receiver.err.log")

  $processes += Start-ServiceProcess `
    -Name "frontend" `
    -FilePath $NpmExe `
    -Arguments @("run", "dev", "--", "--host", "0.0.0.0", "--port", "$FrontendPort") `
    -WorkingDirectory $UiDir `
    -OutLogFile (Join-Path $RuntimeDir "frontend.out.log") `
    -ErrLogFile (Join-Path $RuntimeDir "frontend.err.log")

  $processes += Start-ServiceProcess `
    -Name "ngrok-webhook" `
    -FilePath $NgrokExe `
    -Arguments @("http", "--url=$WebhookNgrokUrl", "--pooling-enabled=true", "$WebhookPort") `
    -WorkingDirectory $RepoRoot `
    -OutLogFile (Join-Path $RuntimeDir "ngrok-webhook.out.log") `
    -ErrLogFile (Join-Path $RuntimeDir "ngrok-webhook.err.log")

  if ($StartFrontendNgrok) {
    $processes += Start-ServiceProcess `
      -Name "ngrok-frontend" `
      -FilePath $NgrokExe `
      -Arguments @("http", "$FrontendPort") `
      -WorkingDirectory $RepoRoot `
      -OutLogFile (Join-Path $RuntimeDir "ngrok-frontend.out.log") `
      -ErrLogFile (Join-Path $RuntimeDir "ngrok-frontend.err.log")
  }

  if (-not $NoLogStream) {
    $logJobs = Start-LogTailJobs -LogFiles $logFiles
    Write-Host "[INFO] Live log streaming enabled. Use -NoLogStream to disable."
  }

  Wait-HttpReady -Name "backend" -Url "http://127.0.0.1:$BackendPort/health" -OnPoll { Flush-LogJobs -Jobs $logJobs }
  Wait-HttpReady -Name "webhook-receiver" -Url "http://127.0.0.1:$WebhookPort/health" -OnPoll { Flush-LogJobs -Jobs $logJobs }
  Wait-HttpReady -Name "frontend" -Url "http://127.0.0.1:$FrontendPort" -OnPoll { Flush-LogJobs -Jobs $logJobs }

  Write-Host "[INFO] Services started."
  Write-Host "[INFO] Backend:   http://127.0.0.1:$BackendPort"
  Write-Host "[INFO] Frontend:  http://127.0.0.1:$FrontendPort"
  Write-Host "[INFO] Webhook:   http://127.0.0.1:$WebhookPort"
  Write-Host "[INFO] Ngrok URL: $WebhookNgrokUrl"
  Write-Host "[INFO] Logs dir:  $RuntimeDir"
  Write-Host "[INFO] Press Ctrl+C to stop."

  while ($true) {
    Flush-LogJobs -Jobs $logJobs
    Start-Sleep -Seconds 2
    $alive = @()
    foreach ($p in $processes) {
      $p.Refresh()
      if ($p.HasExited) {
        $serviceName = if ($p.PSObject.Properties["ServiceName"]) { $p.ServiceName } else { "unknown" }
        $exitCode = $null
        try { $exitCode = $p.ExitCode } catch { }
        if ($serviceName -like "ngrok*") {
          Write-Warning ("Optional process '{0}' exited: PID={1}, ExitCode={2}. Backend/frontend will keep running." -f $serviceName, $p.Id, $(if ($null -ne $exitCode) { $exitCode } else { "unknown" }))
          continue
        }
        throw ("Process exited unexpectedly: service={0}, PID={1}, ExitCode={2}" -f $serviceName, $p.Id, $(if ($null -ne $exitCode) { $exitCode } else { "unknown" }))
      }
      $alive += $p
    }
    $processes = $alive
  }
}
finally {
  Write-Host "[INFO] Stopping services..."
  foreach ($job in $logJobs) {
    try {
      Stop-Job -Job $job -ErrorAction SilentlyContinue
      Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    } catch {
    }
  }
  foreach ($p in $processes) {
    try {
      $p.Refresh()
      if (-not $p.HasExited) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
      }
    } catch {
    }
  }
}
