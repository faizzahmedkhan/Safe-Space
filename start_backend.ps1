# SafeSpace backend launcher for Windows (PowerShell).
#   $env:SAFESPACE_API_KEY = "secret"   # require X-SafeSpace-Key / ?key= on every endpoint except GET /
#   $env:SAFESPACE_LAN = "1"            # listen on all interfaces (default: 127.0.0.1 only)
#   $env:SAFESPACE_CORS_ORIGINS = "http://localhost:3000"
#   $env:SAFESPACE_MODEL = "C:\path\to\model.pth"

Set-Location -Path $PSScriptRoot

$activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) { . $activate } else { Write-Warning "No .venv found - using the Python on PATH." }

$bindHost = "127.0.0.1"
if ($env:SAFESPACE_LAN -eq "1") { $bindHost = "0.0.0.0" }

Write-Host "=============================================="
Write-Host " SafeSpace backend on ${bindHost}:8000"
Write-Host " Open the dashboard at http://localhost:3000"
if ($bindHost -eq "0.0.0.0" -and -not $env:SAFESPACE_API_KEY) {
  Write-Warning "Listening on the LAN with NO API key. Set SAFESPACE_API_KEY first."
}
Write-Host "=============================================="

if (-not (Test-Path "cameras_config.json")) {
  Write-Host "No cameras_config.json yet - add a camera from the dashboard (it is created automatically)."
}

# No --reload: a code reload would restart the AI watcher and drop live cameras mid-incident.
uvicorn main:app --host $bindHost --port 8000
