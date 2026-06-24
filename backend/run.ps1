# AI Media Watch — Windows launcher.
# Creates a local virtualenv, installs the core requirements, and starts the API.
# Usage:  ./run.ps1            (lite mode)
#         pip install -r requirements-ml.txt  inside .venv for full ML mode.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Installing core requirements..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host "Starting AI Media Watch engine on http://127.0.0.1:8000 ..."
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
