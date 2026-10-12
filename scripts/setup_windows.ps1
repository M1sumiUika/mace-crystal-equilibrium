# 只在项目 .venv 中安装，不改变系统 Python/PyTorch。
[CmdletBinding()]
param(
    [switch]$Locked
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDirectory = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE : $Executable $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw "Python launcher 'py' not found. Install Python 3.12 from python.org, or create .venv manually as described in README.md."
        }
        Invoke-Checked -Executable "py" -Arguments @("-3.12", "-m", "venv", $venvDirectory)
    }
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu126")
    if ($Locked) {
        $lockFile = Join-Path $projectRoot "requirements-lock.txt"
        if (-not (Test-Path -LiteralPath $lockFile)) {
            throw "requirements-lock.txt not found; run without -Locked or obtain the recorded lock file."
        }
        Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $lockFile)
        Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "--no-deps", "-e", ".")
    } else {
        Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-e", ".[test]")
    }
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "check")
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "mace_crystal_equilibrium", "check")
    Write-Host "Project environment ready. Run the smoke command in README.md before production."
} finally {
    Pop-Location
}
