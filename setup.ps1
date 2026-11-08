# =====================================================================
#  Quantum ASR - one-time setup (Windows, NVIDIA GPU)
#  Run from this folder:   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# =====================================================================
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Must($what) { if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what" -ForegroundColor Red; exit 1 } }

Step "1/6 Checking NVIDIA driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Write-Host "nvidia-smi not found. Install the latest NVIDIA driver (GeForce Experience / nvidia.com) first." -ForegroundColor Yellow
}

Step "2/6 Finding Python 3.10 - 3.12"
$py = $null
foreach ($v in @("3.11", "3.12", "3.10")) {
    try {
        $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { $py = $exe.Trim(); break }
    } catch {}
}
if (-not $py) {
    $cand = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($cand) {
        $ver = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if (@("3.10", "3.11", "3.12") -contains $ver) { $py = $cand }
    }
}
if (-not $py) { Write-Host "Python 3.11 not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and re-run." -ForegroundColor Red; exit 1 }
Write-Host "Using $py"

Step "3/6 Creating virtual environment .venv"
if (-not (Test-Path ".\.venv\Scripts\python.exe")) { & $py -m venv .venv; Must "venv creation" }
$vpy = (Resolve-Path ".\.venv\Scripts\python.exe").Path
& $vpy -m pip install --upgrade pip wheel setuptools; Must "pip upgrade"

Step "4/6 Installing PyTorch (CUDA 12.4) and requirements"
& $vpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124; Must "torch install"
& $vpy -m pip install -r requirements.txt; Must "requirements install"

Step "5/6 Optional extras (failures here are OK)"
& $vpy -m pip install cupy-cuda12x
if ($LASTEXITCODE -ne 0) { Write-Host "cupy not installed -> torch quantum backend will be used (slower, still works)" -ForegroundColor Yellow }
& $vpy -m pip install IndicTransToolkit
if ($LASTEXITCODE -ne 0) { Write-Host "IndicTransToolkit not installed -> built-in fallback pre-processing will be used" -ForegroundColor Yellow }

Step "6/6 Self-test (GPU + quantum kernels)"
& $vpy -m qasr selftest

Write-Host "`n--- Hugging Face login ---" -ForegroundColor Cyan
$hf = ".\.venv\Scripts\hf.exe"
& $hf auth whoami 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Paste your Hugging Face READ token when asked (https://huggingface.co/settings/tokens)."
    & $hf auth login
}
& $vpy -m qasr check-access

Write-Host "`nSetup finished. Next:  .\train.ps1 --quick   (trial run)   then   .\train.ps1   (full run)" -ForegroundColor Green
