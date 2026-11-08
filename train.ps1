# One command: download data -> features -> train hybrid + classical -> evaluate -> export -> benchmark
#   .\train.ps1            full run
#   .\train.ps1 --quick    small trial run (5% data, 5 epochs)
#   .\train.ps1 --set quantum.n_qubits=8 --set train.epochs=60    (any config override)
# A full log is written to results\logs\ automatically. Re-running resumes where it stopped.
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path ".\.venv\Scripts\python.exe")) { Write-Host "Run setup.ps1 first." -ForegroundColor Red; exit 1 }
& .\.venv\Scripts\python.exe -m qasr all @args
