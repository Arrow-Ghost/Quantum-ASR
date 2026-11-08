# Live Hindi <-> English speech translation from the microphone.
#   .\live.ps1                    default microphone
#   .\live.ps1 --speak            also speak the translation
#   .\live.ps1 --device 3         pick a microphone (list them with: .\live.ps1 devices)
#   .\live.ps1 --file test.wav    replay a recording through the same pipeline
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path ".\.venv\Scripts\python.exe")) { Write-Host "Run setup.ps1 first." -ForegroundColor Red; exit 1 }
if ($args.Count -gt 0 -and $args[0] -eq "devices") { & .\.venv\Scripts\python.exe -m qasr devices; exit }
& .\.venv\Scripts\python.exe -m qasr live @args
