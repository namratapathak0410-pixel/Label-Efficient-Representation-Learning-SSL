# Experiment 2 end to end in its own window (PREREGISTRATION_V2.md): pool -> MAE seed 0 -> tune -> grid -> analysis -> push.
# Safe to re-run at any time: finished encoders, learning-rate choices and runs are skipped (resumes where it stopped).
# Progress and an overall "time left" estimate: logs\experiment.log (lines starting with "ETA |"); latest line alone: logs\STATUS.txt
# Usage (from mae-crossover):  powershell -ExecutionPolicy Bypass -File run_experiment_v2.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\experiment.log"
$py = ".venv\Scripts\python.exe"
$env:PYTHONWARNINGS = "ignore"

function Say($text) {
    Write-Host $text
    Add-Content -Path $log -Value $text -Encoding Ascii
}

function Stage($name) {
    Say "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $name (experiment 2) ====="
    cmd /c "$py -u run.py $name --config config_v2.yaml >> $log 2>&1"
    $code = $LASTEXITCODE
    Say "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') END $name (exit $code) ====="
    if ($code -ne 0) { Write-Host "Stage $name failed - see $log. Re-run this script to resume."; exit $code }
}

Write-Host "Experiment 2 running. Progress: $PSScriptRoot\$log  (keep the laptop plugged in, sleep disabled)"
Stage "v2"
Stage "analyze2"

Set-Location ..
git add mae-crossover/results
git commit -m "Experiment 2 results (auto-commit by run_experiment_v2.ps1)"
git push origin main
Set-Location $PSScriptRoot
Say "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ALL DONE (experiment 2), results pushed ====="
