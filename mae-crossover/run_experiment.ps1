# Runs the main experiment end to end in its own window, independent of any editor or Claude session.
# Safe to re-run at any time: finished MAE encoders, lr choices and runs are skipped (resumes where it stopped).
# Usage (from mae-crossover):  powershell -ExecutionPolicy Bypass -File run_experiment.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\experiment.log"
$py = ".venv\Scripts\python.exe"
$env:PYTHONWARNINGS = "ignore"

function Stage($name) {
    "`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $name =====" | Tee-Object -FilePath $log -Append
    cmd /c "$py -u run.py $name >> $log 2>&1"
    $code = $LASTEXITCODE
    "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') END $name (exit $code) =====" | Tee-Object -FilePath $log -Append
    if ($code -ne 0) { Write-Host "Stage $name failed - see $log. Re-run this script to resume."; exit $code }
}

Write-Host "Experiment running. Progress: $PSScriptRoot\$log  (keep the laptop plugged in, sleep disabled)"
Stage "pretrain"
Stage "tune"
Stage "train"
Stage "analyze"

# publish the results (per-run folders and checkpoints stay local, see .gitignore)
Set-Location ..
git add mae-crossover/results
git commit -m "Main experiment results (auto-commit by run_experiment.ps1)`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ALL DONE, results pushed =====" | Tee-Object -FilePath "mae-crossover\$log" -Append
