$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================="
Write-Host "   STARTING OPENWORLD TEST PIPELINE      "
Write-Host "========================================="

# 1. Run Data Preparation Script (Test Mode)
Write-Host "`n[1/3] Extracting OpenWorld Data (Small sample for testing)..."
$env:PYTHONIOENCODING="utf8"
python .\prepare_openworld.py --test_mode

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error occurred during prepare_openworld.py execution!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Clear old weights to avoid Tensor architecture dimension error
Write-Host "`n[2/3] Deleting old model_weights.weights.h5 (if any)..."
if (Test-Path "model_weights.weights.h5") {
    Remove-Item "model_weights.weights.h5"
}

# 3. Swap config.json (Backup original)
Write-Host "`n[3/3] Starting Training with run_model.py..."
if (Test-Path "config_sample.json") {
    if (Test-Path "config.json") {
        if (Test-Path "config_backup_pipeline.json") {
            Remove-Item "config_backup_pipeline.json" -Force
        }
        Rename-Item -Path "config.json" -NewName "config_backup_pipeline.json" -Force
    }
    Copy-Item "config_sample.json" "config.json"
}

try {
    # Run model
    python .\run_model.py
}
finally {
    # Always restore original config.json regardless of success or failure
    Write-Host "`n[Complete] Restoring original config.json..."
    if (Test-Path "config_backup_pipeline.json") {
        Remove-Item "config.json"
        Rename-Item -Path "config_backup_pipeline.json" -NewName "config.json" -Force
    }
}

Write-Host "========================================="
Write-Host "        TEST PIPELINE COMPLETED!         "
Write-Host "========================================="
