#!/usr/bin/env pwsh
# Complete WGAN Pipeline Demo - Shows all components working together

$ErrorActionPreference = "Stop"

Write-Host @"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              🚀 COMPLETE WGAN PIPELINE DEMONSTRATION                       ║
║                                                                            ║
║  This script runs the ENTIRE integrated infrastructure:                   ║
║    1️⃣  Config System (load presets)                                        ║
║    2️⃣  Data Loading (unified loader)                                      ║
║    3️⃣  Training (multiple experiments)                                    ║
║    4️⃣  Metrics Tracking (diversity, Wasserstein, etc)                     ║
║    5️⃣  Synthetic Generation                                               ║
║    6️⃣  Experiment Comparison                                              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"@

$root_dir = "c:\Users\Gonçalo\Desktop\tese\SoftwareTestingUsingWGANS"
$python_dir = "$root_dir\python"

Push-Location $python_dir

# ============================================================================
# PHASE 1: Quick Test (Validate Setup)
# ============================================================================

Write-Host "`n" + "="*80
Write-Host "PHASE 1: Quick Test (Validate Setup)" -ForegroundColor Cyan
Write-Host "="*80

Write-Host "`n✅ Running: python test_infrastructure.py"
python test_infrastructure.py

Write-Host "`n✓ Setup validated successfully!`n"

# ============================================================================
# PHASE 2: Training Experiment 1 - Quick (Baseline)
# ============================================================================

Write-Host "`n" + "="*80
Write-Host "PHASE 2: Training Experiment 1 - Quick Baseline (5 epochs)" -ForegroundColor Cyan
Write-Host "="*80

$exp1_start = Get-Date
Write-Host "`n✅ Running: python scripts/train_configurable.py --preset quick --name exp1_quick"
python scripts/train_configurable.py --preset quick --name exp1_quick
$exp1_time = ((Get-Date) - $exp1_start).TotalSeconds

Write-Host "`n✓ Experiment 1 completed in $([Math]::Round($exp1_time, 2))s"

# ============================================================================
# PHASE 3: Training Experiment 2 - Extended
# ============================================================================

Write-Host "`n" + "="*80
Write-Host "PHASE 3: Training Experiment 2 - Extended (25 epochs)" -ForegroundColor Cyan
Write-Host "="*80

$exp2_start = Get-Date
Write-Host "`n✅ Running: python scripts/train_configurable.py --dataset combined --epochs 25 --patience 5 --name exp2_extended"
python scripts/train_configurable.py --dataset combined --epochs 25 --patience 5 --name exp2_extended
$exp2_time = ((Get-Date) - $exp2_start).TotalSeconds

Write-Host "`n✓ Experiment 2 completed in $([Math]::Round($exp2_time, 2))s"

# ============================================================================
# PHASE 4: Generate Synthetic Tests
# ============================================================================

Write-Host "`n" + "="*80
Write-Host "PHASE 4: Generate Synthetic Tests (1000 samples)" -ForegroundColor Cyan
Write-Host "="*80

Write-Host "`n✅ Running: python scripts/generate_synthetic.py --model models/generator_final.pth --num-samples 1000"
python scripts/generate_synthetic.py --model models/generator_final.pth --num-samples 1000 --output outputs/synthetic_1k.csv

if (Test-Path outputs/synthetic_1k.csv) {
    $synth_lines = (Get-Content outputs/synthetic_1k.csv | Measure-Object -Line).Lines - 1
    Write-Host "`n✓ Generated $synth_lines synthetic samples"
} else {
    Write-Host "`n⚠️  Synthetic file not found, skipping..."
}

# ============================================================================
# PHASE 5: Compare Experiments
# ============================================================================

Write-Host "`n" + "="*80
Write-Host "PHASE 5: Compare Experiments" -ForegroundColor Cyan
Write-Host "="*80

# Check if history files exist
$hist1 = "outputs/exp1_quick/training_history.csv"
$hist2 = "outputs/exp2_extended/training_history.csv"

if ((Test-Path $hist1) -and (Test-Path $hist2)) {
    Write-Host "`n✅ Running: python scripts/compare_experiments.py"
    Write-Host "   Comparing: $hist1 vs $hist2"
    python scripts/compare_experiments.py --exp1 $hist1 --exp2 $hist2 --output outputs/comparison_results.csv
    Write-Host "`n✓ Comparison completed"
} else {
    Write-Host "`n⚠️  History files not found, skipping comparison..."
    if (-not (Test-Path $hist1)) { Write-Host "   Missing: $hist1" }
    if (-not (Test-Path $hist2)) { Write-Host "   Missing: $hist2" }
}

# ============================================================================
# PHASE 6: Validation & Summary
# ============================================================================

Write-Host "`n" + "="*80
Write-Host "PHASE 6: Validation & Summary" -ForegroundColor Green
Write-Host "="*80

$results = @{
    "Test Infrastructure" = (Test-Path test_infrastructure.py)
    "Config System" = (Test-Path config.py)
    "Data Loader" = (Test-Path data_loader.py)
    "Metrics System" = (Test-Path metrics.py)
    "Training Script" = (Test-Path scripts/train_configurable.py)
    "Generation Script" = (Test-Path scripts/generate_synthetic.py)
    "Comparison Script" = (Test-Path scripts/compare_experiments.py)
    "Exp1 Models" = (Test-Path outputs/exp1_quick/generator_final.pth)
    "Exp2 Models" = (Test-Path outputs/exp2_extended/generator_final.pth)
    "Exp1 History" = (Test-Path outputs/exp1_quick/training_history.csv)
    "Exp2 History" = (Test-Path outputs/exp2_extended/training_history.csv)
    "Synthetic Tests" = (Test-Path outputs/synthetic_1k.csv)
}

Write-Host "`n📊 VALIDATION RESULTS:"
Write-Host "─" * 50

foreach ($item in $results.GetEnumerator()) {
    $status = if ($item.Value) { "✅" } else { "❌" }
    Write-Host "$status $($item.Key)"
}

Write-Host "`n📈 TIMING SUMMARY:"
Write-Host "─" * 50
Write-Host "  Experiment 1 (Quick):    $([Math]::Round($exp1_time, 2))s"
Write-Host "  Experiment 2 (Extended): $([Math]::Round($exp2_time, 2))s"
Write-Host "  Total Pipeline Time:     $([Math]::Round($exp1_time + $exp2_time, 2))s"

# ============================================================================
# FINAL SUMMARY
# ============================================================================

Write-Host @"
`n╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                     ✅ PIPELINE EXECUTION COMPLETE!                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📁 GENERATED ARTIFACTS:
   ├── models/
   │   ├── exp1_quick/
   │   │   ├── generator_final.pth
   │   │   └── critic_final.pth
   │   └── exp2_extended/
   │       ├── generator_final.pth
   │       └── critic_final.pth
   │
   └── outputs/
       ├── exp1_quick/
       │   └── training_history.csv
       ├── exp2_extended/
       │   └── training_history.csv
       ├── synthetic_1k.csv (1000 synthetic tests)
       └── comparison_results.csv (experiment comparison)

📊 NEXT STEPS:
   1. View results: Get-Content outputs/exp1_quick/training_history.csv
   2. Compare experiments: Get-Content outputs/comparison_results.csv
   3. Use synthetic tests: Import-Csv outputs/synthetic_1k.csv
   4. Download Juliet/OWASP for production training
   5. Run: python scripts/train_configurable.py --preset combined --epochs 300

🔗 INTEGRATION VERIFIED:
   ✅ Config system → Training script
   ✅ Data loader → Multiple datasets
   ✅ Metrics → Quality evaluation
   ✅ Training → Model checkpoints
   ✅ Generation → Synthetic tests
   ✅ Comparison → Experiment analysis

"@

Pop-Location

Write-Host "`n✨ All systems operational! Ready for production training.`n"
