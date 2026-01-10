# EvoSuite + WGAN Automated Pipeline
# Generates tests automatically and trains WGAN on the data

Write-Host "╔════════════════════════════════════════════════════════╗"
Write-Host "║    EvoSuite + WGAN Automated Test Generation Pipeline  ║"
Write-Host "╚════════════════════════════════════════════════════════╝"

$ErrorActionPreference = "Stop"

# Step 1: Generate tests with EvoSuite
Write-Host "`n[Step 1/5] Generating tests with EvoSuite..."
try {
    mvn evosuite:generate -q
    Write-Host "✅ EvoSuite tests generated"
} catch {
    Write-Host "⚠️  EvoSuite not installed, skipping generation"
}

# Step 2: Compile
Write-Host "`n[Step 2/5] Compiling project..."
mvn clean compile -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Compilation successful"
} else {
    Write-Host "❌ Compilation failed"
    exit 1
}

# Step 3: Run tests (logs to CSV)
Write-Host "`n[Step 3/5] Running tests (logging to CSV)..."
mvn test -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Tests executed, data logged to CSV"
} else {
    Write-Host "⚠️  Some tests failed, but CSV was generated"
}

# Step 4: Clean CSV data
Write-Host "`n[Step 4/5] Cleaning CSV data..."
Push-Location python
python clean_data.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Data cleaned successfully"
} else {
    Write-Host "❌ Data cleaning failed"
    Pop-Location
    exit 1
}

# Step 5: Train WGAN
Write-Host "`n[Step 5/5] Training WGAN model..."
python train_simple.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ WGAN training completed"
} else {
    Write-Host "❌ WGAN training failed"
    Pop-Location
    exit 1
}

Pop-Location

Write-Host "`n╔════════════════════════════════════════════════════════╗"
Write-Host "║                    ✅ Pipeline Complete!                ║"
Write-Host "╚════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "📊 Results:"
Write-Host "   • CSV Data: python/data/junit_data_cleaned.csv"
Write-Host "   • Models: python/models/generator_epoch1.pth -> generator_epoch50.pth"
Write-Host "   • Next: Generate synthetic tests with python/generate_phase2_tests.py"
Write-Host ""
