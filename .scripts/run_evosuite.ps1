# EvoSuite Test Generator for PowerShell
# Generates automated tests using EvoSuite

Write-Host "`n╔════════════════════════════════════════════════════════╗"
Write-Host "║       EvoSuite Automated Test Generator                ║"
Write-Host "║  Software Testing Using WGAN-GP                        ║"
Write-Host "╚════════════════════════════════════════════════════════╝`n"

$EVOSUITE_JAR = "C:\evosuite\evosuite.jar"
$PROJECT_PATH = "c:\Users\Gonçalo\Desktop\tese\SoftwareTestingUsingWGANS"
$SRC_PATH = "$PROJECT_PATH\src\main\java"
$TEST_PATH = "$PROJECT_PATH\src\test\java"
$TARGET_PATH = "$PROJECT_PATH\target\classes"

# Check if EvoSuite is installed
if (-not (Test-Path $EVOSUITE_JAR)) {
    Write-Host "[ERROR] EvoSuite JAR not found at $EVOSUITE_JAR"
    Write-Host "Please download from: https://github.com/EvoSuite/evosuite/releases"
    exit 1
}

Write-Host "[1/3] Compiling project..."
cd $PROJECT_PATH
mvn clean compile -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed"
    exit 1
}
Write-Host "[OK] Project compiled`n"

Write-Host "[2/3] Generating tests with EvoSuite..."
Write-Host "  - Generating for dataloader.AdvancedTests..."
java -jar $EVOSUITE_JAR -generateSuite -class dataloader.AdvancedTests -projectCP $TARGET_PATH -Dcriterion=branch -Dsearch_budget=60

Write-Host "  - Generating for dataloader.ExampleTests..."
java -jar $EVOSUITE_JAR -generateSuite -class dataloader.ExampleTests -projectCP $TARGET_PATH -Dcriterion=branch -Dsearch_budget=60

Write-Host "  - Generating for dataloader.AdvancedValidationTests..."
java -jar $EVOSUITE_JAR -generateSuite -class dataloader.AdvancedValidationTests -projectCP $TARGET_PATH -Dcriterion=branch -Dsearch_budget=60

Write-Host "`n[3/3] Running all tests (manual + EvoSuite generated)..."
mvn test

Write-Host "`n╔════════════════════════════════════════════════════════╗"
Write-Host "║              ✅ EvoSuite Generation Complete!          ║"
Write-Host "╚════════════════════════════════════════════════════════╝`n"
Write-Host "Generated tests are in: $TEST_PATH"
Write-Host "Look for files like: *_ESTest.java`n"
