@echo off
REM EvoSuite Test Generator for SoftwareTestingUsingWGANS
REM Generates automated tests using EvoSuite

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║       EvoSuite Automated Test Generator                ║
echo ║  Software Testing Using WGAN-GP                        ║
echo ╚════════════════════════════════════════════════════════╝
echo.

set EVOSUITE_JAR=C:\evosuite\evosuite.jar
set PROJECT_PATH=c:\Users\Gonçalo\Desktop\tese\SoftwareTestingUsingWGANS
set SRC_PATH=%PROJECT_PATH%\src\main\java
set TEST_PATH=%PROJECT_PATH%\src\test\java
set TARGET_PATH=%PROJECT_PATH%\target\classes

REM Check if EvoSuite is installed
if not exist %EVOSUITE_JAR% (
    echo [ERROR] EvoSuite JAR not found at %EVOSUITE_JAR%
    echo Please download from: https://github.com/EvoSuite/evosuite/releases
    exit /b 1
)

echo [1/3] Compiling project...
cd %PROJECT_PATH%
call mvn clean compile -q
if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed
    exit /b 1
)
echo [OK] Project compiled

echo.
echo [2/3] Generating tests with EvoSuite for dataloader.AdvancedTests...
java -jar %EVOSUITE_JAR% -generateSuite -class dataloader.AdvancedTests -projectCP %TARGET_PATH% -Dcriterion=branch -Dsearch_budget=60

echo.
echo [2.5/3] Generating tests for dataloader.ExampleTests...
java -jar %EVOSUITE_JAR% -generateSuite -class dataloader.ExampleTests -projectCP %TARGET_PATH% -Dcriterion=branch -Dsearch_budget=60

echo.
echo [2.75/3] Generating tests for dataloader.AdvancedValidationTests...
java -jar %EVOSUITE_JAR% -generateSuite -class dataloader.AdvancedValidationTests -projectCP %TARGET_PATH% -Dcriterion=branch -Dsearch_budget=60

echo.
echo [3/3] Running all tests (manual + EvoSuite generated)...
call mvn test

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║              ✅ EvoSuite Generation Complete!          ║
echo ╚════════════════════════════════════════════════════════╝
echo.
echo Generated tests are in: %TEST_PATH%
echo Look for files like: *_ESTest.java
echo.

endlocal
pause
