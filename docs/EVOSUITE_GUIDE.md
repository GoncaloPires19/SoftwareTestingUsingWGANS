# EvoSuite Integration Guide

## Overview

EvoSuite is an automated test generation tool using genetic algorithms to maximize code coverage.

## Installation & Setup

### 1. Install EvoSuite (One-time)
```powershell
# Download and install EvoSuite
# https://www.evosuite.org/documentation/installation/
```

### 2. Configuration in pom.xml
Already configured! See `<plugin>` section with:
- Criterion: branch + exception coverage
- Search Budget: 60 seconds per class
- Target Module: wgan-tests

## Usage

### Quick Start (Automated Pipeline)
```powershell
.\evosuite_wgan_pipeline.ps1
```

This runs:
1. Generate tests with EvoSuite
2. Compile Java code
3. Execute all tests (logs to CSV)
4. Clean CSV data
5. Train WGAN on generated test data

### Manual Commands

#### Generate Tests for Specific Class
```powershell
mvn evosuite:generate -DtargetClass=dataloader.AdvancedTests
mvn evosuite:generate -DtargetClass=dataloader.AdvancedValidationTests
mvn evosuite:generate -DtargetClass=dataloader.ExampleTests
```

#### Generate Tests for All Classes
```powershell
mvn evosuite:generate
```

#### Run Generated Tests
```powershell
mvn evosuite:test
```

#### Generate Coverage Report
```powershell
mvn surefire-report:report
# Open: target/site/surefire-report.html
```

## Coverage Criteria

Configured in pom.xml:
- **branch**: All decision branches covered
- **exception**: Exception handling paths covered

Other available options:
- `line`: Line coverage
- `weakmutation`: Weak mutations detection
- `strongmutation`: Strong mutations detection
- `method`: Method coverage
- `output`: Output code coverage

## Search Budget

Set in pom.xml as `searchBudget` (seconds):
- 30s: Quick generation
- 60s: Balanced (default)
- 120s: Thorough coverage

## Generated Files

Tests created in:
```
src/test/java/
  ├── dataloader/
  │   ├── AdvancedTests_ESTest.java (EvoSuite generated)
  │   ├── AdvancedValidationTests_ESTest.java
  │   ├── ExampleTests_ESTest.java
  │   └── ...
  └── utils/
      └── JUnitCSVLogger_ESTest.java
```

## CSV Data Integration

All test executions log to:
```
src/test/resources/output/junit_data.csv
```

Format (3 columns from test method results):
```
value1,value2,value3
1,2,3
5,10,15
...
```

## Pipeline: EvoSuite → WGAN

```
EvoSuite Generate Tests
    ↓
Compile (mvn compile)
    ↓
Run All Tests (mvn test)
    ↓
Log Data to CSV
    ↓
Clean Data (python/clean_data.py)
    ↓
Train WGAN (python/train_simple.py)
    ↓
Generate Synthetic Tests (python/generate_phase2_tests.py)
```

## Tips & Tricks

### 1. Speed Up Generation
```powershell
mvn evosuite:generate -DsearchBudget=30
```

### 2. Focus on Exceptions
```powershell
mvn evosuite:generate -Dcriterion=exception
```

### 3. Run Only EvoSuite Tests
```powershell
mvn evosuite:test
```

### 4. Skip EvoSuite (Use Only Manual Tests)
```powershell
mvn test -DskipEvoSuite
```

### 5. Regenerate Tests (Delete Old)
```powershell
Remove-Item src/test/java/*_ESTest.java
mvn evosuite:generate
```

## Benefits for Your Project

1. **Automatic Test Generation**: Create diverse test cases without manual effort
2. **Edge Case Discovery**: Find corner cases you might miss
3. **Increased Coverage**: Branch coverage + exception handling
4. **Better WGAN Training**: More diverse test data = better synthetic tests
5. **Quality Assurance**: Systematically test all code paths

## Troubleshooting

### "EvoSuite plugin not found"
```powershell
# Ensure Maven has internet access or use offline mode
mvn evosuite:generate -o
```

### "No tests generated"
- Check that target classes have public methods
- Increase searchBudget
- Check Maven output for errors

### "CSV not updated"
- Verify JUnitCSVLogger.java is in classpath
- Check test execution with `mvn test -X`

## Next Steps

1. Run pipeline: `.\evosuite_wgan_pipeline.ps1`
2. Check generated tests: `src/test/java/*_ESTest.java`
3. View coverage report: `target/site/surefire-report.html`
4. Check generated synthetic tests: `synthetic_tests.csv`

