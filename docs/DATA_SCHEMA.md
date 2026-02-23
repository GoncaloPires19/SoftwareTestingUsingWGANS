# Dataset Schema and Validation Rules

## Overview
This document defines the schema, semantics, and validation rules for the JUnit test dataset used in WGAN-GP synthetic test generation.

## CSV Schema

### File Format
- **Filename**: `junit_data_cleaned_strict.csv`
- **Format**: CSV without headers
- **Columns**: 3 numerical features
- **Encoding**: UTF-8
- **Delimiter**: comma (`,`)

### Column Definitions

| Column Index | Name | Type | Description | Unit |
|--------------|------|------|-------------|------|
| 0 | methods | float | Number of test methods in JUnit test class | count |
| 1 | branches | float | Number of assertion statements in tests | count |
| 2 | statements | float | Lines of code in test methods | count |

### Semantic Interpretation

**What the columns represent**: Test characteristics (not production code coverage)

- **Column 0 (methods)**: Count of `@Test` annotated methods in a test class
- **Column 1 (branches)**: Count of assertion calls (`assertEquals`, `assertTrue`, `assertNotNull`, etc.) across all tests
- **Column 2 (statements)**: Total lines of executable code within test methods (excluding comments, empty lines)

**Example**:
```csv
5.0,12.0,47.0
```
Represents a test class with:
- 5 test methods (`@Test` annotations)
- 12 assertion statements
- 47 lines of test code

## Validation Rules

### Hard Constraints (MUST be satisfied)

1. **Non-negativity**: All values MUST be >= 0
   - **Rationale**: Test counts cannot be negative
   - **Enforcement**: Data cleaning script rejects negative values

2. **Outlier bounds**: All values MUST be <= 1000
   - **Rationale**: Typical test classes have 1-50 methods, 1-100 assertions, 10-300 lines
   - **Enforcement**: Values > 1000 are outliers from arithmetic test contamination, removed during cleaning

3. **Data types**: All values MUST be numeric (int or float)
   - **Rationale**: Features represent counts
   - **Enforcement**: CSV parsing validates numeric types

4. **No missing values**: No NaN, Inf, or empty cells
   - **Rationale**: Missing data corrupts normalization
   - **Enforcement**: Data cleaning script removes invalid rows

### Soft Constraints (SHOULD be satisfied)

5. **Typical ranges** (95th percentile):
   - methods: 0-100
   - branches: 0-120
   - statements: 0-100
   
   **Note**: Values outside these ranges are valid but uncommon

6. **Minimum dataset size**: >= 500 samples after cleaning
   - **Rationale**: WGAN training requires sufficient data to learn distribution
   - **Enforcement**: Pre-training validation warns if insufficient

## Data Collection Process

### Source: Java JUnit Tests

**Test Suite**: 6 test classes in `src/test/java/dataloader/`
- AdvancedValidationTests.java
- BoundaryTests.java
- ComplexScenarioTests.java
- EdgeCaseTests.java
- RobustTests.java
- IntegrationTests.java

**Logging Mechanism**: `JUnitCSVLogger.log(methods, branches, statements)`

### Original Data Issues (RESOLVED)

**Problem**: Original `junit_data.csv` contained contaminated data:
1. **Extreme outliers** (1M+ values) from arithmetic test cases logging input values instead of test characteristics
2. **Negative values** from subtraction operations being logged
3. **Semantic confusion** between test values and test metrics

**Solution**: `clean_contaminated_data.py` script filters data to [0, 1000] range, removing 8.4% of samples (61/728)

## Normalization Strategy

### Method: Min-Max to [-1, 1]

**Formula**:
```python
x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
```

**Inverse (denormalization)**:
```python
x = ((x_norm + 1) / 2) * (x_max - x_min) + x_min
```

### Normalization Parameters

**Source**: `junit_data_cleaned_norm_stats.json`

**Current values** (from cleaned dataset):
```json
{
  "min": [0.0, 0.0, 0.0],
  "max": [777.0, 1000.0, 1000.0],
  "range": [777.0, 1000.0, 1000.0]
}
```

**Why [-1, 1] range**:
- Matches LeakyReLU activation optimal range
- Symmetric around 0 for stable gradient flow
- Standard practice for WGAN-GP training

## Coverage Metric Calculation

### Purpose
Monitor how much of the test characteristic space is explored during WGAN training.

### Formula
```python
line_cov = mean(min(methods / max_methods, 1.0)) * 100%
branch_cov = mean(min(branches / max_branches, 1.0)) * 100%
method_cov = mean(min(statements / max_statements, 1.0)) * 100%
```

### Max Values for Coverage (Post-Cleanup)

**From instrumentation.py**:
```python
max_methods = 800.0      # Actual max: 777, buffer for edge cases
max_branches = 1000.0    # Actual max: 1000
max_statements = 1000.0  # Actual max: 1000
```

**Important**: These are NOT production code coverage metrics, but exploration metrics for test characteristic space.

## Dataset Statistics (After Cleaning)

### Summary
- **Total samples**: 667
- **Features**: 3 (methods, branches, statements)
- **Retention rate**: 91.62% (61 outliers/negatives removed)

### Distribution Statistics

| Feature | Min | Max | Mean | Median | Std | Q25 | Q75 | Q95 |
|---------|-----|-----|------|--------|-----|-----|-----|-----|
| methods | 0.0 | 777.0 | 23.6 | 6.0 | 62.0 | 3.0 | 15.0 | 100.0 |
| branches | 0.0 | 1000.0 | 20.2 | 2.0 | 83.2 | 1.0 | 8.0 | 95.7 |
| statements | 0.0 | 1000.0 | 16.3 | 1.0 | 64.8 | 0.0 | 9.0 | 76.4 |

### Key Observations
1. **Right-skewed distributions**: Most samples have low values (median << mean)
2. **Low medians**: Typical tests are small (6 methods, 2 assertions, 1 statement)
3. **High std**: Large variability in test complexity
4. **No extreme outliers**: Max values are reasonable (<= 1000)

## Quality Assurance

### Pre-Training Validation

**Script**: `validate_before_training.py`

**Checks**:
1. ✅ Dataset exists and has >= 500 samples
2. ✅ No negative values
3. ✅ No outliers > 1000
4. ✅ No NaN/Inf values
5. ✅ Normalization stats are reasonable (max <= 1100)

**Usage**:
```bash
python python/scripts/validate_before_training.py
```

### Data Cleaning Audit Trail

**Report**: `python/data/cleaning_report.json`

**Contents**:
- Original sample count
- Removed sample count (and reasons)
- Retention rate
- New normalization statistics
- Timestamp of cleaning operation

## Usage in Training Pipeline

### 1. Load Data
```python
import pandas as pd
df = pd.read_csv('python/data/junit_data_cleaned_strict.csv', 
                 header=None, 
                 names=['methods', 'branches', 'statements'])
```

### 2. Normalize
```python
import json
with open('python/data/junit_data_cleaned_norm_stats.json', 'r') as f:
    norm_stats = json.load(f)

data_min = torch.tensor(norm_stats['min'])
data_max = torch.tensor(norm_stats['max'])
data_normalized = 2 * (data - data_min) / (data_max - data_min) - 1
```

### 3. Train WGAN
```python
from junit_datasets import JUnitDataset
dataset = JUnitDataset(
    data_path='python/data/junit_data_cleaned_strict.csv',
    normalize=True
)
```

### 4. Generate Synthetic Tests
```python
python python/scripts/generate_synthetic.py \
  --model outputs/.../generator_final.pth \
  --num-samples 1000 \
  --denormalize \
  --norm-stats python/data/junit_data_cleaned_norm_stats.json
```

### 5. Validate Generation Quality
Synthetic samples MUST satisfy:
- All values >= 0
- All values <= 1000 (or slightly above due to generation noise)
- Distribution similar to real data (KS test, visual inspection)
- No mode collapse (diverse samples)

## Troubleshooting

### Issue: "Dataset contains outliers"
**Cause**: Using old `junit_data_cleaned.csv` instead of `junit_data_cleaned_strict.csv`
**Fix**: Update scripts to use `_strict.csv` variant

### Issue: "Coverage drops from 38% to 2%"
**Cause**: Max values in instrumentation.py are too high (1M+), making typical values appear as 0%
**Fix**: Use realistic max values (800, 1000, 1000) from cleaned dataset

### Issue: "Synthetic tests have values > 1000"
**Cause**: Generator not clipping outputs to valid range
**Fix**: Add `torch.clamp(synthetic, min=0, max=1000)` after denormalization

### Issue: "Training loss diverges"
**Cause**: Dataset still contains extreme outliers or negatives
**Fix**: Re-run `clean_contaminated_data.py` and verify with `validate_before_training.py`

## Change Log

### 2026-02-21: Data Cleaning Regularization
- Created strict cleaning script removing outliers >1000 and negatives
- Updated all training scripts to use `junit_data_cleaned_strict.csv`
- Recalibrated instrumentation.py max values from [1M, 3.1M, 2M] to [800, 1000, 1000]
- Added pre-training validation script
- Retention: 667/728 samples (91.62%)

### Future Improvements
1. **Automated data collection**: Extract from Jacoco instead of manual logging
2. **Balanced sampling**: Generate more samples in underrepresented ranges (100-500)
3. **Multi-project datasets**: Expand beyond single project for better generalization
4. **Semantic validation**: Verify methods >= branches/statements (typical relationship)

## References
- Data cleaning: `python/scripts/clean_contaminated_data.py`
- Validation: `python/scripts/validate_before_training.py`
- Dataset class: `python/junit_datasets.py`
- Training: `python/scripts/train_profile_aware.py`
