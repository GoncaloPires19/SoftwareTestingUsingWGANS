# Copilot Instructions for Software Testing Using WGAN-GP

## Project Overview

This is a **hybrid Java/Python research project** that generates synthetic JUnit test data using Wasserstein GANs with Gradient Penalty (WGAN-GP). The system trains on real JUnit tests (193 total) to generate realistic synthetic test data for improving software testing coverage.

### Core Architecture
- **Java Layer** (Maven): 193 JUnit tests across 6 classes generating test metrics (methods, branches, statements)
- **Python Layer** (PyTorch): WGAN-GP neural networks training on test metrics to generate synthetic test data
- **Data Pipeline**: Java tests → CSV features → PyTorch Dataset → WGAN training → Synthetic test generation

## Critical Workflows

### 1. Full Pipeline Execution (Most Common)
```bash
# 1. Run all Java tests (generates junit_data.csv with test metrics)
mvn clean test

# 2. Train WGAN model (50 epochs on normalized test features)
python python/scripts/train_robust.py

# 3. Generate synthetic tests
python python/scripts/generate_synthetic.py

# 4. Analyze results (Train vs Test metrics, overfitting analysis)
python python/scripts/experimental_analysis.py
```

**Key insight**: The Java tests MUST run first to generate raw metrics. The WGAN trains on feature vectors extracted from these metrics.

### 2. Development Testing
```bash
# Quick system verification (all dependencies, data, models)
python python/test_system.py

# Run only Java tests (no synthetic generation)
mvn clean test

# Run specific Python module test
python -m pytest python/ -v
```

### 3. Training Troubleshooting
If training fails or produces poor results:
- Check `python/data/junit_data_cleaned.csv` exists and has ≥100 samples
- Verify `config.py` hyperparameters (learning_rate, batch_size, num_epochs)
- Monitor `outputs/training_history.csv` for diverging losses
- Look at `outputs/training_curves.png` for gradient stability

## Architecture Patterns & Decision Points

### Dataset Normalization Strategy
Features are normalized to [-1, 1] range **before** GAN training (see [junit_datasets.py](../python/junit_datasets.py#L43-L65)):
- **Why**: Symmetric range matches LeakyReLU activation optimal range; improves numerical stability
- **Formula**: `x_norm = 2 * (x - x_min) / (x_max - x_min) - 1`
- **Gotcha**: Must handle NaN/Inf values before normalization or training diverges

### Gradient Penalty Implementation (WGAN-GP Specific)
The Critic network requires gradient penalty (see [utils.py](../python/utils.py#L15-L50)):
- **Purpose**: Enforces Lipschitz constraint (gradient norm ≤ 1) for training stability
- **Not like traditional GANs**: Critic outputs scores, not probabilities; no sigmoid activation
- **Lambda value**: Default 10 - increasing this stabilizes training but reduces diversity

### Multi-Dataset Support Pattern
The codebase supports combining datasets via `config.py` DatasetMode enum:
- **COMBINED mode** (production): Merges Juliet + Defects4J + EvoSuite 
- **Single modes**: For experimental isolation (JULIET, OWASP, DEFECTS4J, EVOSUITE)
- **Key detail**: Each dataset loader normalizes independently, then combined data is re-normalized

## Project-Specific Conventions

### Naming & File Organization
- **Trained models**: `*_final.pth` (production) vs `*_epoch{N}.pth` (checkpoints)
- **Generated outputs**: `synthetic_*.csv` (always has 3 columns: methods, branches, statements)
- **Analysis results**: Stored in `outputs/experimental/` with JSON metadata + PNG plots
- **Python scripts organization**: `/scripts/` for training/generation, module-level files for core classes

### Data File Contracts
- **Input CSV** (junit_data.csv): No header, 3 columns (numeric features only)
- **Output CSV** (synthetic_tests.csv): Same format as input; can be used as new training data
- **Cleaned data** (junit_data_cleaned.csv): Filtered version with NaN/Inf removed
- **All datasets use same 3-feature schema**: (method_count, branch_count, statement_count)

### Model Configuration Pattern
Configuration lives in `config.py` with DataConfig & ModelConfig dataclasses:
- Always use `config.py` for hyperparameters - hardcoded values in scripts are discouraged
- Individual training scripts (train_simple.py, train_phase2.py, train_robust.py) override specific configs
- Training history + hyperparams saved together in outputs for reproducibility

## Integration Points & Dependencies

### Java ↔ Python Boundary
- Java tests write metrics → `target/surefire-reports/` (raw) → parsed into `junit_data.csv`
- Python reads `junit_data.csv`, never directly accesses surefire-reports
- **Missing file error?** Run `mvn clean test` first to regenerate metrics

### External Datasets Integration
The project integrates three external sources (configured in [config.py](../python/config.py#L29-L40)):
- **Defects4J**: Real bugs; CSV at `data/raw/defects4j/defects4j_normalized.csv`
- **OWASP Benchmark**: Java vulnerabilities; loaded via BenchmarkJava from GitHub
- **EvoSuite**: Auto-generated tests; specific version configured in pom.xml

### PyTorch Model Dependencies
- Generator/Critic architectures defined in [model.py](../python/model.py) (100 → 256 → 512 → 3 neurons)
- Expects exactly 3 output dimensions (must match test metric features)
- Uses LeakyReLU(0.2) consistently - changing this requires retraining all models

## Common Pitfalls & Solutions

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Training loss diverges/NaN | NaN in dataset after normalization | Check `junit_data_cleaned.csv` for invalid values; regenerate with `clean_data.py` |
| Low synthetic quality (< 70%) | Model underfitted; too few epochs/samples | Increase num_epochs in config; ensure ≥100 clean samples in training data |
| "Critic always wins" (loss ratio imbalanced) | Learning rates mismatched or gradient penalty too high | Reduce lambda_gp or increase critic iterations per generator iteration |
| Synthetic tests don't match real test ranges | Denormalization formula incorrect | Verify inverse normalization in generate_synthetic.py uses same min/max as training |

## Key Files Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| [pom.xml](../pom.xml) | Maven build config; EvoSuite plugin settings | Adding new Java dependencies or changing JUnit version |
| [config.py](../python/config.py) | Centralized hyperparameters for all training modes | Tuning learning rates, batch size, dataset mix |
| [model.py](../python/model.py) | Generator & Critic architectures (fixed for this project) | Never - architecture is proven; use config.py for params |
| [junit_datasets.py](../python/junit_datasets.py) | Data loading & normalization (core data pipeline) | Adding new feature types beyond the 3 base metrics |
| [train_robust.py](../python/scripts/train_robust.py) | Main training loop (50 epochs, production quality) | Monitoring hyperparams; don't change unless experimenting |
| [experimental_analysis.py](../python/scripts/experimental_analysis.py) | Validation: Train vs Test metrics, overfitting analysis | Only for research; results inform config.py tuning |

## Testing Strategy

The project uses a **two-level testing approach**:
1. **Java unit tests** (mvn test): Validate test metric extraction and MathOperations class correctness
2. **Python integration tests**: Validate data loading, model creation, and synthetic generation
3. **System-level validation**: Compare generated test distributions vs real test distributions (in outputs/experimental/)

Run the quick verification script before major changes:
```bash
python python/test_system.py
```
This validates all dependencies and data pipelines in <30 seconds.
