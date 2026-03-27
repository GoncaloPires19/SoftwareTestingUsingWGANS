# System Architecture: Software Testing Using WGAN-GP

## 🏗️ High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOFTWARE TESTING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

LAYER 1: JAVA TEST METRICS GENERATION
  ┌──────────────────────────────────────────────┐
  │  src/test/java/dataloader/ (193 Tests)     │
  │  ├─ AdvancedTests (18)                     │
  │  ├─ AdvancedValidationTests (14)           │
  │  ├─ ExampleTests (34)                      │
  │  ├─ ExtendedTests (46)                     │
  │  ├─ MathOperationsTests (45)               │
  │  └─ RobustTests (36)                       │
  └────────────────┬─────────────────────────────┘
                   │ mvn clean test
                   ↓
  ┌──────────────────────────────────────────────┐
  │  target/surefire-reports/                  │
  │  └─ TEST-*.xml                             │
  └────────────────┬─────────────────────────────┘
                   │ Parse metrics
                   ↓
  ┌──────────────────────────────────────────────┐
  │  python/data/junit_data.csv                │
  │  Format: [methods, branches, statements]   │
  │  Samples: 193 (1 per test)                 │
  └──────────────────────────────────────────────┘

LAYER 2: DATA PREPROCESSING
  ┌──────────────────────────────────────────────┐
  │  Data Cleaning & Normalization             │
  │  • Remove NaN/Inf values                   │
  │  • Normalize to [-1, 1] range              │
  │  • Compute min/max/mean/std                │
  └────────────────┬─────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
  ┌──────────────┐     ┌──────────────┐
  │ JUnit V2     │     │ SF110 EvoSuite│
  │ 399 samples  │     │ 125 samples   │
  └──────────────┘     └──────────────┘
        │                     │
        └──────────┬──────────┘
                   │ Combine (optional)
                   ↓
  ┌──────────────────────────────────────────────┐
  │  python/data/junit_data_cleaned_v2.csv     │
  │  + junit_data_cleaned_v2_norm_stats.json   │
  │  Samples: 399 (primary)                    │
  │  OR SF110+JUnit: 513 (combined)            │
  └──────────────────────────────────────────────┘

LAYER 3: WGAN-GP TRAINING
  ┌──────────────────────────────────────────────┐
  │  CONFIGURATION (config.py)                  │
  │  • DataConfig: dataset selection            │
  │  • ModelConfig: architecture params         │
  │  • TrainingConfig: hyperparameters          │
  │  • OutputConfig: logging & save paths       │
  └────────────────┬─────────────────────────────┘
                   │
  ┌────────────────↓─────────────────────────────┐
  │  DATA LOADING (junit_datasets.py)           │
  │  • Load CSV                                 │
  │  • Apply normalization [-1, 1]              │
  │  • Create DataLoader (batch_size=16)        │
  └────────────────┬─────────────────────────────┘
                   │
  ┌────────────────↓─────────────────────────────┐
  │  NEURAL NETWORKS (model.py)                 │
  │                                              │
  │  Generator:                                 │
  │    Input (z_dim=100)                        │
  │      → Linear(256) + LeakyReLU              │
  │      → Linear(512) + LeakyReLU              │
  │      → Linear(3) [output features]          │
  │                                              │
  │  Critic:                                    │
  │    Input (3 features)                       │
  │      → Linear(512) + LeakyReLU              │
  │      → Linear(256) + LeakyReLU              │
  │      → Linear(1) [score]                    │
  └────────────────┬─────────────────────────────┘
                   │
  ┌────────────────↓─────────────────────────────┐
  │  TRAINING LOOP (train_basic_profiles.py)    │
  │  For each epoch (50):                       │
  │    For each batch:                          │
  │      1. Train Critic (7 iterations)         │
  │         • Compute Wasserstein loss          │
  │         • Apply Gradient Penalty (λ=5)     │
  │      2. Train Generator (1 iteration)       │
  │         • Maximize critic fake scores       │
  │      3. Monitor metrics (coverage, etc)     │
  │    Save checkpoint + best model             │
  └────────────────┬─────────────────────────────┘
                   │
  ┌────────────────↓─────────────────────────────┐
  │  OUTPUTS:                                    │
  │  • outputs/*/models/                        │
  │    ├─ generator_best_by_robust_v2.pth       │
  │    ├─ critic_best_by_robust_v2.pth          │
  │    └─ checkpoint_epoch_*.pth (10 files)     │
  │  • outputs/*/training_history.csv           │
  │  • outputs/*/training_analysis.png          │
  └──────────────────────────────────────────────┘

LAYER 4: SYNTHETIC TEST GENERATION
  ┌──────────────────────────────────────────────┐
  │  GENERATION (generate_synthetic.py)         │
  │  1. Load trained generator model            │
  │  2. Sample random noise (N=1000, z_dim=100) │
  │  3. Generate synthetic features             │
  │  4. Denormalize to original scale           │
  │  5. Save to CSV                             │
  └────────────────┬─────────────────────────────┘
                   │
  ┌────────────────↓─────────────────────────────┐
  │  OUTPUT:                                     │
  │  • outputs/*/synthetic_samples_best.csv     │
  │  • Format: [methods, branches, statements]  │
  │  • Samples: 1000                            │
  └──────────────────────────────────────────────┘

LAYER 5: QUALITY ANALYSIS
  ┌──────────────────────────────────────────────┐
  │  EXPERIMENTAL ANALYSIS                      │
  │  (experimental_analysis.py)                 │
  │                                              │
  │  Inputs:                                    │
  │  • Real test data                           │
  │  • Synthetic test data                      │
  │  • Training history                         │
  │                                              │
  │  Analyzes:                                  │
  │  1. Distribution Matching                   │
  │     • Mean/std comparison                   │
  │     • KS test, Wasserstein distance         │
  │     • Coverage metrics (%)                  │
  │                                              │
  │  2. Overfitting Detection                   │
  │     • Train vs Test losses                  │
  │     • Convergence analysis                  │
  │                                              │
  │  3. Coverage Metrics Analysis               │
  │     • Feature statistics                    │
  │     • Coverage tier distribution            │
  │     • Correlation analysis                  │
  │     • Imbalance detection                   │
  │                                              │
  │  4. Overall Quality Score                   │
  │     • Composite metric (0-100%)             │
  │     • Rating: Poor/Fair/Good/Excellent      │
  └────────────────┬─────────────────────────────┘
                   │
  ┌────────────────↓─────────────────────────────┐
  │  OUTPUTS:                                    │
  │  • outputs/experimental/*/                  │
  │    ├─ experimental_analysis_report.json     │
  │    ├─ training_analysis.png                 │
  │    └─ distribution_plots.png                │
  │  • outputs/experimental/                    │
  │    └─ hypothesis_comparison.json (final)    │
  └──────────────────────────────────────────────┘
```

---

## 📊 Data Flow Details

### Feature Definitions (3 Coverage Metrics)

```
methods     = Code/Method Coverage (Cobertura de Código)
              How many methods in the code are executed by the test
              Range: [0-777] (from largest codebase)

branches    = Branch Coverage (Cobertura de Branch)
              How many conditional branches are taken
              Range: [0-1000] (normalized max)

statements  = Line Coverage (Cobertura de Linha)
              How many lines of code are executed
              Range: [0-1000] (normalized max)
```

### Normalization Process

```
Raw Value (from Java):
  methods ∈ [0, 777]

Normalize to [-1, 1]:
  normalized = 2 * (value - min) / (max - min) - 1

Store Statistics:
  junit_data_cleaned_v2_norm_stats.json
  {
    "features": {
      "feature_0": {"min": 0, "max": 777, "mean": 31.5, "std": 69.8},
      "feature_1": {"min": 0, "max": 1000, ...},
      "feature_2": {"min": 0, "max": 1000, ...}
    }
  }

During Generation:
  Denormalize synthetic: value = (normalized + 1) / 2 * (max - min) + min
```

---

## 🔄 Training Process (50 Epochs)

```
Epoch 1:  Initialize Generator & Critic randomly
Epoch 5:  Warmup phase - stabilize gradients
Epoch 10: Monitor first metrics
          ├─ Coverage: ~5%
          ├─ Diversity: 0.95
          └─ Convergence: Early
Epoch 25: Midway checkpoint - saving state
          ├─ Coverage: ~15%
          ├─ Loss stability: improving
          └─ Generator quality: better
Epoch 37: **BEST** checkpoint found
          ├─ Coverage: 43% (H2 model)
          ├─ Robust v2 score: 59.64
          └─ Models saved for generation
Epoch 50: Final epoch
          ├─ Coverage: 43%
          ├─ Diversity: 0.970
          ├─ Final robustness: 59.36
          └─ Generate 1000 synthetic samples

Result: outputs/combined_h2_balanced_20260315_201431/
        Quality Score: 79.2% ✅
```

---

## 📁 Core Components Summary

| Component | File | Purpose |
|-----------|------|---------|
| **Configuration** | `config.py` | Centralized hyperparameters & modes |
| **Architecture** | `model.py` | Generator (100→256→512→3) + Critic (3→512→256→1) |
| **Utilities** | `utils.py` | Gradient Penalty computation (critical for WGAN-GP) |
| **Data Loading** | `junit_datasets.py` | CSV loading, normalization, BatchLoader |
| **Training** | `train_basic_profiles.py` | Main training loop (50 epochs, 3 profiles) |
| **Generation** | `generate_synthetic.py` | Load generator, generate & denormalize 1000 samples |
| **Analysis** | `experimental_analysis.py` | Quality metrics, overfitting detection, visualizations |

---

## 🎯 Key Hyperparameters

**Training Configuration:**
```
Epochs:           50
Batch Size:       16
Learning Rate:    0.00005
Lambda (GP):      5.0
Critic Iterations: 7 per generator step
Optimizer:        Adam (β1=0.5, β2=0.999)
Device:           CPU (CUDA compatible)
```

**Model Architecture:**
```
Generator:
  Input:  100 (random noise)
  Hidden: [256, 512]
  Output: 3 (test features)
  Activation: LeakyReLU(0.2)
  Total Params: 158,979

Critic:
  Input:  3 (test features)
  Hidden: [512, 256]
  Output: 1 (authenticity score)
  Activation: LeakyReLU(0.2)
  Total Params: 133,633
```

---

## 📈 Expected Metrics

**After Training (50 epochs):**

| Metric | Value | Meaning |
|--------|-------|---------|
| Coverage | 43.06% | Synthetic samples cover 43% of real value ranges |
| P95 Coverage | 9.74% | 95th percentile of synthetic features at 9.74% |
| Diversity | 0.970 | High variety in synthetic samples (0=clone, 1=unique) |
| Quality Score | 79.2% | Overall synthetic quality rating |

**Quality Breakdown:**
- Methods:   86.9% ✅ (high code coverage match)
- Branches:  73.1% ⚠️ (moderate branch coverage match)
- Statements: 77.5% ✅ (good line coverage match)

---

## 🚀 Complete Pipeline Command

```bash
# 1. Generate metrics from Java tests
mvn clean test

# 2. Train WGAN with balanced profile
python python/scripts/train_basic_profiles.py --profile balanced --epochs 50

# 3. Generate 1000 synthetic tests
python python/scripts/generate_synthetic.py

# 4. Analyze quality
python python/scripts/experimental_analysis.py

# Optional: Compare hypotheses
# (H1 SF110-only vs H2 SF110+JUnit)
python python/scripts/experimental_analysis.py --compare-hypotheses
```

**Total Runtime:** ~5 minutes (CPU)

---

## ✅ Success Criteria

✓ Training completes without errors
✓ Synthetic samples generated (1000 CSV rows)
✓ Quality score ≥ 70%
✓ No overfitting detected (gap < 0.2)
✓ Synthetic ranges match real data ranges (denormalization correct)
✓ Reports generated with visualizations

---

## 🔗 Related Documentation

- **[QUICK_START.md](QUICK_START.md)** - How to run the pipeline
- **[DATA_SCHEMA.md](DATA_SCHEMA.md)** - Data format specifications
- **[ADVANCED_TRAINING_GUIDE.md](ADVANCED_TRAINING_GUIDE.md)** - Detailed training configurations
