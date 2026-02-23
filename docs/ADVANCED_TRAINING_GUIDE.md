# 🎯 Advanced WGAN Training System - Complete Guide

**Version**: 3.0 (Redesigned for Production-Grade Research)  
**Date**: February 21, 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#architecture)
3. [Training Profiles](#profiles)
4. [Dataset Expansion](#dataset)
5. [Coverage Metrics](#coverage)
6. [Running Experiments](#experiments)
7. [Evaluation](#evaluation)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview {#overview}

This redesigned system addresses the fundamental issue that previous training was too fast and insufficiently challenging:

### Problems Solved

❌ **Before**: Dataset too small → Training converges in minutes  
✅ **After**: 10× dataset expansion → Training takes hours, realistic difficulty

❌ **Before**: Single fixed profile → No experimental flexibility  
✅ **After**: 4 specialized profiles → Different research objectives

❌ **Before**: Loss metrics only → No understanding of actual learning  
✅ **After**: Coverage metrics tracked → See structural exploration progress

---

## 🏗️ System Architecture {#architecture}

### Core Modules

```
python/
├── training_profiles_advanced.py     # 4 configurable profiles
├── dataset_expander.py               # Multi-source aggregation
├── coverage_analysis.py              # Coverage metrics extraction
├── instrumentation.py                # Epoch tracking (to create)
└── scripts/
    └── train_profile_aware.py        # Profile-based training (to create)
```

### Data Flow

```
Multiple Sources
    ↓
[Janitor] → Deduplicate
    ↓
[Augmente] → Stratify by complexity/CWE
    ↓
[Dataset] → 10× expansion
    ↓
[Loader] → Profile-selected subset
    ↓
[Trainer] → Profile-specific rewards
    ↓
[Coverage] → Track line/branch/method
    ↓
[Report] → Compare profiles
```

---

## 🧠 Training Profiles {#profiles}

Four specialized profiles for different research objectives:

### 1. Coverage-Seeking Profile
```yaml
Goal: Maximize structural exploration
Duration: ~4-6 hours
Epochs: 200
Batch Size: 8
Learning Rate: 0.00003 (very slow)
Lambda GP: 10.0 (strong penalty)
n_critic: 12

Dataset: Balanced, heavy augmentation
Rewards: Coverage metrics primary
```

**When to use**: 
- Research on structural learning
- Theoretical analysis
- Understanding what patterns WGAN finds

---

### 2. Vulnerability-Focused Profile
```yaml
Goal: Learn semantic weakness patterns
Duration: ~2-3 hours
Epochs: 150
Batch Size: 12
Learning Rate: 0.00005
Lambda GP: 7.5
n_critic: 8

Dataset: 35% Juliet, 30% OWASP, 35% others
Rewards: Coverage secondary, semantic primary
```

**When to use**:
- Vulnerability detection research
- Security analysis
- Learning weakness-adjacent code

---

### 3. Generalization-Stress Profile
```yaml
Goal: Prevent overfitting via diversity
Duration: ~6-8 hours  
Epochs: 250
Batch Size: 6 (tiny!)
Learning Rate: 0.000025
Lambda GP: 12.0 (maximum)
n_critic: 15 (intensive)

Dataset: 25% synthetic, 30% Defects4J, 45% others
Rewards: Path diversity maximized (35%)
```

**When to use**:
- Testing generalization limits
- Understanding overfitting prevention
- Meta-research on model robustness

---

### 4. Hybrid Realistic Profile
```yaml
Goal: Real-world CI/testing environment
Duration: ~1.5-2 hours
Epochs: 120
Batch Size: 16
Learning Rate: 0.00005
Lambda GP: 5.0
n_critic: 7

Dataset: Balanced, moderate augmentation
Rewards: Realistic mix (structure + semantic)
```

**When to use**:
- Production deployment
- Practical testing scenarios
- Default for most use cases

---

## 📦 Dataset Expansion {#dataset}

### Sources Integrated

| Source | Contribution | Count | Purpose |
|--------|--------------|-------|---------|
| **Juliet** | 25-35% | ~300 | CWE diversity |
| **OWASP** | 20-30% | ~200 | Web vulnerabilities |
| **Defects4J** | 15-25% | ~200 | Real-world bugs |
| **EvoSuite** | 10-15% | ~100 | Generated coverage |
| **Synthetic** | 10-30% | ~150-300 | Controlled complexity |

### Expansion Process

```
Raw Programs
    ↓ (1000 samples from all sources)
Deduplication
    ↓ (-300 duplicates, AST hashing)
    ↓ (700 unique)
Augmentation (2 copies per program)
    ↓ (+1400 augmented versions)
    ↓ (2100 total)
Stratification
    ↓ (By complexity: simple/moderate/complex/very_complex)
    ↓ (By Vulnerability: CWE distribution)
Final Dataset
    ↓ (2100+ realistic, diverse programs)
```

### Augmentation Strategies

Applied per program (2-4 copies each via):

1. **Variable Renaming** - Rename vars preserving semantics
2. **Statement Reordering** - Reorder independent statements
3. **Control Flow Flattening** - Flatten nested control structures
4. **Exception Wrapping** - Add exception handling layers
5. **Field Reordering** - Reorder class fields
6. **Method Extraction** - Extract methods (preserve behavior)

---

## 📊 Coverage Metrics {#coverage}

### Tracked Metrics Per Epoch

```
Line Coverage
├─ Average: % lines covered by generated tests
├─ Unique: Total unique lines across all tests
└─ Δ Delta: Improvement from previous epoch

Branch Coverage
├─ Average: % branches covered
├─ Unique: Total unique branches
└─ Δ Delta: Improvement

Method Coverage
├─ Average: % methods called
├─ Unique: Total unique methods
└─ Δ Delta: Improvement

Diversity Score
├─ Redundancy Ratio: % duplicate execution paths
└─ Path Diversity: 1 - redundancy_ratio
```

### Expected Trajectories

```
Coverage-Seeking (200 epochs):
  Epoch 1:   Coverage ~30% | Diversity ~0.6
  Epoch 50:  Coverage ~65% | Diversity ~0.8
  Epoch 100: Coverage ~80% | Diversity ~0.85
  Epoch 200: Coverage ~92% | Diversity ~0.90

Generalization-Stress (250 epochs):
  Slower convergence, but higher diversity maintained
```

---

## 🧪 Running Experiments {#experiments}

### Quick Start: Single Profile

```bash
cd c:\Users\Gonçalo\Desktop\tese\SoftwareTestingUsingWGANS

# View all profiles
python python/training_profiles_advanced.py

# Run Coverage-Seeking (will take ~4-6h)
python python/scripts/train_profile_aware.py \
  --profile coverage_seeking \
  --device cuda

# Run Vulnerability-Focused (~2-3h)
python python/scripts/train_profile_aware.py \
  --profile vulnerability_focused \
  --device cuda
```

### Parallel Multi-Profile Experiment

```bash
# Launch all 4 profiles in parallel (need GPU memory!)
for profile in coverage_seeking vulnerability_focused generalization_stress hybrid_realistic; do
  python python/scripts/train_profile_aware.py --profile $profile --device cuda &
done

# Wait for all to complete
wait
```

### With Custom Parameters

```bash
python python/scripts/train_profile_aware.py \
  --profile coverage_seeking \
  --epochs 300 \
  --batch-size 4 \
  --augmentation-level high \
  --dedup-strategy semantic_fingerprinting
```

---

## 📈 Evaluation & Analysis {#evaluation}

### Comparing Profiles

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results from all profiles
profiles = ['coverage_seeking', 'vulnerability_focused', 
            'generalization_stress', 'hybrid_realistic']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, profile in enumerate(profiles):
    try:
        df = pd.read_csv(f'python/models/coverage_tracking_{profile}.csv')
        
        ax = axes[idx // 2, idx % 2]
        ax.plot(df['epoch'], df['avg_line_coverage'], label='Line')
        ax.plot(df['epoch'], df['avg_branch_coverage'], label='Branch')
        ax.plot(df['epoch'], df['avg_method_coverage'], label='Method')
        ax.plot(df['epoch'], df['diversity_score'], label='Diversity', linestyle='--')
        
        ax.set_title(f'{profile.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except FileNotFoundError:
        pass

plt.tight_layout()
plt.savefig('profile_comparison.png', dpi=150)
print("✅ Comparison saved: profile_comparison.png")
```

### Coverage Report

```bash
# Generate coverage comparison
python -c "
import json
from pathlib import Path

for profile in ['coverage_seeking', 'vulnerability_focused', 'generalization_stress', 'hybrid_realistic']:
    path = Path(f'python/models/coverage_tracking_{profile}.json')
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f'{profile}:')
        print(f'  Best Line Coverage: {data[\"best_coverage\"]:.2f}%')
        print(f'  Best Diversity: {data[\"best_diversity\"]:.4f}')
"
```

---

## 📚 Examples {#examples}

### Example 1: Quick Experiment

```bash
# Run fast experiment to verify system
python python/scripts/train_profile_aware.py \
  --profile hybrid_realistic \
  --epochs 20 \
  --device cpu
```

**Expected output**:
- ~5 minutes
- Coverage tracking CSV created
- Model checkpoints saved

### Example 2: Full Research Study

```bash
# Compare all profiles
bash experiments/run_all_profiles.sh

# Wait for completion (~16 hours total)
# Then analyze:
python analyze_profile_comparison.py
```

### Example 3: Custom Profile Experiment

Create `custom_profile.py`:
```python
from training_profiles_advanced import TrainingProfileConfig, ProfileType

custom = TrainingProfileConfig(
    profile_type=ProfileType.COVERAGE_SEEKING,  # Start from coverage
    epochs=300,
    batch_size=4,
    learning_rate=0.000015,
    lambda_gp=15.0,
    n_critic=20,
    description="Ultra-conservative profile for theoretical analysis"
)

custom.save(Path("custom_profile.json"))
```

---

## 🔧 Troubleshooting {#troubleshooting}

### Problem: Training starts then stops immediately

**Cause**: Dataset loading fails  
**Solution**:
```bash
python python/dataset_expander.py  #Test data loading
python -c "from python.coverage_analysis import CoverageAnalyzer; print('Coverage OK')"
```

### Problem: GPU memory exceeded

**Cause**: Batch size too large or dataset too big  
**Solution**:
```bash
# Reduce batch size
python python/scripts/train_profile_aware.py \
  --profile coverage_seeking \
  --batch-size 2 \
  --device cuda

# Or use CPU
--device cpu
```

### Problem: Coverage metrics not improving

**Cause**: Learning rate too low or n_critic too high  
**Solution**:
```bash
# Try hybrid_realistic (more balanced)
python python/scripts/train_profile_aware.py \
  --profile hybrid_realistic

# Or adjust manually
--learning-rate 0.0001 \
--n-critic 5
```

### Problem: Training diverges (NaN loss)

**Cause**: Learning rate too high  
**Solution**:
```bash
# Use conservative profile
python python/scripts/train_profile_aware.py \
  --profile generalization_stress  # Lowest LR

# Or manually reduce
--learning-rate 0.000015
```

---

## 📊 Expected Results

### Runtime by Profile

| Profile | GPU | CPU |
|---------|-----|-----|
| Hybrid Realistic (120 epochs) | 35-45 min | 2-3 hours |
| Vulnerability-Focused (150 epochs) | 1-1.5 hours | 4-5 hours |
| Coverage-Seeking (200 epochs) | 1.5-2 hours | 6-8 hours |
| Generalization-Stress (250 epochs) | 2-2.5 hours | 8-12 hours |

### Coverage Improvement

Expected coverage growth (starting from 0%):

```
Epoch 10:  20-30% (random behavior)
Epoch 50:  50-65% (learning phase)
Epoch 100: 75-85% (convergence)
Epoch 150: 85-95% (refinement)
```

### Diversity Improvement

Expected diversity trajectory:

```
Epoch 1:   0.40 (mostly standard paths)
Epoch 30:  0.65 (diverse exploration)
Epoch 60:  0.78 (established patterns)
Epoch 100: 0.85 (stable diversity)
```

---

## 🎓 Scientific Insights

This system now enables research answers like:

### Q: How does dataset diversity affect learning?
**Answer**: Compare Coverage-Seeking (high diversity) vs Hybrid (balanced)

### Q: Can WGAN learn semantic patterns?
**Answer**: Vulnerability-Focused profile shows if CWE patterns are captured

### Q: What causes overfitting?
**Answer**: Generalization-Stress shows diversity collapse prevention

### Q: Is the model really exploring structure?
**Answer**: Coverage metrics show if exploration is real, not memorized

---

## 📝 Next Steps

1. **Implement** `train_profile_aware.py` (main training script)
2. **Implement** `instrumentation.py` (coverage tracking during training)
3. **Create** dataset generation pipeline
4. **Run** initial profile comparison
5. **Analyze** results against expected trajectories
6. **Publish** findings

---

## 📞 Quick Reference

```bash
# List profiles
python python/training_profiles_advanced.py

# Prepare large dataset
python python/dataset_expander.py

# Test coverage analysis
python python/coverage_analysis.py

# Run single profile (adjust --profile)
python python/scripts/train_profile_aware.py --profile hybrid_realistic --epochs 50

# Analyze results
python analyze_profiles.py
```

---

**Status**: ✅ Ready for Implementation  
**Next**: Build `train_profile_aware.py` and `instrumentation.py`
