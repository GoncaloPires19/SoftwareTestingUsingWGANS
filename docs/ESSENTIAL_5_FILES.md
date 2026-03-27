# 5 Essential Files: Understanding the Project in Depth

This guide focuses on the 5 core files that represent 95% of the project logic. Reading these files in order will give you complete understanding of how the system works.

---

## 📖 The Essential 5

### 1. **`python/config.py`** - All Configurations
**Purpose:** Centralized control of all parameters  
**Time to understand:** 15 minutes  
**Key sections:**

```python
# How to read:
1. set_random_seed() - Reproducibility setup
2. DatasetMode Enum - Choose which data to use
3. DataConfig - Where data comes from, how to split
4. ModelConfig - Generator/Critic architecture (100→256→512→3)
5. TrainingConfig - Hyperparameters (epochs=50, lr=0.00005, λ=5.0)
6. WGANConfig - Combines everything
```

**Why it matters:**  
Every training run is controlled by config.py. Change a parameter here, everything adjusts automatically. This is the "control panel" of the system.

**Critical parameters:**
- `epochs=50` - How long to train
- `batch_size=16` - Samples per batch
- `learning_rate=0.00005` - How fast to learn (critical!)
- `lambda_gp=5.0` - Gradient penalty strength
- `n_critic=7` - Train critic 7x per generator step

**Read this first because:** You need to know what ALL the knobs do.

---

### 2. **`python/model.py`** - The Neural Networks
**Purpose:** Define Generator and Critic architectures  
**Time to understand:** 15 minutes  
**Key classes:**

```python
# Generator (learns to create synthetic tests)
class Generator(nn.Module):
    100 (random noise) 
    → Linear(256) + LeakyReLU
    → Linear(512) + LeakyReLU  
    → Linear(3) = [methods, branches, statements]

# Critic (learns to detect fake vs real)
class Critic(nn.Module):
    3 (test features)
    → Linear(512) + LeakyReLU
    → Linear(256) + LeakyReLU
    → Linear(1) = "realness score"
```

**Why it matters:**  
These two networks play a game. Generator creates fakes, Critic learns to detect them. The competition makes both better.

**Key insight:**  
- Generator: Maps randomness → synthetic test features
- Critic: Scores features from 0 (fake) to ∞ (real)
- No sigmoid! (This is WGAN, not traditional GAN)

**Read this second because:** You need to visualize the architecture before training.

---

### 3. **`python/junit_datasets.py`** - Data Pipeline
**Purpose:** Load, normalize, and batch test data  
**Time to understand:** 20 minutes  
**Key sections:**

```python
# Flow:
1. load_csv() - Read [methods, branches, statements]
2. load_normalization_stats() - Get min/max from JSON
3. normalize() - Map [0-777] → [-1, 1] for neural network
4. JUnitDataset - PyTorch Dataset wrapper
5. DataLoader - Batches of 16 samples

# Critical: Denormalization
synthetic_denormalized = (synthetic_normalized + 1) / 2 * (max - min) + min
Result: Synthetic data matches original scale
```

**Why it matters:**  
If normalization is wrong, training fails completely. If denormalization is wrong, synthetic data is useless (wrong scale).

**Key insights:**
- Normalization range is [-1, 1] (optimal for LeakyReLU)
- Statistics are saved in `*_norm_stats.json` for reproducibility
- Denormalization uses exact same stats (critical!)

**Read this third because:** You need to understand data flow before training.

---

### 4. **`python/scripts/train_basic_profiles.py`** - Main Training Loop
**Purpose:** Execute WGAN-GP training for 50 epochs  
**Time to understand:** 25 minutes  
**Key sections:**

```python
# Training loop structure:
for epoch in range(50):
    for batch in dataloader:
        # Train Critic 7 times
        for _ in range(7):
            real_data = batch
            fake_data = generator(noise)
            real_score = critic(real_data)
            fake_score = critic(fake_data)
            
            # Wasserstein loss
            loss = fake_score.mean() - real_score.mean()
            
            # Gradient Penalty (CRITICAL for WGAN-GP)
            gp = compute_gradient_penalty(critic, real, fake)
            loss += 5.0 * gp  # λ=5.0
            
            loss.backward()
            optimizer.step()
        
        # Train Generator 1 time
        fake_data = generator(noise)
        fake_score = critic(fake_data)
        gen_loss = -fake_score.mean()  # Negative: fool the critic
        gen_loss.backward()
        optimizer.step()

# Save best model (by robust_v2 metric)
# Generate 1000 synthetic tests
```

**Why it matters:**  
This is WHERE the learning happens. Understanding this loop explains 80% of WGAN-GP.

**Key insights:**
- Critic trains 7x more than Generator (asymmetric)
- Gradient Penalty is essential (prevents training collapse)
- Best model is chosen by `robust_v2` metric, not just loss
- Auto-generates synthetic data after training

**Read this fourth because:** You need to see the actual training loop.

---

### 5. **`python/scripts/experimental_analysis.py`** - Quality Validation
**Purpose:** Measure synthetic test quality and detect overfitting  
**Time to understand:** 20 minutes  
**Key sections:**

```python
# Analysis pipeline:
1. Load real test data
2. Load generated synthetic data  
3. Load training history (50 epochs of losses)

# Analyzes:
a) Training Convergence
   - Generator loss trend
   - Critic loss trend
   - Divergence detection

b) Distribution Matching
   - Mean/std comparison (real vs synthetic)
   - KS test (statistical difference)
   - Wasserstein distance (optimal transport)

c) Coverage Metrics (THESIS SPECIFIC)
   - methods: Code coverage %
   - branches: Branch coverage %
   - statements: Line coverage %
   - Tier analysis (Very Low/Low/Medium/High/Very High)
   - Correlation between metrics

d) Overall Quality Score
   - Composite metric (0-100%)
   - Final rating (Poor/Fair/Good/Excellent)

# Outputs:
- experimental_analysis_report.json (full metrics)
- training_analysis.png (loss curves + distributions)
- hypothesis_comparison.json (H1 vs H2)
```

**Why it matters:**  
This is how we VALIDATE that the synthetic tests are any good. A model that trains doesn't mean it generates good data.

**Key insights:**
- Quality Score ≥ 70% = successful training
- Overfitting if Train Loss ≫ Test Loss
- Synthetic data must match real data ranges (denormalization check)
- Coverage correlations show if metrics are independent or coupled

**Read this fifth because:** You need to understand success criteria.

---

## 🎓 Reading Order Summary

```
1. config.py (15 min)
   └─ Learn what parameters control the system

2. model.py (15 min)
   └─ See the network architectures

3. junit_datasets.py (20 min)
   └─ Understand data normalization

4. train_basic_profiles.py (25 min)
   └─ Watch the training loop

5. experimental_analysis.py (20 min)
   └─ Measure success

Total: ~95 minutes to understand 95% of the project
```

---

## 🔗 Connected Files (Secondary Understanding)

After the Essential 5, optionally read:

| File | Purpose | Read If |
|------|---------|---------|
| `python/utils.py` | Gradient Penalty implementation | Interested in WGAN-GP theory |
| `python/metrics.py` | Quality metrics computation | Want to add new metrics |
| `python/data_validation.py` | Data integrity checks | Need to diagnose data issues |
| `generate_synthetic.py` | Synthetic generation script | Want to generate more samples |

---

## 💻 Hands-On: Testing Understanding

After reading the 5 files, try this:

```bash
# 1. Modify config.py - change epochs from 50 to 10
python/config.py → TrainingConfig(epochs=10)

# 2. Run training
python python/scripts/train_basic_profiles.py --profile balanced

# 3. Run analysis
python python/scripts/experimental_analysis.py

# 4. Check outputs
# • outputs/*/training_history.csv (10 epochs instead of 50)
# • outputs/*/synthetic_samples_best.csv (1000 samples)
# • outputs/experimental/*/experimental_analysis_report.json
```

If you can:
- ✓ Predict what metrics will be lower (with fewer epochs)
- ✓ Explain why Critic training is 7x Generator training
- ✓ Trace a sample from JSON normalization stats → synthetic generation
- ✓ Interpret the quality score

→ You understand the project!

---

## 📊 Quick Reference: Parameter Impact

| Parameter | Effect | Range | Current |
|-----------|--------|-------|---------|
| `epochs` | More training = better quality → diminishing returns | 10-100 | **50** |
| `batch_size` | Larger = faster but less updates per epoch | 8-64 | **16** |
| `learning_rate` | Higher = faster but unstable, lower = stable but slow | 1e-6 to 1e-3 | **5e-5** |
| `lambda_gp` | Higher = more stable but less diversity | 1-20 | **5** |
| `n_critic` | More critic training = better critic but slower | 3-15 | **7** |

---

## ⚡ Key Algorithms

### Wasserstein Distance (WGAN)
```
Instead of: D(real) vs D(fake) → 0 or 1 (Traditional GAN)
We compute: E[critic(real)] - E[critic(fake)] (Continuous score)

This is more stable because:
- Gradient doesn't vanish
- Training is smoother
- Critic doesn't collapse
```

### Gradient Penalty (GP)
```
Why needed: Without it, Critic just learns constants (bad)
With GP: Forces ||∇Critic(x)|| = 1 (Lipschitz constraint)

Formula: λ * ||∇x_critic - 1||²
where x is interpolation between real and fake

Effect: Critic stays well-behaved during training
```

### Denormalization Formula
```
Real data: X ∈ [min, max]
Train data: X_norm = 2*(X - min)/(max - min) - 1 ∈ [-1, 1]
Generated: Z_norm ∈ [-1, 1] (from generator output)
Result: Z = (Z_norm + 1)/2 * (max - min) + min ∈ [min, max]
```

---

## 🎯 Final Understanding Check

You understand the project if you can explain:

1. **Why 3 features?** (methods, branches, statements = Java coverage metrics)
2. **Why normalize to [-1, 1]?** (Neural network activation range)
3. **Why 100-dim noise?** (Standard WGAN practice, arbitrary but works)
4. **Why Critic trains 7x?** (Better critic = better gradients for generator)
5. **Why Generator output is 3?** (Must match feature count)
6. **Why Quality Score > 70%?** (Threshold for "good enough" synthetic data)
7. **Why save norm_stats.json?** (Needed for denormalization during generation)
8. **Why experimental_analysis.py is critical?** (Validation that synthetic data is realistic)

If you can answer these 8 questions → **You fully understand the project!**
