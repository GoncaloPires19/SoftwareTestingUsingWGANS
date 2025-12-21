# Software Testing Using WGAN-GP

A machine learning system for generating synthetic JUnit test data using Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP).

## 🎯 Objective

Develop an intelligent system capable of generating diverse and realistic test data for Java applications, eliminating the necessity for manual data creation.

## 📊 Project Structure

```
SoftwareTestingUsingWGANS/
├── src/                          # Java source code (Maven)
│   ├── main/java/dataloader/     # Main Java utilities
│   └── test/java/                # 66 JUnit tests (3 test classes)
│
├── python/                       # Python WGAN-GP implementation
│   ├── model.py                  # Generator & Critic networks
│   ├── utils.py                  # Gradient penalty computation
│   ├── junit_datasets.py         # Data loading & normalization
│   ├── scripts/
│   │   ├── clean_data.py         # CSV data cleaning
│   │   ├── train_simple.py       # WGAN training
│   │   └── generate_tests.py     # Synthetic test generation
│   ├── data/
│   │   ├── junit_data.csv        # Raw JUnit test data
│   │   └── junit_data_cleaned.csv# Cleaned data (202 samples)
│   └── models/                   # Trained model checkpoints (50 epochs)
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Python package config
├── pom.xml                       # Maven build config
└── Documentation
    ├── README.md                 # This file
    ├── PHASE2_REPORT.md          # Detailed results
    ├── PROJECT_ANALYSIS.md       # Project structure analysis
    └── TESTING_GUIDE.md          # Step-by-step guide
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS.git
cd SoftwareTestingUsingWGANS

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run JUnit Tests

```bash
mvn clean test
```

✅ **Result**: 66 tests pass, CSV data collected

### 3. Clean Data

```bash
python python/clean_data.py
```

✅ **Result**: `python/data/junit_data_cleaned.csv` (202 valid samples)

### 4. Train WGAN Model

```bash
python python/train_simple.py
```

✅ **Result**: 50 generator checkpoints in `python/models/`

### 5. Generate Synthetic Tests

```bash
python -c "
import torch
from python.model import Generator
import pandas as pd
import numpy as np

gen = Generator(100, 3)
gen.load_state_dict(torch.load('python/models/generator_epoch50.pth', map_location='cpu'))
gen.eval()

z = torch.randn(500, 100)
with torch.no_grad():
    samples = gen(z).numpy()

samples = (samples + 1) / 2 * 255
samples = np.round(np.clip(samples, 0, 255)).astype(int)
df = pd.DataFrame(samples)
df.to_csv('synthetic_tests.csv', index=False, header=False)
print('Generated 500 synthetic test samples')
"
```

✅ **Result**: `synthetic_tests.csv` (500 synthetic samples)

## 📈 Results

### JUnit Test Suite (66 tests)
- **ExampleTests**: 34 tests ✅
  - Math operations (7), Bitwise (7), Number theory (5), Sequences (2), Strings (1), Data (6)
- **AdvancedTests**: 18 tests ✅
  - Array ops (2), Sorting (1), Strings (3), Matrix (2), Complex math (2), Conditionals (2), Loops (2), Recursion (2), Bitwise (2)
- **AdvancedValidationTests**: 14 tests ✅
  - Exceptions (2), Collections (2), Comparisons (1), Edge cases (2), Boundaries (1), Sequences (2), Performance (1), Patterns (2), Combinatorics (1)

### WGAN-GP Model
- **Real Samples**: 202 (cleaned JUnit data)
- **Training**: 50 epochs on 202 samples
- **Generated**: 500 synthetic test samples
- **Architecture**:
  - Generator: 100-dim noise → 256 → 512 → 3-dim (59K params)
  - Critic: 3-dim → 512 → 256 → 1-dim (33K params)

### Training Metrics (Final Epoch 50)
```
Loss_D = 0.8997 (discriminator)
Loss_G = -2.1270 (generator)
Status: Training stable and converged
```

### Data Pipeline
```
Raw JUnit Data (483 samples)
        ↓ [clean_data.py]
Cleaned Data (202 valid samples)
        ↓ [train_simple.py]
Trained WGAN (50 epochs)
        ↓ [generate_tests.py]
Synthetic Tests (500 samples)
```

## 📚 Documentation

- **[PHASE2_REPORT.md](PHASE2_REPORT.md)** - Detailed Phase 2 training results and analysis
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Step-by-step testing instructions
- **[PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md)** - Project structure, issues, and recommendations

## ⚙️ Configuration

### Python Dependencies (requirements.txt)
```
torch==2.0.1
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.64.0
scikit-learn>=1.0.0
pytest>=7.0.0
```

### WGAN Hyperparameters
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Epochs**: 50
- **Critic Iterations**: 5 per generator update
- **Gradient Penalty (λ)**: 10
- **Generator Input Dim**: 100
- **Test Features Output**: 3 (numeric values)
- **Device**: CPU (CUDA compatible)

### Java/Maven
- **Java Version**: 21 (source/target: 1.8)
- **JUnit Version**: 5.10.0 (Jupiter)
- **Maven Version**: 3.9.9+

## 🔗 References

- **GitHub**: https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS
- **WGAN-GP Paper**: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- **PyTorch Documentation**: https://pytorch.org/docs/

## 📝 Status

✅ **Phase 1**: Complete - WGAN trained, 300 synthetic tests generated  
✅ **Phase 2**: Complete - 50 epochs on 502 combined samples, 500 new synthetic tests generated  
✅ **Testing**: All 66 JUnit tests passing  
✅ **Project**: Ready for thesis evaluation

---

**Author**: Gonçalo Pires  
**Created**: 2025  
**License**: Academic Project
