# Software Testing Using WGAN-GP

A machine learning system for generating synthetic JUnit test data using Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP).

## 🎯 Objective

Develop an intelligent system capable of generating diverse and realistic test data for Java applications, eliminating the necessity for manual data creation.

## 📈 Session 3 Achievements

In this session, we achieved significant milestones in project robustness: expanded the test suite from 66 to **193 professional JUnit tests** (3 new classes: RobustTests with 8 nested categories, MathOperationsTests with 45 dedicated tests, and ExtendedTests with 46 boundary case tests), successfully trained the WGAN-GP model on all 193 tests across 50 epochs, and implemented a comprehensive experimental framework for Train vs Test Metrics analysis that validated the **model quality with only 0.164 generalization gap** (LOW overfitting risk) and **74.91% average synthetic test quality**. The final result is a robust system validated experimentally, with all changes (analysis scripts, metrics, visualizations, and CSV files) committed to GitHub and ready for thesis evaluation.

## 📊 Project Structure

```
SoftwareTestingUsingWGANS/
├── 📄 README.md                   # Main documentation
├── 📄 pom.xml                     # Maven configuration
├── 📄 requirements.txt            # Python dependencies
├── 📄 .gitignore                  # Git ignore rules
│
├── 📁 src/                        # Java source code
│   ├── main/java/dataloader/      # Java utilities
│   └── test/java/                 # 193 JUnit tests
│
├── 📁 python/                     # Python WGAN-GP implementation
│   ├── config.py                  # Configuration
│   ├── model.py                   # Generator & Critic networks
│   ├── utils.py                   # Gradient penalty
│   ├── junit_datasets.py          # Dataset loader
│   ├── scripts/                   # Training & analysis scripts
│   │   ├── train_robust.py        # Main training script
│   │   ├── generate_synthetic.py  # Generate synthetic tests
│   │   └── experimental_analysis.py # Validation analysis
│   └── data/                      # Training data
│       ├── junit_data.csv         # Raw test metrics
│       └── junit_data_cleaned.csv # Cleaned dataset
│
├── 📁 docs/                       # Documentation
│   ├── QUICK_START.md             # Quick start guide
│   ├── ADVANCED_TRAINING_GUIDE.md # Training guide
│   └── DATA_SCHEMA.md             # Data format specs
│
├── 📁 models/                     # Trained models (.pth)
├── 📁 outputs/                    # Generated results
├── 📁 data/                       # Additional datasets
│   ├── raw/                       # External datasets
│   └── processed/                 # Processed data
│
└── 📁 .github/                    # GitHub configuration
    └── copilot-instructions.md    # AI assistant instructions
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

### 2. Full Pipeline Execution

```bash
# Step 1: Run Java tests (generates test metrics)
mvn clean test

# Step 2: Train WGAN model with balanced profile
python python/scripts/train_basic_profiles.py --profile balanced

# Step 3: Generate synthetic tests
python python/scripts/generate_synthetic.py --num_samples 500

# Step 4: Analyze results (optional)
python python/scripts/experimental_analysis.py
```

**Training Profiles Available:**
- `aggressive`: Fast training (50 epochs)
- `balanced`: Recommended for production (80 epochs)
- `experimental`: Advanced configuration (80 epochs)
- `conservative`: Maximum stability (150 epochs)

✅ **Expected Results**: 
- 193/193 JUnit tests passing
- Model trained successfully
- 500 synthetic tests generated
- Validation metrics: ~75% quality, LOW overfitting

## 📈 Key Results

| Metric | Value |
|--------|-------|
| **Total JUnit Tests** | 193 |
| **Synthetic Samples Generated** | 500 |
| **Training Epochs** | 50 |
| **Overfitting Risk** | ✅ LOW (gap: 0.164) |
| **Average Synthetic Quality** | 74.91% |
| **Model Status** | ✅ PRODUCTION READY |

## 📚 Documentation

### JUnit Test Suite (193 total)

| Test Class | Count | Type |
|---|---|---|
| **ExampleTests** | 34 | Basic tests |
| **AdvancedTests** | 18 | Advanced scenarios |
| **AdvancedValidationTests** | 14 | Validation tests |
| **ExtendedTests** | 46 | Boundary cases |
| **MathOperationsTests** | 45 | Mathematical operations |
| **RobustTests** | 36 | Professional suite (8 nested) |

### WGAN-GP Model

**Generator Architecture**
- Input: 100 (latent vector dimension)
- Hidden layer 1: 256 neurons
- Hidden layer 2: 512 neurons
- Output: 3 (test features)
- **Total parameters**: 158,979

**Critic Architecture**
- Input: 3 (test features)
- Hidden layer 1: 512 neurons
- Hidden layer 2: 256 neurons
- Output: 1 (discriminator score)
- **Total parameters**: 133,633

### Training Metrics (Final Epoch 50)

```
Discriminator Loss: 0.0391 (stable)
Generator Loss: -0.3658 (converged)
Test Loss: 0.2033 (good generalization)
Generalization Gap: 0.164 (very low)
Overfitting Risk: LOW ✅
```

### Data Pipeline

```
Raw JUnit Data (483 samples)
        ↓ [clean_data.py]
Cleaned Data (202 samples)
        ↓ [train_configurable.py]
WGAN-GP Model (50 epochs)
        ↓ [generate_synthetic.py]
Synthetic Tests (500 samples)
```

## ⚙️ Configuration

### Python Dependencies

```
torch==2.0.1
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.64.0
scikit-learn>=1.0.0
pytest>=7.0.0
```

### WGAN Hyperparameters

- **Learning Rate**: 0.0002
- **Batch Size**: 32
- **Epochs**: 50
- **Critic Iterations**: 5 per generator update
- **Gradient Penalty (λ)**: 10
- **Generator Input Dim**: 100
- **Output Features**: 3 (test features)
- **Device**: CPU (CUDA compatible)
- **Optimizer**: Adam with β1=0.5, β2=0.999

### Java/Maven

- **Java Version**: 21 (source/target: 1.8)
- **JUnit Version**: 5.10.0 (Jupiter)
- **Maven Version**: 3.9.9+

## 🔗 References

- **GitHub Repository**: https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS
- **WGAN-GP Paper**: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- **PyTorch Documentation**: https://pytorch.org/docs/
- **JUnit 5 Documentation**: https://junit.org/junit5/

## 📝 Project Status

✅ **Phase 1** (Complete)
- WGAN trained on initial dataset
- 300 synthetic tests generated
- Baseline validation successful

✅ **Phase 2** (Complete)
- 50 epochs on 502 combined samples
- 500 new synthetic tests generated
- Enhanced model performance

✅ **Phase 3** (Complete)
- Robust test suite: 66 → 193 tests (192% increase)
- WGAN retrained with experimental validation
- Train vs Test metrics comprehensive analysis
- LOW overfitting risk confirmed (gap: 0.164)
- Synthetic quality validated: 74.91%

✅ **Testing**
- All 193 JUnit tests PASSING
- 66 original + 127 new professional tests
- Full coverage of edge cases and boundary conditions

✅ **Experimental Analysis**
- Train vs Test metrics validated
- Overfitting analysis complete
- Generalization gap: 0.164 (excellent)
- Synthetic data quality: 74.91%

✅ **Project Status**
- Ready for thesis evaluation
- Complete documentation
- All code committed to GitHub
- Production-ready model

## 🎓 Academic Context

This project implements a novel approach to automated software testing using Generative Adversarial Networks (GANs). The system learns from existing JUnit tests and generates new, realistic synthetic test cases that maintain the statistical properties of the original test data while introducing novel variations.

### Key Innovation Points

1. **Automated Test Generation**: Uses WGAN-GP to learn test patterns from real JUnit data
2. **Quality Assurance**: Experimental validation framework ensures model quality
3. **Scalability**: Can generate unlimited synthetic tests from limited real data
4. **Reproducibility**: All code, data, and results documented and version-controlled

## 💡 How to Use Generated Tests

The synthetic tests generated by the model are stored in:
```
python/outputs/synthetic_tests_robust.csv
```

These can be used for:
- Augmenting existing test suites
- Testing with diverse input patterns
- Generating edge case scenarios
- Model robustness validation

## 📧 Contact & Support

For questions or issues:
1. Check the [documentation](docs/) folder
2. Review project [GitHub issues](https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS/issues)
3. Consult the [TESTING_GUIDE.md](docs/TESTING_GUIDE.md)

---

**Author**: Gonçalo Pires  
**University**: Universidade de Évora  
**Year**: 2026  
**License**: Academic Project  
**Last Updated**: February 23, 2026
