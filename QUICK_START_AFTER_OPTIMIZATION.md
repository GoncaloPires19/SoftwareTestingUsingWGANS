#  🚀 Quick Start Guide - After Optimization

**Status:** ✅ Repository fully optimized and validated (100% pass rate)  
**Date:** 26 April 2026

---

## 1. Verify Setup (Run Once)
```bash
# Install/update dependencies
pip install -r requirements.txt

# Quick system validation
python validate_refactoring.py

# Verify data pipeline
python python/scripts/validate_before_training.py
```

---

## 2. Train WGAN Model
```bash
# Quick test (5 epochs - ~5 minutes)
python python/scripts/train_basic_profiles.py --profile balanced --epochs 5

# Production training (50 epochs - ~1 hour)
python python/scripts/train_basic_profiles.py --profile optimized --epochs 50
```

---

## 3. Generate Synthetic Tests
```bash
# Generate 1000 synthetic samples
python python/scripts/generate_synthetic.py --num-samples 1000

# Analyze results
python python/scripts/experimental_analysis.py
```

---

## 4. Run Full Pipeline (Complete Workflow)
```bash
# 1. Java tests → extract metrics
mvn clean test

# 2. Train WGAN on metrics
python python/scripts/train_basic_profiles.py --profile optimized --epochs 50

# 3. Generate synthetic tests
python python/scripts/generate_synthetic.py

# 4. Analyze quality
python python/scripts/experimental_analysis.py
```

---

## 5. Advanced: SF110 Experiments
```bash
# Run hypothesis experiments (H1/H2/H3)
python python/scripts/run_sf110_hypotheses.py --epochs 10 --projects 5

# Monitor progress
python python/scripts/tools/monitor_training.py
```

---

## 📊 Key Output Files
- `outputs/training_history.csv` - Training metrics per epoch
- `outputs/synthetic_tests.csv` - Generated synthetic test data
- `outputs/FINAL_RESULTS_TABLE.md` - Consolidated results
- `python/models/*.pth` - Trained WGAN weights

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: No module named 'scipy'" | Run: `pip install scipy>=1.9.0` |
| Training diverges/NaN | Check: `python python/scripts/validate_before_training.py` |
| Out of memory on GPU | Edit `python/config.py` → reduce `batch_size` |
| Cannot find test data | Verify: `python/data/junit_data_cleaned_strict.csv` exists |

---

## 📁 Important Paths
- **Config:** `python/config.py` (hyperparameters)
- **Models:** `python/model.py` (architecture)
- **Data:** `python/data/junit_data_cleaned_strict.csv` (training data)
- **Scripts:** `python/scripts/` (all tools)
- **Results:** `outputs/` (organized by experiment date)

---

## 📚 Documentation
- [Quick Start Guide](docs/QUICK_START.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Advanced Training Guide](docs/ADVANCED_TRAINING_GUIDE.md)
- [Data Schema](docs/DATA_SCHEMA.md)
- [Optimization Report](FINAL_OPTIMIZATION_REPORT.md)

---

**✅ System is ready for immediate use!**
