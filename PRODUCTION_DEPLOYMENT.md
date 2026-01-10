# Production Deployment Plan

## Overview

This document outlines the complete production deployment workflow, from dataset acquisition through synthetic test generation and validation.

**Total Timeline**: 3-4 hours
**Computational Requirements**: CPU/GPU with 16GB+ RAM, ~100GB disk space

---

## Phase 1: Dataset Acquisition (1-2 hours)

### Status
- ✅ Defects4J: Currently downloading via `download_datasets.py`
- ⏳ Juliet: Awaiting manual download
- ⏳ OWASP: Ready for automated clone

### Step-by-Step

#### 1.1 Defects4J (Automated) - IN PROGRESS
```bash
# Already running in background
# Monitor status:
ls -la data/raw/defects4j/ | wc -l
# Should see ~395 project directories
```

Expected size: 2-3 GB
Expected duration: 20-30 minutes

#### 1.2 Juliet Java Test Suite (Manual)

**Required Actions**:
1. Visit: https://samate.nist.gov/SARD/test-suites/81
2. Request or download the Java Test Suite
3. Extract to: `data/raw/juliet/`
4. Verify structure:
   ```bash
   ls data/raw/juliet/Juliet/CWE-*/src/ | wc -l
   # Should see 60+ CWE directories
   ```

Expected size: 5-7 GB (compressed)
Expected duration: 30-60 minutes (mostly download)

#### 1.3 OWASP Benchmark (Automated)

```bash
# Clone OWASP Benchmark
git clone https://github.com/OWASP/benchmark.git data/raw/owasp
```

Or manually from: https://github.com/OWASP/benchmark

Expected size: 2-3 GB
Expected duration: 10-20 minutes

---

## Phase 2: Dataset Processing (15-30 minutes)

Once all datasets are downloaded, process and normalize:

```bash
# Process all datasets
python python/scripts/process_datasets.py --all

# This will:
# - Extract features from Juliet (60,000+ samples)
# - Extract features from OWASP (100+ samples)
# - Extract features from Defects4J (395+ samples)
# - Combine into single normalized CSV
```

Output files:
- `data/raw/juliet_java.csv` - Juliet features
- `data/raw/owasp_benchmark.csv` - OWASP features
- `data/raw/defects4j_tests.csv` - Defects4J features
- `data/raw/combined_datasets.csv` - All combined

---

## Phase 3: Production Training (30-45 minutes)

### Full Dataset Training

```bash
# Train WGAN on all datasets (300 epochs)
python python/scripts/train_production.py --epochs 300 --patience 30

# This will:
# - Load: JUnit (201) + Juliet (60k+) + OWASP (100+) + Defects4J (395+)
# - Total: 60,700+ diverse test samples
# - Train for up to 300 epochs
# - Use early stopping with patience=30
# - Save best models and training history
```

**Expected Metrics**:
- Training time: 30-45 minutes
- Time per epoch: ~6-9 seconds
- Batch size: 32
- Dataset: ~60,700 samples (total)

**Output**:
- `models/generator_final.pth` - Final generator
- `models/critic_final.pth` - Final critic
- `outputs/training_history.csv` - Training metrics

---

## Phase 4: Synthetic Generation (5-10 minutes)

Generate large-scale synthetic test suite:

```bash
# Generate 10,000 synthetic samples
python python/scripts/generate_large_suite.py --num-samples 10000 --output python/outputs/synthetic_10k.csv

# This will:
# - Load trained generator
# - Generate 10,000 synthetic test samples
# - Denormalize features to realistic ranges
# - Save as CSV for deployment
```

**Output**:
- `outputs/synthetic_10k.csv` - 10,000 synthetic samples

---

## Phase 5: Quality Analysis (10-15 minutes)

Compare real vs synthetic distributions:

```bash
# Generate comparison report
python python/scripts/compare_datasets.py \
  --real python/data/junit_data_cleaned.csv \
  --synthetic python/outputs/synthetic_10k.csv \
  --output python/outputs/comparison_report.txt

# This will:
# - Compare statistical properties
# - Analyze diversity metrics
# - Check feature coverage
# - Assess sample quality
# - Generate quality score
```

**Output**:
- `outputs/comparison_report.txt` - Detailed quality analysis

---

## Complete Workflow Script

For convenience, here's the complete command sequence:

```bash
# Phase 1: Dataset Acquisition
# Defects4J (already running)
# Manual: Download Juliet and extract to data/raw/juliet/
git clone https://github.com/OWASP/benchmark.git data/raw/owasp

# Phase 2: Processing
python python/scripts/process_datasets.py --all

# Phase 3: Production Training
python python/scripts/train_production.py --epochs 300 --patience 30

# Phase 4: Synthetic Generation
python python/scripts/generate_large_suite.py --num-samples 10000

# Phase 5: Quality Analysis
python python/scripts/compare_datasets.py

# Final: Commit results
git add -A
git commit -m "Production deployment: 300-epoch training on 60k+ samples, 10k synthetic test generation"
```

---

## Expected Dataset Composition

After all phases complete:

```
REAL TEST DATA (60,700+ samples)
├── JUnit Tests: 201 samples
├── Defects4J Bugs: 395 samples
├── Juliet Dataset: 60,000+ samples
└── OWASP Benchmark: 100+ samples

GENERATED SYNTHETIC DATA (10,000 samples)
└── WGAN Generated: 10,000 samples
    ├── Methods: 250-350 avg
    ├── Branches: 900-1100 avg
    └── Statements: 200-300 avg
```

---

## Quality Targets

| Metric | Target | Acceptable |
|--------|--------|-----------|
| Statistical Similarity | >90% | >85% |
| Feature Coverage | >95% | >85% |
| Sample Validity | >98% | >95% |
| Overall Quality Score | >85/100 | >75/100 |

---

## Deployment After Training

Once training is complete, you can:

1. **Deploy to Testing Tools**:
   ```bash
   # Copy to SonarQube, FindBugs, etc.
   cp outputs/synthetic_10k.csv /path/to/testing/tools/
   ```

2. **Integration Testing**:
   - Use synthetic tests for regression testing
   - Validate against real test behavior
   - Measure false positive rates

3. **Continuous Updates**:
   - Retrain monthly with new real tests
   - Monitor synthetic sample quality
   - Adjust model parameters if needed

---

## Monitoring & Troubleshooting

### Check Download Progress
```bash
# Defects4J
du -sh data/raw/defects4j/

# Juliet
du -sh data/raw/juliet/

# OWASP
du -sh data/raw/owasp/
```

### Check Processing Progress
```bash
# Count generated CSV files
ls -lh data/raw/*.csv

# Check combined dataset size
wc -l data/raw/combined_datasets.csv
```

### Monitor Training
```bash
# Watch training metrics
tail -f outputs/training_history.csv

# Check model sizes
ls -lh models/

# Check GPU/CPU usage (if applicable)
nvidia-smi  # for GPU
```

### Validate Results
```bash
# Check synthetic data
wc -l outputs/synthetic_10k.csv
head outputs/synthetic_10k.csv

# View quality report
cat outputs/comparison_report.txt
```

---

## Common Issues & Solutions

### Issue: Defects4J download fails
**Solution**: 
```bash
# Manual clone
git clone https://github.com/rjust/defects4j.git data/raw/defects4j
```

### Issue: Juliet not accessible
**Solution**:
- Try GitHub mirror: https://github.com/juliet/Juliet-Test-Suite
- Or use academic mirror if available through your institution

### Issue: Out of disk space
**Solution**:
- Compress raw datasets: `gzip -r data/raw/`
- Process datasets incrementally
- Delete intermediate files after processing

### Issue: Out of memory during training
**Solution**:
```bash
# Reduce batch size
python python/scripts/train_production.py --batch-size 16
```

### Issue: Training too slow
**Solution**:
```bash
# Use GPU if available (modify config)
# Or reduce epochs for testing:
python python/scripts/train_production.py --epochs 100
```

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Dataset Acquisition | 1-2h | ⏳ IN PROGRESS |
| 2. Dataset Processing | 15-30m | ⏳ PENDING |
| 3. Production Training | 30-45m | ⏳ PENDING |
| 4. Synthetic Generation | 5-10m | ⏳ PENDING |
| 5. Quality Analysis | 10-15m | ⏳ PENDING |
| **TOTAL** | **2.5-3.5h** | |

---

## Next Steps

1. **Now**: Monitor Defects4J download, prepare for Juliet download
2. **When Defects4J completes**: Clone OWASP Benchmark
3. **When Juliet arrives**: Extract to `data/raw/juliet/`
4. **When all downloaded**: Run processing script
5. **After processing**: Start production training
6. **After training**: Generate synthetic suite
7. **Final**: Run quality analysis and commit

---

## Success Criteria

✅ Deployment is successful when:
- All 4 datasets are loaded and processed
- Training completes 300 epochs on 60k+ samples
- 10,000 synthetic samples generated
- Quality score > 75/100
- All files committed to git

Ready to proceed! 🚀
