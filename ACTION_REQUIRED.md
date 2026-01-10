# 🚀 PRODUCTION DEPLOYMENT - USER ACTION REQUIRED

## Current Status: Phase 1 - Dataset Acquisition IN PROGRESS

Two automated downloads are running:
- ✅ **Defects4J**: Cloning (~20-30 minutes)
- ✅ **OWASP**: Cloning (~10-20 minutes)
- ⏳ **Juliet**: AWAITING MANUAL DOWNLOAD

---

## What You Need to Do NOW

### Step 1: Download Juliet Java Test Suite

**URL**: https://samate.nist.gov/SARD/test-suites/81

**Steps**:
1. Click the download link for "Juliet Test Suite" (Java version)
2. You may need to register or log in
3. Wait for the download to complete (~5-7 GB)
4. Extract the ZIP file to: `data/raw/juliet/`

**Verify** the extraction:
```bash
# Should see CWE directories
ls data/raw/juliet/Juliet/ | grep CWE | wc -l
# Expected: 60+ directories
```

### Alternative (If NIST is unavailable)

Use the GitHub mirror:
```bash
git clone https://github.com/juliet/Juliet-Test-Suite.git data/raw/juliet
```

---

## What's Happening Automatically

### Terminal 1: Defects4J Download
```bash
# Command running in background:
git clone https://github.com/rjust/defects4j.git data/raw/defects4j

# Estimated time: 20-30 minutes
# Size: 2-3 GB
# Contains: ~395 real bugs from 17 Java projects
```

### Terminal 2: OWASP Benchmark Clone
```bash
# Command running in background:
git clone https://github.com/OWASP/benchmark.git data/raw/owasp

# Estimated time: 10-20 minutes
# Size: 2-3 GB
# Contains: ~100+ OWASP test cases
```

---

## Timeline After You Download Juliet

Once all three datasets are ready:

```
Time     | Action
---------|--------------------------------------------------
+0 min   | All downloads complete
+5 min   | Start dataset processing
+35 min  | Processing complete → Start training
+40 min  | Training starts (300 epochs)
+85 min  | Training complete → Generate 10k samples
+95 min  | Synthetic generation complete
+105 min | Quality analysis complete
+110 min | Final commit
---------|
Total: ~110 minutes = 1.8 hours from now
```

---

## Commands to Run (In Order)

### After Juliet is Downloaded

```bash
# 1. Verify all datasets are in place
ls data/raw/defects4j/projects/ | wc -l      # Should be 395
ls data/raw/owasp/src/main/java/org/owasp/benchmark/ | wc -l
ls data/raw/juliet/Juliet/CWE-* | wc -l      # Should be 60+

# 2. Process all datasets (15-30 min)
python python/scripts/process_datasets.py --all

# 3. Production training (30-45 min)
python python/scripts/train_production.py --epochs 300 --patience 30

# 4. Generate 10,000 synthetic samples (5-10 min)
python python/scripts/generate_large_suite.py --num-samples 10000

# 5. Quality analysis (5-10 min)
python python/scripts/compare_datasets.py

# 6. Final commit
git add -A
git commit -m "Production deployment complete: 60k+ training samples, 300-epoch training, 10k synthetic generation"
```

---

## Monitoring Progress

### Check Downloads

```powershell
# Defects4J
(Get-ChildItem data/raw/defects4j/projects/ -Directory).Count
# Should grow from 0 to ~395

# OWASP
(Get-ChildItem data/raw/owasp/ -Recurse -File).Count
# Should be 1000+

# Juliet (after you download)
(Get-ChildItem data/raw/juliet/Juliet/CWE-* -Directory).Count
# Should be ~60+
```

### Check File Sizes

```powershell
# Size of downloads
(Get-Item data/raw/defects4j/ -Force).PSIsContainer
Get-ChildItem data/raw/defects4j/ -Recurse | Measure-Object -Sum Length | Select-Object @{Name="TotalMB"; Expression={[math]::Round($_.Sum/1MB, 2)}}
```

---

## Expected Dataset Composition

After all processing:

```
Training Data:
├── JUnit:          201 samples (0.3%)
├── Defects4J:      395 samples (0.6%)
├── Juliet:     60,000+ samples (99%)
└── OWASP:        100+ samples (0.1%)
                ───────────────────
Total:        60,700+ samples (100%)

Synthetic Output:
└── Generated:   10,000 samples
```

---

## Quality Targets

| Metric | Target | Status |
|--------|--------|--------|
| Dataset size | 60k+ samples | ✅ Will have |
| Training epochs | 300 | ✅ Configured |
| Synthetic samples | 10,000 | ✅ Configured |
| Quality score | >75/100 | ⏳ Pending |

---

## Important Files to Check

After everything completes:

```
outputs/
├── training_history.csv           ← 300 rows of training metrics
├── synthetic_10k.csv              ← 10,000 synthetic samples
└── comparison_report.txt          ← Quality analysis

models/
├── generator_final.pth            ← Final generator model
└── critic_final.pth               ← Final critic model

data/raw/
├── juliet_java.csv                ← Processed Juliet
├── owasp_benchmark.csv            ← Processed OWASP
├── defects4j_tests.csv            ← Processed Defects4J
└── combined_datasets.csv          ← All combined
```

---

## Troubleshooting

### "Can't access Juliet NIST website"
→ Use GitHub mirror: `git clone https://github.com/juliet/Juliet-Test-Suite.git data/raw/juliet`

### "Downloads taking too long"
→ Normal - Defects4J & OWASP are 2-3GB each. Juliet is 5-7GB.
→ Check internet connection and try again if interrupted.

### "Out of disk space"
→ Check available: `Get-Volume C: | Select-Object SizeRemaining`
→ Need ~100GB total free space for all datasets + models

### "Processing script fails"
→ Check datasets are in correct locations
→ Ensure all .java files are readable: `icacls data/raw /grant:r Users:F /t`

---

## Next Steps Summary

1. ✅ **Done**: Infrastructure created, 100-epoch validation
2. ⏳ **Now**: Download Juliet (manual action)
3. ⏳ **Wait**: Defects4J & OWASP finish (20-30 min)
4. ⏳ **Run**: Process datasets (15-30 min)
5. ⏳ **Run**: Production training 300 epochs (30-45 min)
6. ⏳ **Run**: Generate 10k samples (5-10 min)
7. ⏳ **Run**: Quality analysis (5-10 min)
8. ⏳ **Done**: Final commit and completion

---

## Questions?

See:
- `PRODUCTION_DEPLOYMENT.md` - Full workflow guide
- `DATASET_DOWNLOAD_GUIDE.md` - Download instructions
- `PRODUCTION_NEXT_STEPS.md` - Status overview

**Estimated total time: 2.5-3.5 hours** ⏱️

Ready? Download Juliet and we'll proceed! 🚀
