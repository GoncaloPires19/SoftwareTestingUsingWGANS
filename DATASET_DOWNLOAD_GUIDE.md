# Dataset Download Guide

## Overview
This guide explains how to download and set up the public datasets for WGAN training.

**Total size**: ~15-20 GB (compressed)
**Decompressed**: ~50-60 GB
**Time required**: 3-4 hours (mostly download time)

---

## 1. Defects4J (Automated) ✅

**Status**: Automatically downloading via script
**Location**: `data/raw/defects4j/`
**Size**: ~2-3 GB
**Time**: 20-30 minutes

```bash
python python/scripts/download_datasets.py --process-defects4j
```

**What it contains**:
- 395+ real bugs from 17 Java projects
- Test cases and source code
- Bug-fix commits and patches

---

## 2. Juliet Java Test Suite (Manual Download)

**Status**: Requires manual download
**Location**: `data/raw/juliet/`
**Size**: ~5-7 GB (compressed)
**Time**: 30-60 minutes

### Download Steps:

1. **Navigate to NIST SARD**:
   ```
   https://samate.nist.gov/SARD/test-suites/81
   ```

2. **Request access** (may require registration):
   - Email NIST for download link
   - Or use mirror: https://github.com/juliet/Juliet-Test-Suite

3. **Extract to correct location**:
   ```
   data/raw/juliet/
   ```

4. **Expected directory structure**:
   ```
   data/raw/juliet/
   ├── Juliet/
   │   ├── CWE-022/
   │   ├── CWE-023/
   │   ├── ... (60+ CWE types)
   │   └── src/
   │       └── *.java files
   ```

---

## 3. OWASP Benchmark (Manual Download)

**Status**: Requires manual clone
**Location**: `data/raw/owasp/`
**Size**: ~2-3 GB
**Time**: 10-20 minutes

### Download Steps:

1. **Clone from GitHub**:
   ```bash
   git clone https://github.com/OWASP/benchmark.git data/raw/owasp
   ```

   Or manually download from:
   ```
   https://github.com/OWASP/benchmark
   ```

2. **Expected directory structure**:
   ```
   data/raw/owasp/
   ├── src/
   │   ├── main/java/org/owasp/benchmark/
   │   └── helpers/
   ├── testcases/
   └── ...
   ```

---

## 4. Quick Manual Setup

If you want to manually place the datasets:

1. **Create directories**:
   ```bash
   mkdir -p data/raw/{juliet,owasp,defects4j}
   ```

2. **Extract your downloads**:
   ```bash
   # Juliet
   unzip Juliet-Java.zip -d data/raw/juliet
   
   # OWASP
   git clone https://github.com/OWASP/benchmark.git data/raw/owasp
   ```

---

## 5. Verify Downloads

After downloads complete, verify:

```bash
# Check directory structure
ls -R data/raw/

# Check Defects4J
ls data/raw/defects4j/ | wc -l  # Should have 395+ projects

# Check Juliet
ls data/raw/juliet/Juliet/CWE-* | wc -l  # Should have 60+ CWE directories

# Check OWASP
ls data/raw/owasp/src/main/java/org/owasp/benchmark/ | wc -l
```

---

## 6. Next Steps After Download

Once all datasets are in place:

```bash
# Process and normalize all datasets
python python/scripts/process_datasets.py

# Check what was loaded
python python/scripts/validate_datasets.py

# Start production training with 300+ epochs
python python/scripts/train_configurable.py --preset combined --epochs 300
```

---

## Troubleshooting

### Defects4J download fails
- Check internet connection
- Try manual clone: `git clone https://github.com/rjust/defects4j.git data/raw/defects4j`

### Juliet access denied
- Register at NIST SARD website
- Or use GitHub mirror: https://github.com/juliet/Juliet-Test-Suite

### Disk space issues
- Defects4J: Keep compressed to ~2GB
- Juliet: Keep compressed to ~5-7GB
- OWASP: ~2-3GB

### Processing fails
- Ensure all datasets are extracted to correct locations
- Check file permissions: `chmod -R 755 data/raw/`
- Verify Java is installed: `java -version`

---

## Dataset Processing Timeline

| Step | Time | Status |
|------|------|--------|
| 1. Defects4J download | 20-30 min | **IN PROGRESS** |
| 2. Juliet manual download | 30-60 min | ⏳ Waiting |
| 3. OWASP clone | 10-20 min | ⏳ Waiting |
| 4. Dataset processing | 10-15 min | ⏳ Pending |
| 5. Production training (300 epochs) | 30-45 min | ⏳ Pending |
| 6. Synthetic generation (10k samples) | 5-10 min | ⏳ Pending |
| **Total** | **2-3.5 hours** | |

---

## Current Status

- ✅ Defects4J: Downloading...
- ⏳ Juliet: Awaiting manual download
- ⏳ OWASP: Ready for clone
- ⏳ Processing: Pending
- ⏳ Training: Pending

---

Once all datasets are ready, you'll have:
- **201** Java JUnit tests (current)
- **395+** real bugs from Defects4J
- **60,000+** test cases from Juliet
- **100+** OWASP Benchmark test cases
- **Combined**: **60,700+** diverse test samples

This will enable training a robust WGAN that generates realistic and diverse test cases!
