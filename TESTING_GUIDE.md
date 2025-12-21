## Testing Your WGAN-GP JUnit Test System

### Quick Verification (Recommended First Step)

Run the comprehensive test suite to verify everything is working:

```powershell
cd python
python test_system.py
```

This will check:
- ✓ Python and PyTorch installation
- ✓ Required modules (pandas, tqdm, numpy)
- ✓ Data files availability
- ✓ Dataset loading from CSV
- ✓ Model creation (Generator & Critic)
- ✓ Gradient penalty computation
- ✓ Synthetic test generation
- ✓ Java/Maven setup

---

### Step-by-Step Testing

#### 1. **Test Data Loading**
```powershell
cd python
python -c "from datasets import get_junit_data_loader; loader, dataset = get_junit_data_loader('data/junit_data.csv'); print('✓ Data loaded:', dataset.get_feature_statistics())"
```

Expected output: Shows number of samples, features, and their statistics

#### 2. **Test Model Creation**
```powershell
python -c "from model import Generator, Critic; import torch; g = Generator(100, 3); c = Critic(3); print('✓ Models created'); print('Generator params:', sum(p.numel() for p in g.parameters())); print('Critic params:', sum(p.numel() for p in c.parameters()))"
```

Expected output: Shows parameter counts (should be in thousands)

#### 3. **Test Gradient Penalty**
```powershell
python -c "from utils import compute_gradient_penalty; from model import Critic; import torch; c = Critic(3); real = torch.randn(4, 3); fake = torch.randn(4, 3); gp = compute_gradient_penalty(c, real, fake, 'cpu'); print('✓ Gradient penalty:', gp.item())"
```

Expected output: Shows a numerical value (gradient penalty)

---

### Training & Generation Tests

#### 4. **Quick Training Test (1 epoch)**
Edit `train.py` temporarily:
```python
NUM_EPOCHS = 1  # Change from 50 to 1
```

Then run:
```powershell
python train.py
```

This will test if training runs without errors.

#### 5. **Generate Synthetic Tests**
After training (or using existing checkpoints):
```powershell
python generate_synthetic_tests.py --num-samples 50 --output-file synthetic_test_output.csv
```

Check if `synthetic_test_output.csv` is created with 50 rows.

---

### Java Tests Verification

#### 6. **Build Java Project**
```powershell
mvn clean build
```

#### 7. **Run JUnit Tests**
```powershell
mvn test
```

Expected output: Shows test results and CSV logs

#### 8. **Check Generated Test CSV**
After running tests, check:
```powershell
ls junit_test_log.csv
```

---

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Install: `pip install torch pandas tqdm numpy` |
| `FileNotFoundError: data/junit_data.csv` | Ensure you're in the `python/` directory |
| `CUDA out of memory` | Use CPU: Set `DEVICE = torch.device("cpu")` in train.py |
| No checkpoints found | Run training first: `python train.py` |
| Maven not found | Install Maven or check PATH |

---

### What to Look For - Success Indicators

✅ **Test System** passes all checks  
✅ **Data loading** shows: "20 amostras reais de testes JUnit"  
✅ **Training** shows decreasing loss values each epoch  
✅ **Synthetic generation** creates CSV with valid numbers  
✅ **Java tests** complete with 0 failures  
✅ **CSV logs** contain test case data  

---

### Full Pipeline Test

Run everything in sequence:

```powershell
# 1. Verify system
python test_system.py

# 2. Train on real data (if not already trained)
python train.py

# 3. Generate synthetic tests
python generate_synthetic_tests.py --num-samples 100

# 4. Build and run Java tests
cd ..
mvn clean test

# 5. Check results
ls python/junit_test_log.csv
```

---

### Expected File Structure After Testing

```
python/
├── data/
│   └── junit_data.csv          ✓ Real test data
├── models/
│   ├── generator_epoch1.pth     ✓ Trained models
│   ├── critic_epoch1.pth
│   └── ...
├── synthetic_test_output.csv    ✓ Generated synthetic tests
├── junit_test_log.csv          ✓ Test execution log
├── test_system.py
├── train.py
├── generate_synthetic_tests.py
├── datasets.py
├── model.py
├── utils.py
└── ...
```

---

### Performance Metrics to Track

When training, monitor:
- **Loss_D** (Critic): Should stabilize around 0
- **Loss_G** (Generator): Should decrease over epochs
- **Gradient Penalty**: Should stay consistent (around 1-2)

If losses explode → reduce learning rate  
If training is slow → increase batch size or use GPU

---

### Questions?

Check the test output for specific error messages or run individual components in isolation using the step-by-step tests above.
