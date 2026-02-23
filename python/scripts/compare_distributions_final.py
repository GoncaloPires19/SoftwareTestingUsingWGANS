"""Compare real vs synthetic distributions after 200 epochs training."""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
real = pd.read_csv('python/data/junit_data_cleaned_strict.csv', header=None, names=['methods', 'branches', 'statements'])
synthetic = pd.read_csv('outputs/synthetic_continuation_final.csv')

print("="*70)
print("DISTRIBUTION COMPARISON: REAL vs SYNTHETIC (200 epochs)")
print("="*70)

print("\n📊 REAL DATA (667 samples):")
for col in ['methods', 'branches', 'statements']:
    r = real[col]
    print(f"  {col:12s}: min={r.min():6.0f}, Q25={r.quantile(0.25):6.0f}, "
          f"median={r.median():6.0f}, Q75={r.quantile(0.75):6.0f}, max={r.max():6.0f}")
    print(f"                mean={r.mean():6.1f} ± {r.std():6.1f}")

print("\n🤖 SYNTHETIC DATA (1000 samples):")
for col in ['methods', 'branches', 'statements']:
    s = synthetic[col]
    print(f"  {col:12s}: min={s.min():6.0f}, Q25={s.quantile(0.25):6.0f}, "
          f"median={s.median():6.0f}, Q75={s.quantile(0.75):6.0f}, max={s.max():6.0f}")
    print(f"                mean={s.mean():6.1f} ± {s.std():6.1f}")

print("\n📈 SIMILARITY METRICS:")
print(f"{'Feature':<15} {'Mean Ratio':<12} {'Median Ratio':<14} {'KS Statistic':<15} {'p-value'}")
print("-"*70)

for col in ['methods', 'branches', 'statements']:
    mean_ratio = synthetic[col].mean() / real[col].mean()
    median_ratio = synthetic[col].median() / real[col].median()
    ks_stat, p_value = stats.ks_2samp(real[col], synthetic[col])
    
    print(f"{col:<15} {mean_ratio:>11.2f}x {median_ratio:>13.2f}x {ks_stat:>14.3f} {p_value:>12.4f}")

print("\n🎯 MAX VALUE COVERAGE:")
print(f"{'Feature':<15} {'Real Max':<12} {'Synthetic Max':<15} {'Coverage'}")
print("-"*70)
for col in ['methods', 'branches', 'statements']:
    real_max = real[col].max()
    synth_max = synthetic[col].max()
    coverage = (synth_max / real_max) * 100
    status = "✅" if coverage > 90 else "⚠️" if coverage > 70 else "❌"
    print(f"{col:<15} {real_max:>11.0f} {synth_max:>14.0f} {coverage:>11.1f}% {status}")

print("\n💡 INTERPRETATION:")

# Check if distributions are similar (KS test)
all_pvalues = [stats.ks_2samp(real[col], synthetic[col])[1] for col in ['methods', 'branches', 'statements']]
if all(p > 0.05 for p in all_pvalues):
    print("  ✅ All distributions statistically similar (p > 0.05)")
elif any(p > 0.01 for p in all_pvalues):
    print("  ⚠️  Some distributions differ slightly (0.01 < p < 0.05)")
else:
    print("  ❌ Distributions differ significantly (p < 0.01)")

# Check mean ratios
mean_ratios = [synthetic[col].mean() / real[col].mean() for col in ['methods', 'branches', 'statements']]
if all(0.8 < r < 1.2 for r in mean_ratios):
    print("  ✅ Mean values well-matched (0.8x - 1.2x range)")
elif all(0.5 < r < 2.0 for r in mean_ratios):
    print("  ⚠️  Mean values acceptable (0.5x - 2.0x range)")
else:
    print("  ❌ Mean values poorly matched")

# Check max coverage
max_coverages = [(synthetic[col].max() / real[col].max()) * 100 for col in ['methods', 'branches', 'statements']]
avg_coverage = np.mean(max_coverages)
print(f"  📊 Average max coverage: {avg_coverage:.1f}%")
if avg_coverage > 90:
    print("  ✅ Excellent coverage of extreme values")
elif avg_coverage > 70:
    print("  ⚠️  Good coverage of extreme values")
else:
    print("  ❌ Poor coverage of extreme values")
