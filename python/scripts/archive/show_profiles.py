"""Display available training profiles and their characteristics."""

print("="*80)
print("AVAILABLE TRAINING PROFILES")
print("="*80)

profiles = [
    {
        'name': 'balanced',
        'epochs': 100,
        'batch': 16,
        'lr': 0.00005,
        'lambda_gp': 5.0,
        'n_critic': 7,
        'time': '30-50min',
        'status': '✅ TESTED',
        'description': 'Production-ready, stable training',
        'best_for': 'General purpose, balanced quality'
    },
    {
        'name': 'aggressive',
        'epochs': 50,
        'batch': 32,
        'lr': 0.0001,
        'lambda_gp': 2.0,
        'n_critic': 5,
        'time': '10-20min',
        'status': '⏱️  FAST',
        'description': 'Quick training, may be unstable',
        'best_for': 'Rapid prototyping, quick tests'
    },
    {
        'name': 'experimental',
        'epochs': 80,
        'batch': 12,
        'lr': 0.00008,
        'lambda_gp': 7.5,
        'n_critic': 8,
        'time': '20-40min',
        'status': '🧪 UNTESTED',
        'description': 'Improved hybrid configuration',
        'best_for': 'Better quality than aggressive'
    },
    {
        'name': 'conservative',
        'epochs': 150,
        'batch': 8,
        'lr': 0.000025,
        'lambda_gp': 10.0,
        'n_critic': 12,
        'time': '2h+',
        'status': '🛡️ STABLE',
        'description': 'Maximum stability, very slow',
        'best_for': 'Critical applications requiring stability'
    },
    {
        'name': 'coverage-seeking',
        'epochs': 150,
        'batch': 16,
        'lr': 0.00003,
        'lambda_gp': 10.0,
        'n_critic': 5,
        'time': '40-60min',
        'status': '🎯 SPECIALIZED',
        'description': 'Focus on maximizing coverage',
        'best_for': 'When max coverage is priority'
    },
    {
        'name': 'hybrid-realistic',
        'epochs': 100,
        'batch': 16,
        'lr': 0.00005,
        'lambda_gp': 5.0,
        'n_critic': 7,
        'time': '30-50min',
        'status': '✅ TESTED (200 epochs)',
        'description': 'Balance between coverage and realism',
        'best_for': 'Synthetic data generation with realism',
        'result': '86% max coverage, but distributions diverged at 200 epochs'
    }
]

print("\nProfile Comparison Table:")
print("-"*80)
print(f"{'Profile':<20} {'Status':<15} {'Epochs':<8} {'Time':<12} {'Best For'}")
print("-"*80)

for p in profiles:
    print(f"{p['name']:<20} {p['status']:<15} {p['epochs']:<8} {p['time']:<12} {p['best_for']}")

print("\n" + "="*80)
print("DETAILED PROFILES")
print("="*80)

for p in profiles:
    print(f"\n{p['status']} {p['name'].upper()}")
    print(f"   Description: {p['description']}")
    print(f"   Configuration: epochs={p['epochs']}, batch={p['batch']}, lr={p['lr']}, "
          f"λ_gp={p['lambda_gp']}, n_critic={p['n_critic']}")
    print(f"   Training time: {p['time']} (CPU)")
    print(f"   Best for: {p['best_for']}")
    if 'result' in p:
        print(f"   Previous result: {p['result']}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
Based on your goal (realistic synthetic test generation):

1. 🥇 RECOMMENDED: 'experimental' (80 epochs, 20-40min)
   - Better than aggressive (more stable)
   - Faster than conservative
   - Likely better distribution match than hybrid-realistic at 200 epochs

2. 🥈 ALTERNATIVE: 'aggressive' (50 epochs, 10-20min)
   - Quick baseline for comparison
   - Fast iteration for hyperparameter tuning

3. 🥉 IF TIME PERMITS: 'coverage-seeking' (150 epochs, 40-60min)
   - Specialized for max coverage
   - Compare with hybrid-realistic results

AVOID for now:
- 'conservative': Too slow (2h+) for incremental testing
- 'balanced': Same as hybrid-realistic already tested

NEXT STEPS:
Run these commands to test recommended profiles:

# Quick test (10-20min)
python python/scripts/train_profile_aware.py --profile aggressive --epochs 50 --seed 42

# Balanced test (20-40min)  
python python/scripts/train_profile_aware.py --profile experimental --epochs 80 --seed 42

# Specialized test (40-60min)
python python/scripts/train_profile_aware.py --profile coverage-seeking --epochs 150 --seed 42
""")
