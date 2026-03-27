#!/usr/bin/env python3
"""
Monitor optimized training progress in real-time
"""

import subprocess
import sys
from pathlib import Path
import time

output_dir = Path("outputs/h2_v2_optimized_20260316_231903")
config_path = output_dir / "config.json"
history_path = output_dir / "training_history.csv"

print("⏱️  Real-time Training Monitor - Optimized Profile")
print("="*70)

while True:
    if history_path.exists():
        import pandas as pd
        df = pd.read_csv(history_path)
        
        latest = df.iloc[-1]
        epoch = int(latest['epoch'])
        
        print(f"\n📊 Epoch {epoch}/100", end="")
        print(f"  | G: {latest['gen_loss']:7.4f} | C: {latest['critic_loss']:7.4f}")
        print(f"   Coverage: {latest['coverage']:.2f}% | Robust V2: {latest['robust_v2_score']:.2f}")
        print(f"   Diversity: {latest['diversity']:.3f} | Tail penalty: {latest['tail_excess_penalty']:.4f}")
        
        if epoch >= 100:
            print("\n✅ Training complete! Run analysis script now.")
            break
    
    time.sleep(30)
