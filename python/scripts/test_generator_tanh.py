"""
Quick test to verify Generator now produces values in [-1, 1] with tanh activation.
"""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Generator

print("Testing Generator with tanh activation...")
print("=" * 60)

# Create generator
gen = Generator(z_dim=100, output_dim=3)
print(f"✓ Generator created with {sum(p.numel() for p in gen.parameters())} params")

# Generate some random samples
gen.eval()
with torch.no_grad():
    z = torch.randn(1000, 100)
    synthetic = gen(z)

# Check range
min_val = synthetic.min().item()
max_val = synthetic.max().item()
mean_val = synthetic.mean().item()

print(f"\nGenerated 1000 samples:")
print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
print(f"  Mean: {mean_val:.4f}")
print(f"  Shape: {synthetic.shape}")

# Verify in [-1, 1]
if min_val >= -1.0 and max_val <= 1.0:
    print(f"\n✅ SUCCESS: All values in [-1, 1] range!")
    print(f"   This will prevent negative denormalized values.")
else:
    print(f"\n❌ FAIL: Values outside [-1, 1] range!")
    print(f"   Min: {min_val} (should be >= -1)")
    print(f"   Max: {max_val} (should be <= 1)")

print("=" * 60)
