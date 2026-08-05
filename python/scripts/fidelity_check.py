#!/usr/bin/env python3
"""
Distributional Fidelity and Overfitting Check for the WGAN-GP (E1)

Loads a generator trained ONLY on a train split, generates synthetic samples,
and compares them against the train split (seen) and a held-out test split
(never seen during training) in the normalised [-1, 1] / log1p space the
generator actually operates in.

If synthetic-vs-test fidelity is close to synthetic-vs-train fidelity, the
generator learned the underlying distribution rather than memorising the
training rows.

Usage:
    python python/scripts/fidelity_check.py \
        --run-dir outputs/fidelity_check/run \
        --train outputs/fidelity_check/train.csv \
        --test outputs/fidelity_check/test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

from model import Generator

SEP = "=" * 72
FEATURE_NAMES = ["methods", "branches", "statements"]


def _normalize_like_train(raw: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Apply the *train* log1p + min-max stats to another raw array (no leakage)."""
    x = raw.astype(np.float64)
    if norm_stats.get("log_transform", False):
        x = np.log1p(x)
    out = np.zeros_like(x)
    for i in range(x.shape[1]):
        fs = norm_stats["features"][f"feature_{i}"]
        rng = fs["range"] if fs["range"] != 0 else 1.0
        out[:, i] = 2 * (x[:, i] - fs["min"]) / rng - 1
    return out


def _js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    return float(jensenshannon(pa, pb) ** 2)  # squared JS = the divergence (JS distance is its sqrt)


def main() -> None:
    parser = argparse.ArgumentParser(description="WGAN-GP distributional fidelity / overfitting check")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        run_config = json.load(f)
    model_cfg = run_config.get("model", {})

    ckpt = torch.load(run_dir / "checkpoint_best.pth", map_location="cpu")
    gen_state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    best_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None

    z_dim = model_cfg.get("z_dim", 100)
    output_dim = model_cfg.get("output_dim", 3)
    hidden_dims_g = model_cfg.get("hidden_dims_g", [256, 512])

    generator = Generator(z_dim=z_dim, output_dim=output_dim, hidden_dims=hidden_dims_g)
    generator.load_state_dict(gen_state)
    generator.eval()

    torch.manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(args.n_samples, z_dim)
        synthetic = generator(z).numpy()
    synthetic = np.clip(synthetic, -1, 1)

    # Find the norm stats saved alongside the train split used for this run.
    norm_stats_path = Path(args.train).with_name(Path(args.train).stem + "_norm_stats.json")
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    train_raw = pd.read_csv(args.train, header=None).values
    test_raw = pd.read_csv(args.test, header=None).values

    train_norm = _normalize_like_train(train_raw, norm_stats)
    test_norm = _normalize_like_train(test_raw, norm_stats)

    print(SEP)
    print("  FIDELIDADE DISTRIBUCIONAL E SOBREAJUSTE — WGAN-GP (espaço normalizado [-1,1])")
    print(SEP)
    print(f"\n  Checkpoint: epoch {best_epoch} | sintéticos gerados: {args.n_samples}")
    print(f"  Treino: {len(train_raw)} amostras | Teste (nunca visto): {len(test_raw)} amostras\n")

    results = {"per_feature": {}, "overall": {}}
    header = f"  {'Feature':<12} {'KS(train)':>10} {'KS(test)':>9} {'JS(train)':>10} {'JS(test)':>9} {'W1(train)':>10} {'W1(test)':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    ks_train, ks_test, js_train, js_test, w1_train, w1_test = [], [], [], [], [], []
    for i, name in enumerate(FEATURE_NAMES):
        s, tr, te = synthetic[:, i], train_norm[:, i], test_norm[:, i]

        ks_tr = ks_2samp(s, tr).statistic
        ks_te = ks_2samp(s, te).statistic
        js_tr = _js_divergence(s, tr)
        js_te = _js_divergence(s, te)
        w1_tr = wasserstein_distance(s, tr)
        w1_te = wasserstein_distance(s, te)

        ks_train.append(ks_tr); ks_test.append(ks_te)
        js_train.append(js_tr); js_test.append(js_te)
        w1_train.append(w1_tr); w1_test.append(w1_te)

        results["per_feature"][name] = {
            "ks_train": ks_tr, "ks_test": ks_te,
            "js_train": js_tr, "js_test": js_te,
            "w1_train": w1_tr, "w1_test": w1_te,
        }
        print(f"  {name:<12} {ks_tr:>10.3f} {ks_te:>9.3f} {js_tr:>10.3f} {js_te:>9.3f} {w1_tr:>10.3f} {w1_te:>9.3f}")

    mean_ks_train, mean_ks_test = float(np.mean(ks_train)), float(np.mean(ks_test))
    mean_js_train, mean_js_test = float(np.mean(js_train)), float(np.mean(js_test))
    mean_w1_train, mean_w1_test = float(np.mean(w1_train)), float(np.mean(w1_test))

    print("  " + "-" * (len(header) - 2))
    print(f"  {'MÉDIA':<12} {mean_ks_train:>10.3f} {mean_ks_test:>9.3f} {mean_js_train:>10.3f} {mean_js_test:>9.3f} {mean_w1_train:>10.3f} {mean_w1_test:>9.3f}")

    results["overall"] = {
        "mean_ks_train": mean_ks_train, "mean_ks_test": mean_ks_test,
        "mean_js_train": mean_js_train, "mean_js_test": mean_js_test,
        "mean_w1_train": mean_w1_train, "mean_w1_test": mean_w1_test,
        "gap_ks": mean_ks_test - mean_ks_train,
        "gap_js": mean_js_test - mean_js_train,
        "gap_w1": mean_w1_test - mean_w1_train,
    }

    print(f"\n  Gap teste-treino (positivo = pior no teste): "
          f"KS={results['overall']['gap_ks']:+.3f}  JS={results['overall']['gap_js']:+.3f}  W1={results['overall']['gap_w1']:+.3f}")

    out_path = Path(args.output) if args.output else run_dir / "fidelity_check.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados: {out_path}")


if __name__ == "__main__":
    main()
