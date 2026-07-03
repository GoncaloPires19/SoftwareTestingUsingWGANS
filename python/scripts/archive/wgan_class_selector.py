#!/usr/bin/env python3
"""
WGAN-Based Guided Class Selector

The WGAN-central version of guided_class_selector.py: instead of using a
RandomForest, this script trains the project's WGAN-GP on the structural
profiles of TESTABLE classes (coverage >= threshold) and uses the trained
CRITIC as the testability score for every class in every SF110 project.

This is the legitimate WGAN <-> mutation-testing link:
  - The WGAN learns the manifold of testable-class profiles (generative).
  - The Critic scores any new class by how well it fits that manifold
    (in-distribution detection).
  - Classes the Critic ranks high are kept as PIT targets.

ANTI-BIAS:
  - The WGAN is trained on testable-class profiles from the bridge dataset
    (only classes for which we have ground-truth coverage). It then scores
    every class in the SF110 source tree (most of which it has never seen).
  - No project identity is used as a feature.

Usage:
    python python/scripts/wgan_class_selector.py \
        --bridge outputs/mutation_experiments/bridge_dataset_111.csv \
        --structural outputs/mutation_experiments/class_structural_metrics_111.csv \
        --output-dir outputs/mutation_experiments/wgan_guided_targets \
        --epochs 300 --top-percentile 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.optim as optim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import Critic, Generator
from sf110_lib import FEATURES, load_bridge
from utils import compute_feature_matching_loss, compute_gradient_penalty

Z_DIM = 32
SEP = "=" * 72


def _train_wgan_on_testable(
    X_testable: np.ndarray, n_features: int, epochs: int,
    device: torch.device, seed: int, fm_weight: float,
) -> Critic:
    """Train a WGAN-GP on testable-class profiles; return the trained critic."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    g = Generator(z_dim=Z_DIM, output_dim=n_features, hidden_dims=[64, 128]).to(device)
    c = Critic(input_dim=n_features, hidden_dims=[128, 64], use_minibatch_std=False).to(device)
    g_opt = optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_opt = optim.Adam(c.parameters(), lr=1e-4, betas=(0.5, 0.9))

    data = torch.tensor(X_testable, dtype=torch.float32, device=device)
    n = data.size(0)
    batch_size = min(32, n)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            real = data[perm[i:i + batch_size]]
            bs = real.size(0)
            if bs < 2:
                continue
            for _ in range(5):
                c_opt.zero_grad()
                z = torch.randn(bs, Z_DIM, device=device)
                fake = g(z).detach()
                loss = c(fake).mean() - c(real).mean()
                gp = compute_gradient_penalty(c, real, fake, device, bs, 10.0)
                (loss + gp).backward()
                c_opt.step()
            g_opt.zero_grad()
            z = torch.randn(bs, Z_DIM, device=device)
            fake = g(z)
            g_loss = -c(fake).mean()
            if fm_weight > 0:
                g_loss = g_loss + fm_weight * compute_feature_matching_loss(real, fake)
            g_loss.backward()
            g_opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}/{epochs}", flush=True)

    c.eval()
    return c


@torch.no_grad()
def _score(critic: Critic, X: np.ndarray, device: torch.device) -> np.ndarray:
    return critic(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy().ravel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WGAN-guided per-project class selector for PIT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset_111.csv",
                        help="Bridge dataset (used to identify TESTABLE training profiles)")
    parser.add_argument("--structural", type=str, default="outputs/mutation_experiments/class_structural_metrics_111.csv",
                        help="Structural metrics of every class to be scored")
    parser.add_argument("--output-dir", type=str, default="outputs/mutation_experiments/wgan_guided_targets")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fm-weight", type=float, default=0.3)
    parser.add_argument("--top-percentile", type=float, default=60.0,
                        help="Keep classes whose critic score is in the top X%% per project (default 60)")
    parser.add_argument("--min-classes-per-project", type=int, default=3,
                        help="Floor: keep at least N top-ranked classes even if below the percentile cut")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")

    # ----- Load testable training profiles from the bridge dataset -----
    X_bridge, y, _, _ = load_bridge(Path(args.bridge))
    X_testable = X_bridge[y == 1]
    print(SEP)
    print("  WGAN CLASS SELECTOR — critic in-distribution scoring")
    print(SEP)
    print(f"\n  Treino: {len(X_testable)} perfis de classes testáveis (do bridge)")
    print(f"  Features: {len(FEATURES)} | epochs: {args.epochs} | fm_weight: {args.fm_weight}")

    # Scale to [-1, 1] to match the Generator Tanh range. Stats fit on TESTABLE only.
    lo, hi = X_testable.min(axis=0), X_testable.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)

    def scale(X):
        return np.clip(2 * (X - lo) / rng - 1, -3, 3)

    Xtr_scaled = scale(X_testable)

    # ----- Train the WGAN-GP -----
    print(f"\n  A treinar WGAN-GP no manifold dos perfis testáveis...", flush=True)
    critic = _train_wgan_on_testable(Xtr_scaled, len(FEATURES), args.epochs, device, args.seed, args.fm_weight)

    # ----- Score every class in the SF110 source tree -----
    rows = []
    with open(args.structural, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    X_all = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float64)
    X_all_scaled = scale(X_all)
    scores = _score(critic, X_all_scaled, device)

    print(f"\n  Scored {len(rows)} classes de {len({r['project'] for r in rows})} projetos")
    print(f"  Critic score: min={scores.min():.2f} mean={scores.mean():.2f} max={scores.max():.2f}")

    # ----- Per-project selection by percentile of critic score -----
    by_project: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r, s in zip(rows, scores):
        by_project[r["project"]].append((float(s), r["class_name"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    total_kept = 0
    total_classes = 0

    for proj, scored in sorted(by_project.items()):
        scored.sort(key=lambda t: -t[0])  # highest score first
        total_classes += len(scored)

        # Keep the top X% by critic score (per project, so each project gets some targets)
        cutoff = int(np.ceil(len(scored) * args.top_percentile / 100.0))
        cutoff = max(cutoff, args.min_classes_per_project)
        kept = [c for _, c in scored[:cutoff]]

        out_file = out_dir / f"{proj}.classes"
        out_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
        total_kept += len(kept)
        summary.append((proj, len(scored), len(kept)))

    print(f"\n  {'Projeto':<28} {'Total':>6} {'Mantidas':>9} {'%':>5}")
    print("  " + "-" * 54)
    for proj, total, kept in summary[:15]:
        pct = kept / total * 100 if total else 0
        print(f"  {proj:<28} {total:>6} {kept:>9} {pct:>4.0f}%")
    if len(summary) > 15:
        print(f"  ... mais {len(summary)-15} projetos")
    print(f"\n  Total: {total_kept}/{total_classes} classes ({total_kept/total_classes*100:.0f}%)")
    print(f"  Limiar (top X%): {args.top_percentile}% | floor: {args.min_classes_per_project}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
