#!/usr/bin/env python3
"""
WGAN-Based Testability Scorer (the genuine WGAN <-> test-generation link)

This is the honest connection between the project's WGAN and mutation-test
targeting that earlier scripts only approximated with a RandomForest.

Method (AnoGAN-style anomaly detection with a WGAN-GP critic):
  1. Train the WGAN-GP ONLY on the structural profiles of TESTABLE classes
     (coverage >= threshold). The generator learns to synthesise "what a
     testable class looks like"; the critic learns the manifold of testable
     profiles.
  2. Use the trained CRITIC as an in-distribution score. A class whose profile
     lies on the testable manifold gets a high critic score; an anomalous
     (untestable: UI/network) class gets a low score.
  3. The critic score becomes the target-selection signal: target the classes
     the WGAN recognises as testable.

ANTI-BIAS DESIGN:
  - Leave-projects-out (GroupKFold). The WGAN is trained on testable classes of
    the TRAINING projects only, then scores classes of unseen TEST projects.
  - Critic scores are z-normalised per fold before pooling (the absolute critic
    scale differs across folds; ranking is what matters).

Outputs a direct AUC comparison: WGAN-critic vs RandomForest, plus the
score-vs-coverage tradeoff at the chosen operating points.

Usage:
    python python/scripts/wgan_testability_scorer.py \
        --bridge outputs/mutation_experiments/bridge_dataset.csv \
        --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from model import Critic, Generator
from utils import compute_gradient_penalty
from sf110_lib import FEATURES, killed_relevant, load_bridge

SEP = "=" * 72


def _minmax_to_tanh(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Scale features to [-1, 1] to match the Tanh generator output range."""
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return 2 * (X - lo) / rng - 1


def _train_wgan_on_testable(
    X_testable: np.ndarray,
    n_features: int,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Critic:
    """Train a WGAN-GP on testable-class profiles; return the trained critic."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    z_dim = 32
    generator = Generator(z_dim=z_dim, output_dim=n_features, hidden_dims=[64, 128]).to(device)
    # use_minibatch_std=False: the critic must score samples independently
    critic = Critic(input_dim=n_features, hidden_dims=[128, 64], use_minibatch_std=False).to(device)

    g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_opt = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    data = torch.tensor(X_testable, dtype=torch.float32, device=device)
    n = data.size(0)
    batch_size = min(32, n)
    n_critic = 5
    lambda_gp = 10.0

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            real = data[idx]
            bs = real.size(0)
            if bs < 2:
                continue

            for _ in range(n_critic):
                c_opt.zero_grad()
                z = torch.randn(bs, z_dim, device=device)
                fake = generator(z).detach()
                c_loss = critic(fake).mean() - critic(real).mean()
                gp = compute_gradient_penalty(critic, real, fake, device, bs, lambda_gp)
                (c_loss + gp).backward()
                c_opt.step()

            g_opt.zero_grad()
            z = torch.randn(bs, z_dim, device=device)
            g_loss = -critic(generator(z)).mean()
            g_loss.backward()
            g_opt.step()

    critic.eval()
    return critic


@torch.no_grad()
def _critic_scores(critic: Critic, X: np.ndarray, device: torch.device) -> np.ndarray:
    t = torch.tensor(X, dtype=torch.float32, device=device)
    return critic(t).cpu().numpy().ravel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WGAN-critic testability scorer vs RandomForest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bridge = Path(args.bridge)
    if not bridge.exists():
        print(f"[ERRO] Bridge dataset not found: {bridge}")
        sys.exit(1)

    device = torch.device("cpu")
    X, y, groups, rows = load_bridge(bridge)

    print(SEP)
    print("  WGAN TESTABILITY SCORER — critic como detetor in-distribution")
    print(SEP)
    print(f"\n  Classes: {len(y)} | Testaveis: {y.sum()} ({y.mean()*100:.0f}%)")
    print(f"  Projetos: {len(set(groups))} | WGAN epochs/fold: {args.epochs}")
    print(f"  Features: {len(FEATURES)}")

    n_splits = min(args.n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    oof_wgan = np.zeros(len(y))
    oof_rf = np.zeros(len(y))
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"\n  Fold {fold}/{n_splits} — a treinar WGAN nos perfis testaveis...", flush=True)

        # Scaling fit on TRAINING testable classes only (no leakage)
        train_testable_mask = y[train_idx] == 1
        X_train_testable = X[train_idx][train_testable_mask]
        if len(X_train_testable) < 10:
            print(f"    (poucas classes testaveis no treino: {len(X_train_testable)}) — a saltar fold")
            continue

        lo = X_train_testable.min(axis=0)
        hi = X_train_testable.max(axis=0)
        Xtr_scaled = _minmax_to_tanh(X_train_testable, lo, hi)
        Xte_scaled = _minmax_to_tanh(X[test_idx], lo, hi)
        Xte_scaled = np.clip(Xte_scaled, -3, 3)  # guard against extreme OOD values

        # --- WGAN critic ---
        critic = _train_wgan_on_testable(Xtr_scaled, len(FEATURES), args.epochs, device, args.seed)
        wgan_scores = _critic_scores(critic, Xte_scaled, device)
        # z-normalise within fold (absolute critic scale is fold-specific)
        wgan_scores = (wgan_scores - wgan_scores.mean()) / (wgan_scores.std() + 1e-8)
        oof_wgan[test_idx] = wgan_scores

        # --- RandomForest baseline (same fold) ---
        rf_scaler = StandardScaler().fit(X[train_idx])
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=args.seed, n_jobs=-1
        )
        rf.fit(rf_scaler.transform(X[train_idx]), y[train_idx])
        rf_scores = rf.predict_proba(rf_scaler.transform(X[test_idx]))[:, 1]
        oof_rf[test_idx] = rf_scores

        yte = y[test_idx]
        wgan_auc = roc_auc_score(yte, wgan_scores) if len(set(yte)) > 1 else float("nan")
        rf_auc = roc_auc_score(yte, rf_scores) if len(set(yte)) > 1 else float("nan")
        fold_rows.append((fold, len(test_idx), wgan_auc, rf_auc))
        print(f"    Fold {fold}: WGAN-critic AUC={wgan_auc:.3f} | RandomForest AUC={rf_auc:.3f}")

    # --- Summary ---
    print(f"\n{SEP}")
    print("  COMPARACAO AUC (cross-project, out-of-fold)")
    print(SEP)
    print(f"\n  {'Fold':>4} {'Classes':>8} {'WGAN-critic':>12} {'RandomForest':>13}")
    print(f"  {'-'*40}")
    for fold, n_test, w, r in fold_rows:
        print(f"  {fold:>4} {n_test:>8} {w:>12.3f} {r:>13.3f}")

    valid = (oof_wgan != 0) | (oof_rf != 0)
    global_wgan = roc_auc_score(y[valid], oof_wgan[valid]) if len(set(y[valid])) > 1 else float("nan")
    global_rf = roc_auc_score(y[valid], oof_rf[valid]) if len(set(y[valid])) > 1 else float("nan")
    print(f"  {'-'*40}")
    print(f"  AUC GLOBAL out-of-fold:")
    print(f"    WGAN-critic : {global_wgan:.3f}")
    print(f"    RandomForest: {global_rf:.3f}")

    # --- Operating-point tradeoff for the WGAN scorer ---
    print(f"\n{SEP}")
    print("  TRADE-OFF do WGAN-critic (selecao por percentil de score)")
    print(SEP)

    print(f"\n  {'Percentil':>9} {'Classes':>8} {'%Mut':>6} {'Score glob':>11} {'Media/proj':>11}")
    print(f"  {'-'*50}")
    all_rel = sum(killed_relevant(r)[1] for r in rows)
    order = np.argsort(-oof_wgan)  # highest critic score first
    for pct in [100, 80, 60, 40, 30, 20, 10]:
        keep_n = max(1, int(len(rows) * pct / 100))
        keep_idx = set(order[:keep_n].tolist())
        kept = [rows[i] for i in range(len(rows)) if i in keep_idx]
        k = sum(killed_relevant(r)[0] for r in kept)
        rel = sum(killed_relevant(r)[1] for r in kept)
        glob = k / rel * 100 if rel else 0
        by_proj = {}
        for r in kept:
            by_proj.setdefault(r["project"], []).append(r)
        proj_scores = []
        for rs in by_proj.values():
            kk = sum(killed_relevant(x)[0] for x in rs)
            rr = sum(killed_relevant(x)[1] for x in rs)
            if rr > 0:
                proj_scores.append(kk / rr * 100)
        avg = sum(proj_scores) / len(proj_scores) if proj_scores else 0
        pct_mut = rel / all_rel * 100 if all_rel else 0
        print(f"  {pct:>8}% {len(kept):>8} {pct_mut:>5.0f}% {glob:>10.1f}% {avg:>10.1f}%")

    print(f"\n  Conclusao: comparar a coluna WGAN-critic vs RandomForest acima.")
    print(f"  Se RF >= WGAN, a abordagem discriminativa vence (resultado valido p/ a tese).")


if __name__ == "__main__":
    main()
