#!/usr/bin/env python3
"""
WGAN-Central Testability Prediction via Synthetic Data Augmentation

This is the centrepiece linking the project's WGAN to the mutation-testing goal.
It combines the WGAN's original generative purpose with data augmentation:

  (3) GENERATE  — A WGAN-GP learns the distribution of class structural profiles
                  and generates SYNTHETIC profiles for each testability class
                  (testable / untestable). This is the WGAN doing its core job.
  (1) AUGMENT   — The synthetic profiles enlarge the small (532-class) training
                  set. The testability predictor is then trained on real +
                  synthetic data, addressing the data-scarcity bottleneck.

We measure whether WGAN augmentation improves the CROSS-PROJECT testability
predictor (AUC on unseen projects) and the resulting projected mutation score
after guided target selection.

ANTI-BIAS DESIGN:
  - Leave-projects-out (GroupKFold). Per fold, the WGAN is trained ONLY on
    training-project classes; synthetic data is generated from that; the
    predictor is evaluated on unseen test projects. No test data ever touches
    WGAN training or augmentation.
  - Two WGANs per fold (one per class) so augmentation is class-conditional and
    does not distort the class ratio.

Usage:
    python python/scripts/wgan_augmented_predictor.py \
        --bridge outputs/mutation_experiments/bridge_dataset.csv \
        --epochs 150 --synth-per-class 300
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

from model import Critic, Generator
from utils import (
    compute_feature_matching_loss,
    compute_gradient_penalty,
    compute_wasserstein_distance,
)
from sf110_lib import FEATURES, killed_relevant, load_bridge

SEP = "=" * 72
Z_DIM = 32


class _MinMaxTanh:
    """Scale to [-1, 1] (generator Tanh range) and back."""
    def __init__(self, X: np.ndarray):
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        self.rng = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.clip(2 * (X - self.lo) / self.rng - 1, -3, 3)

    def inverse(self, Xs: np.ndarray) -> np.ndarray:
        return (Xs + 1) / 2 * self.rng + self.lo


def _train_wgan(
    X_scaled: np.ndarray, n_features: int, epochs: int, device: torch.device,
    seed: int, fm_weight: float = 0.2,
) -> Generator:
    """
    Train a WGAN-GP on a single class's profiles; return the generator.

    Quality improvements over the bare baseline:
      - Feature matching loss (weight fm_weight) so the generator reproduces the
        real per-feature mean/std — directly improves synthetic fidelity.
      - More critic capacity and proper Adam betas for WGAN-GP.
    """
    torch.manual_seed(seed)
    g = Generator(z_dim=Z_DIM, output_dim=n_features, hidden_dims=[64, 128]).to(device)
    c = Critic(input_dim=n_features, hidden_dims=[128, 64], use_minibatch_std=False).to(device)
    g_opt = optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_opt = optim.Adam(c.parameters(), lr=1e-4, betas=(0.5, 0.9))

    data = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    n = data.size(0)
    batch_size = min(32, n)

    for _ in range(epochs):
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

    g.eval()
    return g


@torch.no_grad()
def _generate(g: Generator, n: int, device: torch.device, seed: int) -> np.ndarray:
    torch.manual_seed(seed + 999)
    z = torch.randn(n, Z_DIM, device=device)
    return g(z).cpu().numpy()


def _fit_eval_rf(X_tr, y_tr, X_te, y_te, seed):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
    return auc, proba


def _run_once(X, y, groups, rows, n_splits, epochs, synth_ratio, fm_weight, seed, device, verbose):
    """
    Run one full cross-project experiment for a single seed.

    Returns:
        g_base, g_aug : global out-of-fold AUC (baseline / WGAN-augmented)
        mean_fidelity : mean synthetic W1 fidelity
        oof_base, oof_aug : out-of-fold probability arrays (for downstream scoring)
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_base = np.zeros(len(y))
    oof_aug = np.zeros(len(y))
    fidelity_log: list[float] = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        if verbose:
            print(f"\n  Fold {fold}/{n_splits}: a treinar WGANs e a augmentar...", flush=True)

        scaler = _MinMaxTanh(X[train_idx])
        Xtr = scaler.forward(X[train_idx])
        Xte = scaler.forward(X[test_idx])
        ytr = y[train_idx]
        yte = y[test_idx]

        base_auc, base_proba = _fit_eval_rf(Xtr, ytr, Xte, yte, seed)
        oof_base[test_idx] = base_proba

        synth_X, synth_y = [], []
        for label in (0, 1):
            X_class = Xtr[ytr == label]
            if len(X_class) < 10:
                continue
            g = _train_wgan(X_class, len(FEATURES), epochs, device, seed + label, fm_weight)
            n_gen = max(10, int(len(X_class) * synth_ratio))
            gen = _generate(g, n_gen, device, seed + label)
            fid = compute_wasserstein_distance(gen, X_class)
            fidelity_log.append(fid)
            synth_X.append(gen)
            synth_y.append(np.full(len(gen), label))
            if verbose:
                print(f"    classe {label}: {len(X_class)} reais -> +{len(gen)} sinteticos (W1={fid:.3f})", flush=True)

        if synth_X:
            Xtr_aug = np.vstack([Xtr] + synth_X)
            ytr_aug = np.concatenate([ytr] + synth_y)
        else:
            Xtr_aug, ytr_aug = Xtr, ytr

        aug_auc, aug_proba = _fit_eval_rf(Xtr_aug, ytr_aug, Xte, yte, seed)
        oof_aug[test_idx] = aug_proba

        fold_rows.append((fold, len(test_idx), base_auc, aug_auc))
        if verbose:
            delta = aug_auc - base_auc
            arrow = "UP" if delta > 0.005 else ("DOWN" if delta < -0.005 else "==")
            print(f"    AUC: baseline={base_auc:.3f} -> augmentado={aug_auc:.3f}  ({arrow} {delta:+.3f})")

    valid = (oof_base != 0) | (oof_aug != 0)
    g_base = roc_auc_score(y[valid], oof_base[valid]) if len(set(y[valid])) > 1 else float("nan")
    g_aug = roc_auc_score(y[valid], oof_aug[valid]) if len(set(y[valid])) > 1 else float("nan")

    if verbose:
        print(f"\n  {'Fold':>4} {'Classes':>8} {'Baseline':>10} {'WGAN-aug':>10} {'Delta':>8}")
        print(f"  {'-'*44}")
        for fold, n_test, b, a in fold_rows:
            print(f"  {fold:>4} {n_test:>8} {b:>10.3f} {a:>10.3f} {a-b:>+8.3f}")

    return g_base, g_aug, float(np.mean(fidelity_log)) if fidelity_log else 0.0, oof_base, oof_aug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WGAN-augmented cross-project testability prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset.csv")
    parser.add_argument("--epochs", type=int, default=200, help="WGAN epochs per class per fold")
    parser.add_argument("--synth-ratio", type=float, default=1.0,
                        help="Synthetic samples per class = ratio x real count (default 1.0)")
    parser.add_argument("--fm-weight", type=float, default=0.2,
                        help="Feature matching loss weight for synthetic fidelity (0=off)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Run the whole experiment with N seeds and report mean +/- std (stability check)")
    args = parser.parse_args()

    bridge = Path(args.bridge)
    if not bridge.exists():
        print(f"[ERRO] Bridge dataset not found: {bridge}")
        sys.exit(1)

    device = torch.device("cpu")
    X, y, groups, rows = load_bridge(bridge)
    n_splits = min(args.n_splits, len(set(groups)))

    print(SEP)
    print("  WGAN-CENTRAL — augmentacao sintetica para previsao de testabilidade")
    print(SEP)
    print(f"\n  Classes reais: {len(y)} | Testaveis: {y.sum()} ({y.mean()*100:.0f}%)")
    print(f"  Projetos: {len(set(groups))}")
    print(f"  WGAN: {args.epochs} epochs/classe/fold | ratio={args.synth_ratio}x | feature_matching={args.fm_weight}")

    # -------------------------------------------------------------------
    # Multi-seed stability check
    # -------------------------------------------------------------------
    if args.seeds > 1:
        seeds = [args.seed + i for i in range(args.seeds)]
        print(f"  Validacao de estabilidade: {len(seeds)} seeds {seeds}\n")
        records = []
        for s in seeds:
            gb, ga, fid, _, _ = _run_once(
                X, y, groups, rows, n_splits, args.epochs,
                args.synth_ratio, args.fm_weight, s, device, verbose=False,
            )
            records.append((s, gb, ga, ga - gb, fid))
            print(f"  seed {s}: baseline={gb:.3f}  WGAN={ga:.3f}  delta={ga-gb:+.3f}  fidelidade={fid:.3f}", flush=True)

        deltas = np.array([r[3] for r in records])
        bases = np.array([r[1] for r in records])
        augs = np.array([r[2] for r in records])
        n_pos = int((deltas > 0).sum())

        print(f"\n{SEP}")
        print("  ESTABILIDADE (media +/- desvio sobre seeds)")
        print(SEP)
        print(f"\n  AUC baseline    : {bases.mean():.3f} +/- {bases.std():.3f}")
        print(f"  AUC WGAN-aug    : {augs.mean():.3f} +/- {augs.std():.3f}")
        print(f"  Delta (ganho)   : {deltas.mean():+.3f} +/- {deltas.std():.3f}")
        print(f"  Seeds com ganho : {n_pos}/{len(seeds)}")
        # Simple consistency verdict
        if deltas.mean() > 0 and deltas.mean() > deltas.std():
            print(f"\n  >>> Ganho CONSISTENTE: a media do delta excede o desvio-padrao.")
        elif n_pos >= len(seeds) * 0.6:
            print(f"\n  >>> Ganho MAioritariamente positivo mas com variancia — reportar media +/- desvio.")
        else:
            print(f"\n  >>> Ganho NAO robusto entre seeds — o efeito da augmentacao e ruidoso.")
        return

    # -------------------------------------------------------------------
    # Single-seed run with full reporting
    # -------------------------------------------------------------------
    g_base, g_aug, mean_fid, oof_base, oof_aug = _run_once(
        X, y, groups, rows, n_splits, args.epochs,
        args.synth_ratio, args.fm_weight, args.seed, device, verbose=True,
    )

    print(f"\n{SEP}")
    print("  RESULTADO — augmentacao WGAN (cross-project, out-of-fold)")
    print(SEP)
    print(f"  AUC GLOBAL out-of-fold:")
    print(f"    Baseline (sem WGAN) : {g_base:.3f}")
    print(f"    WGAN-augmentado     : {g_aug:.3f}   ({g_aug-g_base:+.3f})")
    print(f"  Fidelidade media dos sinteticos (W1, menor=melhor): {mean_fid:.3f}")

    # --- Downstream mutation-score impact of WGAN-augmented selection ---
    print(f"\n{SEP}")
    print("  IMPACTO NO MUTATION SCORE — selecao guiada pelo preditor WGAN-augmentado")
    print(SEP)

    all_rel = sum(killed_relevant(r)[1] for r in rows)

    def score_at(proba, thr):
        kept = [rows[i] for i in range(len(rows)) if proba[i] >= thr]
        if not kept:
            return 0, 0, 0
        k = sum(killed_relevant(r)[0] for r in kept)
        rel = sum(killed_relevant(r)[1] for r in kept)
        glob = k / rel * 100 if rel else 0
        by_proj = {}
        for r in kept:
            by_proj.setdefault(r["project"], []).append(r)
        ps = []
        for rs in by_proj.values():
            kk = sum(killed_relevant(x)[0] for x in rs)
            rr = sum(killed_relevant(x)[1] for x in rs)
            if rr > 0:
                ps.append(kk / rr * 100)
        avg = sum(ps) / len(ps) if ps else 0
        return glob, avg, rel / all_rel * 100

    print(f"\n  {'Limiar':>7} {'%Mut(base)':>11} {'Media/proj base':>16} {'Media/proj WGAN':>16}")
    print(f"  {'-'*54}")
    for thr in [0.4, 0.5, 0.6, 0.7]:
        gb, ab, mb = score_at(oof_base, thr)
        ga, aa, ma = score_at(oof_aug, thr)
        print(f"  {thr:>7.1f} {mb:>10.0f}% {ab:>15.1f}% {aa:>15.1f}%")

    print(f"\n  Leitura: se a coluna WGAN >= baseline, a augmentacao da WGAN melhora a")
    print(f"  selecao de alvos -> a WGAN contribui diretamente para o mutation testing.")


if __name__ == "__main__":
    main()
