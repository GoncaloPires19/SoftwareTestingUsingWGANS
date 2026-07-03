#!/usr/bin/env python3
"""
WGAN Ablation Study — Causal Contribution to Testability Prediction

Isolates the WGAN's contribution to the testability predictor by comparing
three conditions under identical leave-projects-out (GroupKFold) evaluation:

  COND-A  RF_only          RandomForest on 15 structural features (no WGAN)
  COND-B  RF_wgan_score    RF on 15 features + WGAN critic score as 16th feature
  COND-C  RF_wgan_aug      RF trained on real + WGAN-generated synthetic data

If COND-B > COND-A: the WGAN's *representation* helps (critic captures signal
  not in raw features).
If COND-C > COND-A: the WGAN's *generative* capacity helps (augmentation
  reduces overfitting).
If both == COND-A: the WGAN is not causally necessary for this task.

Statistical significance: Wilcoxon signed-rank test on per-fold AUC pairs
(paired, non-parametric — appropriate for small fold counts).

Usage:
    python python/scripts/wgan_ablation.py \
        --bridge outputs/mutation_experiments/bridge_dataset_111.csv \
        [--epochs 150] [--synth-ratio 1.0] [--seeds 3] [--n-splits 5]
"""

from __future__ import annotations

import argparse
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
from scipy.stats import wilcoxon

from model import Critic, Generator
from utils import compute_feature_matching_loss, compute_gradient_penalty
from sf110_lib import FEATURES, load_bridge

SEP = "=" * 72
Z_DIM = 32


class _MinMaxScaler:
    def __init__(self, X: np.ndarray):
        self.lo = X.min(axis=0)
        self.rng = np.where(X.max(axis=0) - X.min(axis=0) == 0, 1.0, X.max(axis=0) - X.min(axis=0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(2 * (X - self.lo) / self.rng - 1, -3, 3)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(2 * (X - self.lo) / self.rng - 1, -3, 3)


def _train_wgan(
    X_scaled: np.ndarray, n_features: int, epochs: int, device: torch.device, seed: int,
    fm_weight: float = 0.2,
) -> tuple[Generator, Critic]:
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
    c.eval()
    return g, c


@torch.no_grad()
def _generate(g: Generator, n: int, device: torch.device, seed: int) -> np.ndarray:
    torch.manual_seed(seed + 999)
    z = torch.randn(n, Z_DIM, device=device)
    return g(z).cpu().numpy()


@torch.no_grad()
def _critic_scores(c: Critic, X_scaled: np.ndarray, device: torch.device) -> np.ndarray:
    """Extract Wasserstein critic scores (higher = more 'in-distribution')."""
    x = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    scores = c(x).cpu().numpy().squeeze()
    return scores


def _fit_rf(X_tr, y_tr, X_te, y_te, seed):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
    return auc


def _run_ablation(
    X, y, groups, n_splits, epochs, synth_ratio, fm_weight, seed, device, verbose
) -> tuple[list[float], list[float], list[float]]:
    """
    One full ablation run for a given seed.
    Returns per-fold AUCs for (COND-A, COND-B, COND-C).
    """
    gkf = GroupKFold(n_splits=n_splits)
    aucs_a, aucs_b, aucs_c = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        scaler_std = StandardScaler().fit(X[train_idx])
        Xtr_std = scaler_std.transform(X[train_idx])
        Xte_std = scaler_std.transform(X[test_idx])
        ytr = y[train_idx]
        yte = y[test_idx]

        # COND-A: RF on structural features only
        auc_a = _fit_rf(Xtr_std, ytr, Xte_std, yte, seed)
        aucs_a.append(auc_a)

        # Train one WGAN per class (testable / not testable) for COND-B and COND-C
        mm = _MinMaxScaler(X[train_idx])
        Xtr_mm = mm.fit_transform(X[train_idx])
        Xte_mm = mm.transform(X[test_idx])

        # Train two WGANs (one per class label)
        wgans: dict[int, tuple[Generator, Critic]] = {}
        synth_X, synth_y = [], []

        for label in (0, 1):
            X_class = Xtr_mm[ytr == label]
            if len(X_class) < 10:
                continue
            g, c = _train_wgan(X_class, len(FEATURES), epochs, device, seed + label * 100, fm_weight)
            wgans[label] = (g, c)

            # Synthetic data for COND-C
            n_gen = max(10, int(len(X_class) * synth_ratio))
            gen = _generate(g, n_gen, device, seed + label)
            synth_X.append(gen)
            synth_y.append(np.full(len(gen), label))

        # COND-B: RF on structural features + critic score as extra feature
        # Use the testable-class WGAN critic (higher score = more like testable classes)
        if 1 in wgans:
            _, c_testable = wgans[1]
            tr_scores = _critic_scores(c_testable, Xtr_mm, device)
            te_scores = _critic_scores(c_testable, Xte_mm, device)
            # Standardize the added critic score feature
            score_mean = tr_scores.mean()
            score_std = max(tr_scores.std(), 1e-8)
            tr_scores_std = (tr_scores - score_mean) / score_std
            te_scores_std = (te_scores - score_mean) / score_std
            Xtr_b = np.column_stack([Xtr_std, tr_scores_std])
            Xte_b = np.column_stack([Xte_std, te_scores_std])
            auc_b = _fit_rf(Xtr_b, ytr, Xte_b, yte, seed)
        else:
            auc_b = auc_a  # no WGAN trained, fall back
        aucs_b.append(auc_b)

        # COND-C: RF on real + synthetic data (WGAN augmented)
        if synth_X:
            Xtr_aug = np.vstack([Xtr_std] + [
                StandardScaler().fit(Xtr_std).transform(
                    np.column_stack([s]) if s.ndim > 1 else s.reshape(-1, 1)
                ) if s.shape[1] != Xtr_std.shape[1] else s
                for s in synth_X
            ])
            # Simpler: augment in mm-scaled space, then re-standardize
            Xtr_aug_mm = np.vstack([Xtr_mm] + synth_X)
            ytr_aug = np.concatenate([ytr] + synth_y)
            # Re-scale augmented set via its own standard scaler
            scaler_aug = StandardScaler().fit(Xtr_aug_mm)
            Xtr_aug_std = scaler_aug.transform(Xtr_aug_mm)
            Xte_aug_std = scaler_aug.transform(Xte_mm)
            auc_c = _fit_rf(Xtr_aug_std, ytr_aug, Xte_aug_std, yte, seed)
        else:
            auc_c = auc_a
        aucs_c.append(auc_c)

        if verbose:
            print(f"    Fold {fold}: A={auc_a:.3f}  B={auc_b:.3f}({auc_b-auc_a:+.3f})  "
                  f"C={auc_c:.3f}({auc_c-auc_a:+.3f})", flush=True)

    return aucs_a, aucs_b, aucs_c


def _wilcoxon_or_na(a: list[float], b: list[float]) -> str:
    """Return p-value string for Wilcoxon test, or 'n/a' if not applicable."""
    diffs = [x - y for x, y in zip(b, a)]
    if all(d == 0 for d in diffs):
        return "p=1.000 (identical)"
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return f"p={p:.3f}"
    except Exception:
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WGAN ablation study — isolates WGAN causal contribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset_111.csv")
    parser.add_argument("--epochs", type=int, default=150, help="WGAN epochs per class per fold")
    parser.add_argument("--synth-ratio", type=float, default=1.0,
                        help="Synthetic per class = ratio x real count")
    parser.add_argument("--fm-weight", type=float, default=0.2, help="Feature matching weight")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for stability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bridge = Path(args.bridge)
    if not bridge.exists():
        print(f"[ERRO] Bridge not found: {bridge}")
        sys.exit(1)

    device = torch.device("cpu")
    X, y, groups, _ = load_bridge(bridge)
    n_splits = min(args.n_splits, len(set(groups)))

    print(SEP)
    print("  WGAN ABLATION STUDY — Contribuicao causal para a previsao de testabilidade")
    print(SEP)
    print(f"\n  Classes: {len(y)} | Projetos: {len(set(groups))} | Splits: {n_splits}")
    print(f"  Condicoes:")
    print(f"    COND-A  RF_only       : RF nos 15 features estruturais (sem WGAN)")
    print(f"    COND-B  RF+critic     : RF nos 15 features + score do critico WGAN")
    print(f"    COND-C  RF+augmented  : RF nos dados reais + dados sinteticos da WGAN")
    print(f"  Seeds: {args.seeds} | Epochs WGAN/fold: {args.epochs}")

    seed_results: list[tuple[list, list, list]] = []
    for i in range(args.seeds):
        s = args.seed + i
        print(f"\n  [Seed {s}]", flush=True)
        aucs_a, aucs_b, aucs_c = _run_ablation(
            X, y, groups, n_splits, args.epochs,
            args.synth_ratio, args.fm_weight, s, device, args.verbose,
        )
        seed_results.append((aucs_a, aucs_b, aucs_c))
        ga = np.nanmean(aucs_a)
        gb = np.nanmean(aucs_b)
        gc = np.nanmean(aucs_c)
        print(f"  seed {s}: COND-A={ga:.3f}  COND-B={gb:.3f}({gb-ga:+.3f})  "
              f"COND-C={gc:.3f}({gc-ga:+.3f})", flush=True)

    # Aggregate across seeds
    all_a = [auc for r in seed_results for auc in r[0]]
    all_b = [auc for r in seed_results for auc in r[1]]
    all_c = [auc for r in seed_results for auc in r[2]]

    mean_a = np.nanmean(all_a)
    mean_b = np.nanmean(all_b)
    mean_c = np.nanmean(all_c)
    std_a = np.nanstd(all_a)
    std_b = np.nanstd(all_b)
    std_c = np.nanstd(all_c)

    print(f"\n{SEP}")
    print("  RESULTADO FINAL — AUC cross-project, media sobre seeds e folds")
    print(SEP)
    print(f"\n  {'Condicao':<22} {'AUC medio':>12} {'±std':>8} {'vs A':>8}")
    print(f"  {'-'*52}")
    print(f"  {'COND-A  RF_only':<22} {mean_a:>12.3f} {std_a:>7.3f}")
    print(f"  {'COND-B  RF+critic':<22} {mean_b:>12.3f} {std_b:>7.3f} {mean_b-mean_a:>+7.3f}")
    print(f"  {'COND-C  RF+augmented':<22} {mean_c:>12.3f} {std_c:>7.3f} {mean_c-mean_a:>+7.3f}")

    # Wilcoxon on pooled fold-level AUCs
    p_ab = _wilcoxon_or_na(all_a, all_b)
    p_ac = _wilcoxon_or_na(all_a, all_c)

    print(f"\n  Teste de Wilcoxon (A vs B): {p_ab}")
    print(f"  Teste de Wilcoxon (A vs C): {p_ac}")

    print(f"\n{SEP}")
    print("  INTERPRETACAO")
    print(SEP)

    delta_b = mean_b - mean_a
    delta_c = mean_c - mean_a

    if delta_b > 0.01 and "p=0." in p_ab and float(p_ab.split("p=")[1].split(" ")[0]) < 0.05:
        print("\n  COND-B > COND-A (significativo): o score do critico WGAN captura")
        print("  informacao adicional alem das features estruturais brutas.")
        print("  A WGAN contribui atraves da sua representacao aprendida.")
    elif delta_b > 0.005:
        print("\n  COND-B > COND-A (tendencia, nao significativo): o critico WGAN")
        print("  pode capturar sinal adicional, mas os dados sao insuficientes para")
        print("  confirmacao estatistica.")
    else:
        print("\n  COND-B ≈ COND-A: o score do critico WGAN nao acrescenta informacao")
        print("  alem das 15 features estruturais. A representacao latente da WGAN")
        print("  nao e superior a engenharia de features directa para esta tarefa.")

    if delta_c > 0.01 and "p=0." in p_ac and float(p_ac.split("p=")[1].split(" ")[0]) < 0.05:
        print("\n  COND-C > COND-A (significativo): a augmentacao com dados sinteticos")
        print("  WGAN melhora o preditor. A WGAN contribui atraves da sua capacidade")
        print("  generativa para superar a escassez de dados.")
    elif delta_c > 0.005:
        print("\n  COND-C > COND-A (tendencia): augmentacao mostra ganho positivo mas")
        print("  abaixo do limiar de significancia estatistica.")
    else:
        print("\n  COND-C ≈ COND-A: augmentacao WGAN nao melhora o preditor de forma")
        print("  robusta. O RF ja generaliza bem com os dados reais disponiveis.")
        print("  Resultado negativo valido: discriminativo vence one-class quando ha labels.")

    print(f"\n  Conclusao para a tese:")
    if delta_b <= 0.005 and delta_c <= 0.005:
        print("  A WGAN nao tem contribuicao causal directa no preditor de testabilidade.")
        print("  A contribuicao e INDIRECTA: os perfis gerados pela WGAN guiaram a")
        print("  seleccao de features e a compreensao da distribuicao de testabilidade,")
        print("  mas o RandomForest sobre features estruturais e suficiente para o")
        print("  desempenho obtido. Este e um resultado negativo valido e honesto.")
    elif max(delta_b, delta_c) > 0.01:
        print("  A WGAN tem contribuicao causal mensuravel (COND-{} > COND-A).".format(
            "B" if delta_b > delta_c else "C"
        ))
        print("  O ganho justifica a sua inclusao no pipeline.")
    else:
        print("  A contribuicao da WGAN e marginal. Reportar como resultado parcialmente")
        print("  positivo com as ressalvas do tamanho do efeito e variancia entre seeds.")


if __name__ == "__main__":
    main()
