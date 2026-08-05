#!/usr/bin/env python3
"""
Testability Regressor — continuous alternative to the binary predictor.

train_testability_predictor.py predicts the binary `testable` label
(coverage >= 0.5). On Defects4J that label is almost always 1 (98% of classes
have real developer tests covering the bug fix), so the classifier has no
negative-class signal to learn from — AUC collapses to ~random.

This regresses directly on a continuous mutation-testing outcome instead
(mutation_score by default, or coverage/test_strength), which still carries
real variance even when every class clears the coverage>=0.5 threshold.
Same anti-bias design as the classifier: leave-projects-out (GroupKFold), so a
good score proves the structural signal generalises across unseen projects.

Usage:
    python python/scripts/train_testability_regressor.py \
        --bridge outputs/mutation_experiments/bridge_dataset_111.csv \
        --target mutation_score
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from sf110_lib import FEATURES, load_bridge

SEP = "=" * 72


def _cross_project_eval(X, y, groups, model_name: str, n_splits: int):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        if model_name == "ridge":
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = RandomForestRegressor(
                n_estimators=300, max_depth=6, random_state=42, n_jobs=-1,
            )
        model.fit(X_tr, y[train_idx])

        pred = model.predict(X_te)
        oof_pred[test_idx] = pred

        rho, _ = spearmanr(y[test_idx], pred) if len(test_idx) > 1 else (float("nan"), None)
        fold_metrics.append({
            "fold": fold,
            "n_projects": len(set(groups[test_idx])),
            "n_test": len(test_idx),
            "r2": r2_score(y[test_idx], pred) if len(test_idx) > 1 else float("nan"),
            "spearman": rho,
            "mae": mean_absolute_error(y[test_idx], pred),
        })

    return fold_metrics, oof_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-project testability regressor (continuous target, no class-imbalance issue)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset.csv")
    parser.add_argument("--target", type=str, default="mutation_score",
                        choices=["mutation_score", "coverage", "test_strength"])
    parser.add_argument("--model", choices=["rf", "ridge"], default="rf")
    parser.add_argument("--model-out", type=str, default="")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        print(f"[ERRO] Bridge dataset not found: {bridge_path}")
        sys.exit(1)

    X, y, groups, rows = load_bridge(bridge_path, target=args.target)
    n_splits = min(args.n_splits, len(set(groups)))

    print(SEP)
    print("  PREDITOR DE TESTABILIDADE (REGRESSAO) — validacao leave-projects-out")
    print(SEP)
    print(f"\n  Classes: {len(y)} | Alvo: {args.target} (media={y.mean():.3f}, desvio={y.std():.3f})")
    print(f"  Projetos: {len(set(groups))} | Modelo: {args.model}")
    print(f"  Features: {FEATURES}")

    fold_metrics, oof_pred = _cross_project_eval(X, y, groups, args.model, n_splits)

    print(f"\n  {'Fold':>4} {'Projetos':>9} {'Classes':>8} {'R2':>8} {'Spearman':>9} {'MAE':>8}")
    print(f"  {'-'*52}")
    for m in fold_metrics:
        print(
            f"  {m['fold']:>4} {m['n_projects']:>9} {m['n_test']:>8} "
            f"{m['r2']:>8.3f} {m['spearman']:>9.3f} {m['mae']:>8.3f}"
        )

    mean_r2 = np.nanmean([m["r2"] for m in fold_metrics])
    mean_rho = np.nanmean([m["spearman"] for m in fold_metrics])
    mean_mae = np.mean([m["mae"] for m in fold_metrics])
    overall_r2 = r2_score(y, oof_pred)
    overall_rho, _ = spearmanr(y, oof_pred)
    print(f"  {'-'*52}")
    print(f"  Media     {'':>9} {'':>8} {mean_r2:>8.3f} {mean_rho:>9.3f} {mean_mae:>8.3f}")
    print(f"\n  R2 out-of-fold global:       {overall_r2:.3f}")
    print(f"  Spearman out-of-fold global: {overall_rho:.3f}  (cross-project, sem leakage)")

    # Feature importance via a full-data fit (interpretability)
    scaler = StandardScaler().fit(X)
    if args.model == "ridge":
        final = Ridge(alpha=1.0, random_state=42)
    else:
        final = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    final.fit(scaler.transform(X), y)

    print(f"\n  Importancia das features:")
    if args.model == "ridge":
        importances = np.abs(final.coef_)
    else:
        importances = final.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda t: -t[1]):
        bar = "#" * int(imp / max(importances) * 30) if max(importances) else ""
        print(f"    {feat:<12} {bar} {imp:.3f}")

    if args.model_out:
        model_out = Path(args.model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        with open(model_out, "wb") as f:
            pickle.dump({"model": final, "scaler": scaler, "features": FEATURES, "target": args.target}, f)
        print(f"\n  Modelo serializado (pickle): {model_out}")


if __name__ == "__main__":
    main()
