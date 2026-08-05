#!/usr/bin/env python3
"""
Predictor Complexity Ladder — Does the Random Forest Actually Earn Its Keep?

Compares the production RandomForest testability predictor against
progressively simpler baselines, all evaluated under the exact same
leave-projects-out (GroupKFold) protocol as train_testability_predictor.py:

    majority  -> always predicts the majority class (floor, AUC ~ 0.5)
    stump     -> a single-feature decision tree (depth 1): the simplest rule
                 a human could plausibly write by hand from the structural
                 features alone
    logreg    -> linear model over all 15 features
    rf        -> the production RandomForest (200 trees, depth 6)

If the RF does not clearly beat the stump/logreg, the added complexity is
not earning its keep.

Usage:
    python python/scripts/predictor_baselines.py --bridge outputs/mutation_experiments/bridge_dataset_111.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sf110_lib import FEATURES, load_bridge

SEP = "=" * 72

MODELS = {
    "majority": lambda seed: DummyClassifier(strategy="most_frequent"),
    "stump": lambda seed: DecisionTreeClassifier(max_depth=1, random_state=seed),
    "logreg": lambda seed: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=seed, n_jobs=-1
    ),
}


def _cross_project_auc(X, y, groups, model_name: str, n_splits: int, seed: int):
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros(len(y))
    fold_aucs = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        clf = MODELS[model_name](seed)
        clf.fit(X_tr, y[train_idx])

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_te)[:, 1]
        else:
            proba = clf.decision_function(X_te)

        oof_proba[test_idx] = proba
        if len(set(y[test_idx])) > 1:
            fold_aucs.append(roc_auc_score(y[test_idx], proba))

    overall_auc = roc_auc_score(y, oof_proba) if len(set(y)) > 1 else float("nan")
    return overall_auc, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictor complexity ladder (naive baselines vs RF)")
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset_111.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/mutation_experiments/predictor_baselines.json")
    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        print(f"[ERRO] Bridge dataset not found: {bridge_path}")
        sys.exit(1)

    X, y, groups, rows = load_bridge(bridge_path)
    n_splits = min(args.n_splits, len(set(groups)))

    print(SEP)
    print("  PREDICTOR COMPLEXITY LADDER — cross-project (leave-projects-out)")
    print(SEP)
    print(f"\n  Classes: {len(y)} | Testaveis: {y.sum()} ({y.mean()*100:.0f}%)")
    print(f"  Projetos: {len(set(groups))} | Features: {FEATURES}")

    results = {}
    print(f"\n  {'Modelo':<12} {'AUC (out-of-fold)':>18} {'Media fold':>12} {'Std fold':>10}")
    print(f"  {'-'*54}")
    for model_name in ("majority", "stump", "logreg", "rf"):
        overall_auc, mean_fold, std_fold = _cross_project_auc(X, y, groups, model_name, n_splits, args.seed)
        results[model_name] = {
            "overall_auc": overall_auc,
            "mean_fold_auc": mean_fold,
            "std_fold_auc": std_fold,
        }
        print(f"  {model_name:<12} {overall_auc:>18.3f} {mean_fold:>12.3f} {std_fold:>10.3f}")

    delta_rf_logreg = results["rf"]["overall_auc"] - results["logreg"]["overall_auc"]
    delta_rf_stump = results["rf"]["overall_auc"] - results["stump"]["overall_auc"]
    print(f"\n  RF - logreg (linear, 15 features) : {delta_rf_logreg:+.3f}")
    print(f"  RF - stump  (1 feature, depth 1)   : {delta_rf_stump:+.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados: {output_path}")


if __name__ == "__main__":
    main()
