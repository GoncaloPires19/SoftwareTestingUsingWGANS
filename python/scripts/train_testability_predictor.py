#!/usr/bin/env python3
"""
Testability Predictor — WGAN-Guided Target Selection

Trains a model that predicts, from a class's STRUCTURE alone (no PIT run needed),
whether unit tests can meaningfully cover it. This drives smarter PIT target
selection: skip the inherently untestable classes (network/IO/UI) that only add
no_coverage noise, and focus mutation testing on classes where the score
reflects genuine test quality.

ANTI-BIAS DESIGN (critical for the dissertation):
  - Validation is LEAVE-PROJECTS-OUT (GroupKFold by project). The model is always
    evaluated on projects it never saw in training, so a high score proves the
    structural signal generalises — it is NOT memorising specific projects.
  - Features are source-structural only (methods, branches, statements, loc,
    n_fields, has_io). No project identity, no post-hoc leakage.

It then quantifies the projected dual-metric improvement: if PIT only targets
classes the model predicts testable, what happens to (a) the global score and
(b) the per-project average — measured on held-out folds only.

Usage:
    python python/scripts/train_testability_predictor.py \
        --bridge outputs/mutation_experiments/bridge_dataset.csv \
        --model-out outputs/mutation_experiments/testability_model.json
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from sf110_lib import FEATURES, killed_relevant, load_bridge, per_project_average, total_relevant

SEP = "=" * 72


def _cross_project_eval(X, y, groups, model_name: str, n_splits: int = 5):
    """Leave-projects-out cross-validation. Returns fold metrics and out-of-fold preds."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(y))
    oof_proba = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        if model_name == "logreg":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        else:
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=6, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
        clf.fit(X_tr, y[train_idx])

        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        oof_pred[test_idx] = pred
        oof_proba[test_idx] = proba

        test_projects = sorted(set(groups[test_idx]))
        auc = roc_auc_score(y[test_idx], proba) if len(set(y[test_idx])) > 1 else float("nan")
        fold_metrics.append({
            "fold": fold,
            "n_test": len(test_idx),
            "n_projects": len(test_projects),
            "auc": auc,
            "accuracy": accuracy_score(y[test_idx], pred),
            "precision": precision_score(y[test_idx], pred, zero_division=0),
            "recall": recall_score(y[test_idx], pred, zero_division=0),
        })

    return fold_metrics, oof_pred, oof_proba


def _threshold_sweep(rows: list[dict], oof_proba: np.ndarray) -> None:
    """
    Score-vs-coverage tradeoff. As the probability threshold rises, we keep only
    classes the model is confident are testable: the score on the kept subset
    rises, but we cover fewer classes. Honest operating-point selection.
    """
    print(f"\n{SEP}")
    print("  TRADE-OFF SCORE vs COBERTURA (out-of-fold, cross-project)")
    print(SEP)

    all_relevant = sum(killed_relevant(r)[1] for r in rows)

    print(f"\n  {'Limiar':>7} {'Classes':>8} {'%Mut':>6} {'Score glob':>11} {'Media/proj':>11} {'>=70%?':>7}")
    print(f"  {'-'*56}")
    for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        kept = [r for r, p in zip(rows, oof_proba) if p >= thr]
        if not kept:
            continue
        k = sum(killed_relevant(r)[0] for r in kept)
        rel = sum(killed_relevant(r)[1] for r in kept)
        glob = k / rel * 100 if rel else 0
        avg, n_proj = per_project_average(kept)
        pct_mut = rel / all_relevant * 100 if all_relevant else 0
        flag = "SIM" if avg >= 70 else ""
        print(f"  {thr:>7.1f} {len(kept):>8} {pct_mut:>5.0f}% {glob:>10.1f}% {avg:>10.1f}% {flag:>7}")

    print(f"\n  Leitura: limiares mais altos => menos classes, mas score mais alto.")
    print(f"  O ponto de operacao reporta-se de forma transparente (score X% cobrindo Y% das mutacoes).")


def _projected_improvement(rows: list[dict], oof_pred: np.ndarray) -> None:
    """
    Quantify the dual-metric impact of WGAN-guided target selection, measured on
    out-of-fold predictions only (so it reflects unseen-project generalisation).
    """
    print(f"\n{SEP}")
    print("  IMPACTO PROJETADO — selecao de alvos guiada (out-of-fold, limiar 0.5)")
    print(SEP)

    all_killed = sum(killed_relevant(r)[0] for r in rows)
    all_relevant = sum(killed_relevant(r)[1] for r in rows)
    base_global = all_killed / all_relevant * 100 if all_relevant else 0

    kept = [r for r, p in zip(rows, oof_pred) if p == 1]
    guided_killed = sum(killed_relevant(r)[0] for r in kept)
    guided_relevant = sum(killed_relevant(r)[1] for r in kept)
    guided_global = guided_killed / guided_relevant * 100 if guided_relevant else 0

    print(f"\n  {'':<34}{'BASELINE':>12}{'GUIADO':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Classes alvo':<34}{len(rows):>12}{len(kept):>12}")
    print(f"  {'Mutacoes relevantes':<34}{all_relevant:>12}{guided_relevant:>12}")
    print(f"  {'Score GLOBAL ponderado':<34}{base_global:>11.1f}%{guided_global:>11.1f}%")

    base_avg, base_n = per_project_average(rows)
    guided_avg, guided_n = per_project_average(kept)
    print(f"  {'Media por-projeto':<34}{base_avg:>11.1f}%{guided_avg:>11.1f}%")
    print(f"  {'Projetos com alvos':<34}{base_n:>12}{guided_n:>12}")

    print(f"\n  >>> Selecao guiada (limiar 0.5) sobe a media por-projeto de {base_avg:.1f}% para {guided_avg:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train cross-project testability predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/bridge_dataset.csv")
    parser.add_argument("--model", choices=["logreg", "rf"], default="rf")
    parser.add_argument("--model-out", type=str, default="outputs/mutation_experiments/testability_model.json")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        print(f"[ERRO] Bridge dataset not found: {bridge_path}")
        sys.exit(1)

    X, y, groups, rows = load_bridge(bridge_path)

    print(SEP)
    print("  PREDITOR DE TESTABILIDADE — validacao leave-projects-out")
    print(SEP)
    print(f"\n  Classes: {len(y)} | Testaveis: {y.sum()} ({y.mean()*100:.0f}%)")
    print(f"  Projetos: {len(set(groups))} | Modelo: {args.model}")
    print(f"  Features: {FEATURES}")

    n_splits = min(args.n_splits, len(set(groups)))
    fold_metrics, oof_pred, oof_proba = _cross_project_eval(X, y, groups, args.model, n_splits)

    print(f"\n  {'Fold':>4} {'Projetos':>9} {'Classes':>8} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7}")
    print(f"  {'-'*56}")
    for m in fold_metrics:
        print(
            f"  {m['fold']:>4} {m['n_projects']:>9} {m['n_test']:>8} "
            f"{m['auc']:>7.3f} {m['accuracy']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f}"
        )

    mean_auc = np.nanmean([m["auc"] for m in fold_metrics])
    mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
    overall_auc = roc_auc_score(y, oof_proba) if len(set(y)) > 1 else float("nan")
    print(f"  {'-'*56}")
    print(f"  Media     {'':>9} {'':>8} {mean_auc:>7.3f} {mean_acc:>7.3f}")
    print(f"\n  AUC out-of-fold global: {overall_auc:.3f}  (cross-project, sem leakage)")

    # Feature importance via a full-data fit (interpretability)
    scaler = StandardScaler().fit(X)
    if args.model == "logreg":
        final = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        final = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    final.fit(scaler.transform(X), y)

    print(f"\n  Importancia das features:")
    if args.model == "logreg":
        importances = np.abs(final.coef_[0])
    else:
        importances = final.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda t: -t[1]):
        bar = "#" * int(imp / max(importances) * 30)
        print(f"    {feat:<12} {bar} {imp:.3f}")

    _projected_improvement(rows, oof_pred)
    _threshold_sweep(rows, oof_proba)

    # Persist model spec (for the guided re-run)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "model_type": args.model,
        "features": FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cross_project_auc": float(overall_auc),
        "mean_fold_accuracy": float(mean_acc),
        "n_classes": int(len(y)),
        "n_projects": int(len(set(groups))),
    }
    if args.model == "logreg":
        spec["coef"] = final.coef_[0].tolist()
        spec["intercept"] = float(final.intercept_[0])
    with open(model_out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\n  Modelo guardado: {model_out}")

    # Serialise the fitted sklearn estimator + scaler so that downstream guided runs
    # can call predict_proba directly without re-training.
    import pickle
    pickle_out = model_out.with_suffix(".pkl")
    with open(pickle_out, "wb") as f:
        pickle.dump({"model": final, "scaler": scaler, "features": FEATURES}, f)
    print(f"  Modelo serializado (pickle): {pickle_out}")


if __name__ == "__main__":
    main()
