#!/usr/bin/env python3
"""
Cross-benchmark validation: score Defects4J with the SF110-trained testability model.

train_testability_predictor.py validates only within SF110 (leave-projects-out).
This applies that same fitted model + scaler, with zero retraining, to
bridge_dataset_d4j.csv (produced by run_defects4j_mutation_testing.py) to see
whether the structural signal generalises to a genuinely different benchmark
(Defects4J: mature real-world projects, real bugs, developer-written tests).

Usage:
    python python/scripts/score_defects4j_with_sf110_model.py \
        --bridge outputs/mutation_experiments/defects4j/bridge_dataset_d4j.csv \
        --model outputs/mutation_experiments/testability_model_111.pkl
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

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from sf110_lib import killed_relevant, load_bridge, per_project_average, weighted_score

SEP = "=" * 72


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Defects4J bridge dataset with the SF110-trained testability model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bridge", type=str, default="outputs/mutation_experiments/defects4j/bridge_dataset_d4j.csv")
    parser.add_argument("--model", type=str, default="outputs/mutation_experiments/testability_model_111.pkl")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        print(f"[ERRO] Bridge dataset not found: {bridge_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERRO] Model pickle not found: {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        spec = pickle.load(f)
    model, scaler, features = spec["model"], spec["scaler"], spec["features"]

    X, y, groups, rows = load_bridge(bridge_path, features=features)

    print(SEP)
    print("  VALIDACAO CROSS-BENCHMARK — modelo SF110 aplicado ao Defects4J")
    print(SEP)
    print(f"\n  Classes D4J: {len(y)} | Testaveis: {y.sum()} ({y.mean()*100:.0f}%)")
    print(f"  Projetos D4J: {len(set(groups))}")
    print(f"  Modelo: {model_path.name} (treinado no SF110, zero retraining)")

    proba = model.predict_proba(scaler.transform(X))[:, 1]
    pred = (proba >= args.threshold).astype(int)

    if len(set(y)) > 1:
        auc = roc_auc_score(y, proba)
        print(f"\n  AUC (SF110 -> D4J, sem retreino): {auc:.3f}")
    else:
        auc = float("nan")
        print(f"\n  AUC indefinido: todas as classes D4J tem o mesmo label (testable={y[0]})")

    print(f"  Accuracy:  {accuracy_score(y, pred):.3f}")
    print(f"  Precision: {precision_score(y, pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y, pred, zero_division=0):.3f}")

    base_global = weighted_score(rows)
    base_avg, base_n = per_project_average(rows)
    print(f"\n  Score GLOBAL ponderado (D4J, todas as classes): {base_global:.1f}%")
    print(f"  Media por-projeto (D4J, todas as classes):      {base_avg:.1f}% ({base_n} projetos)")

    kept = [r for r, p in zip(rows, pred) if p == 1]
    if kept:
        guided_global = weighted_score(kept)
        guided_avg, guided_n = per_project_average(kept)
        print(f"\n  Selecao guiada pelo modelo (limiar {args.threshold}):")
        print(f"    Classes mantidas: {len(kept)}/{len(rows)}")
        print(f"    Score GLOBAL ponderado: {guided_global:.1f}%")
        print(f"    Media por-projeto:      {guided_avg:.1f}% ({guided_n} projetos)")
    else:
        print(f"\n  Selecao guiada (limiar {args.threshold}): nenhuma classe mantida.")

    print(f"\n{SEP}")
    print("  Por-projeto (D4J):")
    print(SEP)
    by_project: dict[str, list[dict]] = {}
    for r in rows:
        by_project.setdefault(r["project"], []).append(r)
    print(f"\n  {'Projeto':<18}{'Classes':>8}{'Testaveis':>11}{'Score':>9}")
    print(f"  {'-'*46}")
    for project, project_rows in sorted(by_project.items()):
        k = sum(killed_relevant(r)[0] for r in project_rows)
        rel = sum(killed_relevant(r)[1] for r in project_rows)
        score = k / rel * 100 if rel else 0.0
        n_testable = sum(int(r["testable"]) for r in project_rows)
        print(f"  {project:<18}{len(project_rows):>8}{n_testable:>11}{score:>8.1f}%")

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
