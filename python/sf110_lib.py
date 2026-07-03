"""
SF110 Testability Analysis — Shared Library

Single source of truth for the SF110 testability-prediction pipeline. Centralises
the structural feature set, bridge-dataset loading, and mutation-score arithmetic
that were previously duplicated across the analysis scripts.

Pipeline scripts that consume this module:
    - train_testability_predictor.py
    - wgan_testability_scorer.py
    - wgan_augmented_predictor.py
    - analyze_class_level.py (metrics helpers)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Structural features used to predict class testability (source-derived, pre-PIT).
# Order matters — downstream scalers and models rely on it.
FEATURES: list[str] = [
    "methods", "branches", "statements", "loc", "n_fields", "has_io",
    "extends_framework", "implements_callback", "is_abstract",
    "n_static_methods", "static_ratio", "has_default_ctor",
    "n_imports", "n_external_deps", "max_ctor_params",
]


def load_bridge(path: str | Path, features: list[str] | None = None):
    """
    Load the bridge dataset (structure -> testability).

    Returns:
        X       : (N, F) float feature matrix in FEATURES order.
        y       : (N,) int testability labels (1 = covered/testable).
        groups  : (N,) project name per row (for leave-projects-out CV).
        rows    : list of raw dict rows (for downstream score arithmetic).
    """
    features = features or FEATURES
    path = Path(path)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    X = np.array([[float(r[c]) for c in features] for r in rows], dtype=np.float64)
    y = np.array([int(r["testable"]) for r in rows], dtype=np.int64)
    groups = np.array([r["project"] for r in rows])
    return X, y, groups, rows


def killed_relevant(row: dict) -> tuple[int, int]:
    """
    Recover (killed, relevant) mutation counts for one class row.

    The bridge dataset stores mutation_score = killed / relevant and n_mutations
    = relevant, so killed is reconstructed as round(score * relevant).
    """
    relevant = int(row["n_mutations"])
    killed = round(float(row["mutation_score"]) * relevant)
    return killed, relevant


def weighted_score(rows: list[dict]) -> float:
    """Mutation-weighted global score (%) over a set of class rows."""
    k = sum(killed_relevant(r)[0] for r in rows)
    rel = sum(killed_relevant(r)[1] for r in rows)
    return k / rel * 100 if rel else 0.0


def per_project_average(rows: list[dict]) -> tuple[float, int]:
    """
    Average of per-project weighted mutation scores (%).

    Returns (average, n_projects). Each project contributes one score equal to its
    own mutation-weighted score, so small and large projects weigh equally — this
    is the 'average above 70%' metric the dissertation targets.
    """
    by_project: dict[str, list[dict]] = {}
    for r in rows:
        by_project.setdefault(r["project"], []).append(r)

    scores = []
    for project_rows in by_project.values():
        k = sum(killed_relevant(r)[0] for r in project_rows)
        rel = sum(killed_relevant(r)[1] for r in project_rows)
        if rel > 0:
            scores.append(k / rel * 100)

    return (sum(scores) / len(scores) if scores else 0.0), len(scores)


def total_relevant(rows: list[dict]) -> int:
    """Total relevant mutations across all rows (denominator for coverage %)."""
    return sum(killed_relevant(r)[1] for r in rows)
