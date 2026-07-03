"""
Unit tests for sf110_lib — the shared SF110 testability-analysis helpers.

Run with:
    pytest python/tests/test_sf110_lib.py -v
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sf110_lib import (
    FEATURES,
    killed_relevant,
    load_bridge,
    per_project_average,
    total_relevant,
    weighted_score,
)


@pytest.fixture
def bridge_csv(tmp_path: Path) -> Path:
    """A minimal bridge dataset with 2 projects and known scores."""
    p = tmp_path / "bridge.csv"
    header = FEATURES + ["n_mutations", "coverage", "mutation_score", "test_strength", "testable", "project", "class_name"]
    rows = [
        # projA: 2 classes — scores 0.8 (100 mut) and 0.4 (100 mut) -> weighted 0.6
        {**{f: 1 for f in FEATURES}, "n_mutations": 100, "coverage": 0.9, "mutation_score": 0.8, "test_strength": 0.9, "testable": 1, "project": "projA", "class_name": "A1"},
        {**{f: 1 for f in FEATURES}, "n_mutations": 100, "coverage": 0.6, "mutation_score": 0.4, "test_strength": 0.7, "testable": 1, "project": "projA", "class_name": "A2"},
        # projB: 1 class — score 0.2 (50 mut)
        {**{f: 2 for f in FEATURES}, "n_mutations": 50, "coverage": 0.2, "mutation_score": 0.2, "test_strength": 0.5, "testable": 0, "project": "projB", "class_name": "B1"},
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return p


class TestLoadBridge:
    def test_shapes(self, bridge_csv: Path):
        X, y, groups, rows = load_bridge(bridge_csv)
        assert X.shape == (3, len(FEATURES))
        assert y.tolist() == [1, 1, 0]
        assert list(groups) == ["projA", "projA", "projB"]
        assert len(rows) == 3

    def test_feature_order_preserved(self, bridge_csv: Path):
        X, _, _, _ = load_bridge(bridge_csv)
        # projB rows have all features = 2
        assert all(v == 2 for v in X[2])


class TestKilledRelevant:
    def test_reconstruction(self):
        row = {"n_mutations": 100, "mutation_score": 0.8}
        killed, relevant = killed_relevant(row)
        assert relevant == 100
        assert killed == 80

    def test_rounding(self):
        row = {"n_mutations": 30, "mutation_score": 0.333}
        killed, relevant = killed_relevant(row)
        assert relevant == 30
        assert killed == 10  # round(9.99)

    def test_zero_score(self):
        killed, relevant = killed_relevant({"n_mutations": 50, "mutation_score": 0.0})
        assert killed == 0
        assert relevant == 50


class TestWeightedScore:
    def test_weighted(self, bridge_csv: Path):
        _, _, _, rows = load_bridge(bridge_csv)
        # killed = 80 + 40 + 10 = 130 ; relevant = 250 -> 52%
        assert weighted_score(rows) == pytest.approx(52.0, abs=0.5)

    def test_empty(self):
        assert weighted_score([]) == 0.0


class TestPerProjectAverage:
    def test_average(self, bridge_csv: Path):
        _, _, _, rows = load_bridge(bridge_csv)
        # projA weighted: (80+40)/200 = 60% ; projB: 10/50 = 20% ; avg = 40%
        avg, n_proj = per_project_average(rows)
        assert n_proj == 2
        assert avg == pytest.approx(40.0, abs=0.5)

    def test_equal_weight_per_project(self, bridge_csv: Path):
        """Small project (projB) weighs equally to large project (projA)."""
        _, _, _, rows = load_bridge(bridge_csv)
        avg, _ = per_project_average(rows)
        # If it were mutation-weighted instead, projA (200 mut) would dominate
        # and the average would be ~52%, not 40%.
        assert avg == pytest.approx(40.0, abs=0.5)

    def test_empty(self):
        avg, n = per_project_average([])
        assert avg == 0.0
        assert n == 0


class TestTotalRelevant:
    def test_total(self, bridge_csv: Path):
        _, _, _, rows = load_bridge(bridge_csv)
        assert total_relevant(rows) == 250
