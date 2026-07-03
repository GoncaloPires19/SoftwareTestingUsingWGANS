"""
Pytest unit tests for the WGAN-GP Python modules.

Covers: model, utils, junit_datasets, config, EarlyStopper.

Run with:
    pytest python/tests/test_modules.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Make sure the python/ package is importable regardless of where pytest is run
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ModelConfig,
    TrainingConfig,
    WGANConfig,
    MetricType,
    set_random_seed,
)
from model import Critic, Generator
from utils import (
    compute_composite_score,
    compute_diversity_score,
    compute_feature_matching_loss,
    compute_gradient_penalty,
    compute_wasserstein_distance,
    get_model_gradient_stats,
)
from junit_datasets import JUnitDataset, get_junit_data_loader, load_normalization_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_csv(tmp_path: Path) -> Path:
    """A minimal 3-column JUnit CSV with 20 samples."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=(20, 3))
    p = tmp_path / "data.csv"
    pd.DataFrame(data).to_csv(p, header=False, index=False)
    return p


@pytest.fixture
def batch_real() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(16, 3) * 2 - 1  # uniform in [-1, 1]


@pytest.fixture
def batch_fake() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(16, 3) * 2 - 1


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.early_stopping_enabled is True
        assert cfg.early_stopping_patience == 20
        assert cfg.feature_matching_weight == 0.1
        assert cfg.use_lr_scheduler is True

    def test_no_duplicate_early_stopping_fields(self):
        """Regression: early_stopping_enabled must default to True, not False."""
        cfg = TrainingConfig()
        assert cfg.early_stopping_enabled is True, (
            "Duplicate dataclass field bug: early_stopping_enabled was overwritten to False"
        )

    def test_model_config_hidden_dims(self):
        cfg = ModelConfig(hidden_dims_g=[128, 256], hidden_dims_c=[256, 128])
        assert cfg.hidden_dims_g == [128, 256]
        assert cfg.hidden_dims_c == [256, 128]

    def test_model_config_minibatch_std_default(self):
        cfg = ModelConfig()
        assert cfg.use_minibatch_std is True

    def test_wgan_config_save_load(self, tmp_path: Path):
        cfg = WGANConfig()
        cfg.experiment_name = "test_run"
        path = tmp_path / "config.json"
        cfg.save(path)
        assert path.exists()
        loaded = WGANConfig.load(path)
        assert loaded.training.epochs == cfg.training.epochs

    def test_from_preset(self):
        cfg = WGANConfig.from_preset("quick")
        assert cfg.training.epochs == 5

    def test_set_random_seed(self):
        set_random_seed(42)
        a = torch.randn(5)
        set_random_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_output_shape_default(self):
        g = Generator(z_dim=100, output_dim=3)
        z = torch.randn(8, 100)
        out = g(z)
        assert out.shape == (8, 3)

    def test_output_range(self):
        g = Generator(z_dim=100, output_dim=3)
        z = torch.randn(64, 100)
        out = g(z)
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_custom_hidden_dims(self):
        g = Generator(z_dim=50, output_dim=4, hidden_dims=[64, 128, 256])
        out = g(torch.randn(4, 50))
        assert out.shape == (4, 4)

    def test_batch_norm_variant(self):
        g = Generator(z_dim=100, output_dim=3, use_batch_norm=True)
        out = g(torch.randn(8, 100))
        assert out.shape == (8, 3)

    def test_single_sample(self):
        g = Generator()
        out = g(torch.randn(1, 100))
        assert out.shape == (1, 3)


class TestCritic:
    def test_output_shape(self):
        c = Critic(input_dim=3)
        x = torch.randn(16, 3)
        out = c(x)
        assert out.shape == (16, 1)

    def test_no_sigmoid(self):
        """Critic output must not be bounded to [0,1] — Wasserstein requires unbounded."""
        c = Critic(input_dim=3)
        x = torch.randn(64, 3) * 10  # extreme values
        out = c(x)
        assert out.abs().max().item() > 1.0, "Critic should produce unbounded scores"

    def test_minibatch_std(self):
        c = Critic(input_dim=3, use_minibatch_std=True)
        x = torch.randn(16, 3)
        out = c(x)
        assert out.shape == (16, 1)

    def test_spectral_norm_warning(self):
        with pytest.warns(UserWarning, match="incompatible"):
            Critic(input_dim=3, use_spectral_norm=True)

    def test_custom_hidden_dims(self):
        c = Critic(input_dim=4, hidden_dims=[256, 128, 64])
        out = c(torch.randn(8, 4))
        assert out.shape == (8, 1)

    def test_gradient_flows_through_critic(self):
        c = Critic(input_dim=3)
        x = torch.randn(8, 3, requires_grad=True)
        out = c(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------

class TestGradientPenalty:
    def test_shape_and_sign(self, batch_real, batch_fake):
        c = Critic(input_dim=3)
        gp = compute_gradient_penalty(c, batch_real, batch_fake, torch.device("cpu"), 16)
        assert gp.shape == ()
        assert gp.item() >= 0.0

    def test_lambda_scales_penalty(self, batch_real, batch_fake):
        """Higher lambda_gp must produce a proportionally larger penalty."""
        c = Critic(input_dim=3)
        # Fix the seed so both calls draw the same interpolation alpha
        torch.manual_seed(99)
        gp1 = compute_gradient_penalty(c, batch_real, batch_fake, torch.device("cpu"), 16, lambda_gp=1.0)
        torch.manual_seed(99)
        gp10 = compute_gradient_penalty(c, batch_real, batch_fake, torch.device("cpu"), 16, lambda_gp=10.0)
        assert abs(gp10.item() - 10.0 * gp1.item()) < 1e-3

    def test_smaller_batch_size(self):
        c = Critic(input_dim=3)
        real = torch.randn(5, 3)
        fake = torch.randn(5, 3)
        gp = compute_gradient_penalty(c, real, fake, torch.device("cpu"), 5)
        assert gp.item() >= 0.0


class TestFeatureMatchingLoss:
    def test_identical_distributions(self):
        x = torch.randn(64, 3)
        loss = compute_feature_matching_loss(x, x)
        assert loss.item() < 0.1

    def test_different_distributions(self, batch_real, batch_fake):
        loss = compute_feature_matching_loss(batch_real, batch_fake * 5)
        assert loss.item() > 0.0

    def test_output_is_scalar(self, batch_real, batch_fake):
        loss = compute_feature_matching_loss(batch_real, batch_fake)
        assert loss.shape == ()


class TestDiversityScore:
    def test_range(self):
        real = np.random.randn(100, 3)
        synth = np.random.randn(100, 3)
        score = compute_diversity_score(synth, real)
        assert 0.0 <= score <= 1.0

    def test_zero_variance_synth(self):
        real = np.random.randn(50, 3)
        synth = np.zeros((50, 3))  # no diversity
        score = compute_diversity_score(synth, real)
        assert score == pytest.approx(0.0)

    def test_similar_distribution(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        score = compute_diversity_score(data, data)
        assert score == pytest.approx(0.5, abs=0.05)


class TestWassersteinDistance:
    def test_same_distribution(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 3))
        dist = compute_wasserstein_distance(data, data)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_different_distributions(self):
        rng = np.random.default_rng(0)
        real = rng.standard_normal((100, 3))
        shifted = real + 5.0
        dist = compute_wasserstein_distance(shifted, real)
        assert dist > 1.0

    def test_output_is_float(self):
        real = np.random.randn(50, 3)
        synth = np.random.randn(50, 3)
        dist = compute_wasserstein_distance(synth, real)
        assert isinstance(dist, float)


class TestCompositeScore:
    def test_perfect_quality_and_diversity(self):
        score = compute_composite_score(diversity=0.5, wasserstein=0.0, w1_reference=1.0)
        assert score == pytest.approx(0.8, abs=0.01)  # 0.6 * 1.0 + 0.4 * 0.5

    def test_zero_quality_high_diversity(self):
        """Epoch 1 scenario: random noise has high diversity but W1 >> w1_reference."""
        score = compute_composite_score(diversity=0.7, wasserstein=2.0, w1_reference=1.0)
        assert score < 0.5, "Random noise should not be picked as best model"

    def test_score_increases_as_w1_decreases(self):
        w1_ref = 0.5
        score_bad = compute_composite_score(0.5, 0.5, w1_ref)
        score_good = compute_composite_score(0.5, 0.1, w1_ref)
        assert score_good > score_bad

    def test_range(self):
        score = compute_composite_score(diversity=0.4, wasserstein=0.3, w1_reference=0.5)
        assert 0.0 <= score <= 1.0


class TestGradientStats:
    def test_no_gradients_returns_zeros(self):
        g = Generator()
        stats = get_model_gradient_stats(g)
        assert stats["mean_grad_norm"] == 0.0
        assert stats["max_grad_norm"] == 0.0

    def test_after_backward(self):
        g = Generator()
        z = torch.randn(8, 100)
        loss = g(z).sum()
        loss.backward()
        stats = get_model_gradient_stats(g)
        assert stats["mean_grad_norm"] > 0.0
        assert stats["max_grad_norm"] >= stats["mean_grad_norm"]
        assert stats["min_grad_norm"] <= stats["mean_grad_norm"]


# ---------------------------------------------------------------------------
# junit_datasets.py
# ---------------------------------------------------------------------------

class TestJUnitDataset:
    def test_loads_correctly(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        assert len(ds) == 20
        assert ds.norm_stats["num_features"] == 3

    def test_normalized_range(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        data = ds.normalized_data
        assert data.min().item() >= -1.0 - 1e-5
        assert data.max().item() <= 1.0 + 1e-5

    def test_norm_stats_saved(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        stats_path = small_csv.parent / f"{small_csv.stem}_norm_stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        assert "features" in stats
        assert stats["log_transform"] is True

    def test_denormalize_round_trip(self, small_csv: Path):
        ds = JUnitDataset(small_csv, use_log_transform=False)
        norm = ds.normalized_data.numpy()
        recovered = ds.denormalize(norm)
        expected = ds.raw_data
        np.testing.assert_allclose(recovered, expected, atol=1.0)

    def test_log_transform_denormalize(self, small_csv: Path):
        ds = JUnitDataset(small_csv, use_log_transform=True)
        norm = ds.normalized_data.numpy()
        recovered = ds.denormalize(norm)
        np.testing.assert_allclose(recovered, ds.raw_data, atol=1.0)

    def test_split(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        train, val = ds.split(val_fraction=0.2, seed=42)
        assert len(train) + len(val) == len(ds)
        assert len(val) >= 1

    def test_split_reproducible(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        t1, v1 = ds.split(seed=7)
        t2, v2 = ds.split(seed=7)
        assert t1.indices == t2.indices
        assert v1.indices == v2.indices

    def test_getitem_returns_tensor(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3,)

    def test_nan_rows_dropped(self, tmp_path: Path):
        data = pd.DataFrame([[1, 2, 3], [float("nan"), 5, 6], [7, 8, 9]])
        p = tmp_path / "nan_data.csv"
        data.to_csv(p, header=False, index=False)
        ds = JUnitDataset(p)
        assert len(ds) == 2  # NaN row dropped

    def test_dataloader_integration(self, small_csv: Path):
        loader, ds = get_junit_data_loader(small_csv, batch_size=4)
        batch = next(iter(loader))
        assert batch.shape[0] <= 4
        assert batch.shape[1] == 3

    def test_load_normalization_stats(self, small_csv: Path):
        ds = JUnitDataset(small_csv)
        stats_path = small_csv.parent / f"{small_csv.stem}_norm_stats.json"
        stats = load_normalization_stats(stats_path)
        assert stats["num_features"] == 3


# ---------------------------------------------------------------------------
# EarlyStopper (imported inline from train_optimized)
# ---------------------------------------------------------------------------

class TestEarlyStopper:
    def _make(self, patience=3, min_delta=0.01, mode="max"):
        # Import here to avoid module-level side effects
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_optimized import EarlyStopper
        return EarlyStopper(patience=patience, min_delta=min_delta, mode=mode)

    def test_does_not_stop_while_improving(self):
        es = self._make(patience=3)
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1):
            assert es.step(v, i) is False

    def test_stops_after_patience(self):
        es = self._make(patience=3)
        es.step(0.5, 1)
        es.step(0.5, 2)
        es.step(0.5, 3)
        assert es.step(0.5, 4) is True

    def test_resets_counter_on_improvement(self):
        es = self._make(patience=3)
        es.step(0.5, 1)
        es.step(0.5, 2)
        es.step(0.6, 3)  # improvement — counter resets
        assert es.epochs_without_improvement == 0

    def test_best_epoch_tracked(self):
        es = self._make(patience=5)
        es.step(0.2, 1)
        es.step(0.8, 2)
        es.step(0.7, 3)
        assert es.best_epoch == 2
        assert es.best_value == pytest.approx(0.8)

    def test_min_mode(self):
        es = self._make(patience=2, min_delta=0.01, mode="min")
        es.step(1.0, 1)
        es.step(0.9, 2)
        es.step(0.9, 3)  # no improvement
        assert es.step(0.9, 4) is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            self._make(mode="invalid")
