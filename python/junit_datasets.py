"""
JUnit Test Dataset Handling

PyTorch Dataset and DataLoader utilities for loading and preprocessing
JUnit test metrics for WGAN-GP training.

Features:
    - CSV loading with automatic NaN/Inf filtering.
    - Optional log1p transform for heavy-tailed count distributions.
    - Min-max normalisation to [-1, 1] with saved stats for denormalisation.
    - Skewness report at load time to guide transform decisions.
    - Train/validation split via `split()`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class JUnitDataset(Dataset):
    """
    PyTorch Dataset for JUnit test metric data.

    Loads a header-less CSV (columns: methods, branches, statements, …),
    filters invalid rows, optionally applies log1p, normalises to [-1, 1],
    and persists the normalisation statistics for denormalisation of
    synthetic samples.

    Attributes:
        raw_data (np.ndarray): Original data before any transform.
        normalized_data (torch.Tensor): Data normalised to [-1, 1].
        norm_stats (dict): Min/max per feature plus transform metadata.
        use_log_transform (bool): Whether log1p was applied.
    """

    def __init__(
        self,
        csv_path: str | Path,
        stats_output_path: str | Path | None = None,
        use_log_transform: bool = True,
    ):
        """
        Args:
            csv_path: Path to CSV (no header, numeric values).
            stats_output_path: Where to save normalisation stats JSON.
                Defaults to <csv_stem>_norm_stats.json in the same directory.
            use_log_transform: Apply log1p before normalising. Strongly
                recommended for count features with skew > 2.
        """
        df = pd.read_csv(csv_path, header=None)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        n_dropped = 0  # compute after
        original_len = len(df) + df.isnull().any(axis=1).sum()

        self.raw_data = df.values.astype(np.float64)
        self.use_log_transform = use_log_transform

        self._log_data_summary(df)

        if use_log_transform:
            transformed = np.log1p(self.raw_data)
            self.data = torch.tensor(transformed, dtype=torch.float32)
            logger.info("log1p transform applied to all features")
        else:
            self.data = torch.tensor(self.raw_data, dtype=torch.float32)

        self.norm_stats = self._compute_normalization_stats()
        self.norm_stats["log_transform"] = use_log_transform

        self.normalized_data = self._normalize()

        self._save_normalization_stats(csv_path, stats_output_path)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.normalized_data[idx]

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _compute_normalization_stats(self) -> dict:
        stats: dict = {
            "num_features": self.data.shape[1],
            "num_samples": len(self.data),
            "features": {},
        }
        for i in range(self.data.shape[1]):
            col = self.data[:, i]
            stats["features"][f"feature_{i}"] = {
                "min": float(col.min().item()),
                "max": float(col.max().item()),
                "range": float((col.max() - col.min()).item()),
                "mean": float(col.mean().item()),
                "std": float(col.std().item()),
            }
        return stats

    def _normalize(self) -> torch.Tensor:
        """Normalise each feature to [-1, 1] using min-max from norm_stats."""
        normalized = self.data.clone()
        for i in range(self.data.shape[1]):
            fs = self.norm_stats["features"][f"feature_{i}"]
            feature_range = fs["range"] if fs["range"] != 0 else 1.0
            normalized[:, i] = 2 * (self.data[:, i] - fs["min"]) / feature_range - 1

        normalized = torch.where(torch.isnan(normalized), torch.zeros_like(normalized), normalized)
        normalized = torch.where(torch.isinf(normalized), torch.zeros_like(normalized), normalized)
        return normalized

    def denormalize(self, normalized_data: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Invert normalisation (and log1p if applied) to recover original scale.

        Args:
            normalized_data: Data in [-1, 1] — numpy array or torch.Tensor.

        Returns:
            numpy array in original raw scale (rounded non-negative values).
        """
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()

        result = normalized_data.copy().astype(np.float64)

        if result.ndim == 1:
            result = result.reshape(1, -1)

        for i in range(result.shape[1]):
            fs = self.norm_stats["features"][f"feature_{i}"]
            feature_range = fs["range"] if fs["range"] != 0 else 1.0
            result[:, i] = (result[:, i] + 1) / 2 * feature_range + fs["min"]

        if self.norm_stats.get("log_transform", False):
            result = np.expm1(np.clip(result, 0.0, None))

        return np.clip(result, 0, None)

    # ------------------------------------------------------------------
    # Train/validation split
    # ------------------------------------------------------------------

    def split(
        self, val_fraction: float = 0.15, seed: int = 42
    ) -> tuple[Subset, Subset]:
        """
        Split the dataset into train and validation subsets.

        Shuffles indices deterministically using `seed`.

        Args:
            val_fraction: Fraction of data reserved for validation [0, 1).
            seed: Random seed for reproducibility.

        Returns:
            (train_subset, val_subset) — both are torch.utils.data.Subset.
        """
        n = len(self)
        n_val = max(1, int(n * val_fraction))
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        val_indices = indices[:n_val].tolist()
        train_indices = indices[n_val:].tolist()
        logger.info(f"Split: {len(train_indices)} train / {n_val} val")
        return Subset(self, train_indices), Subset(self, val_indices)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_feature_statistics(self) -> dict:
        """Return per-feature statistics of the (possibly transformed) data."""
        return {
            "num_samples": len(self.data),
            "mean": self.data.mean(dim=0).tolist(),
            "std": self.data.std(dim=0).tolist(),
            "min": self.data.min(dim=0).values.tolist(),
            "max": self.data.max(dim=0).values.tolist(),
        }

    def _log_data_summary(self, df: pd.DataFrame) -> None:
        """Log skewness and basic stats per feature at load time."""
        logger.info(f"Dataset loaded: {len(df)} samples, {df.shape[1]} features")
        for i in range(df.shape[1]):
            col = df.iloc[:, i]
            skew = float(col.skew())
            flag = "  [HIGH SKEW — log1p recommended]" if abs(skew) > 2 else ""
            logger.info(
                f"  Feature {i}: "
                f"mean={col.mean():.2f}, std={col.std():.2f}, "
                f"min={col.min():.0f}, max={col.max():.0f}, "
                f"skew={skew:.2f}{flag}"
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_normalization_stats(
        self,
        csv_path: str | Path,
        stats_output_path: str | Path | None,
    ) -> None:
        if stats_output_path is None:
            p = Path(csv_path)
            stats_output_path = p.parent / f"{p.stem}_norm_stats.json"
        else:
            stats_output_path = Path(stats_output_path)

        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_output_path, "w") as f:
            json.dump(self.norm_stats, f, indent=2)
        logger.info(f"Normalisation statistics saved to {stats_output_path}")


# ---------------------------------------------------------------------------
# Standalone loaders
# ---------------------------------------------------------------------------

def load_normalization_stats(stats_path: str | Path) -> dict:
    """Load saved normalisation statistics from a JSON file."""
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalisation stats not found: {stats_path}")
    with open(stats_path) as f:
        stats = json.load(f)
    logger.info(f"Loaded normalisation statistics from {stats_path}")
    return stats


def get_junit_data_loader(
    csv_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    use_log_transform: bool = True,
) -> tuple[DataLoader, JUnitDataset]:
    """
    Convenience loader: returns (DataLoader, JUnitDataset).

    Args:
        csv_path: Path to CSV file.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle every epoch.
        num_workers: DataLoader worker processes (0 = main process).
        use_log_transform: Apply log1p before normalising.
    """
    dataset = JUnitDataset(csv_path, use_log_transform=use_log_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
