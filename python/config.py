"""
Centralized configuration for WGAN training with multiple datasets.
Supports: Juliet, OWASP, Defects4J, Evosuite (synthetic), and combinations.

REPRODUCIBILITY: All random seeds must be set to ensure deterministic training.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Literal
from enum import Enum
import json
import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42):
    """
    Set random seed for complete reproducibility.

    Must be called BEFORE any random operations!

    Args:
        seed: Random seed value (default: 42 for research reproducibility)
    """
    logger.info(f"Setting random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Random seed set successfully")


class DatasetMode(Enum):
    """Dataset configuration modes"""
    JULIET = "juliet"
    OWASP = "owasp"
    DEFECTS4J = "defects4j"
    EVOSUITE = "evosuite"
    COMBINED = "combined"


class MetricType(Enum):
    """Available metrics for monitoring"""
    WASSERSTEIN_DISTANCE = "wasserstein"
    FID = "fid"
    DIVERSITY = "diversity"
    CWE_COVERAGE = "cwe_coverage"
    CODE_SIMILARITY = "code_similarity"


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_mode: DatasetMode = DatasetMode.COMBINED
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))

    juliet_path: Path = field(default_factory=lambda: Path("data/raw/juliet"))
    owasp_path: Path = field(default_factory=lambda: Path("data/raw/owasp"))
    defects4j_path: Path = field(default_factory=lambda: Path("data/raw/defects4j"))
    evosuite_path: Path = field(default_factory=lambda: Path("data/raw/evosuite"))

    train_data_path: Path = field(default_factory=lambda: Path("data/processed/train.csv"))
    val_data_path: Path = field(default_factory=lambda: Path("data/processed/val.csv"))
    test_data_path: Path = field(default_factory=lambda: Path("data/processed/test.csv"))

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    min_test_length: int = 10
    max_test_length: int = 500
    include_vulnerable: bool = True
    include_clean: bool = True

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.juliet_path = Path(self.juliet_path)
        self.owasp_path = Path(self.owasp_path)
        self.defects4j_path = Path(self.defects4j_path)
        self.evosuite_path = Path(self.evosuite_path)
        self.train_data_path = Path(self.train_data_path)
        self.val_data_path = Path(self.val_data_path)
        self.test_data_path = Path(self.test_data_path)

        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    z_dim: int = 100
    output_dim: int = 3  # methods, branches, statements
    hidden_dims_g: List[int] = field(default_factory=lambda: [256, 512])
    hidden_dims_c: List[int] = field(default_factory=lambda: [512, 256])
    use_batch_norm_g: bool = False
    use_spectral_norm_c: bool = False  # Incompatible with GP — keep False
    use_minibatch_std: bool = True  # Anti-mode-collapse; appends batch std to Critic


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for WGAN-GP.

    REPRODUCIBILITY: seed MUST be set before training begins.
    """
    epochs: int = 100
    batch_size: int = 8

    learning_rate: float = 0.00002
    beta1: float = 0.5
    beta2: float = 0.999

    # WGAN-GP
    lambda_gp: float = 10.0
    n_critic: int = 10

    # Feature matching loss weight (0 = disabled)
    feature_matching_weight: float = 0.1

    # Early stopping — monitor composite score; stops if no improvement for `patience` epochs
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: MetricType = MetricType.DIVERSITY
    early_stopping_threshold: float = 0.005

    # LR scheduler (cosine annealing from lr → eta_min over all epochs)
    use_lr_scheduler: bool = True
    lr_eta_min: float = 1e-6

    # Checkpointing
    checkpoint_interval: int = 10
    save_best_only: bool = True

    # Reproducibility
    seed: int = 42

    device: str = "cpu"


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    metrics: List[MetricType] = field(default_factory=lambda: [
        MetricType.DIVERSITY,
        MetricType.CWE_COVERAGE,
        MetricType.WASSERSTEIN_DISTANCE,
    ])

    num_samples_for_metrics: int = 500
    codebertmodel_path: str = "microsoft/codebert-base"

    track_cwe_distribution: bool = True
    min_cwe_coverage: float = 0.8


@dataclass
class OutputConfig:
    """Output and logging configuration"""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    save_training_history: bool = True
    save_synthetic_samples: bool = True
    save_visualizations: bool = True

    verbose: int = 1
    log_interval: int = 1
    plot_interval: int = 10

    track_gradient_stats: bool = True
    log_gradient_interval: int = 5

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.models_dir = Path(self.models_dir)
        self.logs_dir = Path(self.logs_dir)

        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


@dataclass
class WGANConfig:
    """Complete WGAN configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    experiment_name: str = "wgan_default"
    description: str = ""
    seed: int = 42

    def apply_seed(self):
        """Apply random seeds from configuration. Must be called before any random operations."""
        set_random_seed(self.training.seed)
        self.seed = self.training.seed

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "seed": self.training.seed,
            "data": {
                "dataset_mode": self.data.dataset_mode.value,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "test_split": self.data.test_split,
            },
            "model": {
                "z_dim": self.model.z_dim,
                "output_dim": self.model.output_dim,
                "hidden_dims_g": self.model.hidden_dims_g,
                "hidden_dims_c": self.model.hidden_dims_c,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "lambda_gp": self.training.lambda_gp,
                "n_critic": self.training.n_critic,
                "feature_matching_weight": self.training.feature_matching_weight,
                "seed": self.training.seed,
                "early_stopping_enabled": self.training.early_stopping_enabled,
                "early_stopping_patience": self.training.early_stopping_patience,
                "early_stopping_threshold": self.training.early_stopping_threshold,
            },
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "WGANConfig":
        with open(path) as f:
            data = json.load(f)
        config = cls()
        if "training" in data:
            t = data["training"]
            config.training.epochs = t.get("epochs", config.training.epochs)
            config.training.batch_size = t.get("batch_size", config.training.batch_size)
            config.training.learning_rate = t.get("learning_rate", config.training.learning_rate)
            config.training.lambda_gp = t.get("lambda_gp", config.training.lambda_gp)
            config.training.n_critic = t.get("n_critic", config.training.n_critic)
            config.training.seed = t.get("seed", config.training.seed)
            config.training.feature_matching_weight = t.get(
                "feature_matching_weight", config.training.feature_matching_weight
            )
        return config

    @classmethod
    def from_preset(cls, preset_name: str) -> "WGANConfig":
        presets = {
            "quick": cls(
                experiment_name="quick_test",
                training=TrainingConfig(epochs=5, batch_size=16, early_stopping_enabled=False),
            ),
            "juliet_only": cls(
                experiment_name="juliet_baseline",
                data=DataConfig(dataset_mode=DatasetMode.JULIET),
                training=TrainingConfig(epochs=100, early_stopping_patience=15),
            ),
            "combined": cls(
                experiment_name="full_combined",
                data=DataConfig(dataset_mode=DatasetMode.COMBINED),
                training=TrainingConfig(epochs=200, early_stopping_patience=20),
            ),
            "evosuite_synthetic": cls(
                experiment_name="synthetic_only",
                data=DataConfig(dataset_mode=DatasetMode.EVOSUITE),
                training=TrainingConfig(epochs=150, early_stopping_patience=15),
            ),
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
            )

        return presets[preset_name]


def get_config(
    dataset_mode: str = "combined",
    epochs: int = 50,
    batch_size: int = 32,
    experiment_name: str = "wgan_experiment",
    early_stopping: bool = True,
    preset: str = None,
) -> WGANConfig:
    """Create a WGAN config with optional overrides"""
    if preset:
        return WGANConfig.from_preset(preset)

    return WGANConfig(
        experiment_name=experiment_name,
        data=DataConfig(dataset_mode=DatasetMode(dataset_mode)),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_enabled=early_stopping,
        ),
    )


if __name__ == "__main__":
    config = WGANConfig.from_preset("combined")
    print(f"Preset: {config.experiment_name}")
    print(f"Dataset mode: {config.data.dataset_mode.value}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Early stopping: {config.training.early_stopping_enabled} (patience={config.training.early_stopping_patience})")
    print(f"Feature matching weight: {config.training.feature_matching_weight}")
    config.save(Path("outputs/config_example.json"))
