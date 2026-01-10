"""
Centralized configuration for WGAN training with multiple datasets.
Supports: Juliet, OWASP, Defects4J, Evosuite (synthetic), and combinations.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Literal
from enum import Enum
import json
import os


class DatasetMode(Enum):
    """Dataset configuration modes"""
    JULIET = "juliet"
    OWASP = "owasp"
    DEFECTS4J = "defects4j"
    EVOSUITE = "evosuite"
    COMBINED = "combined"  # Juliet + Defects4J + Evosuite


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
    
    # Individual dataset paths
    juliet_path: Path = field(default_factory=lambda: Path("data/raw/juliet"))
    owasp_path: Path = field(default_factory=lambda: Path("data/raw/owasp"))
    defects4j_path: Path = field(default_factory=lambda: Path("data/raw/defects4j"))
    evosuite_path: Path = field(default_factory=lambda: Path("data/raw/evosuite"))
    
    # Output paths
    train_data_path: Path = field(default_factory=lambda: Path("data/processed/train.csv"))
    val_data_path: Path = field(default_factory=lambda: Path("data/processed/val.csv"))
    test_data_path: Path = field(default_factory=lambda: Path("data/processed/test.csv"))
    
    # Data split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Filtering
    min_test_length: int = 10  # Minimum lines of code
    max_test_length: int = 500
    include_vulnerable: bool = True
    include_clean: bool = True
    
    def __post_init__(self):
        """Convert strings to Path objects if needed"""
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
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    z_dim: int = 100
    output_dim: int = 3  # Number of features (methods, branches, statements)
    generator_hidden: List[int] = field(default_factory=lambda: [256, 512])
    critic_hidden: List[int] = field(default_factory=lambda: [512, 256])
    batch_norm_g: bool = True
    batch_norm_c: bool = False
    

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_gp: float = 10.0
    n_critic: int = 5  # Critic iterations per epoch
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: MetricType = MetricType.DIVERSITY
    early_stopping_threshold: float = 0.01  # Minimum improvement
    
    # Checkpointing
    checkpoint_interval: int = 5
    save_best_only: bool = True
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    

@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    metrics: List[MetricType] = field(default_factory=lambda: [
        MetricType.DIVERSITY,
        MetricType.CWE_COVERAGE,
        MetricType.WASSERSTEIN_DISTANCE
    ])
    
    # FID/diversity computation
    num_samples_for_metrics: int = 500
    codebertmodel_path: str = "microsoft/codebert-base"  # For embeddings
    
    # CWE coverage (for vulnerability-aware evaluation)
    track_cwe_distribution: bool = True
    min_cwe_coverage: float = 0.8  # At least 80% of CWEs represented


@dataclass
class OutputConfig:
    """Output and logging configuration"""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Artifacts
    save_training_history: bool = True
    save_synthetic_samples: bool = True
    save_visualizations: bool = True
    
    # Verbose
    verbose: int = 1  # 0=silent, 1=normal, 2=debug
    log_interval: int = 5  # Log every N epochs
    plot_interval: int = 10  # Generate plots every N epochs
    
    def __post_init__(self):
        """Create output directories"""
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
    
    # Experiment metadata
    experiment_name: str = "wgan_default"
    description: str = ""
    seed: int = 42
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "seed": self.seed,
            "data": {
                "dataset_mode": self.data.dataset_mode.value,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "test_split": self.data.test_split,
            },
            "model": {
                "z_dim": self.model.z_dim,
                "output_dim": self.model.output_dim,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "lambda_gp": self.training.lambda_gp,
                "n_critic": self.training.n_critic,
                "early_stopping_enabled": self.training.early_stopping_enabled,
                "early_stopping_patience": self.training.early_stopping_patience,
            }
        }
    
    def save(self, path: Path):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Config saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'WGANConfig':
        """Load config from JSON file"""
        with open(path) as f:
            data = json.load(f)
        config = cls()
        # Update fields from loaded data if needed
        return config
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'WGANConfig':
        """Load a preset configuration"""
        presets = {
            "quick": cls(
                experiment_name="quick_test",
                training=TrainingConfig(epochs=5, batch_size=16),
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
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return presets[preset_name]


# Convenience function to get config
def get_config(
    dataset_mode: str = "combined",
    epochs: int = 50,
    batch_size: int = 32,
    experiment_name: str = "wgan_experiment",
    early_stopping: bool = True,
    preset: str = None
) -> WGANConfig:
    """Create a WGAN config with optional overrides"""
    
    if preset:
        config = WGANConfig.from_preset(preset)
    else:
        config = WGANConfig(
            experiment_name=experiment_name,
            data=DataConfig(dataset_mode=DatasetMode(dataset_mode)),
            training=TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                early_stopping_enabled=early_stopping,
            )
        )
    
    return config


if __name__ == "__main__":
    # Test configs
    print("🔧 Testing configuration system...")
    
    config = WGANConfig.from_preset("combined")
    print(f"\n✅ Loaded preset: {config.experiment_name}")
    print(f"   Dataset mode: {config.data.dataset_mode.value}")
    print(f"   Epochs: {config.training.epochs}")
    print(f"   Early stopping: {config.training.early_stopping_enabled}")
    
    # Save example config
    config.save(Path("outputs/config_example.json"))
    print("\n✅ Configuration system ready!")
