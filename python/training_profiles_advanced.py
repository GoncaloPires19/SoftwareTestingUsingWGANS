"""
Training Profile System - Configuration-Driven Training Objectives
==================================================================

Define configurable training profiles for different analytical goals:
1. Coverage-Seeking: Maximize structural exploration
2. Vulnerability-Focused: Learn semantic weakness patterns
3. Generalization-Stress: Prevent overfitting
4. Hybrid Realistic: Real-world CI-like usage
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json
from pathlib import Path


class ProfileType(Enum):
    """Training profile types"""
    COVERAGE_SEEKING = "coverage_seeking"
    VULNERABILITY_FOCUSED = "vulnerability_focused"
    GENERALIZATION_STRESS = "generalization_stress"
    HYBRID_REALISTIC = "hybrid_realistic"


@dataclass
class RewardConfig:
    """Reward signal configuration"""
    # Coverage rewards
    line_coverage_weight: float = 0.25
    branch_coverage_weight: float = 0.25
    method_coverage_weight: float = 0.25
    
    # Diversity rewards
    path_diversity_weight: float = 0.15
    
    # Penalization
    redundancy_penalty: float = 0.10  # Penalize duplicate paths
    

@dataclass
class DatasetConfig:
    """Dataset composition configuration"""
    # Source weights (sum should be 1.0)
    juliet_weight: float = 0.3
    owasp_weight: float = 0.2
    defects4j_weight: float = 0.2
    evosuite_weight: float = 0.15
    synthetic_weight: float = 0.15
    
    # Sampling strategy
    use_stratification: bool = True
    stratify_by_cwe: bool = True
    stratify_by_complexity: bool = True
    
    # Deduplication
    enable_deduplication: bool = True
    dedup_strategy: str = "ast_hashing"  # or "semantic_fingerprinting"
    
    # Data augmentation
    augment: bool = True
    augmentation_transforms: List[str] = field(default_factory=lambda: [
        "variable_renaming",
        "statement_reordering",
        "code_refactoring",
        "exception_wrapping"
    ])
    
    # Total samples to generate
    total_samples: int = 0  # 0 means use default
    
    @property
    def source_weights(self) -> Dict[str, float]:
        """Get source weights as dict"""
        return {
            'juliet': self.juliet_weight,
            'owasp': self.owasp_weight,
            'defects4j': self.defects4j_weight,
            'evosuite': self.evosuite_weight,
            'synthetic': self.synthetic_weight,
        }


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    # Coverage metrics
    track_line_coverage: bool = True
    track_branch_coverage: bool = True
    track_method_coverage: bool = True
    
    # Diversity metrics
    track_path_diversity: bool = True
    track_execution_similarity: bool = True
    
    # Efficiency metrics
    track_coverage_per_test: bool = True
    track_redundancy_ratio: bool = True
    track_time_to_coverage: bool = True
    
    # Generalization metrics
    track_cross_project_performance: bool = True
    track_transformed_code_coverage: bool = True
    
    # Logging interval (epochs)
    evaluation_interval: int = 5


@dataclass
class TrainingProfileConfig:
    """Complete training profile configuration"""
    profile_type: ProfileType
    
    # Model hyperparameters
    epochs: int
    batch_size: int
    learning_rate: float
    lambda_gp: float
    n_critic: int
    
    # Rewards
    rewards: RewardConfig = field(default_factory=RewardConfig)
    
    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Evaluation
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Metadata
    description: str = ""
    
    @property
    def dataset_config(self):
        """Alias for dataset attribute"""
        return self.dataset
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "profile_type": self.profile_type.value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lambda_gp": self.lambda_gp,
            "n_critic": self.n_critic,
            "description": self.description,
        }
    
    def save(self, path: Path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# PREDEFINED PROFILES
# ============================================================================

COVERAGE_SEEKING_PROFILE = TrainingProfileConfig(
    profile_type=ProfileType.COVERAGE_SEEKING,
    epochs=200,
    batch_size=8,
    learning_rate=0.00003,
    lambda_gp=10.0,
    n_critic=12,
    description="""
    Coverage-Seeking Profile
    ========================
    
    Objective: Maximize structural exploration and code coverage
    
    Characteristics:
    - Very slow learning rate (0.00003) for careful exploration
    - High gradient penalty (10.0) for stable critic
    - Many critic iterations (12) for refined discrimination
    - Small batches (8) for more frequent updates
    - 200 epochs for thorough convergence
    
    Dataset Strategy:
    - Balanced across all sources
    - Heavy augmentation via code transformations
    - Stratification by cyclomatic complexity
    - Deduplication via AST hashing
    
    Reward Signals:
    - High weight on coverage (line + branch + method)
    - Path diversity strongly rewarded
    - Redundant paths penalized
    
    Best For: Research on structure learning, theoretical analysis
    Expected Duration: ~4-6 hours
    """,
    rewards=RewardConfig(
        line_coverage_weight=0.30,
        branch_coverage_weight=0.30,
        method_coverage_weight=0.30,
        path_diversity_weight=0.10,
    ),
    dataset=DatasetConfig(
        juliet_weight=0.25,
        owasp_weight=0.25,
        defects4j_weight=0.25,
        evosuite_weight=0.125,
        synthetic_weight=0.125,
        augment=True,
        augmentation_transforms=[
            "variable_renaming",
            "statement_reordering",
            "control_flow_flattening",
            "exception_wrapping",
        ]
    ),
    evaluation=EvaluationConfig(
        track_line_coverage=True,
        track_branch_coverage=True,
        track_method_coverage=True,
        track_path_diversity=True,
        evaluation_interval=5,
    )
)


VULNERABILITY_FOCUSED_PROFILE = TrainingProfileConfig(
    profile_type=ProfileType.VULNERABILITY_FOCUSED,
    epochs=150,
    batch_size=12,
    learning_rate=0.00005,
    lambda_gp=7.5,
    n_critic=8,
    description="""
    Vulnerability-Focused Profile
    ==============================
    
    Objective: Learn semantic patterns related to security weaknesses
    
    Characteristics:
    - Moderate learning rate (0.00005) for semantic stability
    - Strong gradient penalty (7.5) for critic robustness
    - Balanced critic iterations (8)
    - Medium batches (12)
    - 150 epochs for convergence
    
    Dataset Strategy:
    - Heavy bias toward Juliet (30%) and OWASP (30%)
    - Hard-negative mining (safe-but-similar code)
    - CWE-stratified sampling
    - Semantic deduplication
    
    Reward Signals:
    - Coverage rewards present but secondary
    - Emphasis on discovering weakness-adjacent code
    - Penalize trivial safe code generation
    
    Best For: Vulnerability detection research, security analysis
    Expected Duration: ~2-3 hours
    """,
    rewards=RewardConfig(
        line_coverage_weight=0.10,
        branch_coverage_weight=0.10,
        method_coverage_weight=0.10,
        path_diversity_weight=0.20,
    ),
    dataset=DatasetConfig(
        juliet_weight=0.35,
        owasp_weight=0.30,
        defects4j_weight=0.15,
        evosuite_weight=0.10,
        synthetic_weight=0.10,
        augment=True,
        stratify_by_cwe=True,
    ),
    evaluation=EvaluationConfig(
        track_line_coverage=True,
        track_cross_project_performance=True,
        evaluation_interval=5,
    )
)


GENERALIZATION_STRESS_PROFILE = TrainingProfileConfig(
    profile_type=ProfileType.GENERALIZATION_STRESS,
    epochs=250,
    batch_size=6,
    learning_rate=0.000025,
    lambda_gp=12.0,
    n_critic=15,
    description="""
    Generalization-Stress Profile
    ==============================
    
    Objective: Prevent overfitting via heavy augmentation and diversity
    
    Characteristics:
    - Very slow learning rate (0.000025) for extrapolation safety
    - Maximum gradient penalty (12.0) for critic constraint
    - Many critic iterations (15) for strict discrimination
    - Tiny batches (6) for maximum updates
    - 250 epochs for exhaustive convergence
    
    Dataset Strategy:
    - Heavy synthetic and Defects4J (30% each)
    - Aggressive code transformations
    - Cross-project validation splits
    - Maximal augmentation pipeline
    
    Reward Signals:
    - Strong path diversity (reduce memorization)
    - Penalize seen-before patterns heavily
    
    Best For: Generalization capability, overfitting prevention
    Expected Duration: ~6-8 hours
    """,
    rewards=RewardConfig(
        line_coverage_weight=0.15,
        branch_coverage_weight=0.15,
        method_coverage_weight=0.15,
        path_diversity_weight=0.35,
        redundancy_penalty=0.20,
    ),
    dataset=DatasetConfig(
        juliet_weight=0.15,
        owasp_weight=0.15,
        defects4j_weight=0.30,
        evosuite_weight=0.15,
        synthetic_weight=0.25,
        augment=True,
        augmentation_transforms=[
            "variable_renaming",
            "statement_reordering",
            "control_flow_flattening",
            "exception_wrapping",
            "field_reordering",
            "method_extraction",
        ]
    ),
    evaluation=EvaluationConfig(
        track_line_coverage=True,
        track_branch_coverage=True,
        track_method_coverage=True,
        track_path_diversity=True,
        track_cross_project_performance=True,
        track_transformed_code_coverage=True,
        evaluation_interval=5,
    )
)


HYBRID_REALISTIC_PROFILE = TrainingProfileConfig(
    profile_type=ProfileType.HYBRID_REALISTIC,
    epochs=120,
    batch_size=16,
    learning_rate=0.00005,
    lambda_gp=5.0,
    n_critic=7,
    description="""
    Hybrid Realistic Profile
    ========================
    
    Objective: Approximate real-world CI/testing environment
    
    Characteristics:
    - Balanced learning rate (0.00005) for production use
    - Standard gradient penalty (5.0)
    - Standard critic iterations (7)
    - Standard batches (16)
    - 120 epochs for practical convergence
    
    Dataset Strategy:
    - Balanced distribution across all sources
    - Moderate augmentation (common transforms),
    - Noise injection (incomplete coverage, flaky-style)
    - Realistic class distributions
    
    Reward Signals:
    - Balanced structural + semantic rewards
    - Realistic penalty structure
    
    Best For: Production deployment, practical testing
    Expected Duration: ~1.5-2 hours
    """,
    rewards=RewardConfig(
        line_coverage_weight=0.20,
        branch_coverage_weight=0.20,
        method_coverage_weight=0.20,
        path_diversity_weight=0.25,
        redundancy_penalty=0.15,
    ),
    dataset=DatasetConfig(
        juliet_weight=0.25,
        owasp_weight=0.20,
        defects4j_weight=0.25,
        evosuite_weight=0.15,
        synthetic_weight=0.15,
        augment=True,
        augmentation_transforms=[
            "variable_renaming",
            "statement_reordering",
            "exception_wrapping",
        ]
    ),
    evaluation=EvaluationConfig(
        track_line_coverage=True,
        track_branch_coverage=True,
        track_method_coverage=True,
        track_path_diversity=True,
        track_cross_project_performance=True,
        evaluation_interval=10,
    )
)


# ============================================================================
# PROFILE REGISTRY
# ============================================================================

PROFILES: Dict[ProfileType, TrainingProfileConfig] = {
    ProfileType.COVERAGE_SEEKING: COVERAGE_SEEKING_PROFILE,
    ProfileType.VULNERABILITY_FOCUSED: VULNERABILITY_FOCUSED_PROFILE,
    ProfileType.GENERALIZATION_STRESS: GENERALIZATION_STRESS_PROFILE,
    ProfileType.HYBRID_REALISTIC: HYBRID_REALISTIC_PROFILE,
}


def get_profile(profile_type: ProfileType) -> TrainingProfileConfig:
    """Get profile by type"""
    return PROFILES[profile_type]


def get_profile_config(profile_name: str) -> TrainingProfileConfig:
    """
    Get profile config by string name.
    
    Args:
        profile_name: Profile name (e.g. 'coverage-seeking', 'hybrid-realistic')
        
    Returns:
        TrainingProfileConfig for the profile
    """
    # Convert kebab-case to SNAKE_CASE
    profile_key = profile_name.upper().replace('-', '_')
    
    # Map string names to ProfileType
    name_map = {
        'COVERAGE_SEEKING': ProfileType.COVERAGE_SEEKING,
        'VULNERABILITY_FOCUSED': ProfileType.VULNERABILITY_FOCUSED,
        'GENERALIZATION_STRESS': ProfileType.GENERALIZATION_STRESS,
        'HYBRID_REALISTIC': ProfileType.HYBRID_REALISTIC,
    }
    
    if profile_key not in name_map:
        available = ', '.join(name_map.keys())
        raise ValueError(f"Unknown profile: {profile_name}. Available: {available}")
    
    profile_type = name_map[profile_key]
    return PROFILES[profile_type]


# Profile name descriptions (string key -> display name)
PROFILE_DESCRIPTIONS = {
    'coverage-seeking': COVERAGE_SEEKING_PROFILE.description,
    'vulnerability-focused': VULNERABILITY_FOCUSED_PROFILE.description,
    'generalization-stress': GENERALIZATION_STRESS_PROFILE.description,
    'hybrid-realistic': HYBRID_REALISTIC_PROFILE.description,
}


def list_profiles():
    """List all available profiles"""
    for profile_type in ProfileType:
        config = PROFILES[profile_type]
        print(f"\n{'='*70}")
        print(config.description)
        print(f"{'='*70}")


if __name__ == "__main__":
    list_profiles()
