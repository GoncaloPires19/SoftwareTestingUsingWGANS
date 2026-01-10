"""
Advanced metrics for WGAN evaluation:
- Diversity: n-gram and structural diversity
- FID-like: distance between generated and real distributions
- CWE Coverage: vulnerability coverage from synthetic samples
- Code Similarity: embedding-based similarity (CodeBERT)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import Counter
import torch
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class DiversityMetric:
    """Measure diversity of generated tests using n-grams and structural features"""
    
    def __init__(self, n: int = 2):
        """
        Args:
            n: n-gram size for diversity measurement
        """
        self.n = n
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2) -> set:
        """Extract n-grams from test method names/comments"""
        tokens = text.lower().split()
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def compute_diversity(self, samples: List[str]) -> float:
        """
        Compute diversity as unique n-grams / total n-grams
        Higher = more diverse
        """
        all_ngrams = []
        unique_ngrams = set()
        
        for sample in samples:
            ngrams = self.extract_ngrams(sample, self.n)
            all_ngrams.extend(ngrams)
            unique_ngrams.update(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        diversity = len(unique_ngrams) / len(all_ngrams)
        return float(diversity)
    
    def compute_from_dataframe(self, df: pd.DataFrame, text_column: str = "test_code") -> float:
        """Compute diversity from DataFrame"""
        samples = df[text_column].fillna("").tolist()
        return self.compute_diversity(samples)


class FeatureDiversityMetric:
    """Measure diversity in numerical features (methods, branches, statements)"""
    
    @staticmethod
    def compute_feature_diversity(features: np.ndarray) -> Dict[str, float]:
        """
        Compute diversity metrics on feature vectors
        Args:
            features: (N, D) array of feature vectors
        Returns:
            Dictionary with entropy, variance, correlation metrics
        """
        if len(features) < 2:
            return {"entropy": 0.0, "variance": 0.0}
        
        # Normalize features
        features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
        
        # Entropy per feature
        entropies = []
        for col in range(features_norm.shape[1]):
            hist, _ = np.histogram(features_norm[:, col], bins=10, range=(0, 1))
            hist = hist / hist.sum()
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(entropy)
        
        # Average variance
        variance = np.var(features_norm, axis=0).mean()
        
        # Pairwise correlation (lower = more diverse)
        if features_norm.shape[1] > 1:
            corr_matrix = np.corrcoef(features_norm.T)
            avg_corr = np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]).mean()
        else:
            avg_corr = 0.0
        
        return {
            "entropy_mean": float(np.mean(entropies)),
            "variance": float(variance),
            "correlation": float(avg_corr),
        }


class WassersteinDistanceMetric:
    """Compute Wasserstein distance between real and generated distributions"""
    
    @staticmethod
    def compute_1d_wasserstein(real_samples: np.ndarray, fake_samples: np.ndarray) -> float:
        """
        Compute 1D Wasserstein distance
        Good approximation: sort and compare CDFs
        """
        real_sorted = np.sort(real_samples.flatten())
        fake_sorted = np.sort(fake_samples.flatten())
        
        # Pad shorter array
        max_len = max(len(real_sorted), len(fake_sorted))
        real_sorted = np.pad(real_sorted, (0, max_len - len(real_sorted)), mode='edge')
        fake_sorted = np.pad(fake_sorted, (0, max_len - len(fake_sorted)), mode='edge')
        
        # Wasserstein distance
        distance = np.mean(np.abs(real_sorted - fake_sorted))
        return float(distance)
    
    @staticmethod
    def compute_sliced_wasserstein(real_samples: np.ndarray, fake_samples: np.ndarray, 
                                   num_projections: int = 50) -> float:
        """
        Compute sliced Wasserstein distance for high-dimensional data
        """
        if real_samples.ndim == 1:
            return WassersteinDistanceMetric.compute_1d_wasserstein(real_samples, fake_samples)
        
        distances = []
        for _ in range(num_projections):
            # Random projection direction
            theta = np.random.randn(real_samples.shape[1])
            theta = theta / np.linalg.norm(theta)
            
            # Project
            real_proj = real_samples @ theta
            fake_proj = fake_samples @ theta
            
            # 1D Wasserstein
            dist = WassersteinDistanceMetric.compute_1d_wasserstein(real_proj, fake_proj)
            distances.append(dist)
        
        return float(np.mean(distances))


class FIDMetric:
    """
    Fréchet Inception Distance adapted for test code.
    Uses embedding distance instead of Inception features.
    """
    
    def __init__(self, embedding_model_name: str = "microsoft/codebert-base"):
        """
        Args:
            embedding_model_name: Name of CodeBERT or similar model
        """
        self.model_name = embedding_model_name
        self.embeddings_cache = {}
    
    def compute_statistics(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of embeddings"""
        mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings.T)
        return mean, cov
    
    def compute_fid(self, real_embeddings: np.ndarray, fake_embeddings: np.ndarray) -> float:
        """
        Compute FID between real and generated embeddings
        Lower = better (distributions are closer)
        """
        real_mean, real_cov = self.compute_statistics(real_embeddings)
        fake_mean, fake_cov = self.compute_statistics(fake_embeddings)
        
        # Euclidean distance between means
        mean_diff = np.sum((real_mean - fake_mean) ** 2)
        
        # Trace of covariance difference
        try:
            # Matrix square root
            from scipy.linalg import sqrtm
            cov_diff = sqrtm(real_cov @ fake_cov)
            trace_diff = np.trace(real_cov + fake_cov - 2 * cov_diff)
        except:
            # Fallback: use frobenius norm
            trace_diff = np.sum((real_cov - fake_cov) ** 2)
        
        fid = mean_diff + trace_diff
        return float(np.sqrt(max(fid, 0)))


class CWECoverageMetric:
    """Track coverage of CWE (Common Weakness Enumeration) types in generated samples"""
    
    def __init__(self):
        self.cwe_counts = Counter()
    
    def track_cwe_distribution(self, df: pd.DataFrame, cwe_column: str = "cwe") -> Dict[str, float]:
        """
        Analyze CWE distribution in data
        Args:
            df: DataFrame with CWE information
            cwe_column: Column name containing CWE IDs/types
        Returns:
            Dictionary with coverage statistics
        """
        if cwe_column not in df.columns:
            return {"coverage": 0.0, "num_cwe_types": 0}
        
        cwes = df[cwe_column].dropna().unique()
        num_cwe_types = len(cwes)
        
        # If we have reference data, compute coverage
        return {
            "num_cwe_types_generated": num_cwe_types,
            "cwe_distribution": dict(df[cwe_column].value_counts()),
        }
    
    def compute_coverage_ratio(self, generated_cwes: set, reference_cwes: set) -> float:
        """
        Compute what % of reference CWEs are represented in generated data
        """
        if not reference_cwes:
            return 0.0
        
        coverage = len(generated_cwes & reference_cwes) / len(reference_cwes)
        return float(coverage)


class MetricsTracker:
    """Track all metrics during training"""
    
    def __init__(self, metrics_list: List[str] = None):
        """
        Args:
            metrics_list: List of metric names to track
                         (diversity, fid, wasserstein, cwe_coverage, feature_diversity)
        """
        self.metrics_list = metrics_list or ["diversity", "feature_diversity", "wasserstein"]
        self.history = {metric: [] for metric in self.metrics_list}
        self.epoch_count = 0
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update tracked metrics"""
        for metric_name, value in metrics_dict.items():
            if metric_name in self.history:
                self.history[metric_name].append(value)
        self.epoch_count += 1
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame"""
        data = {"epoch": list(range(len(next(iter(self.history.values())))))}
        data.update(self.history)
        return pd.DataFrame(data)
    
    def get_best_epoch(self, metric: str, higher_is_better: bool = True) -> Tuple[int, float]:
        """Get epoch with best metric value"""
        if metric not in self.history or not self.history[metric]:
            return 0, 0.0
        
        values = self.history[metric]
        if higher_is_better:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx, float(values[best_idx])
    
    def print_summary(self):
        """Print metrics summary"""
        print("\n" + "="*70)
        print("📊 METRICS SUMMARY")
        print("="*70)
        
        for metric_name in self.metrics_list:
            if metric_name in self.history and self.history[metric_name]:
                values = self.history[metric_name]
                print(f"\n{metric_name}:")
                print(f"  Current: {values[-1]:.4f}")
                print(f"  Best: {np.max(values):.4f}")
                print(f"  Worst: {np.min(values):.4f}")
                print(f"  Mean: {np.mean(values):.4f}")
                print(f"  Trend: {'↑ improving' if values[-1] > np.mean(values[:-1]) else '↓ declining'}")


def compute_all_metrics(
    real_samples: np.ndarray,
    fake_samples: np.ndarray,
    real_df: Optional[pd.DataFrame] = None,
    fake_df: Optional[pd.DataFrame] = None,
    metrics_to_compute: List[str] = None
) -> Dict[str, float]:
    """
    Compute all requested metrics
    Args:
        real_samples: (N, D) array of real feature vectors
        fake_samples: (M, D) array of generated feature vectors
        real_df: DataFrame with real test data (optional, for diversity)
        fake_df: DataFrame with synthetic test data (optional, for diversity)
        metrics_to_compute: List of metric names
    Returns:
        Dictionary with computed metrics
    """
    
    if metrics_to_compute is None:
        metrics_to_compute = ["diversity", "feature_diversity", "wasserstein"]
    
    metrics = {}
    
    # Feature diversity
    if "feature_diversity" in metrics_to_compute:
        try:
            fd = FeatureDiversityMetric.compute_feature_diversity(fake_samples)
            metrics.update({f"feature_diversity_{k}": v for k, v in fd.items()})
        except Exception as e:
            logger.warning(f"Failed to compute feature diversity: {e}")
    
    # Wasserstein distance
    if "wasserstein" in metrics_to_compute:
        try:
            wd = WassersteinDistanceMetric.compute_sliced_wasserstein(real_samples, fake_samples)
            metrics["wasserstein_distance"] = wd
        except Exception as e:
            logger.warning(f"Failed to compute Wasserstein: {e}")
    
    # Diversity (text-based, if DataFrames provided)
    if "diversity" in metrics_to_compute and fake_df is not None:
        try:
            div = DiversityMetric(n=2)
            # Estimate from feature values as proxy
            fake_str = fake_df.astype(str).agg(' '.join, axis=1).tolist() if hasattr(fake_df, 'agg') else []
            if fake_str:
                div_score = div.compute_diversity(fake_str)
            else:
                # Fallback: use feature diversity
                div_score = metrics.get("feature_diversity_entropy_mean", 0.5)
            metrics["diversity"] = div_score
        except Exception as e:
            logger.warning(f"Failed to compute diversity: {e}")
    
    # CWE coverage
    if "cwe_coverage" in metrics_to_compute and fake_df is not None and "cwe" in fake_df.columns:
        try:
            cwe_tracker = CWECoverageMetric()
            cwe_info = cwe_tracker.track_cwe_distribution(fake_df)
            metrics["cwe_types"] = cwe_info.get("num_cwe_types_generated", 0)
        except Exception as e:
            logger.warning(f"Failed to compute CWE coverage: {e}")
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    print("🧪 Testing metrics system...")
    
    # Synthetic data
    real_data = np.random.randn(100, 3)
    fake_data = np.random.randn(50, 3) + 0.5  # Slightly different distribution
    
    metrics = compute_all_metrics(
        real_data,
        fake_data,
        metrics_to_compute=["feature_diversity", "wasserstein"]
    )
    
    print("\n✅ Computed metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
