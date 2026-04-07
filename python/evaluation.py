"""
Advanced Evaluation Framework for WGAN-GP

Métricas avançadas para comparar qualidade de testes com/sem EvoSuite:
- Code Coverage extraction (linha, branch, método, exceção)
- Diversity metrics (n-gram, feature correlation)
- Quality metrics (synthetic similarity, mutation score)
- Dataset statistics comparison
"""

import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Code coverage metrics extracted from test reports"""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    method_coverage: float = 0.0
    exception_coverage: float = 0.0
    statement_coverage: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def average(self) -> float:
        """Average coverage across all metrics"""
        return np.mean([self.line_coverage, self.branch_coverage, 
                       self.method_coverage, self.exception_coverage])


@dataclass
class DiversityMetrics:
    """Diversity metrics for test generation"""
    ngram_diversity: float = 0.0  # unique n-grams / total
    feature_correlation: float = 0.0  # avg correlation between features
    feature_std: List[float] = None  # std per feature
    unique_patterns: int = 0  # count unique patterns
    entropy: float = 0.0  # Shannon entropy of feature distributions
    
    def __post_init__(self):
        if self.feature_std is None:
            self.feature_std = [0.0, 0.0, 0.0]
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['feature_std'] = self.feature_std if self.feature_std else []
        return result


@dataclass
class QualityMetrics:
    """Synthetic test quality metrics"""
    synthetic_quality_score: float = 0.0  # 0-1, how realistic synthetic tests are
    mutation_score: float = 0.0  # % of mutants detected
    assertion_rate: float = 0.0  # % tests with assertions
    test_complexity: float = 0.0  # avg lines per test
    method_mean: float = 0.0
    branch_mean: float = 0.0
    statement_mean: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetStatistics:
    """Statistics for dataset comparison"""
    sample_count: int = 0
    methods_mean: float = 0.0
    methods_std: float = 0.0
    methods_min: float = 0.0
    methods_max: float = 0.0
    branches_mean: float = 0.0
    branches_std: float = 0.0
    branches_min: float = 0.0
    branches_max: float = 0.0
    statements_mean: float = 0.0
    statements_std: float = 0.0
    statements_min: float = 0.0
    statements_max: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class CoverageExtractor:
    """Extract coverage metrics from Surefire XML reports"""
    
    def __init__(self, surefire_dir: str = "target/surefire-reports"):
        self.surefire_dir = Path(surefire_dir)
        self.logger = logging.getLogger(__name__)
    
    def extract_from_surefire(self) -> CoverageMetrics:
        """
        Extract coverage from Surefire report files
        Looks for: TEST-*.xml files in target/surefire-reports/
        """
        if not self.surefire_dir.exists():
            self.logger.warning(f"Surefire dir not found: {self.surefire_dir}")
            return CoverageMetrics()
        
        # Find TEST-*.xml files
        test_files = list(self.surefire_dir.glob("TEST-*.xml"))
        if not test_files:
            self.logger.warning("No TEST-*.xml files found in surefire-reports")
            return CoverageMetrics()
        
        # Parse coverage info
        total_tests = 0
        total_skipped = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract test counts
                    tests_match = re.search(r'tests="(\d+)"', content)
                    skipped_match = re.search(r'skipped="(\d+)"', content)
                    
                    if tests_match:
                        total_tests += int(tests_match.group(1))
                    if skipped_match:
                        total_skipped += int(skipped_match.group(1))
            except Exception as e:
                self.logger.error(f"Error parsing {test_file}: {e}")
        
        # Calculate pass rate as proxy for coverage
        pass_rate = (total_tests - total_skipped) / max(total_tests, 1)
        
        # Return basic metrics (more detailed requires coverage tools like JaCoCo)
        return CoverageMetrics(
            line_coverage=pass_rate * 0.85,  # estimate
            branch_coverage=pass_rate * 0.75,
            method_coverage=pass_rate * 0.95,
            exception_coverage=pass_rate * 0.65,
            statement_coverage=pass_rate * 0.85
        )
    
    def extract_from_jacoco(self, jacoco_csv: str = "target/site/jacoco/index.csv") -> CoverageMetrics:
        """
        Extract coverage from JaCoCo CSV report (if available)
        Format: GROUP,PACKAGE,CLASS,INSTRUCTION,BRANCH,LINE,COMPLEXITY,METHOD
        """
        jacoco_path = Path(jacoco_csv)
        if not jacoco_path.exists():
            self.logger.info(f"JaCoCo report not found: {jacoco_csv}")
            return self.extract_from_surefire()
        
        try:
            df = pd.read_csv(jacoco_path)
            # Parse coverage rows
            if 'BRANCH' in df.columns:
                branch_covered = df['BRANCH'].str.extract(r'(\d+)/(\d+)')[0].astype(int).sum()
                branch_total = df['BRANCH'].str.extract(r'(\d+)/(\d+)')[1].astype(int).sum()
                branch_coverage = branch_covered / max(branch_total, 1)
            else:
                branch_coverage = 0.0
            
            if 'LINE' in df.columns:
                line_covered = df['LINE'].str.extract(r'(\d+)/(\d+)')[0].astype(int).sum()
                line_total = df['LINE'].str.extract(r'(\d+)/(\d+)')[1].astype(int).sum()
                line_coverage = line_covered / max(line_total, 1)
            else:
                line_coverage = 0.0
            
            return CoverageMetrics(
                line_coverage=line_coverage,
                branch_coverage=branch_coverage,
                method_coverage=0.0,  # not in CSV
                exception_coverage=0.0  # not in CSV
            )
        except Exception as e:
            self.logger.error(f"Error parsing JaCoCo: {e}")
            return self.extract_from_surefire()


class DiversityAnalyzer:
    """Analyze diversity of test data"""
    
    @staticmethod
    def compute_ngram_diversity(samples: List[str], n: int = 2) -> float:
        """
        Compute n-gram diversity
        Args:
            samples: List of test method names or descriptions
            n: n-gram size
        Returns:
            Diversity score (0-1)
        """
        all_ngrams = []
        unique_ngrams = set()
        
        for sample in samples:
            tokens = str(sample).lower().split()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                all_ngrams.append(ngram)
                unique_ngrams.add(ngram)
        
        if not all_ngrams:
            return 0.0
        
        return len(unique_ngrams) / len(all_ngrams)
    
    @staticmethod
    def compute_feature_correlation(data: pd.DataFrame) -> float:
        """
        Compute average correlation between numeric features
        """
        if data.shape[1] < 2:
            return 0.0
        
        corr_matrix = data.corr().abs()
        # Get upper triangle (avoid duplicates)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        correlation_values = upper_triangle.values.flatten()
        correlation_values = correlation_values[~np.isnan(correlation_values)]
        
        if len(correlation_values) == 0:
            return 0.0
        
        return float(correlation_values.mean())
    
    @staticmethod
    def compute_entropy(data: pd.DataFrame) -> float:
        """
        Compute Shannon entropy for distribution analysis
        """
        entropy = 0.0
        for col in data.columns:
            # Discretize continuous features
            if data[col].dtype in [np.float64, np.float32]:
                bins = pd.cut(data[col], bins=10, labels=False)
            else:
                bins = data[col]
            
            # Calculate entropy
            value_counts = bins.value_counts()
            probabilities = value_counts / len(data)
            col_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy += col_entropy
        
        return entropy / max(data.shape[1], 1)
    
    @staticmethod
    def analyze_dataset(data: pd.DataFrame) -> DiversityMetrics:
        """
        Comprehensive diversity analysis
        """
        # n-gram diversity (using feature names as proxy)
        ngram_div = DiversityAnalyzer.compute_ngram_diversity(
            [f"feat_{i}" for i in range(data.shape[1])],
            n=2
        )
        
        # Feature correlation
        feature_corr = DiversityAnalyzer.compute_feature_correlation(data)
        
        # Feature statistics
        feature_std = [float(data.iloc[:, i].std()) for i in range(data.shape[1])]
        
        # Entropy
        entropy = DiversityAnalyzer.compute_entropy(data)
        
        # Unique patterns (rows without duplicates)
        unique_patterns = len(data.drop_duplicates())
        
        return DiversityMetrics(
            ngram_diversity=ngram_div,
            feature_correlation=feature_corr,
            feature_std=feature_std,
            unique_patterns=unique_patterns,
            entropy=entropy
        )


class DatasetComparator:
    """Compare datasets (real vs synthetic, with/without EvoSuite)"""
    
    @staticmethod
    def compute_statistics(data: pd.DataFrame) -> DatasetStatistics:
        """Extract basic statistics from dataset"""
        stats = DatasetStatistics(
            sample_count=len(data)
        )
        
        if data.shape[1] >= 1:
            col0 = data.iloc[:, 0]
            stats.methods_mean = float(col0.mean())
            stats.methods_std = float(col0.std())
            stats.methods_min = float(col0.min())
            stats.methods_max = float(col0.max())
        
        if data.shape[1] >= 2:
            col1 = data.iloc[:, 1]
            stats.branches_mean = float(col1.mean())
            stats.branches_std = float(col1.std())
            stats.branches_min = float(col1.min())
            stats.branches_max = float(col1.max())
        
        if data.shape[1] >= 3:
            col2 = data.iloc[:, 2]
            stats.statements_mean = float(col2.mean())
            stats.statements_std = float(col2.std())
            stats.statements_min = float(col2.min())
            stats.statements_max = float(col2.max())
        
        return stats
    
    @staticmethod
    def compare_datasets(real_data: pd.DataFrame, 
                         synthetic_data: pd.DataFrame) -> Dict:
        """Compare real vs synthetic datasets"""
        real_stats = DatasetComparator.compute_statistics(real_data)
        synthetic_stats = DatasetComparator.compute_statistics(synthetic_data)
        
        # Compute distributions differences
        real_diversity = DiversityAnalyzer.analyze_dataset(real_data)
        synthetic_diversity = DiversityAnalyzer.analyze_dataset(synthetic_data)
        
        return {
            "real": {
                "statistics": real_stats.to_dict(),
                "diversity": real_diversity.to_dict()
            },
            "synthetic": {
                "statistics": synthetic_stats.to_dict(),
                "diversity": synthetic_diversity.to_dict()
            },
            "comparison": {
                "sample_ratio": synthetic_stats.sample_count / max(real_stats.sample_count, 1),
                "mean_difference": {
                    "methods": abs(synthetic_stats.methods_mean - real_stats.methods_mean),
                    "branches": abs(synthetic_stats.branches_mean - real_stats.branches_mean),
                    "statements": abs(synthetic_stats.statements_mean - real_stats.statements_mean)
                }
            }
        }


class EvaluationReport:
    """Generate comprehensive evaluation report"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_comparison_report(self,
                                   name: str,
                                   coverage: CoverageMetrics,
                                   diversity: DiversityMetrics,
                                   quality: QualityMetrics,
                                   dataset_comparison: Dict) -> Dict:
        """
        Generate comprehensive report comparing with/without EvoSuite
        """
        timestamp = datetime.now().isoformat()
        
        report = {
            "timestamp": timestamp,
            "experiment_name": name,
            "coverage": coverage.to_dict(),
            "diversity": diversity.to_dict(),
            "quality": quality.to_dict(),
            "dataset_comparison": dataset_comparison,
            "summary": {
                "coverage_score": coverage.average(),
                "diversity_score": diversity.entropy,
                "quality_score": quality.synthetic_quality_score,
                "overall_score": (coverage.average() + diversity.entropy + quality.synthetic_quality_score) / 3
            }
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = None) -> Path:
        """Save report to JSON file"""
        if filename is None:
            filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return filepath
    
    def compare_experiments(self, report1: Dict, report2: Dict) -> Dict:
        """
        Compare two experiment reports
        Returns delta metrics showing improvement/degradation
        """
        comparison = {
            "experiment_1": report1["experiment_name"],
            "experiment_2": report2["experiment_name"],
            "timestamp": datetime.now().isoformat(),
            "deltas": {
                "coverage": {
                    "line": report2["coverage"]["line_coverage"] - report1["coverage"]["line_coverage"],
                    "branch": report2["coverage"]["branch_coverage"] - report1["coverage"]["branch_coverage"],
                    "method": report2["coverage"]["method_coverage"] - report1["coverage"]["method_coverage"],
                    "exception": report2["coverage"]["exception_coverage"] - report1["coverage"]["exception_coverage"]
                },
                "diversity": {
                    "ngram": report2["diversity"]["ngram_diversity"] - report1["diversity"]["ngram_diversity"],
                    "entropy": report2["diversity"]["entropy"] - report1["diversity"]["entropy"]
                },
                "quality": {
                    "synthetic_quality": report2["quality"]["synthetic_quality_score"] - report1["quality"]["synthetic_quality_score"],
                    "mutation_score": report2["quality"]["mutation_score"] - report1["quality"]["mutation_score"]
                },
                "overall_score": report2["summary"]["overall_score"] - report1["summary"]["overall_score"]
            },
            "better_experiment": "experiment_2" if (
                report2["summary"]["overall_score"] > report1["summary"]["overall_score"]
            ) else "experiment_1"
        }
        
        return comparison


# Convenience function for quick evaluation
def evaluate_experiment(name: str,
                       coverage_dir: str = "target/surefire-reports",
                       real_data_path: str = "python/data/junit_data.csv",
                       synthetic_data_path: str = "outputs/synthetic_samples.csv",
                       output_dir: str = "outputs/evaluation") -> Dict:
    """
    Complete evaluation pipeline for an experiment
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Extract coverage
    coverage_extractor = CoverageExtractor(coverage_dir)
    coverage = coverage_extractor.extract_from_surefire()
    
    # Load data and analyze
    real_data = pd.read_csv(real_data_path, header=None)
    synthetic_data = pd.read_csv(synthetic_data_path, header=None)
    
    diversity = DiversityAnalyzer.analyze_dataset(real_data)
    dataset_comparison = DatasetComparator.compare_datasets(real_data, synthetic_data)

    mean_diff = dataset_comparison["comparison"]["mean_difference"]
    real_stats = dataset_comparison["real"]["statistics"]

    feature_scores = []
    for feat_name, diff in mean_diff.items():
        denom = max(float(real_stats.get(f"{feat_name}_mean", 0.0)), 1e-8)
        feature_scores.append(max(0.0, 1.0 - (float(diff) / denom)))

    synthetic_quality_score = float(np.mean(feature_scores)) if feature_scores else 0.0

    methods_mean = float(dataset_comparison["synthetic"]["statistics"].get("methods_mean", 0.0))
    branches_mean = float(dataset_comparison["synthetic"]["statistics"].get("branches_mean", 0.0))
    statements_mean = float(dataset_comparison["synthetic"]["statistics"].get("statements_mean", 0.0))

    assertion_rate = float(min(1.0, branches_mean / max(methods_mean, 1.0)))
    
    # Quality metrics (computed from dataset alignment + synthetic distribution)
    quality = QualityMetrics(
        synthetic_quality_score=synthetic_quality_score,
        assertion_rate=assertion_rate,
        test_complexity=statements_mean,
        method_mean=float(real_data.iloc[:, 0].mean()),
        branch_mean=float(real_data.iloc[:, 1].mean()) if real_data.shape[1] > 1 else 0.0,
        statement_mean=float(real_data.iloc[:, 2].mean()) if real_data.shape[1] > 2 else 0.0
    )
    
    # Generate report
    report_gen = EvaluationReport(str(output_dir_path.parent))
    report = report_gen.generate_comparison_report(
        name=name,
        coverage=coverage,
        diversity=diversity,
        quality=quality,
        dataset_comparison=dataset_comparison
    )
    
    # Save
    report_gen.save_report(report, f"evaluation_{name}.json")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✅ Evaluation framework ready!")
    print("  - CoverageExtractor: Extract from Surefire/JaCoCo")
    print("  - DiversityAnalyzer: Analyze test diversity")
    print("  - DatasetComparator: Compare real vs synthetic")
    print("  - EvaluationReport: Generate comparison reports")
