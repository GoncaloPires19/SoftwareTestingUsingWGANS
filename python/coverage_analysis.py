"""
Coverage Analysis Module - Extract and Track Program Coverage Metrics
=====================================================================

Provides standardized coverage analysis across all test sources:
- Line Coverage
- Branch Coverage  
- Method Coverage
- Execution Path Tracking
- Coverage-based Metrics

Uses JaCoCo-compatible format for consistency.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from pathlib import Path
import json
from enum import Enum
import hashlib


class CoverageType(Enum):
    """Types of code coverage"""
    LINE = "line"
    BRANCH = "branch"
    METHOD = "method"
    PATH = "path"


@dataclass
class CoverageMetrics:
    """Coverage metrics for a single test/program"""
    test_id: str
    
    # Coverage counts
    lines_covered: Set[int] = field(default_factory=set)
    branches_covered: Set[int] = field(default_factory=set)
    methods_covered: Set[str] = field(default_factory=set)
    execution_paths: List[str] = field(default_factory=list)
    
    # Totals (for reference)
    total_lines_available: int = 0
    total_branches_available: int = 0
    total_methods_available: int = 0
    
    # Metadata
    execution_time_ms: float = 0.0
    test_passed: bool = True
    
    @property
    def line_coverage_pct(self) -> float:
        """Line coverage percentage"""
        if self.total_lines_available == 0:
            return 0.0
        return (len(self.lines_covered) / self.total_lines_available) * 100
    
    @property
    def branch_coverage_pct(self) -> float:
        """Branch coverage percentage"""
        if self.total_branches_available == 0:
            return 0.0
        return (len(self.branches_covered) / self.total_branches_available) * 100
    
    @property
    def method_coverage_pct(self) -> float:
        """Method coverage percentage"""
        if self.total_methods_available == 0:
            return 0.0
        return (len(self.methods_covered) / self.total_methods_available) * 100
    
    @property
    def combined_coverage(self) -> float:
        """Combined coverage score (average of all metrics)"""
        metrics = [
            self.line_coverage_pct,
            self.branch_coverage_pct,
            self.method_coverage_pct,
        ]
        return sum(metrics) / len(metrics) if metrics else 0.0
    
    def path_hash(self) -> str:
        """Generate hash of execution path for deduplication"""
        path_str = "->".join(self.execution_paths)
        return hashlib.md5(path_str.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "line_coverage": self.line_coverage_pct,
            "branch_coverage": self.branch_coverage_pct,
            "method_coverage": self.method_coverage_pct,
            "combined_coverage": self.combined_coverage,
            "execution_time_ms": self.execution_time_ms,
            "test_passed": self.test_passed,
        }


@dataclass
class AggregatedCoverage:
    """Aggregated coverage across multiple tests"""
    test_metrics: List[CoverageMetrics] = field(default_factory=list)
    
    @property
    def total_line_coverage(self) -> float:
        """Average line coverage across all tests"""
        if not self.test_metrics:
            return 0.0
        return sum(m.line_coverage_pct for m in self.test_metrics) / len(self.test_metrics)
    
    @property
    def total_branch_coverage(self) -> float:
        """Average branch coverage"""
        if not self.test_metrics:
            return 0.0
        return sum(m.branch_coverage_pct for m in self.test_metrics) / len(self.test_metrics)
    
    @property
    def total_method_coverage(self) -> float:
        """Average method coverage"""
        if not self.test_metrics:
            return 0.0
        return sum(m.method_coverage_pct for m in self.test_metrics) / len(self.test_metrics)
    
    @property
    def unique_lines_covered(self) -> int:
        """Total unique lines covered by any test"""
        all_lines = set()
        for m in self.test_metrics:
            all_lines.update(m.lines_covered)
        return len(all_lines)
    
    @property
    def unique_branches_covered(self) -> int:
        """Total unique branches covered"""
        all_branches = set()
        for m in self.test_metrics:
            all_branches.update(m.branches_covered)
        return len(all_branches)
    
    @property
    def unique_methods_covered(self) -> int:
        """Total unique methods covered"""
        all_methods = set()
        for m in self.test_metrics:
            all_methods.update(m.methods_covered)
        return len(all_methods)
    
    @property
    def redundancy_ratio(self) -> float:
        """Ratio of redundant tests (paths already covered)"""
        if len(self.test_metrics) <= 1:
            return 0.0
        
        seen_paths = set()
        redundant_count = 0
        
        for metric in self.test_metrics:
            path_hash = metric.path_hash()
            if path_hash in seen_paths:
                redundant_count += 1
            seen_paths.add(path_hash)
        
        return redundant_count / len(self.test_metrics)
    
    @property
    def diversity_score(self) -> float:
        """Diversity score: 1 - redundancy_ratio"""
        return 1.0 - self.redundancy_ratio
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "total_tests": len(self.test_metrics),
            "avg_line_coverage": self.total_line_coverage,
            "avg_branch_coverage": self.total_branch_coverage,
            "avg_method_coverage": self.total_method_coverage,
            "unique_lines_covered": self.unique_lines_covered,
            "unique_branches_covered": self.unique_branches_covered,
            "unique_methods_covered": self.unique_methods_covered,
            "redundancy_ratio": self.redundancy_ratio,
            "diversity_score": self.diversity_score,
        }


class CoverageAnalyzer:
    """Analyze and track code coverage"""
    
    @staticmethod
    def parse_jacoco_report(jacoco_xml_path: Path) -> Optional[CoverageMetrics]:
        """Parse JaCoCo XML coverage report"""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(jacoco_xml_path)
            root = tree.getroot()
            
            metrics = CoverageMetrics(test_id=jacoco_xml_path.stem)
            
            # Parse line coverage
            for counter in root.findall(".//counter[@type='LINE']"):
                covered = int(counter.get('covered', 0))
                missed = int(counter.get('missed', 0))
                metrics.total_lines_available = covered + missed
                metrics.lines_covered = set(range(covered))  # Simplified
            
            # Parse branch coverage
            for counter in root.findall(".//counter[@type='BRANCH']"):
                covered = int(counter.get('covered', 0))
                missed = int(counter.get('missed', 0))
                metrics.total_branches_available = covered + missed
                metrics.branches_covered = set(range(covered))
            
            # Parse method coverage
            for counter in root.findall(".//counter[@type='METHOD']"):
                covered = int(counter.get('covered', 0))
                missed = int(counter.get('missed', 0))
                metrics.total_methods_available = covered + missed
                metrics.methods_covered = set(range(covered))
            
            return metrics
        
        except Exception as e:
            print(f"Error parsing JaCoCo report: {e}")
            return None
    
    @staticmethod
    def create_from_dict(data: dict) -> CoverageMetrics:
        """Create metrics from dictionary"""
        metrics = CoverageMetrics(test_id=data.get("test_id", "unknown"))
        
        if "lines_covered" in data:
            metrics.lines_covered = set(data["lines_covered"])
        if "branches_covered" in data:
            metrics.branches_covered = set(data["branches_covered"])
        if "methods_covered" in data:
            metrics.methods_covered = set(data["methods_covered"])
        if "total_lines_available" in data:
            metrics.total_lines_available = data["total_lines_available"]
        if "total_branches_available" in data:
            metrics.total_branches_available = data["total_branches_available"]
        if "total_methods_available" in data:
            metrics.total_methods_available = data["total_methods_available"]
        
        return metrics
    
    @staticmethod
    def aggregate(metrics_list: List[CoverageMetrics]) -> AggregatedCoverage:
        """Aggregate multiple coverage metrics"""
        agg = AggregatedCoverage()
        agg.test_metrics = metrics_list
        return agg
    
    @staticmethod
    def calculate_delta_coverage(before: AggregatedCoverage, 
                                 after: AggregatedCoverage) -> Dict[str, float]:
        """Calculate coverage delta between two states"""
        return {
            "delta_line_coverage": after.total_line_coverage - before.total_line_coverage,
            "delta_branch_coverage": after.total_branch_coverage - before.total_branch_coverage,
            "delta_method_coverage": after.total_method_coverage - before.total_method_coverage,
            "delta_unique_lines": after.unique_lines_covered - before.unique_lines_covered,
            "delta_unique_branches": after.unique_branches_covered - before.unique_branches_covered,
            "delta_diversity": after.diversity_score - before.diversity_score,
        }


class EpochCoverageTracker:
    """Track coverage metrics across training epochs"""
    
    def __init__(self):
        self.epoch_data: Dict[int, Dict] = {}
        self.best_coverage: float = 0.0
        self.best_diversity: float = 0.0
    
    def record_epoch(self, epoch: int, coverage_metrics: AggregatedCoverage):
        """Record coverage metrics for an epoch"""
        self.epoch_data[epoch] = coverage_metrics.to_dict()
        
        combined = (coverage_metrics.total_line_coverage +
                   coverage_metrics.total_branch_coverage +
                   coverage_metrics.total_method_coverage) / 3
        
        if combined > self.best_coverage:
            self.best_coverage = combined
        if coverage_metrics.diversity_score > self.best_diversity:
            self.best_diversity = coverage_metrics.diversity_score
    
    def get_coverage_trajectory(self) -> List[Tuple[int, float]]:
        """Get coverage improvement trajectory"""
        trajectory = []
        for epoch in sorted(self.epoch_data.keys()):
            data = self.epoch_data[epoch]
            combined = (data["avg_line_coverage"] +
                       data["avg_branch_coverage"] +
                       data["avg_method_coverage"]) / 3
            trajectory.append((epoch, combined))
        return trajectory
    
    def save(self, path: Path):
        """Save to JSON"""
        with open(path, 'w') as f:
            json.dump({
                "epoch_data": self.epoch_data,
                "best_coverage": self.best_coverage,
                "best_diversity": self.best_diversity,
            }, f, indent=2)
    
    def report(self) -> str:
        """Generate text report"""
        lines = [
            "COVERAGE TRACKING REPORT",
            "=" * 50,
            f"Best Combined Coverage: {self.best_coverage:.2f}%",
            f"Best Diversity Score: {self.best_diversity:.4f}",
            f"Total Epochs Tracked: {len(self.epoch_data)}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    metrics = CoverageMetrics(
        test_id="test_1",
        lines_covered={1, 2, 3, 5, 7},
        total_lines_available=10,
        branches_covered={1, 3},
        total_branches_available=5,
        methods_covered={"methodA", "methodB"},
        total_methods_available=3,
    )
    print(f"Line coverage: {metrics.line_coverage_pct:.1f}%")
    print(f"Branch coverage: {metrics.branch_coverage_pct:.1f}%")
    print(f"Method coverage: {metrics.method_coverage_pct:.1f}%")
    print(f"Combined: {metrics.combined_coverage:.1f}%")
