"""
Real-time coverage instrumentation for WGAN training.

Tracks coverage metrics during training epochs and provides
feedback on structural exploration vs memorization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import csv
from pathlib import Path
import torch
import numpy as np
from hashlib import md5
import json


@dataclass
class EpochCoverageSnapshot:
    """Coverage metrics snapshot at a specific epoch."""
    epoch: int
    line_coverage: float
    branch_coverage: float
    method_coverage: float
    path_diversity: float
    unique_paths: int
    total_paths: int
    generator_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'line_coverage': self.line_coverage,
            'branch_coverage': self.branch_coverage,
            'method_coverage': self.method_coverage,
            'path_diversity': self.path_diversity,
            'unique_paths': self.unique_paths,
            'total_paths': self.total_paths,
            'generator_loss': self.generator_loss,
            'critic_loss': self.critic_loss,
            'timestamp': self.timestamp
        }


class TrainingInstrumentation:
    """
    Hook coverage tracking into training loop.
    
    Extracts execution paths from synthetic test programs
    and tracks coverage metrics across epochs.
    """
    
    def __init__(self, output_dir: Path = None, max_values: Optional[Dict[str, float]] = None):
        """
        Initialize instrumentation.
        
        Args:
            output_dir: Directory to save coverage reports
            max_values: Dict with 'methods', 'branches', 'statements' max values from training data.
                       If None, uses reasonable defaults based on typical test code.
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set max values for coverage calculation
        if max_values:
            self.max_methods = max_values.get('methods', 800.0)
            self.max_branches = max_values.get('branches', 1000.0)
            self.max_statements = max_values.get('statements', 1000.0)
        else:
            # Use realistic defaults from cleaned dataset (max observed: 777, 1000, 1000)
            # Adding small buffer for edge cases
            self.max_methods = 800.0
            self.max_branches = 1000.0
            self.max_statements = 1000.0
        
        self.coverage_history: List[EpochCoverageSnapshot] = []
        self.path_cache: Dict[int, set] = {}  # epoch -> set of unique paths
        
    def track_epoch_coverage(self, 
                            epoch: int,
                            generated_tests: torch.Tensor,
                            gen_loss: Optional[float] = None,
                            critic_loss: Optional[float] = None) -> Dict:
        """
        Extract coverage from generated tests.
        
        Args:
            epoch: Current epoch number
            generated_tests: Tensor of shape (batch_size, 3) with test metrics
            gen_loss: Generator loss (optional)
            critic_loss: Critic loss (optional)
            
        Returns:
            Dict with coverage metrics
        """
        # Convert tensor to numpy for analysis
        if isinstance(generated_tests, torch.Tensor):
            tests = generated_tests.detach().cpu().numpy()
        else:
            tests = np.array(generated_tests)
        
        # Calculate coverage metrics from test features
        # Features are: [method_count, branch_count, statement_count]
        methods = tests[:, 0] if tests.shape[1] > 0 else np.zeros((tests.shape[0],))
        branches = tests[:, 1] if tests.shape[1] > 1 else np.zeros((tests.shape[0],))
        statements = tests[:, 2] if tests.shape[1] > 2 else np.zeros((tests.shape[0],))
        
        # Normalize to coverage percentages using actual dataset max values
        # Avoid division by zero
        line_cov = float(np.mean(np.minimum(methods / (self.max_methods + 1e-8), 1.0)) * 100)
        branch_cov = float(np.mean(np.minimum(branches / (self.max_branches + 1e-8), 1.0)) * 100)
        method_cov = float(np.mean(np.minimum(statements / (self.max_statements + 1e-8), 1.0)) * 100)
        
        # Calculate path diversity (unique execution patterns)
        unique_paths, total_paths, diversity = self._calculate_path_diversity(tests)
        
        # Create snapshot
        snapshot = EpochCoverageSnapshot(
            epoch=epoch,
            line_coverage=line_cov,
            branch_coverage=branch_cov,
            method_coverage=method_cov,
            path_diversity=diversity,
            unique_paths=unique_paths,
            total_paths=total_paths,
            generator_loss=gen_loss,
            critic_loss=critic_loss,
            timestamp=datetime.now().isoformat()
        )
        
        self.coverage_history.append(snapshot)
        self.path_cache[epoch] = set(range(unique_paths))
        
        return snapshot.to_dict()
    
    def _calculate_path_diversity(self, tests: np.ndarray) -> tuple:
        """
        Calculate execution path diversity.
        
        Uses MD5 hashing to identify unique execution patterns.
        """
        if len(tests) == 0:
            return 0, 0, 0.0
        
        # Hash each test to create unique path identifier
        unique_hashes = set()
        for test in tests:
            # Create a hash from test characteristics
            test_str = f"{test[0]:.2f}_{test[1]:.2f}_{test[2]:.2f}"
            test_hash = md5(test_str.encode()).hexdigest()[:8]
            unique_hashes.add(test_hash)
        
        unique_paths = len(unique_hashes)
        total_paths = len(tests)
        
        # Diversity = unique / total (higher = more diverse)
        diversity = float(unique_paths / max(total_paths, 1))
        
        return unique_paths, total_paths, diversity
    
    def export_trajectory_csv(self, filename: str) -> Path:
        """
        Save coverage trajectory to CSV.
        
        Args:
            filename: Output CSV filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = [
                'epoch', 'line_cov', 'branch_cov', 'method_cov',
                'diversity', 'unique_paths', 'total_paths',
                'gen_loss', 'critic_loss', 'timestamp'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for snap in self.coverage_history:
                writer.writerow({
                    'epoch': snap.epoch,
                    'line_cov': f"{snap.line_coverage:.2f}",
                    'branch_cov': f"{snap.branch_coverage:.2f}",
                    'method_cov': f"{snap.method_coverage:.2f}",
                    'diversity': f"{snap.path_diversity:.4f}",
                    'unique_paths': snap.unique_paths,
                    'total_paths': snap.total_paths,
                    'gen_loss': f"{snap.generator_loss:.6f}" if snap.generator_loss else "N/A",
                    'critic_loss': f"{snap.critic_loss:.6f}" if snap.critic_loss else "N/A",
                    'timestamp': snap.timestamp
                })
        
        print(f"✓ Coverage trajectory saved to {filepath}")
        return filepath
    
    def export_json(self, filename: str) -> Path:
        """
        Export coverage history as JSON.
        
        Args:
            filename: Output JSON filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        data = {
            'total_epochs': len(self.coverage_history),
            'coverage_history': [snap.to_dict() for snap in self.coverage_history],
            'summary': self._generate_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Coverage JSON saved to {filepath}")
        return filepath
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics from coverage history."""
        if not self.coverage_history:
            return {}
        
        snapshots = self.coverage_history
        coverages = [s.line_coverage for s in snapshots]
        diversities = [s.path_diversity for s in snapshots]
        
        return {
            'epochs_tracked': len(snapshots),
            'initial_coverage': coverages[0] if coverages else 0,
            'final_coverage': coverages[-1] if coverages else 0,
            'max_coverage': max(coverages) if coverages else 0,
            'avg_coverage': float(np.mean(coverages)) if coverages else 0,
            'initial_diversity': diversities[0] if diversities else 0,
            'final_diversity': diversities[-1] if diversities else 0,
            'max_diversity': max(diversities) if diversities else 0,
            'avg_diversity': float(np.mean(diversities)) if diversities else 0,
            'coverage_growth': (coverages[-1] - coverages[0]) if len(coverages) > 1 else 0,
        }
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """Get metrics from latest epoch."""
        if not self.coverage_history:
            return None
        return self.coverage_history[-1].to_dict()
    
    def get_coverage_delta(self, epoch_a: int, epoch_b: int) -> Optional[Dict]:
        """
        Get coverage delta between two epochs.
        
        Args:
            epoch_a: First epoch
            epoch_b: Second epoch
            
        Returns:
            Dict with delta coverage metrics
        """
        snap_a = next((s for s in self.coverage_history if s.epoch == epoch_a), None)
        snap_b = next((s for s in self.coverage_history if s.epoch == epoch_b), None)
        
        if snap_a is None or snap_b is None:
            return None
        
        return {
            'epoch_a': epoch_a,
            'epoch_b': epoch_b,
            'line_cov_delta': snap_b.line_coverage - snap_a.line_coverage,
            'branch_cov_delta': snap_b.branch_coverage - snap_a.branch_coverage,
            'method_cov_delta': snap_b.method_coverage - snap_a.method_coverage,
            'diversity_delta': snap_b.path_diversity - snap_a.path_diversity,
        }


class CoverageReporter:
    """Generate training reports with coverage analysis."""
    
    def __init__(self, instrumentation: TrainingInstrumentation):
        """
        Initialize reporter.
        
        Args:
            instrumentation: TrainingInstrumentation instance
        """
        self.instrumentation = instrumentation
    
    def generate_report(self, profile_name: str, run_dir: Path = None) -> Dict:
        """
        Generate comprehensive coverage report.
        
        Args:
            profile_name: Name of training profile
            run_dir: Directory to save report
            
        Returns:
            Report dict
        """
        run_dir = Path(run_dir) if run_dir else self.instrumentation.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'profile': profile_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.instrumentation._generate_summary(),
            'coverage_trajectory': [
                s.to_dict() for s in self.instrumentation.coverage_history
            ]
        }
        
        # Save report
        report_file = run_dir / f"report_{profile_name}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {report_file}")
        return report


if __name__ == '__main__':
    # Demo usage
    print("Instrumentation Module")
    
    instr = TrainingInstrumentation()
    
    # Simulate 10 epochs of training
    for epoch in range(10):
        fake_tests = torch.randn(32, 3) * 50  # 32 tests, 3 features
        metrics = instr.track_epoch_coverage(
            epoch,
            fake_tests,
            gen_loss=1.5 - epoch*0.1,
            critic_loss=0.5 + epoch*0.02
        )
        print(f"Epoch {epoch}: Coverage={metrics['line_coverage']:.1f}%, "
              f"Diversity={metrics['path_diversity']:.3f}")
    
    # Export results
    instr.export_trajectory_csv('coverage_demo.csv')
    instr.export_json('coverage_demo.json')
    
    print("\n✓ Instrumentation demo complete")
