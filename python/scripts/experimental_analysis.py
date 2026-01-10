#!/usr/bin/env python3
"""
Experimental Analysis: Train vs Test Metrics Comparison
Analyzes WGAN-GP model effectiveness by comparing training and test metrics
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import json
from typing import Dict, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Generator, Critic
from utils import compute_gradient_penalty
from junit_datasets import JUnitDataset
from torch.utils.data import DataLoader, random_split

class ExperimentalAnalysis:
    """Comprehensive experimental analysis of WGAN-GP training"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_metrics = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'd_real': [],
            'd_fake': [],
            'gradient_penalty': []
        }
        self.test_metrics = {
            'epoch': [],
            'd_loss': [],
            'synthetic_quality': [],
            'distribution_diff': []
        }
        # Batch-level accumulation
        self.batch_d_real = []
        self.batch_d_fake = []
        self.batch_gp = []
        print(f"✓ Initialized on device: {self.device}")
    
    def load_data(self, data_path: str, batch_size: int = 32):
        """Load and split data"""
        print(f"\n📂 Loading data from: {data_path}")
        dataset = JUnitDataset(data_path)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  Train: {len(train_dataset)} samples ({len(self.train_loader)} batches)")
        print(f"  Test: {len(test_dataset)} samples ({len(self.test_loader)} batches)")
        print(f"  Data range: [{dataset.data.min():.4f}, {dataset.data.max():.4f}]")
        
        return dataset
    
    def create_models(self, z_dim: int = 100, output_dim: int = 3):
        """Create models"""
        print(f"\n🧠 Creating models...")
        self.generator = Generator(z_dim=z_dim, output_dim=output_dim).to(self.device)
        self.critic = Critic(input_dim=output_dim).to(self.device)
        self.z_dim = z_dim
        
        gen_params = sum(p.numel() for p in self.generator.parameters())
        crit_params = sum(p.numel() for p in self.critic.parameters())
        print(f"  Generator: {gen_params:,} params")
        print(f"  Critic: {crit_params:,} params")
    
    def store_batch_metrics(self, real_output: torch.Tensor, 
                           fake_output: torch.Tensor, gp: torch.Tensor):
        """Store batch-level metrics for later aggregation"""
        self.batch_d_real.append(real_output.mean().item())
        self.batch_d_fake.append(fake_output.mean().item())
        self.batch_gp.append(gp.item())
    
    def evaluate_synthetic_quality(self, real_data: torch.Tensor, 
                                   fake_data: torch.Tensor) -> float:
        """
        Compute synthetic data quality score
        Based on feature-wise distribution similarity (KL divergence proxy)
        """
        real_mean = real_data.mean(dim=0)
        real_std = real_data.std(dim=0)
        fake_mean = fake_data.mean(dim=0)
        fake_std = fake_data.std(dim=0)
        
        # Mean absolute difference (normalized)
        mean_diff = torch.abs(real_mean - fake_mean).mean().item()
        std_diff = torch.abs(real_std - fake_std).mean().item()
        
        # Quality score: lower is better (0-1 normalized)
        quality = 1.0 / (1.0 + mean_diff + std_diff)
        return quality
    
    def compute_distribution_difference(self, real_data: torch.Tensor, 
                                       fake_data: torch.Tensor) -> float:
        """
        Compute Wasserstein distance approximation between real and fake distributions
        Using mean and std as proxy for full distribution
        """
        real_mean = real_data.mean()
        real_std = real_data.std()
        fake_mean = fake_data.mean()
        fake_std = fake_data.std()
        
        # Approximate Wasserstein distance
        w_dist = torch.abs(real_mean - fake_mean) + torch.abs(real_std - fake_std)
        return w_dist.item()
    
    def train_epoch(self, epoch: int, lr_g: float = 0.0002, lr_c: float = 0.0002,
                   lambda_gp: float = 10, n_critic: int = 5):
        """Train for one epoch with detailed metrics"""
        self.generator.train()
        self.critic.train()
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c, betas=(0.5, 0.999))
        
        # Reset batch accumulators
        self.batch_d_real = []
        self.batch_d_fake = []
        self.batch_gp = []
        
        self.train_metrics['epoch'].append(epoch)
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count = 0
        
        for batch_idx, real_data in enumerate(self.train_loader):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Critic
            for _ in range(n_critic):
                opt_c.zero_grad()
                
                real_output = self.critic(real_data)
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_data = self.generator(z)
                fake_output = self.critic(fake_data.detach())
                
                gp = compute_gradient_penalty(self.critic, real_data, fake_data, 
                                            self.device, batch_size, lambda_gp)
                
                d_loss = -torch.mean(real_output) + torch.mean(fake_output) + gp
                d_loss.backward()
                opt_c.step()
                
                self.store_batch_metrics(real_output, fake_output, gp)
            
            # Train Generator
            opt_g.zero_grad()
            z = torch.randn(batch_size, self.z_dim, device=self.device)
            fake_data = self.generator(z)
            fake_output = self.critic(fake_data)
            g_loss = -torch.mean(fake_output)
            g_loss.backward()
            opt_g.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            batch_count += 1
        
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        
        # Store aggregated epoch metrics
        self.train_metrics['d_loss'].append(avg_d_loss)
        self.train_metrics['g_loss'].append(avg_g_loss)
        self.train_metrics['d_real'].append(sum(self.batch_d_real) / len(self.batch_d_real) if self.batch_d_real else 0.0)
        self.train_metrics['d_fake'].append(sum(self.batch_d_fake) / len(self.batch_d_fake) if self.batch_d_fake else 0.0)
        self.train_metrics['gradient_penalty'].append(sum(self.batch_gp) / len(self.batch_gp) if self.batch_gp else 0.0)
        
        return avg_d_loss, avg_g_loss
    
    def evaluate(self, epoch: int) -> Tuple[float, float, float]:
        """Evaluate model on test set"""
        self.generator.eval()
        self.critic.eval()
        
        self.test_metrics['epoch'].append(epoch)
        
        total_d_loss = 0.0
        total_quality = 0.0
        total_dist_diff = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for real_data in self.test_loader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Critic loss on test set
                real_output = self.critic(real_data)
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_data = self.generator(z)
                fake_output = self.critic(fake_data)
                
                d_loss = -torch.mean(real_output) + torch.mean(fake_output)
                total_d_loss += d_loss.item()
                
                # Quality metrics
                quality = self.evaluate_synthetic_quality(real_data, fake_data)
                dist_diff = self.compute_distribution_difference(real_data, fake_data)
                
                total_quality += quality
                total_dist_diff += dist_diff
                batch_count += 1
        
        avg_d_loss = total_d_loss / batch_count
        avg_quality = total_quality / batch_count
        avg_dist_diff = total_dist_diff / batch_count
        
        self.test_metrics['d_loss'].append(avg_d_loss)
        self.test_metrics['synthetic_quality'].append(avg_quality)
        self.test_metrics['distribution_diff'].append(avg_dist_diff)
        
        return avg_d_loss, avg_quality, avg_dist_diff
    
    def compute_overfitting_metrics(self) -> Dict:
        """Compute overfitting indicators"""
        if len(self.train_metrics['d_loss']) < 2:
            return {}
        
        train_d_loss = np.array(self.train_metrics['d_loss'])
        test_d_loss = np.array(self.test_metrics['d_loss'])
        
        # Metrics
        final_train_loss = train_d_loss[-1]
        final_test_loss = test_d_loss[-1]
        
        # Generalization gap
        gen_gap = abs(final_test_loss - final_train_loss)
        
        # Loss variance
        train_var = np.var(train_d_loss[-5:]) if len(train_d_loss) >= 5 else np.var(train_d_loss)
        test_var = np.var(test_d_loss[-5:]) if len(test_d_loss) >= 5 else np.var(test_d_loss)
        
        # Stability (lower variance = more stable)
        stability = min(train_var, test_var)
        
        return {
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'generalization_gap': gen_gap,
            'train_variance': train_var,
            'test_variance': test_var,
            'stability': stability,
            'overfitting_risk': 'HIGH' if gen_gap > 0.5 else ('MEDIUM' if gen_gap > 0.2 else 'LOW')
        }
    
    def compute_effectiveness_score(self) -> Dict:
        """Compute overall model effectiveness"""
        if len(self.test_metrics['synthetic_quality']) == 0:
            return {}
        
        quality_scores = np.array(self.test_metrics['synthetic_quality'])
        dist_diffs = np.array(self.test_metrics['distribution_diff'])
        
        return {
            'avg_quality': float(quality_scores.mean()),
            'max_quality': float(quality_scores.max()),
            'min_quality': float(quality_scores.min()),
            'quality_trend': 'improving' if quality_scores[-1] > quality_scores[0] else 'degrading',
            'avg_distribution_diff': float(dist_diffs.mean()),
            'distribution_convergence': 'converging' if dist_diffs[-1] < dist_diffs[0] else 'diverging'
        }
    
    def train(self, epochs: int = 50, lr_g: float = 0.0002, lr_c: float = 0.0002):
        """Complete training loop with experimental analysis"""
        print("\n" + "="*80)
        print("🔬 EXPERIMENTAL ANALYSIS: WGAN-GP Train vs Test Metrics")
        print("="*80)
        
        print(f"\n⚙️  Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate (G/C): {lr_g}/{lr_c}")
        print(f"  Lambda (GP): 10")
        print(f"  Critic iterations: 5")
        
        print(f"\n{'Epoch':<8} {'Train D':<12} {'Train G':<12} {'Test D':<12} {'Quality':<12} {'Dist Diff':<12}")
        print("-" * 80)
        
        for epoch in range(1, epochs + 1):
            train_d, train_g = self.train_epoch(epoch, lr_g, lr_c)
            test_d, quality, dist_diff = self.evaluate(epoch)
            
            print(f"{epoch:<8} {train_d:<12.4f} {train_g:<12.4f} {test_d:<12.4f} {quality:<12.4f} {dist_diff:<12.4f}")
        
        print("-" * 80)
        print("✅ Training completed")
        
        return self.train_metrics, self.test_metrics
    
    def generate_report(self, output_dir: str = "outputs/experimental"):
        """Generate comprehensive experimental report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Generating experimental analysis report...")
        
        # Save metrics
        train_df = pd.DataFrame(self.train_metrics)
        test_df = pd.DataFrame(self.test_metrics)
        
        train_df.to_csv(f"{output_dir}/train_metrics.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_metrics.csv", index=False)
        
        # Compute analysis metrics
        overfitting = self.compute_overfitting_metrics()
        effectiveness = self.compute_effectiveness_score()
        
        # Save analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'overfitting_analysis': overfitting,
            'effectiveness_analysis': effectiveness
        }
        
        with open(f"{output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("📈 EXPERIMENTAL ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n🔴 Overfitting Analysis:")
        for key, value in overfitting.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n🟢 Model Effectiveness:")
        for key, value in effectiveness.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Generate plots
        self.plot_train_vs_test(output_dir)
        self.plot_quality_metrics(output_dir)
        self.plot_generalization_gap(output_dir)
        
        print(f"\n📁 Report saved to: {output_dir}/")
        print(f"  - train_metrics.csv")
        print(f"  - test_metrics.csv")
        print(f"  - analysis.json")
        print(f"  - train_vs_test.png")
        print(f"  - quality_metrics.png")
        print(f"  - generalization_gap.png")
    
    def plot_train_vs_test(self, output_dir: str = "outputs/experimental"):
        """Plot train vs test losses"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = self.train_metrics['epoch']
        
        # Critic loss
        axes[0].plot(epochs, self.train_metrics['d_loss'], 'b-', label='Train D Loss', linewidth=2, marker='o', markersize=4)
        axes[0].plot(self.test_metrics['epoch'], self.test_metrics['d_loss'], 'r--', label='Test D Loss', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Critic Loss: Train vs Test', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Generator loss
        axes[1].plot(epochs, self.train_metrics['g_loss'], 'g-', label='Generator Loss', linewidth=2, marker='o', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Generator Loss Over Time', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/train_vs_test.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_quality_metrics(self, output_dir: str = "outputs/experimental"):
        """Plot synthetic data quality metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = self.test_metrics['epoch']
        
        # Quality score
        axes[0].plot(epochs, self.test_metrics['synthetic_quality'], 'g-', linewidth=2, marker='o', markersize=6)
        axes[0].fill_between(epochs, self.test_metrics['synthetic_quality'], alpha=0.3, color='green')
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Synthetic Data Quality (Higher is Better)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Distribution difference
        axes[1].plot(epochs, self.test_metrics['distribution_diff'], 'r-', linewidth=2, marker='s', markersize=6)
        axes[1].fill_between(epochs, self.test_metrics['distribution_diff'], alpha=0.3, color='red')
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Distribution Difference', fontsize=12, fontweight='bold')
        axes[1].set_title('Real vs Synthetic Distribution Gap (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quality_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_generalization_gap(self, output_dir: str = "outputs/experimental"):
        """Plot generalization gap analysis"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        train_losses = self.train_metrics['d_loss']
        test_losses = self.test_metrics['d_loss']
        epochs = self.test_metrics['epoch']
        
        # Generalization gap
        gen_gap = np.array(test_losses) - np.array(train_losses)
        
        # Plot
        ax.bar(epochs, gen_gap, color=['red' if x > 0 else 'green' for x in gen_gap], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=0.2, color='orange', linestyle='--', linewidth=1, label='Medium Risk Threshold')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='High Risk Threshold')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Generalization Gap (Test - Train)', fontsize=12, fontweight='bold')
        ax.set_title('Generalization Gap Analysis (Overfitting Indicator)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/generalization_gap.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main experimental analysis"""
    print("\n🔬 Experimental WGAN Analysis - Train vs Test Metrics\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis = ExperimentalAnalysis(device=device)
    
    # Load data
    dataset = analysis.load_data("data/junit_data_cleaned.csv", batch_size=32)
    
    # Create models
    analysis.create_models(z_dim=100, output_dim=3)
    
    # Train with analysis
    try:
        train_metrics, test_metrics = analysis.train(epochs=50)
        
        # Generate report
        analysis.generate_report("python/experimental")
        
        print("\n" + "="*80)
        print("✅ Experimental analysis completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
