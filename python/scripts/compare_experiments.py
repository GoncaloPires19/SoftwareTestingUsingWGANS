"""
Compare multiple WGAN training experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np
from typing import List, Optional


def load_experiment(exp_path: Path) -> pd.DataFrame:
    """Load training history from an experiment"""
    history_file = exp_path / "training_history.csv"
    
    if not history_file.exists():
        # Try alternative location
        history_file = exp_path if exp_path.suffix == ".csv" else None
    
    if not history_file or not history_file.exists():
        raise FileNotFoundError(f"No training_history.csv found in {exp_path}")
    
    df = pd.read_csv(history_file)
    return df


def compare_experiments(
    experiment_paths: List[str],
    metric: str = "diversity",
    output_dir: Path = Path("outputs"),
) -> None:
    """
    Compare multiple experiments
    
    Args:
        experiment_paths: List of paths to experiment directories or CSV files
        metric: Metric to compare (diversity, wasserstein_distance, etc)
        output_dir: Where to save comparison plots
    """
    
    output_dir.mkdir(exist_ok=True)
    
    # Load all experiments
    experiments = {}
    for path_str in experiment_paths:
        path = Path(path_str)
        
        try:
            df = load_experiment(path)
            exp_name = path.stem if path.suffix == ".csv" else path.name
            experiments[exp_name] = df
            print(f"✅ Loaded {exp_name}: {len(df)} epochs")
        except Exception as e:
            print(f"❌ Failed to load {path_str}: {e}")
    
    if not experiments:
        print("No experiments loaded!")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Critic loss
    ax = axes[0, 0]
    for name, df in experiments.items():
        ax.plot(df['epoch'], df['train_d_loss'], label=name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("D Loss")
    ax.set_title("Critic Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Generator loss
    ax = axes[0, 1]
    for name, df in experiments.items():
        ax.plot(df['epoch'], df['train_g_loss'], label=name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("G Loss")
    ax.set_title("Generator Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Metric (diversity, etc)
    ax = axes[1, 0]
    if metric in df.columns:
        for name, df in experiments.items():
            if metric in df.columns:
                ax.plot(df['epoch'], df[metric], label=name, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.title()} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation loss
    ax = axes[1, 1]
    for name, df in experiments.items():
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], label=name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"comparison_{metric}.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n📊 Comparison plot saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📈 EXPERIMENT COMPARISON SUMMARY")
    print("="*70)
    
    metrics_to_show = ['train_d_loss', 'train_g_loss', 'val_loss']
    if metric in experiments[list(experiments.keys())[0]].columns:
        metrics_to_show.append(metric)
    
    for metric_name in metrics_to_show:
        print(f"\n{metric_name}:")
        print(f"{'Experiment':<30} {'Final':<12} {'Best':<12} {'Epochs':<8}")
        print("-" * 62)
        
        for name, df in sorted(experiments.items()):
            if metric_name in df.columns:
                final_val = df[metric_name].iloc[-1]
                best_val = df[metric_name].min()
                best_epoch = df[metric_name].idxmin() + 1
                
                print(f"{name:<30} {final_val:>11.4f} {best_val:>11.4f} {best_epoch:>7d}")


def main():
    parser = argparse.ArgumentParser(description="Compare WGAN experiments")
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Paths to experiment directories or CSV files"
    )
    parser.add_argument(
        "--metric",
        default="diversity",
        help="Metric to compare (default: diversity)"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    compare_experiments(
        args.experiments,
        metric=args.metric,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
