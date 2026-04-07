"""
Train with basic profile configurations (AGGRESSIVE, EXPERIMENTAL, CONSERVATIVE).
Wrapper around train_profile_aware.py that converts basic profiles to advanced profile parameters.
"""
import sys
import argparse
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
from datetime import datetime

from model import Generator, Critic
from junit_datasets import JUnitDataset
from config import ModelConfig, set_random_seed
from utils import compute_gradient_penalty

# Profile definitions
PROFILES = {
    'aggressive': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'lambda_gp': 2.0,
        'n_critic': 5,
        'description': 'Fast training, may be unstable'
    },
    'experimental': {
        'epochs': 80,
        'batch_size': 12,
        'learning_rate': 0.00008,
        'lambda_gp': 7.5,
        'n_critic': 8,
        'description': 'Improved hybrid configuration'
    },
    'conservative': {
        'epochs': 150,
        'batch_size': 8,
        'learning_rate': 0.000025,
        'lambda_gp': 10.0,
        'n_critic': 12,
        'description': 'Maximum stability, very slow'
    },
    'balanced': {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.00005,
        'lambda_gp': 5.0,
        'n_critic': 7,
        'description': 'Production-ready, stable training'
    },
    'optimized': {
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.00002,
        'lambda_gp': 10.0,
        'n_critic': 10,
        'description': 'Fixes mode collapse: stricter Lipschitz (λ=10), more critic (n=10), smaller batch/lr'
    },
    'generalization-stress': {
        'batch_size': 6,
        'learning_rate': 0.000025,
        'lambda_gp': 12.0,
        'n_critic': 15,
        'epochs': 150,
        'description': 'Designed specifically for extremely small datasets (<150 samples) to prevent mode collapse.'
    }
}


def _compute_real_feature_stats(real_data: np.ndarray):
    """Compute baseline statistics from real data for robust distribution matching."""
    if real_data.ndim != 2:
        raise ValueError("Expected 2D real_data array")

    return {
        "mean": real_data.mean(axis=0),
        "std": real_data.std(axis=0),
        "q50": np.quantile(real_data, 0.50, axis=0),
        "q75": np.quantile(real_data, 0.75, axis=0),
        "q90": np.quantile(real_data, 0.90, axis=0),
        "q95": np.quantile(real_data, 0.95, axis=0),
        "q99": np.quantile(real_data, 0.99, axis=0),
    }


def _resolve_dataset_path(data_path: str | None) -> str:
    """Resolve dataset path using explicit input or repository fallbacks."""
    candidates = []
    if data_path:
        candidates.append(Path(data_path))

    candidates.extend([
        Path("python/data/junit_data_cleaned_strict.csv"),
        Path("python/data/junit_data_cleaned.csv"),
        Path("data/processed/train.csv"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Keep original explicit value for clearer downstream error if it was provided.
    if data_path:
        return data_path
    return str(candidates[0])


def _evaluate_synthetic_batch(fake_eval: np.ndarray, real_stats: dict, max_values: np.ndarray):
    """Compute robust metrics for model selection.

    Uses p95/p99 plus distance to real mean/std to avoid selecting early outlier-heavy epochs.
    """
    eps = 1e-8
    feature_count = fake_eval.shape[1]

    max_achieved = fake_eval.max(axis=0)
    p95_achieved = np.quantile(fake_eval, 0.95, axis=0)
    p99_achieved = np.quantile(fake_eval, 0.99, axis=0)

    coverage_max = float((max_achieved / np.maximum(max_values, eps) * 100).mean())
    coverage_p95 = float((p95_achieved / np.maximum(max_values, eps) * 100).mean())
    coverage_p99 = float((p99_achieved / np.maximum(max_values, eps) * 100).mean())

    fake_mean = fake_eval.mean(axis=0)
    fake_std = fake_eval.std(axis=0)
    mean_z = np.abs((fake_mean - real_stats["mean"]) / np.maximum(real_stats["std"], eps))
    std_z = np.abs((fake_std - real_stats["std"]) / np.maximum(real_stats["std"], eps))

    mean_distance_z = float(np.linalg.norm(mean_z) / np.sqrt(feature_count))
    std_distance_z = float(np.linalg.norm(std_z) / np.sqrt(feature_count))

    # Higher is better: reward high quantile coverage, penalize distribution mismatch.
    robust_score = (
        0.15 * coverage_max +
        0.50 * coverage_p95 +
        0.35 * coverage_p99 -
        20.0 * ((mean_distance_z + std_distance_z) / 2.0)
    )

    return {
        "max_achieved": max_achieved,
        "p95_achieved": p95_achieved,
        "p99_achieved": p99_achieved,
        "coverage_max": coverage_max,
        "coverage_p95": coverage_p95,
        "coverage_p99": coverage_p99,
        "mean_distance_z": mean_distance_z,
        "std_distance_z": std_distance_z,
        "robust_score": float(robust_score),
    }


def _evaluate_synthetic_batch_v2(fake_eval: np.ndarray, real_stats: dict, max_values: np.ndarray):
    """Compute stricter robust score focused on distribution matching.

    This score avoids rewarding raw extremes and instead prioritizes:
    - quantile alignment (q50/q75/q90/q95/q99)
    - controlled tails (q99 relative to q95)
    - stable mean/std proximity to the real dataset
    """
    eps = 1e-8

    q50 = np.quantile(fake_eval, 0.50, axis=0)
    q75 = np.quantile(fake_eval, 0.75, axis=0)
    q90 = np.quantile(fake_eval, 0.90, axis=0)
    q95 = np.quantile(fake_eval, 0.95, axis=0)
    q99 = np.quantile(fake_eval, 0.99, axis=0)
    max_achieved = fake_eval.max(axis=0)

    fake_mean = fake_eval.mean(axis=0)
    fake_std = fake_eval.std(axis=0)
    mean_z = np.abs((fake_mean - real_stats["mean"]) / np.maximum(real_stats["std"], eps))
    std_z = np.abs((fake_std - real_stats["std"]) / np.maximum(real_stats["std"], eps))
    mean_distance_z = float(np.linalg.norm(mean_z) / np.sqrt(fake_eval.shape[1]))
    std_distance_z = float(np.linalg.norm(std_z) / np.sqrt(fake_eval.shape[1]))

    quantile_pairs = [
        (q50, real_stats["q50"]),
        (q75, real_stats["q75"]),
        (q90, real_stats["q90"]),
        (q95, real_stats["q95"]),
        (q99, real_stats["q99"]),
    ]

    quantile_rel_errors = []
    for fake_q, real_q in quantile_pairs:
        rel = np.abs(fake_q - real_q) / np.maximum(real_q, eps)
        quantile_rel_errors.append(rel)
    quantile_match_error = float(np.mean(np.vstack(quantile_rel_errors)))

    real_tail_ratio = real_stats["q99"] / np.maximum(real_stats["q95"], eps)
    fake_tail_ratio = q99 / np.maximum(q95, eps)
    tail_excess = np.maximum(0.0, fake_tail_ratio - real_tail_ratio) / np.maximum(real_tail_ratio, eps)
    tail_excess_penalty = float(np.mean(tail_excess))

    over_real_q99 = np.maximum(0.0, max_achieved - real_stats["q99"]) / np.maximum(real_stats["q99"], eps)
    max_excess_penalty = float(np.mean(over_real_q99))

    coverage_p95 = float((q95 / np.maximum(max_values, eps) * 100).mean())
    coverage_p99 = float((q99 / np.maximum(max_values, eps) * 100).mean())

    # Higher is better. Main objective is low mismatch, not high extremes.
    robust_v2_score = 100.0 - (
        65.0 * quantile_match_error +
        12.0 * mean_distance_z +
        8.0 * std_distance_z +
        18.0 * tail_excess_penalty +
        12.0 * max_excess_penalty
    )

    return {
        "q50_achieved": q50,
        "q75_achieved": q75,
        "q90_achieved": q90,
        "q95_achieved": q95,
        "q99_achieved": q99,
        "max_achieved": max_achieved,
        "coverage_p95": coverage_p95,
        "coverage_p99": coverage_p99,
        "quantile_match_error": quantile_match_error,
        "tail_excess_penalty": tail_excess_penalty,
        "max_excess_penalty": max_excess_penalty,
        "mean_distance_z": mean_distance_z,
        "std_distance_z": std_distance_z,
        "robust_v2_score": float(robust_v2_score),
    }


def train_basic_profile(
    profile_name,
    epochs=None,
    seed=42,
    device='cpu',
    data_path=None,
    norm_stats_path=None,
    output_root='outputs',
    run_name=None,
    auto_generate_synthetic=False,
    synthetic_samples=1000,
    synthetic_checkpoint='auto',
    synthetic_output_name=None,
    eval_samples=1000,
    eval_seed=12345,
    best_metric='robust_v2',
    warmup_epochs=5,
    batch_log_interval=0,
):
    """Train with a basic profile configuration."""

    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Choose from: {list(PROFILES.keys())}")

    profile = PROFILES[profile_name]
    if epochs is None:
        epochs = profile['epochs']

    set_random_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        output_dir = Path(output_root) / f"{run_name}_{profile_name}_{timestamp}"
    else:
        output_dir = Path(output_root) / f"{profile_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    dataset_path = _resolve_dataset_path(data_path)
    if norm_stats_path is None:
        dataset_path_obj = Path(dataset_path)
        norm_stats_path = str(dataset_path_obj.parent / f"{dataset_path_obj.stem}_norm_stats.json")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"{'='*70}")
    print(f"TRAINING: {profile_name.upper()}")
    print(f"{'='*70}")
    print(f"Description: {profile['description']}")
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {profile['batch_size']}")
    print(f"  Learning rate: {profile['learning_rate']}")
    print(f"  Lambda GP: {profile['lambda_gp']}")
    print(f"  Critic iterations: {profile['n_critic']}")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Norm stats: {norm_stats_path}")
    print(f"  Eval samples: {eval_samples}")
    print(f"  Eval seed: {eval_seed}")
    print(f"  Best metric: {best_metric}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    dataset = JUnitDataset(dataset_path, norm_stats_path)
    dataloader = DataLoader(
        dataset,
        batch_size=profile['batch_size'],
        shuffle=True,
        drop_last=True
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}\n")

    model_config = ModelConfig()
    generator = Generator(z_dim=model_config.z_dim, output_dim=model_config.output_dim).to(device)
    critic = Critic(input_dim=model_config.output_dim).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=profile['learning_rate'], betas=(0.5, 0.999))
    optimizer_c = optim.Adam(critic.parameters(), lr=profile['learning_rate'], betas=(0.5, 0.999))

    with open(norm_stats_path, 'r', encoding='utf-8') as f:
        norm_stats = json.load(f)

    num_features = norm_stats['num_features']
    max_values = np.array([norm_stats['features'][f'feature_{i}']['max'] for i in range(num_features)])
    min_values = np.array([norm_stats['features'][f'feature_{i}']['min'] for i in range(num_features)])

    def denormalize(normalized_data):
        data_01 = (normalized_data + 1) / 2
        return data_01 * (max_values - min_values) + min_values

    real_stats = _compute_real_feature_stats(dataset.raw_data)

    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(eval_seed)
    z_eval_fixed = torch.randn(eval_samples, model_config.z_dim, generator=eval_generator, device=device)

    history = []
    best_coverage = float("-inf")
    best_epoch = 0
    best_robust_score = float("-inf")
    best_robust_epoch = 0
    best_robust_v2_score = float("-inf")
    best_robust_v2_epoch = 0

    print("Starting training...\n")

    for epoch in range(epochs):
        generator.train()
        critic.train()

        epoch_g_loss = 0
        epoch_c_loss = 0
        num_batches = 0

        for batch_idx, real_data in enumerate(dataloader, start=1):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            for _ in range(profile['n_critic']):
                optimizer_c.zero_grad()

                z = torch.randn(batch_size, model_config.z_dim).to(device)
                fake_data = generator(z)

                critic_real = critic(real_data)
                critic_fake = critic(fake_data.detach())

                gp = compute_gradient_penalty(
                    critic, real_data, fake_data, device, batch_size,
                    lambda_gp=profile['lambda_gp']
                )

                critic_loss = critic_fake.mean() - critic_real.mean() + gp
                critic_loss.backward()
                optimizer_c.step()

                epoch_c_loss += critic_loss.item()

            optimizer_g.zero_grad()

            z = torch.randn(batch_size, model_config.z_dim).to(device)
            fake_data = generator(z)

            critic_fake = critic(fake_data)
            gen_loss = -critic_fake.mean()

            gen_loss.backward()
            optimizer_g.step()

            epoch_g_loss += gen_loss.item()
            num_batches += 1

            if batch_log_interval > 0 and (batch_idx % batch_log_interval) == 0:
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Batch {batch_idx:3d}/{len(dataloader)} | "
                    f"G Loss: {gen_loss.item():7.4f} | "
                    f"C Loss: {critic_loss.item():7.4f}"
                )

        avg_g_loss = epoch_g_loss / num_batches
        avg_c_loss = epoch_c_loss / (num_batches * profile['n_critic'])

        with torch.no_grad():
            fake_eval_norm = generator(z_eval_fixed).cpu().numpy()
            fake_eval = denormalize(fake_eval_norm)

        metrics = _evaluate_synthetic_batch(fake_eval, real_stats, max_values)
        metrics_v2 = _evaluate_synthetic_batch_v2(fake_eval, real_stats, max_values)
        max_achieved = metrics['max_achieved']
        coverage_pct = metrics['coverage_max']
        diversity = len(np.unique(fake_eval.round(1), axis=0)) / len(fake_eval)

        history.append({
            'epoch': epoch + 1,
            'gen_loss': avg_g_loss,
            'critic_loss': avg_c_loss,
            'coverage': coverage_pct,
            'coverage_p95': metrics['coverage_p95'],
            'coverage_p99': metrics['coverage_p99'],
            'robust_score': metrics['robust_score'],
            'robust_v2_score': metrics_v2['robust_v2_score'],
            'quantile_match_error': metrics_v2['quantile_match_error'],
            'tail_excess_penalty': metrics_v2['tail_excess_penalty'],
            'max_excess_penalty': metrics_v2['max_excess_penalty'],
            'mean_distance_z': metrics['mean_distance_z'],
            'std_distance_z': metrics['std_distance_z'],
            'diversity': diversity,
            'max_methods': max_achieved[0],
            'max_branches': max_achieved[1],
            'max_statements': max_achieved[2],
            'p95_methods': metrics['p95_achieved'][0],
            'p95_branches': metrics['p95_achieved'][1],
            'p95_statements': metrics['p95_achieved'][2],
            'p99_methods': metrics['p99_achieved'][0],
            'p99_branches': metrics['p99_achieved'][1],
            'p99_statements': metrics['p99_achieved'][2],
            'q50_methods': metrics_v2['q50_achieved'][0],
            'q50_branches': metrics_v2['q50_achieved'][1],
            'q50_statements': metrics_v2['q50_achieved'][2],
        })

        should_select_best = (epoch + 1) > warmup_epochs

        # Persist best-performing checkpoints after warmup to avoid early outlier selection.
        if should_select_best and coverage_pct > best_coverage:
            best_coverage = coverage_pct
            best_epoch = epoch + 1
            torch.save(generator.state_dict(), models_dir / 'generator_best_by_coverage.pth')
            torch.save(critic.state_dict(), models_dir / 'critic_best_by_coverage.pth')

        if should_select_best and metrics['robust_score'] > best_robust_score:
            best_robust_score = metrics['robust_score']
            best_robust_epoch = epoch + 1
            torch.save(generator.state_dict(), models_dir / 'generator_best_by_robust.pth')
            torch.save(critic.state_dict(), models_dir / 'critic_best_by_robust.pth')

        if should_select_best and metrics_v2['robust_v2_score'] > best_robust_v2_score:
            best_robust_v2_score = metrics_v2['robust_v2_score']
            best_robust_v2_epoch = epoch + 1
            torch.save(generator.state_dict(), models_dir / 'generator_best_by_robust_v2.pth')
            torch.save(critic.state_dict(), models_dir / 'critic_best_by_robust_v2.pth')

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"G Loss: {avg_g_loss:7.4f} | "
                  f"C Loss: {avg_c_loss:7.4f} | "
                  f"CovMax/P95/P99: {coverage_pct:.2f}/{metrics['coverage_p95']:.2f}/{metrics['coverage_p99']:.2f}% | "
                  f"R1/R2: {metrics['robust_score']:.2f}/{metrics_v2['robust_v2_score']:.2f} | "
                  f"Max: [{max_achieved[0]:.0f}, {max_achieved[1]:.0f}, {max_achieved[2]:.0f}]")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_c': optimizer_c.state_dict(),
            }, models_dir / f'checkpoint_epoch_{epoch+1}.pth')

    torch.save(generator.state_dict(), models_dir / 'generator_final.pth')
    torch.save(critic.state_dict(), models_dir / 'critic_final.pth')

    # Fallback if warmup excluded all epochs or metric did not improve.
    if best_epoch == 0:
        best_row = max(history, key=lambda x: x['coverage'])
        best_epoch = int(best_row['epoch'])
        best_coverage = float(best_row['coverage'])
        torch.save(generator.state_dict(), models_dir / 'generator_best_by_coverage.pth')
        torch.save(critic.state_dict(), models_dir / 'critic_best_by_coverage.pth')

    if best_robust_epoch == 0:
        best_row = max(history, key=lambda x: x['robust_score'])
        best_robust_epoch = int(best_row['epoch'])
        best_robust_score = float(best_row['robust_score'])
        torch.save(generator.state_dict(), models_dir / 'generator_best_by_robust.pth')
        torch.save(critic.state_dict(), models_dir / 'critic_best_by_robust.pth')

    if best_robust_v2_epoch == 0:
        best_row = max(history, key=lambda x: x['robust_v2_score'])
        best_robust_v2_epoch = int(best_row['epoch'])
        best_robust_v2_score = float(best_row['robust_v2_score'])
        torch.save(generator.state_dict(), models_dir / 'generator_best_by_robust_v2.pth')
        torch.save(critic.state_dict(), models_dir / 'critic_best_by_robust_v2.pth')

    df_history = pd.DataFrame(history)
    df_history.to_csv(output_dir / 'training_history.csv', index=False)

    config_dict = {
        'profile': profile_name,
        'profile_config': profile,
        'epochs': epochs,
        'seed': seed,
        'device': device,
        'dataset_path': dataset_path,
        'norm_stats_path': norm_stats_path,
        'timestamp': timestamp,
        'eval_samples': eval_samples,
        'eval_seed': eval_seed,
        'best_metric': best_metric,
        'warmup_epochs': warmup_epochs,
        'best_by_coverage': {
            'epoch': best_epoch,
            'score': best_coverage,
        },
        'best_by_robust': {
            'epoch': best_robust_epoch,
            'score': best_robust_score,
        },
        'best_by_robust_v2': {
            'epoch': best_robust_v2_epoch,
            'score': best_robust_v2_score,
        },
    }
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)

    final_metrics = history[-1]
    if synthetic_output_name is None:
        if synthetic_checkpoint == 'final':
            synthetic_output_name = 'synthetic_samples_final.csv'
        else:
            synthetic_output_name = 'synthetic_samples_best.csv'

    selected_checkpoint = models_dir / 'generator_final.pth'
    if synthetic_checkpoint in ('auto', 'best'):
        if best_metric == 'coverage':
            best_ckpt = models_dir / 'generator_best_by_coverage.pth'
        elif best_metric == 'robust_v2':
            best_ckpt = models_dir / 'generator_best_by_robust_v2.pth'
        else:
            best_ckpt = models_dir / 'generator_best_by_robust.pth'

        if best_ckpt.exists():
            selected_checkpoint = best_ckpt
        elif synthetic_checkpoint == 'best':
            raise FileNotFoundError(f"Best checkpoint requested but not found: {best_ckpt}")

    print(f"\n{'='*70}")
    print("✅ Training completed!")
    print(f"   Final coverage: {final_metrics['coverage']:.2f}%")
    print(f"   Final p95/p99: {final_metrics['coverage_p95']:.2f}% / {final_metrics['coverage_p99']:.2f}%")
    print(f"   Final robust score: {final_metrics['robust_score']:.2f}")
    print(f"   Final robust v2 score: {final_metrics['robust_v2_score']:.2f}")
    print(f"   Best coverage: {best_coverage:.2f}% (epoch {best_epoch})")
    print(f"   Best robust score: {best_robust_score:.2f} (epoch {best_robust_epoch})")
    print(f"   Best robust v2 score: {best_robust_v2_score:.2f} (epoch {best_robust_v2_epoch})")
    print(f"   Final diversity: {final_metrics['diversity']:.3f}")
    print(f"   Final max values: [{final_metrics['max_methods']:.0f}, "
          f"{final_metrics['max_branches']:.0f}, {final_metrics['max_statements']:.0f}]")
    print(f"   Output: {output_dir}")

    if auto_generate_synthetic:
        synthetic_output_path = output_dir / synthetic_output_name
        print("\n🧬 Auto-generating synthetic samples...")
        print(f"   Checkpoint: {selected_checkpoint.name}")
        print(f"   Samples: {synthetic_samples}")
        print(f"   Output CSV: {synthetic_output_path}")

        cmd = [
            sys.executable,
            "python/scripts/generate_synthetic.py",
            "--model", str(selected_checkpoint).replace("\\", "/"),
            "--num-samples", str(synthetic_samples),
            "--output", str(synthetic_output_path).replace("\\", "/"),
            "--norm-stats", str(norm_stats_path).replace("\\", "/"),
            "--device", device,
        ]
        subprocess.run(cmd, check=True)

    print(f"{'='*70}\n")

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, required=True,
                        choices=list(PROFILES.keys()),
                        help='Training profile to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from profile)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to input CSV (auto-resolved when omitted)')
    parser.add_argument('--norm-stats-path', type=str, default=None,
                        help='Path to normalization stats JSON (auto-generated if omitted)')
    parser.add_argument('--output-root', type=str, default='outputs',
                        help='Output root directory for training runs')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional prefix to group experiment runs')
    parser.add_argument('--auto-generate-synthetic', action='store_true',
                        help='Generate synthetic samples automatically after training')
    parser.add_argument('--synthetic-samples', type=int, default=1000,
                        help='Number of synthetic samples to generate when auto-generate is enabled')
    parser.add_argument('--synthetic-checkpoint', type=str, default='auto',
                        choices=['auto', 'best', 'final'],
                        help='Checkpoint to use for synthetic generation (auto=best if available)')
    parser.add_argument('--synthetic-output-name', type=str, default=None,
                        help='Output filename for synthetic CSV inside the training output folder')
    parser.add_argument('--eval-samples', type=int, default=1000,
                        help='Fixed number of synthetic samples used for deterministic epoch evaluation')
    parser.add_argument('--eval-seed', type=int, default=12345,
                        help='Seed for deterministic evaluation latent vectors')
    parser.add_argument('--best-metric', type=str, default='robust_v2',
                        choices=['robust', 'robust_v2', 'coverage'],
                        help='Metric used for selecting the best checkpoint when synthetic-checkpoint is auto/best')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Do not update best checkpoints before this epoch (exclusive)')
    parser.add_argument('--batch-log-interval', type=int, default=0,
                        help='Optional interval for batch-level progress logs (0 disables)')

    args = parser.parse_args()

    train_basic_profile(
        args.profile,
        args.epochs,
        args.seed,
        args.device,
        data_path=args.data_path,
        norm_stats_path=args.norm_stats_path,
        output_root=args.output_root,
        run_name=args.run_name,
        auto_generate_synthetic=args.auto_generate_synthetic,
        synthetic_samples=args.synthetic_samples,
        synthetic_checkpoint=args.synthetic_checkpoint,
        synthetic_output_name=args.synthetic_output_name,
        eval_samples=args.eval_samples,
        eval_seed=args.eval_seed,
        best_metric=args.best_metric,
        warmup_epochs=args.warmup_epochs,
        batch_log_interval=args.batch_log_interval,
    )
