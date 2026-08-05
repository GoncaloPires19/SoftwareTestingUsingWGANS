#!/usr/bin/env python3
"""
WGAN-GP Training — Optimized Pipeline

Improvements over the baseline:
- Composite score (quality + diversity) used for early stopping and best-model tracking.
  Prevents epoch 1 (random noise) from being selected as the best model.
- MinibatchStdDev in Critic: proven technique against mode collapse.
- Feature matching loss in generator step.
- Cosine annealing LR scheduler for both optimizers.
- Best checkpoint saved by composite score, not by epoch.
- Diversity, Wasserstein distance, and composite score tracked every epoch.
- Uses JUnitDataset for proper log1p + min-max normalisation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import ModelConfig, TrainingConfig, set_random_seed
from junit_datasets import JUnitDataset
from model import Critic, Generator
from utils import (
    compute_composite_score,
    compute_diversity_score,
    compute_feature_matching_loss,
    compute_gradient_penalty,
    compute_wasserstein_distance,
    get_model_gradient_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopper:
    """
    Stops training when a monitored metric does not improve by at least
    `min_delta` for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.005, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float = -float("inf") if mode == "max" else float("inf")
        self.best_epoch: int = 0
        self._counter: int = 0

    def step(self, value: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience

    @property
    def epochs_without_improvement(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WGANGPTrainer:
    """
    WGAN-GP trainer with MinibatchStdDev, composite score, LR scheduler,
    feature matching loss, and best-model tracking.

    History columns:
        epoch, gen_loss, critic_loss, gp_loss, feature_matching_loss,
        diversity_score, wasserstein_distance, composite_score,
        mean_grad_norm_g, mean_grad_norm_c, lr_g
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_config: ModelConfig,
        device: str = "cpu",
    ):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device)

        self.generator = Generator(
            z_dim=model_config.z_dim,
            output_dim=model_config.output_dim,
            hidden_dims=model_config.hidden_dims_g,
            use_batch_norm=model_config.use_batch_norm_g,
        ).to(self.device)

        self.critic = Critic(
            input_dim=model_config.output_dim,
            hidden_dims=model_config.hidden_dims_c,
            use_spectral_norm=model_config.use_spectral_norm_c,
            use_minibatch_std=model_config.use_minibatch_std,
        ).to(self.device)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.c_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

        self.g_scheduler: CosineAnnealingLR | None = None
        self.c_scheduler: CosineAnnealingLR | None = None
        if config.use_lr_scheduler:
            self.g_scheduler = CosineAnnealingLR(
                self.g_optimizer, T_max=config.epochs, eta_min=config.lr_eta_min
            )
            self.c_scheduler = CosineAnnealingLR(
                self.c_optimizer, T_max=config.epochs, eta_min=config.lr_eta_min
            )

        self.history: dict[str, list] = {
            "epoch": [],
            "gen_loss": [],
            "critic_loss": [],
            "gp_loss": [],
            "feature_matching_loss": [],
            "diversity_score": [],
            "wasserstein_distance": [],
            "composite_score": [],
            "mean_grad_norm_g": [],
            "mean_grad_norm_c": [],
            "lr_g": [],
        }

        self._best_composite: float = -1.0
        self._best_epoch: int = 0
        # W1 reference for quality normalisation.
        # Fixed at 1.0 (maximum plausible W1 in [-1,1] space) — NOT calibrated from
        # epoch 1, because at epoch 1 the generator is random noise and W1 happens to
        # be low (Tanh noise ≈ uniform ≈ normalised real data), which would make all
        # subsequent epochs score near 0.
        self._w1_reference: float = 1.0

        g_params = sum(p.numel() for p in self.generator.parameters())
        c_params = sum(p.numel() for p in self.critic.parameters())
        logger.info("WGAN-GP trainer initialised")
        logger.info(f"  Device           : {self.device}")
        logger.info(f"  Output dim       : {model_config.output_dim}")
        logger.info(f"  Generator params : {g_params:,}")
        logger.info(f"  Critic params    : {c_params:,}")
        logger.info(f"  lambda_gp        : {config.lambda_gp}  n_critic: {config.n_critic}")
        logger.info(f"  Feature matching : weight={config.feature_matching_weight}")
        logger.info(f"  MinibatchStdDev  : {model_config.use_minibatch_std}")
        logger.info(f"  LR scheduler     : {config.use_lr_scheduler} (cosine)")
        logger.info(f"  Early stopping   : {config.early_stopping_enabled} (patience={config.early_stopping_patience})")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> tuple[float, float, float, float]:
        gen_losses, critic_losses, gp_losses, fm_losses = [], [], [], []

        for batch in train_loader:
            real_data = batch[0] if isinstance(batch, (list, tuple)) else batch
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)

            # ── Critic ─────────────────────────────────────────────────
            for _ in range(self.config.n_critic):
                self.c_optimizer.zero_grad()

                real_score = self.critic(real_data)
                z = torch.randn(batch_size, self.model_config.z_dim, device=self.device)
                fake_data = self.generator(z).detach()
                fake_score = self.critic(fake_data)

                critic_loss = fake_score.mean() - real_score.mean()
                gp = compute_gradient_penalty(
                    self.critic, real_data, fake_data,
                    self.device, batch_size, self.config.lambda_gp,
                )
                (critic_loss + gp).backward()
                self.c_optimizer.step()

                critic_losses.append(critic_loss.item())
                gp_losses.append(gp.item())

            # ── Generator ──────────────────────────────────────────────
            self.g_optimizer.zero_grad()
            z = torch.randn(batch_size, self.model_config.z_dim, device=self.device)
            fake_data = self.generator(z)
            fake_score = self.critic(fake_data)

            gen_loss = -fake_score.mean()
            fm_loss = torch.tensor(0.0, device=self.device)
            if self.config.feature_matching_weight > 0:
                fm_loss = compute_feature_matching_loss(real_data, fake_data)
                total_g = gen_loss + self.config.feature_matching_weight * fm_loss
            else:
                total_g = gen_loss

            total_g.backward()
            self.g_optimizer.step()

            gen_losses.append(gen_loss.item())
            fm_losses.append(fm_loss.item())

        return (
            float(np.mean(gen_losses)),
            float(np.mean(critic_losses)),
            float(np.mean(gp_losses)),
            float(np.mean(fm_losses)),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, real_data_np: np.ndarray, n_samples: int = 500) -> tuple[float, float, float]:
        """
        Generate synthetic samples and compute diversity, Wasserstein distance,
        and composite score.

        Returns:
            (diversity, wasserstein, composite_score)
        """
        self.generator.eval()
        z = torch.randn(n_samples, self.model_config.z_dim, device=self.device)
        synthetic_np = self.generator(z).cpu().numpy()
        self.generator.train()

        diversity = compute_diversity_score(synthetic_np, real_data_np)
        wasserstein = compute_wasserstein_distance(synthetic_np, real_data_np)
        composite = compute_composite_score(
            diversity, wasserstein, w1_reference=self._w1_reference
        )
        return diversity, wasserstein, composite

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        output_dir: Path,
        real_data_np: np.ndarray | None = None,
    ) -> dict:
        """
        Full training loop with composite-score early stopping and best-model tracking.

        Args:
            train_loader: DataLoader with normalised training data.
            output_dir: Directory to save checkpoints, history, and plots.
            real_data_np: Normalised real data for metric computation.
                If None, metrics are skipped (not recommended).

        Returns:
            Training history dictionary.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        early_stopper = (
            EarlyStopper(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_threshold,
                mode="max",
            )
            if self.config.early_stopping_enabled
            else None
        )

        logger.info("=" * 70)
        logger.info("STARTING OPTIMIZED WGAN-GP TRAINING")
        logger.info("=" * 70)

        for epoch in range(1, self.config.epochs + 1):
            gen_loss, critic_loss, gp_loss, fm_loss = self._train_epoch(train_loader)

            diversity, wasserstein, composite = 0.0, 0.0, 0.0
            if real_data_np is not None:
                diversity, wasserstein, composite = self._evaluate(real_data_np)

            lr_g = self.g_optimizer.param_groups[0]["lr"]

            # Gradient stats
            g_stats = get_model_gradient_stats(self.generator)
            c_stats = get_model_gradient_stats(self.critic)

            self.history["epoch"].append(epoch)
            self.history["gen_loss"].append(gen_loss)
            self.history["critic_loss"].append(critic_loss)
            self.history["gp_loss"].append(gp_loss)
            self.history["feature_matching_loss"].append(fm_loss)
            self.history["diversity_score"].append(diversity)
            self.history["wasserstein_distance"].append(wasserstein)
            self.history["composite_score"].append(composite)
            self.history["mean_grad_norm_g"].append(g_stats["mean_grad_norm"])
            self.history["mean_grad_norm_c"].append(c_stats["mean_grad_norm"])
            self.history["lr_g"].append(lr_g)

            # Best model: composite score (quality + diversity)
            if composite > self._best_composite:
                self._best_composite = composite
                self._best_epoch = epoch
                self._save_checkpoint(output_dir / "checkpoint_best.pth", epoch)
                logger.info(
                    f"  [NEW BEST] epoch={epoch} composite={composite:.4f} "
                    f"(div={diversity:.4f}, W1={wasserstein:.4f})"
                )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>4}/{self.config.epochs} | "
                    f"G={gen_loss:+.4f} | C={critic_loss:+.4f} | "
                    f"GP={gp_loss:.4f} | FM={fm_loss:.4f} | "
                    f"div={diversity:.4f} | W1={wasserstein:.4f} | "
                    f"score={composite:.4f} | lr={lr_g:.2e}"
                )

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:04d}.pth", epoch)

            # Step LR schedulers AFTER optimizer update
            if self.g_scheduler:
                self.g_scheduler.step()
            if self.c_scheduler:
                self.c_scheduler.step()

            if early_stopper is not None:
                should_stop = early_stopper.step(composite, epoch)
                if should_stop:
                    logger.info(
                        f"Early stopping at epoch {epoch} — composite score unchanged "
                        f"for {self.config.early_stopping_patience} epochs. "
                        f"Best: epoch {early_stopper.best_epoch} "
                        f"(score={early_stopper.best_value:.4f})"
                    )
                    break

        logger.info("Training complete.")
        logger.info(
            f"Best model: epoch {self._best_epoch}, "
            f"composite={self._best_composite:.4f}"
        )

        self._save_final_models(output_dir)
        self._save_history(output_dir)
        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "g_optimizer": self.g_optimizer.state_dict(),
                "c_optimizer": self.c_optimizer.state_dict(),
                "best_composite": self._best_composite,
                "w1_reference": self._w1_reference,
            },
            path,
        )

    def save_config(self, output_dir: Path) -> None:
        """Persist training and model config for diagnostics and reproducibility."""
        cfg = {
            "training": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "lambda_gp": self.config.lambda_gp,
                "n_critic": self.config.n_critic,
                "feature_matching_weight": self.config.feature_matching_weight,
                "seed": self.config.seed,
                "early_stopping_enabled": self.config.early_stopping_enabled,
                "early_stopping_patience": self.config.early_stopping_patience,
                "early_stopping_threshold": self.config.early_stopping_threshold,
                "use_lr_scheduler": self.config.use_lr_scheduler,
            },
            "model": {
                "z_dim": self.model_config.z_dim,
                "output_dim": self.model_config.output_dim,
                "hidden_dims_g": self.model_config.hidden_dims_g,
                "hidden_dims_c": self.model_config.hidden_dims_c,
                "use_batch_norm_g": self.model_config.use_batch_norm_g,
                "use_minibatch_std": self.model_config.use_minibatch_std,
            },
            "results": {
                "best_epoch": self._best_epoch,
                "best_composite": self._best_composite,
                "w1_reference": self._w1_reference,
            },
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    def _save_final_models(self, output_dir: Path) -> None:
        torch.save(self.generator.state_dict(), output_dir / "generator_final.pth")
        torch.save(self.critic.state_dict(), output_dir / "critic_final.pth")
        self.save_config(output_dir)
        logger.info(f"Final models saved to {output_dir}")

    def _save_history(self, output_dir: Path) -> None:
        pd.DataFrame(self.history).to_csv(output_dir / "training_history.csv", index=False)
        self._plot_training_curves(output_dir)

    def _plot_training_curves(self, output_dir: Path) -> None:
        epochs = self.history["epoch"]
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))

        axes[0, 0].plot(epochs, self.history["gen_loss"])
        axes[0, 0].set_title("Generator Loss")

        axes[0, 1].plot(epochs, self.history["critic_loss"])
        axes[0, 1].set_title("Critic Loss")

        axes[1, 0].plot(epochs, self.history["gp_loss"], label="GP")
        axes[1, 0].plot(epochs, self.history["feature_matching_loss"], label="FM")
        axes[1, 0].set_title("Gradient Penalty & Feature Matching Loss")
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, self.history["composite_score"], color="purple")
        if self._best_epoch > 0:
            axes[1, 1].axvline(
                self._best_epoch, color="red", linestyle="--",
                label=f"best epoch {self._best_epoch} ({self._best_composite:.3f})",
            )
            axes[1, 1].legend()
        axes[1, 1].set_title("Composite Score (higher = better)")

        axes[2, 0].plot(epochs, self.history["diversity_score"])
        axes[2, 0].set_title("Diversity Score")

        axes[2, 1].plot(epochs, self.history["wasserstein_distance"])
        axes[2, 1].set_title("Wasserstein Distance (lower = better)")

        for ax in axes.flat:
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WGAN-GP with optimized hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str,
        default="python/data/junit_data_cleaned_strict.csv",
        help="Path to training CSV (no header, numeric features)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default from config: 8; recomendado para 398 amostras: 32-64)")
    parser.add_argument("--fm-weight", type=float, default=None,
                        help="Override feature matching loss weight (default 0.1; runs anteriores usaram 0.3)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate (default 2e-5)")
    parser.add_argument("--no-feature-matching", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--no-minibatch-std", action="store_true")
    parser.add_argument("--no-lr-scheduler", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        logger.error(f"Data file not found: {csv_path}")
        return

    config = TrainingConfig()
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.fm_weight is not None:
        config.feature_matching_weight = args.fm_weight
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.no_feature_matching:
        config.feature_matching_weight = 0.0
    if args.no_early_stopping:
        config.early_stopping_enabled = False
    if args.no_lr_scheduler:
        config.use_lr_scheduler = False

    set_random_seed(config.seed)

    logger.info(f"Loading training data: {csv_path}")
    dataset = JUnitDataset(csv_path, use_log_transform=True)
    logger.info(f"Loaded {len(dataset)} samples, {dataset.norm_stats['num_features']} features")

    real_data_np = dataset.normalized_data.numpy()
    model_config = ModelConfig(output_dim=dataset.norm_stats["num_features"])
    if args.no_minibatch_std:
        model_config.use_minibatch_std = False

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    trainer = WGANGPTrainer(config, model_config, device=args.device)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"optimized_{timestamp}"

    trainer.train(train_loader, output_dir, real_data_np=real_data_np)


if __name__ == "__main__":
    main()
