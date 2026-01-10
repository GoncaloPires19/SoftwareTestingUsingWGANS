"""
Production WGAN training script
Full training with 300+ epochs on combined datasets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import torch
import pandas as pd
from datetime import datetime

from config import get_config
from data_loader import UnifiedDataLoader
from train_configurable import WGANTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_production():
    """
    Production training workflow:
    1. Load all available datasets (Juliet, OWASP, Defects4J, JUnit)
    2. Train WGAN for 300+ epochs
    3. Track metrics and early stopping
    4. Save best model and history
    """
    
    logger.info("=" * 70)
    logger.info("PRODUCTION WGAN TRAINING - 300+ EPOCHS")
    logger.info("=" * 70)
    
    # Configuration
    config = get_config(
        dataset_mode="combined",
        epochs=300,
        batch_size=32,
        experiment_name="production_final",
        early_stopping=True
    )
    
    # Override early stopping settings
    config.training.early_stopping_enabled = True
    config.training.early_stopping_patience = 30  # Stop if no improvement for 30 epochs
    config.training.checkpoint_interval = 10
    config.training.log_interval = 20
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Experiment: {config.experiment_name}")
    logger.info(f"   Epochs: {config.training.epochs}")
    logger.info(f"   Batch size: {config.training.batch_size}")
    logger.info(f"   Learning rate: {config.training.learning_rate}")
    logger.info(f"   Early stopping: {config.training.early_stopping_enabled}")
    logger.info(f"   Early stopping patience: {config.training.early_stopping_patience}")
    
    # Load data
    logger.info("\n📂 Loading datasets...")
    data_loader = UnifiedDataLoader()
    
    train_data, val_data, test_data, metadata = data_loader.load_combined(
        include_juliet=True,
        include_owasp=True,
        include_defects4j=True,
        include_evosuite=False,
        include_current_junit=True,
    )
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.training.batch_size,
    )
    
    logger.info(f"\n📊 Data Summary:")
    logger.info(f"   Total training samples: {len(train_data)}")
    logger.info(f"   Total validation samples: {len(val_data)}")
    logger.info(f"   Total test samples: {len(test_data)}")
    
    # Initialize trainer
    trainer = WGANTrainer(config)
    
    # Train
    logger.info("\n🚀 Starting training...")
    start_time = datetime.now()
    
    history = trainer.train(train_loader, val_loader, train_data)
    
    elapsed = datetime.now() - start_time
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Total time: {elapsed}")
    logger.info(f"   Epochs completed: {len(history)}")
    
    # Summary
    logger.info("\n📊 Training Summary:")
    df_history = pd.DataFrame(history)
    
    if len(df_history) > 0:
        logger.info(f"   Best validation loss: {df_history['val_loss'].min():.4f}")
        logger.info(f"   Final D loss: {df_history['train_d_loss'].iloc[-1]:.4f}")
        logger.info(f"   Final G loss: {df_history['train_g_loss'].iloc[-1]:.4f}")
    
    logger.info(f"\n💾 Models saved to: {config.output.models_dir}/")
    logger.info(f"   History saved to: {config.output.output_dir}/training_history.csv")
    
    return trainer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production WGAN training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    
    args = parser.parse_args()
    
    trainer, history = train_production()
