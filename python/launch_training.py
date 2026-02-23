"""
Training Launcher - Interface simples para treinar com diferentes perfis
========================================================================

Uso:
  python launch_training.py --profile balanced
  python launch_training.py --profile aggressive
  python launch_training.py --scenario quick_test
  python launch_training.py --list-profiles
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
from torch.utils.data import DataLoader
import torch

from config import set_random_seed, ModelConfig
from training_profiles import (
    TrainingProfile, get_training_config, get_scenario_profile,
    print_profile_info, print_comparison_table, PROFILES
)
from scripts.train_optimized import OptimizedWGANTrainer
from junit_datasets import JUnitDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="🚀 WGAN Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_training.py --profile balanced
  python launch_training.py --scenario quick_test
  python launch_training.py --profile aggressive --epochs 60
  python launch_training.py --list-profiles
        """
    )
    
    # Main options
    parser.add_argument(
        '--profile',
        type=str,
        choices=['balanced', 'aggressive', 'conservative', 'experimental'],
        default='balanced',
        help='Training profile to use'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['quick_test', 'production', 'research', 'optimization'],
        help='Use predefined scenario (overrides --profile)'
    )
    
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List all available profiles and exit'
    )
    
    # Override options
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--lambda-gp', type=float, help='Override lambda GP')
    parser.add_argument('--n-critic', type=int, help='Override n critic')
    
    # Other options
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='python/data/junit_data_cleaned_strict.csv',
        help='Path to training data'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # List profiles if requested
    if args.list_profiles:
        print_comparison_table()
        for profile in TrainingProfile:
            print(f"\n{'='*70}")
            print_profile_info(profile)
        return
    
    # Determine profile
    if args.scenario:
        profile = get_scenario_profile(args.scenario)
        logger.info(f"📌 Using scenario: {args.scenario} → {profile.value}")
    else:
        profile = TrainingProfile(args.profile)
    
    # Get training config
    config = get_training_config(profile)
    
    # Override with command-line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.lambda_gp:
        config.lambda_gp = args.lambda_gp
    if args.n_critic:
        config.n_critic = args.n_critic
    
    # Set seed
    set_random_seed(args.seed)
    
    # Print configuration
    print_profile_info(profile)
    print("\n" + "="*70)
    print("🔧 EFFECTIVE CONFIGURATION")
    print("="*70)
    print(f"   Epochs:          {config.epochs}")
    print(f"   Batch Size:      {config.batch_size}")
    print(f"   Learning Rate:   {config.learning_rate:.5f}")
    print(f"   Lambda GP:       {config.lambda_gp}")
    print(f"   n_critic:        {config.n_critic}")
    print(f"   Device:          {args.device}")
    print(f"   Data:            {args.data_path}")
    print("="*70 + "\n")
    
    # Load data
    logger.info(f"📂 Loading data from {args.data_path}...")
    try:
        dataset = JUnitDataset(args.data_path)
        logger.info(f"✅ Loaded {len(dataset)} samples")
    except FileNotFoundError:
        logger.error(f"❌ Data file not found: {args.data_path}")
        logger.info("\n💡 Run the following to generate data:")
        logger.info("   cd python")
        logger.info("   python -c 'from junit_datasets import JUnitDataset; JUnitDataset.generate_dummy_data()'")
        return
    
    if len(dataset) < 20:
        logger.warning(f"⚠️  Very small dataset: {len(dataset)} samples")
        logger.info("   Consider using more data for better training")
    
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize trainer
    model_config = ModelConfig()
    trainer = OptimizedWGANTrainer(config, model_config, device=args.device)
    
    # Train
    logger.info("⏱️  Starting training...\n")
    history = trainer.train(train_loader)
    
    # Save results
    generator_path, critic_path = trainer.save_models(suffix=profile.value)
    history_path = trainer.save_training_history(suffix=profile.value)
    
    # Print summary
    print("\n" + "="*70)
    print("[SUCCESS] TRAINING COMPLETED")
    print("="*70)
    print(f"Profile used:    {profile.value}")
    print(f"Generator:       {generator_path}")
    print(f"Critic:          {critic_path}")
    print(f"History:         {history_path}")
    print("="*70 + "\n")
    
    # Next steps
    print("📋 NEXT STEPS:")
    print("   1. Analyze training curves:")
    print(f"      python -c \"import pandas as pd; pd.read_csv('{history_path}').plot()\"")
    print()
    print("   2. Generate synthetic tests:")
    print("      python python/scripts/generate_synthetic.py")
    print()
    print("   3. Evaluate results:")
    print("      python python/scripts/experimental_analysis.py")


if __name__ == "__main__":
    main()
