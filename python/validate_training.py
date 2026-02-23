"""
Pre-Training Validation Script
==============================

Valida se tudo está pronto para treino:
- ✅ Dataset existe e é carregável
- ✅ Modelos podem ser inicializados
- ✅ Dataloader funciona
- ✅ GPU detectada (se disponível)
- ✅ Configurações são válidas

Use este script antes de iniciar qualquer treino longo!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def validate_environment():
    """Validar ambiente de desenvolvimento"""
    logger.info("\n" + "="*70)
    logger.info("1. ENVIRONMENT VALIDATION")
    logger.info("="*70)
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"   Python: {py_version} ✅")
    
    # PyTorch
    torch_version = torch.__version__
    logger.info(f"   PyTorch: {torch_version} ✅")
    
    # GPU
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"   GPU: {gpu_name} ✅")
    else:
        logger.warning("   GPU: Not available (will use CPU)")
    
    return True


def validate_dataset():
    """Validar dataset"""
    logger.info("\n" + "="*70)
    logger.info("2. DATASET VALIDATION")
    logger.info("="*70)
    
    try:
        from junit_datasets import JUnitDataset
        
        data_path = Path("python/data/junit_data_cleaned_strict.csv")
        if not data_path.exists():
            logger.error(f"   ✗ Data file not found: {data_path}")
            return False
        
        logger.info(f"   Loading from: {data_path}")
        dataset = JUnitDataset(str(data_path))
        logger.info(f"   Samples: {len(dataset)} ✅")
        
        # Check a batch
        if len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, torch.Tensor):
                logger.info(f"   Sample shape: {sample.shape} ✅")
                logger.info(f"   Sample dtype: {sample.dtype} ✅")
                logger.info(f"   Sample values: min={sample.min():.4f}, max={sample.max():.4f} ✅")
            else:
                logger.error(f"   ✗ Invalid sample type: {type(sample)}")
                return False
        
        return True, dataset
    
    except Exception as e:
        logger.error(f"   ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def validate_models():
    """Validar modelos"""
    logger.info("\n" + "="*70)
    logger.info("3. MODEL VALIDATION")
    logger.info("="*70)
    
    try:
        from model import Generator, Critic
        from config import ModelConfig
        
        config = ModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generator
        logger.info("   Generator:")
        gen = Generator(z_dim=config.z_dim, output_dim=config.output_dim).to(device)
        z = torch.randn(1, config.z_dim).to(device)
        out = gen(z)
        logger.info(f"      Input: ({1}, {config.z_dim})")
        logger.info(f"      Output: {out.shape} ✅")
        
        # Critic
        logger.info("   Critic:")
        crit = Critic(input_dim=config.output_dim).to(device)
        real_data = torch.randn(1, config.output_dim).to(device)
        score = crit(real_data)
        logger.info(f"      Input: {real_data.shape}")
        logger.info(f"      Output: {score.shape} ✅")
        
        return True
    
    except Exception as e:
        logger.error(f"   ✗ Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dataloader(dataset):
    """Validar DataLoader"""
    logger.info("\n" + "="*70)
    logger.info("4. DATALOADER VALIDATION")
    logger.info("="*70)
    
    try:
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        logger.info(f"   BatchSize: 16")
        logger.info(f"   Batches: {len(loader)}")
        
        # Get first batch
        batch = next(iter(loader))
        logger.info(f"   Batch shape: {batch.shape} ✅")
        logger.info(f"   Batch dtype: {batch.dtype} ✅")
        logger.info(f"   Batch range: [{batch.min():.4f}, {batch.max():.4f}] ✅")
        
        return True
    
    except Exception as e:
        logger.error(f"   ✗ Error with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_training_config():
    """Validar configurações de treino"""
    logger.info("\n" + "="*70)
    logger.info("5. TRAINING CONFIG VALIDATION")
    logger.info("="*70)
    
    try:
        from training_profiles import TrainingProfile, get_training_config, PROFILES
        
        logger.info("   Available profiles:")
        for profile in TrainingProfile:
            hp = PROFILES[profile]
            logger.info(f"      - {profile.value:15} (epochs={hp.epochs}, lr={hp.learning_rate:.5f})")
        
        # Test getting config
        config = get_training_config(TrainingProfile.BALANCED)
        logger.info(f"\n   Sample config (BALANCED):")
        logger.info(f"      Epochs: {config.epochs}")
        logger.info(f"      Batch size: {config.batch_size}")
        logger.info(f"      Learning rate: {config.learning_rate}")
        logger.info(f"      Lambda GP: {config.lambda_gp}")
        logger.info(f"      n_critic: {config.n_critic} ✅")
        
        return True
    
    except Exception as e:
        logger.error(f"   ✗ Error with training config: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_recommendations(dataset_size):
    """Imprimir recomendações baseadas na validação"""
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)
    
    if dataset_size < 30:
        logger.warning(f"   ⚠️  Small dataset: {dataset_size} samples")
        logger.info("      Recommendation: Use CONSERVATIVE profile")
        logger.info("      Command: python launch_training.py --scenario research")
    
    elif dataset_size < 100:
        logger.info(f"   📊 Moderate dataset: {dataset_size} samples")
        logger.info("      Recommendation: Use BALANCED profile")
        logger.info("      Command: python launch_training.py --profile balanced")
    
    else:
        logger.info(f"   ✅ Good dataset: {dataset_size} samples")
        logger.info("      Recommendation: Use BALANCED or EXPERIMENTAL profile")
        logger.info("      Command: python launch_training.py --profile experimental")
    
    if torch.cuda.is_available():
        logger.info("\n   GPU detected! Use it for faster training:")
        logger.info("      Command: python launch_training.py --profile balanced --device cuda")
    
    logger.info("\n   To start training:")
    logger.info("      python launch_training.py --profile balanced")


def main():
    """Run all validations"""
    logger.info("\n"*2)
    logger.info("🔍 PRE-TRAINING VALIDATION")
    logger.info("="*70)
    logger.info("This script validates that everything is ready for WGAN training")
    logger.info("="*70)
    
    # Run validations
    checks = {
        "Environment": validate_environment,
        "Dataset": None,  # Will be handled specially
        "Models": validate_models,
        "Dataloader": None,  # Depends on dataset
        "Training Config": validate_training_config,
    }
    
    results = []
    dataset = None
    
    # 1. Environment
    if not validate_environment():
        logger.error("   Environment check failed!")
        return False
    results.append(("Environment", True))
    
    # 2. Dataset
    dataset_ok, dataset = validate_dataset()
    if not dataset_ok:
        logger.error("   Dataset check failed!")
        return False
    results.append(("Dataset", True))
    
    # 3. Models
    if not validate_models():
        logger.error("   Model check failed!")
        return False
    results.append(("Models", True))
    
    # 4. DataLoader
    if not validate_dataloader(dataset):
        logger.error("   DataLoader check failed!")
        return False
    results.append(("DataLoader", True))
    
    # 5. Training Config
    if not validate_training_config():
        logger.error("   Training config check failed!")
        return False
    results.append(("Training Config", True))
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    for name, ok in results:
        status = "✅ OK" if ok else "❌ FAILED"
        logger.info(f"   {name:20} {status}")
    
    # Recommendations
    print_recommendations(len(dataset))
    
    logger.info("\n" + "="*70)
    logger.info("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
    logger.info("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
