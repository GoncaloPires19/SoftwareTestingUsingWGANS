"""
WGAN Training Profiles - Diferentes configurações de hiperparâmetros
===================================================================

Este módulo define diferentes perfis de treino otimizados para diferentes
cenários e objetivos.

Profiles disponíveis:
- BALANCED: Bem equilibrado, bom para produção
- AGGRESSIVE: Treino mais rápido, maior risco de instabilidade
- CONSERVATIVE: Treino mais lento, maior estabilidade
- EXPERIMENTAL: Testes com novos hiperparâmetros
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict
from config import TrainingConfig

class TrainingProfile(Enum):
    """Available training profiles"""
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    EXPERIMENTAL = "experimental"


@dataclass
class TrainingHyperparams:
    """Container for training hyperparameters"""
    epochs: int
    batch_size: int
    learning_rate: float
    lambda_gp: float
    n_critic: int
    description: str


# ============================================================================
# PROFILE DEFINITIONS
# ============================================================================

PROFILES: Dict[TrainingProfile, TrainingHyperparams] = {
    
    TrainingProfile.BALANCED: TrainingHyperparams(
        epochs=100,
        batch_size=16,
        learning_rate=0.00005,
        lambda_gp=5.0,
        n_critic=7,
        description="""
        🎯 BALANCED - Recomendado para produção
        
        Características:
        - 100 épocas para convergência adequada
        - Batch size 16 para bom balanço memória/atualizações
        - LR moderada para aprendizagem estável
        - Lambda GP 5.0 para boa penalização de gradiente
        - 7 iterações de critic por generator
        
        Melhor para: Treino balanceado com bons resultados
        Tempo estimado: ~30-50 min (CPU) | ~5-10 min (GPU)
        """
    ),
    
    TrainingProfile.AGGRESSIVE: TrainingHyperparams(
        epochs=50,
        batch_size=32,
        learning_rate=0.0001,
        lambda_gp=2.0,
        n_critic=5,
        description="""
        ⚡ AGGRESSIVE - Treino rápido, pode ser instável
        
        Características:
        - 50 épocas apenas (mais rápido)
        - Batch size 32 para atualizações menos frequentes
        - LR mais alta para aprendizagem mais rápida
        - Lambda GP 2.0 para penalização mais fraca
        - 5 iterações de critic (menos refinamento)
        
        ⚠️  Riscos:
        - Mode collapse mais provável
        - Menos exploração do espaço latente
        - Qualidade das amostras pode ser pior
        
        Melhor para: Prototipagem rápida, testes
        Tempo estimado: ~10-20 min
        """
    ),
    
    TrainingProfile.CONSERVATIVE: TrainingHyperparams(
        epochs=150,
        batch_size=8,
        learning_rate=0.000025,
        lambda_gp=10.0,
        n_critic=12,
        description="""
        🛡️  CONSERVATIVE - Máxima estabilidade
        
        Características:
        - 150 épocas para convergência lenta e estável
        - Batch size 8 para muitas atualizações
        - LR reduzida para aprendizagem muito cautelosa
        - Lambda GP 10.0 para forte penalização
        - 12 iterações de critic (muito refinamento)
        
        Vantagens:
        - Muito estável
        - Menos risco de mode collapse
        - Melhor convergência teórica
        
        Desvantagens:
        - Muito lento
        - Pode não atingir boas soluções
        - Uso excessivo de recursos
        
        Melhor para: Testes de estabilidade
        Tempo estimado: ~2h+ (CPU) | ~20+ min (GPU)
        """
    ),
    
    TrainingProfile.EXPERIMENTAL: TrainingHyperparams(
        epochs=80,
        batch_size=12,
        learning_rate=0.00008,
        lambda_gp=7.5,
        n_critic=8,
        description="""
        🧪 EXPERIMENTAL - Configuração melhorada
        
        Características:
        - 80 épocas (meio ponto entre balanced e agg)
        - Batch size 12 (bom balanço)
        - LR 0.00008 (ligeiramente mais agressiva)
        - Lambda GP 7.5 (forte penalização)
        - 8 iterações de critic (mais que o balanced)
        
        Objetivos:
        - Melhor qualidade que aggressive
        - Mais rápido que conservative
        - Teste de novos hiperparâmetros
        
        Melhor para: Otimização e experimentos
        Tempo estimado: ~20-40 min
        """
    ),
}


def get_profile(profile: TrainingProfile) -> TrainingHyperparams:
    """Get hyperparameters for a specific profile"""
    return PROFILES[profile]


def get_training_config(profile: TrainingProfile) -> TrainingConfig:
    """Convert profile to TrainingConfig"""
    hyp = PROFILES[profile]
    return TrainingConfig(
        epochs=hyp.epochs,
        batch_size=hyp.batch_size,
        learning_rate=hyp.learning_rate,
        lambda_gp=hyp.lambda_gp,
        n_critic=hyp.n_critic
    )


def print_profile_info(profile: TrainingProfile):
    """Print detailed information about a profile"""
    hyp = PROFILES[profile]
    print(hyp.description)


def list_all_profiles():
    """List all available profiles"""
    print("\n" + "="*70)
    print("📋 WGAN TRAINING PROFILES")
    print("="*70)
    
    for profile in TrainingProfile:
        hyp = PROFILES[profile]
        print(f"\n{profile.value.upper()}")
        print(f"  Epochs: {hyp.epochs} | Batch: {hyp.batch_size} | LR: {hyp.learning_rate}")
        print(f"  Lambda GP: {hyp.lambda_gp} | n_critic: {hyp.n_critic}")


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def print_comparison_table():
    """Print comparison of all profiles"""
    print("\n" + "="*100)
    print("WGAN PROFILES - COMPARISON TABLE")
    print("="*100)
    print(f"{'Profile':<15} {'Epochs':<10} {'Batch':<8} {'LR':<12} {'LambdaGP':<10} {'n_critic':<10} {'Speed':<10}")
    print("-"*100)
    
    for profile in TrainingProfile:
        hyp = PROFILES[profile]
        speed = "⚡ Fast" if hyp.epochs <= 50 else ("🐢 Slow" if hyp.epochs >= 150 else "🚀 Medium")
        print(f"{profile.value:<15} {hyp.epochs:<10} {hyp.batch_size:<8} {hyp.learning_rate:<12.5f} {hyp.lambda_gp:<8} {hyp.n_critic:<10} {speed:<10}")
    
    print("="*100 + "\n")


# ============================================================================
# QUICK PROFILES FOR COMMON SCENARIOS
# ============================================================================

SCENARIO_PROFILES = {
    "quick_test": TrainingProfile.AGGRESSIVE,
    "production": TrainingProfile.BALANCED,
    "research": TrainingProfile.CONSERVATIVE,
    "optimization": TrainingProfile.EXPERIMENTAL,
}


def get_scenario_profile(scenario: str) -> TrainingProfile:
    """Get profile for a common scenario"""
    return SCENARIO_PROFILES.get(scenario, TrainingProfile.BALANCED)


if __name__ == "__main__":
    print_comparison_table()
    print_profile_info(TrainingProfile.BALANCED)
