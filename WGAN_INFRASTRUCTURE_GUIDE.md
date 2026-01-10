# WGAN Training Infrastructure - Complete Guide

## Overview

Este documento descreve a nova infraestrutura configurável para treinar WGAN com múltiplos datasets:
- **Juliet** (Java vulnerability benchmarks)
- **OWASP Benchmark** (web security tests)
- **Defects4J** (real bugs de código open-source)
- **EvoSuite** (testes gerados automaticamente)
- **Combinações** (mistura de todos)

## Architecture

```
python/
├── config.py                 # Sistema de configuração centralizado
├── metrics.py               # Métricas avançadas (FID, diversity, CWE coverage)
├── data_loader.py           # Carregador unificado de datasets
├── model.py                 # Arquitetura WGAN-GP (já existia)
├── utils.py                 # Utilidades (gradient penalty, etc)
└── scripts/
    ├── train_configurable.py  # 🆕 Treino com configação completa
    ├── train_robust.py         # (mantido para compatibilidade)
    ├── download_datasets.py    # 🆕 Download/normalização de datasets
    ├── generate_evosuite.py    # 🆕 Geração de testes com EvoSuite
    └── ...outros scripts
```

## Step 1: Setup Inicial

### 1.1 Instalar dependências

```bash
pip install pandas numpy torch pytorch scikit-learn scipy
```

### 1.2 Verificar estrutura de dados

```
data/
├── raw/
│   ├── juliet/          (manual download)
│   ├── owasp/           (manual download)
│   ├── defects4j/       (auto-download via git)
│   └── evosuite/        (auto-gerado via EvoSuite)
└── processed/
    ├── juliet_normalized.csv
    ├── owasp_normalized.csv
    ├── defects4j_normalized.csv
    ├── evosuite_normalized.csv
    └── train.csv, val.csv, test.csv  (splits)
```

## Step 2: Preparar Datasets

### 2.1 Juliet Java Test Suite (OBRIGATÓRIO - Manual)

1. Descarregar de: https://samate.nist.gov/SARD/test-suites/81
2. Extrair para: `data/raw/juliet/`
3. Estrutura esperada:
   ```
   data/raw/juliet/
   └── Juliet/
       └── v1.3/
           └── Java/
               ├── CWE-78/
               ├── CWE-79/
               ├── CWE-89/
               └── ... (mais CWEs)
   ```

### 2.2 OWASP Benchmark (OBRIGATÓRIO - Manual)

1. Descarregar de: https://github.com/OWASP/benchmark
2. Extrair para: `data/raw/owasp/`

### 2.3 Defects4J (Automático)

```bash
cd python
python scripts/download_datasets.py --process-defects4j
```

Este script:
- Clona Defects4J via git
- Extrai testes reais
- Normaliza para formato comum

### 2.4 EvoSuite (Opcional - Automático)

Requer EvoSuite instalado. Duas opções:

**Option A: CLI EvoSuite**
```bash
# Instalar EvoSuite standalone
# Download: https://www.evosuite.org/download/

python scripts/generate_evosuite.py --classes 10 --time-per-class 60
```

**Option B: Maven Plugin**
```bash
# Adicionar ao pom.xml:
<plugin>
    <groupId>org.evosuite.plugins</groupId>
    <artifactId>evosuite-maven-plugin</artifactId>
    <version>1.0.6</version>
</plugin>

# Gerar:
mvn evosuite:generate -DtimeInMinutesPerClass=1
```

### 2.5 Normalizar Todos os Datasets

```bash
cd python
python scripts/download_datasets.py
```

Output:
```
data/processed/
├── juliet_normalized.csv        (CWE-based vulnerabilities)
├── owasp_normalized.csv         (web security)
├── defects4j_normalized.csv     (real bugs)
└── evosuite_normalized.csv      (synthetic tests)
```

## Step 3: Treinar com Diferentes Configurações

### 3.1 Quick Test (5 épocas)

```bash
cd python
python scripts/train_configurable.py --preset quick
```

### 3.2 Só Juliet (Baseline)

```bash
python scripts/train_configurable.py \
  --dataset juliet \
  --epochs 100 \
  --name juliet_baseline
```

Características:
- Foca em vulnerabilidades (CWEs)
- Dados etiquetados
- ~500-1000 amostras

### 3.3 Só Defects4J (Bugs Reais)

```bash
python scripts/train_configurable.py \
  --dataset defects4j \
  --epochs 100 \
  --name defects4j_realbugs
```

### 3.4 Só EvoSuite (Sintéticos)

```bash
python scripts/train_configurable.py \
  --dataset evosuite \
  --epochs 150 \
  --name evosuite_synthetic
```

### 3.5 COMBINADO (Recomendado)

```bash
python scripts/train_configurable.py \
  --dataset combined \
  --epochs 200 \
  --patience 20 \
  --name full_combined_v1 \
  --early-stopping
```

Este modo:
- Mistura Juliet + OWASP + Defects4J + EvoSuite + JUnit atual
- Tira vantagem de:
  - **Juliet**: CWEs etiquetadas (vulnerability awareness)
  - **OWASP**: Contexto web/segurança
  - **Defects4J**: Bugs reais e complexos
  - **EvoSuite**: Cobertura ampla + diversidade
  - **JUnit atual**: Dados conhecidos
- Early stopping automático

## Step 4: Configurações Avançadas

### 4.1 Via arquivo de configuração

```python
from config import WGANConfig, DatasetMode

config = WGANConfig(
    experiment_name="advanced_experiment",
    data=DataConfig(
        dataset_mode=DatasetMode.COMBINED,
        train_split=0.7,
        val_split=0.15,
    ),
    training=TrainingConfig(
        epochs=200,
        batch_size=64,
        learning_rate=0.0002,
        lambda_gp=10,
        n_critic=5,
        early_stopping_enabled=True,
        early_stopping_patience=15,
        early_stopping_metric=MetricType.DIVERSITY,
    ),
    evaluation=EvaluationConfig(
        metrics=[
            MetricType.DIVERSITY,
            MetricType.WASSERSTEIN_DISTANCE,
            MetricType.CWE_COVERAGE,
        ]
    ),
)

# Salvar config
config.save(Path("outputs/my_config.json"))
```

### 4.2 Métricas Avançadas

O novo sistema rastreia:

1. **Diversity** (diversidade de código gerado)
   - N-gram diversity
   - Feature diversity (entropia + variância)
   - Correlação entre features

2. **Wasserstein Distance** (distância distribuição real vs gerada)
   - 1D Wasserstein
   - Sliced Wasserstein (high-dimensional)

3. **FID-like** (adaptado para código)
   - Embedding distance (CodeBERT)
   - Distribuição estatística

4. **CWE Coverage** (cobertura de vulnerabilidades)
   - % de CWEs representados
   - Distribuição por tipo

Logs durante treino:
```
[100] D: -0.4235 G: -1.2145 Val: -0.5831 | diversity: 0.7234 | wasserstein: 0.4521 (0.23s)
```

### 4.3 Early Stopping

Automático quando o modelo não melhora:

```python
config.training.early_stopping_enabled = True
config.training.early_stopping_patience = 15  # Stop após 15 épocas sem melhoria
config.training.early_stopping_metric = MetricType.DIVERSITY  # Métrica para monitorizar
```

## Step 5: Análise de Resultados

### 5.1 Training history

```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/training_history.csv')
print(df[['epoch', 'train_d_loss', 'train_g_loss', 'diversity']].tail(10))
"
```

### 5.2 Comparar múltiplos treinos

```bash
python scripts/compare_experiments.py \
  --experiments juliet_baseline defects4j_realbugs combined_v1 \
  --metric diversity
```

### 5.3 Visualizar métricas

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/training_history.csv')

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0,0].plot(df['epoch'], df['train_d_loss'], label='D Loss')
ax[0,0].set_title('Critic Loss')

ax[0,1].plot(df['epoch'], df['train_g_loss'], label='G Loss')
ax[0,1].set_title('Generator Loss')

ax[1,0].plot(df['epoch'], df['diversity'], label='Diversity')
ax[1,0].set_title('Code Diversity')

ax[1,1].plot(df['epoch'], df['wasserstein_distance'], label='Wasserstein')
ax[1,1].set_title('Distribution Distance')

plt.tight_layout()
plt.savefig('outputs/analysis.png', dpi=150)
```

## Step 6: Gerar Testes Sintéticos com Diferentes Modos

### 6.1 Com modelo treinado em Juliet

```bash
python scripts/generate_synthetic.py \
  --model models/generator_final.pth \
  --dataset juliet \
  --num-samples 1000 \
  --output outputs/synthetic_juliet.csv
```

Resultado: Testes com características de vulnerabilidade

### 6.2 Com modelo treinado em Defects4J

```bash
python scripts/generate_synthetic.py \
  --model models/generator_final.pth \
  --dataset defects4j \
  --num-samples 1000 \
  --output outputs/synthetic_defects4j.csv
```

Resultado: Testes com bugs complexos

### 6.3 Com modelo combinado

```bash
python scripts/generate_synthetic.py \
  --model models/generator_final.pth \
  --dataset combined \
  --num-samples 5000 \
  --output outputs/synthetic_combined.csv
```

Resultado: Testes diversos com vulnerabilidades + bugs

## Step 7: Experimentos Recomendados

### Experimento 1: Baseline (2-3 horas)

```bash
# Treino individual
python scripts/train_configurable.py --dataset juliet --epochs 100 --name exp1_juliet
python scripts/train_configurable.py --dataset defects4j --epochs 100 --name exp1_defects4j
python scripts/train_configurable.py --dataset evosuite --epochs 100 --name exp1_evosuite
```

**Objetivo:** Comparar qualidade de síntese por dataset individual

### Experimento 2: Mixtures (3-4 horas)

```bash
# Mistura 2 a 2
python scripts/train_configurable.py --dataset combined --epochs 150 --name exp2_juliet_defects4j
# Modificar config para juliet+owasp, etc...
```

**Objetivo:** Encontrar melhor combinação

### Experimento 3: Full Pipeline (5-6 horas)

```bash
# Treino completo com early stopping
python scripts/train_configurable.py \
  --dataset combined \
  --epochs 300 \
  --batch-size 64 \
  --patience 20 \
  --early-stopping \
  --name exp3_full_combined
```

**Objetivo:** Modelo final otimizado

### Experimento 4: Avaliação em Benchmark Real

```bash
# Gerar sintéticos com cada modelo
python scripts/generate_synthetic.py --model exp1_juliet/generator_final.pth --output out1.csv
python scripts/generate_synthetic.py --model exp1_defects4j/generator_final.pth --output out2.csv
python scripts/generate_synthetic.py --model exp3_full_combined/generator_final.pth --output out3.csv

# Testar em ferramentas reais (SonarQube, FindBugs, etc)
# Medir: detecção real vs falsos positivos
```

## Troubleshooting

### Problema: "Dataset not found"

```bash
# Verificar se datasets estão em data/raw/
ls data/raw/

# Se falta Juliet ou OWASP:
# 1. Descarregar manual
# 2. Extrair para data/raw/juliet/ ou data/raw/owasp/
# 3. Rodar normalize script
```

### Problema: "CUDA out of memory"

```bash
# Reduzir batch size
python scripts/train_configurable.py --batch-size 16

# Ou forçar CPU
python scripts/train_configurable.py --device cpu
```

### Problema: "Early stopping muito cedo"

```bash
# Aumentar patience
python scripts/train_configurable.py --patience 30

# Ou desabilitar
python scripts/train_configurable.py --no-early-stopping
```

## Next Steps

1. **Download Juliet e OWASP** (manual, ~1-2 horas)
2. **Rodar `download_datasets.py`** para Defects4J e normalizar
3. **Executar Experimento 1** (baselines) - ~2 horas
4. **Comparar resultados** - qual dataset + modelo melhor
5. **Experimento 3** (full combined) com melhor combinação
6. **Gerar sintéticos** em escala (5000-10000 amostras)
7. **Validar** em ferramentas reais (SonarQube, etc)

## References

- Arjovsky et al., "Wasserstein GAN" (2017)
- Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
- Khare et al., "Understanding the Effectiveness of LLMs in Detecting Security Vulnerabilities" (ICST 2025)
- Defects4J: https://github.com/rjust/defects4j
- Juliet: https://samate.nist.gov/SARD/
- OWASP: https://github.com/OWASP/benchmark
- EvoSuite: https://www.evosuite.org/

---

**Status:** ✅ Infraestrutura completa e pronta para uso

**Próxima ação:** Download manual de Juliet e OWASP, depois começar Experimento 1
