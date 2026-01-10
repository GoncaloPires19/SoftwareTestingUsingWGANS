# 🚀 QUICK START - WGAN Multi-Dataset Training

## O que foi implementado (5 componentes)

### 1️⃣ **config.py** - Sistema de Configuração Centralizado
- Presets: `quick`, `juliet_only`, `combined`, `evosuite_synthetic`
- Configurável: epochs, batch_size, learning rate, early stopping, métricas
- Salva/carrega configurations em JSON

### 2️⃣ **metrics.py** - Métricas Avançadas
- **Diversity**: Diversidade de código (n-grams + features)
- **Wasserstein Distance**: Distância distribuição real vs gerada
- **Feature Diversity**: Entropia, variância, correlação
- **CWE Coverage**: Cobertura de vulnerabilidades
- **Early Stopping**: Com métricas customizadas

### 3️⃣ **data_loader.py** - Carregador Unificado
- Suporta: Juliet, OWASP, Defects4J, EvoSuite, JUnit atual
- Normaliza tudo para formato comum (métodos, branches, statements)
- Splits automáticos: train/val/test
- PyTorch DataLoaders integrados

### 4️⃣ **Scripts de Dados**
- `download_datasets.py` - Baixa/normaliza Defects4J, Juliet, OWASP
- `generate_evosuite.py` - Gera testes com EvoSuite
- `generate_synthetic.py` - Gera testes sintéticos com modelo treinado
- `compare_experiments.py` - Compara múltiplos treinos

### 5️⃣ **train_configurable.py** - Novo Script de Treino
- Suporta múltiplos datasets
- Early stopping inteligente
- Rastreia múltiplas métricas
- Checkpoints automáticos

---

## ⚡ Uso Rápido

### Test 1: Quick (5 épocas, 1 minuto)
```bash
cd python
python scripts/train_configurable.py --preset quick
```

### Test 2: Só JUnit Atual
```bash
python scripts/train_configurable.py \
  --dataset combined \
  --epochs 50 \
  --name test_current_junit
```

### Test 3: Com Early Stopping
```bash
python scripts/train_configurable.py \
  --dataset combined \
  --epochs 200 \
  --patience 15 \
  --early-stopping \
  --name full_with_early_stop
```

### Test 4: Gerar Sintéticos
```bash
python scripts/generate_synthetic.py \
  --model models/generator_final.pth \
  --num-samples 5000 \
  --output outputs/synthetic_5k.csv
```

---

## 📊 Exemplo com Defects4J (Real)

```bash
# 1. Baixar/normalizar Defects4J
python scripts/download_datasets.py --process-defects4j

# 2. Treinar com Defects4J
python scripts/train_configurable.py \
  --dataset defects4j \
  --epochs 150 \
  --name defects4j_experiment

# 3. Comparar com baseline
python scripts/compare_experiments.py \
  outputs/training_history.csv \
  --metric diversity
```

---

## 🎯 Plano para Próximas Semanas

### Semana 1: Juliet + OWASP (Manual)
1. Download Juliet: https://samate.nist.gov/SARD/test-suites/81
2. Download OWASP: https://github.com/OWASP/benchmark
3. Extrair para `data/raw/juliet/` e `data/raw/owasp/`

### Semana 2: Primeiros Experimentos
```bash
# Treinar 3 baselines
python scripts/train_configurable.py --dataset juliet --epochs 100 --name juliet_baseline
python scripts/train_configurable.py --dataset defects4j --epochs 100 --name defects4j_baseline
python scripts/train_configurable.py --dataset evosuite --epochs 100 --name evosuite_baseline

# Comparar resultados
python scripts/compare_experiments.py \
  outputs/juliet_baseline/training_history.csv \
  outputs/defects4j_baseline/training_history.csv \
  outputs/evosuite_baseline/training_history.csv \
  --metric diversity
```

### Semana 3: Otimização
```bash
# Melhor combinação
python scripts/train_configurable.py \
  --dataset combined \
  --epochs 200 \
  --patience 20 \
  --name optimized_combined
```

### Semana 4: Validação
```bash
# Gerar sintéticos em escala
python scripts/generate_synthetic.py \
  --model models/generator_final.pth \
  --num-samples 10000 \
  --output outputs/synthetic_10k.csv

# Testar em ferramentas reais (SonarQube, FindBugs)
```

---

## 📈 Métricas Monitoradas

Durante treino:
```
[100] D: -0.4235 G: -1.2145 Val: -0.5831 | diversity: 0.7234 | wasserstein: 0.4521
```

Significado:
- **D Loss**: Critic loss (deve oscilar, não convergir)
- **G Loss**: Generator loss (geralmente negativo)
- **Val Loss**: Validation critic loss
- **Diversity**: Diversidade do código gerado (0-1, maior é melhor)
- **Wasserstein**: Distância distribuição (menor é melhor)

---

## 🔧 Troubleshooting

| Problema | Solução |
|----------|---------|
| "Dataset not found" | Download manual Juliet/OWASP, rodar `download_datasets.py` |
| Early stopping muito cedo | `--patience 30` ou `--no-early-stopping` |
| GPU out of memory | `--batch-size 16` ou `--device cpu` |
| Métricas zeradas | Verificar se dados carregados corretamente |

---

## 📚 Estrutura Atual

```
python/
├── config.py              ✅ Novo
├── metrics.py             ✅ Novo
├── data_loader.py         ✅ Novo
├── model.py               (existia)
├── utils.py               (existia)
└── scripts/
    ├── train_configurable.py    ✅ Novo
    ├── train_robust.py          (mantido)
    ├── download_datasets.py      ✅ Novo
    ├── generate_evosuite.py      ✅ Novo
    ├── generate_synthetic.py     ✅ Novo
    └── compare_experiments.py    ✅ Novo

data/
├── raw/
│   ├── juliet/         (manual)
│   ├── owasp/          (manual)
│   ├── defects4j/      (auto via git)
│   └── evosuite/       (auto via EvoSuite)
└── processed/
    ├── *_normalized.csv
    └── train.csv, val.csv, test.csv

outputs/
├── training_history.csv
├── models/
│   ├── generator_final.pth
│   └── critic_final.pth
└── synthetic_*.csv
```

---

## ✨ Diferenças vs Script Original (train_robust.py)

| Aspecto | Original | Novo |
|---------|----------|------|
| **Datasets** | Só JUnit | Juliet, OWASP, Defects4J, EvoSuite, Combinado |
| **Configuração** | Hard-coded | Parametrizada (CLI + config.py) |
| **Métricas** | Só loss | Diversity, Wasserstein, CWE coverage, etc |
| **Early Stopping** | Manual | Automático com paciência |
| **Checkpoints** | Cada 5 épocas | Configurável |
| **Presets** | N/A | quick, juliet_only, combined, evosuite_synthetic |
| **Comparação** | Manual | Script compare_experiments.py |

---

## 🎓 Próximos Passos (Ordre Sugerida)

1. ✅ **Hoje**: Testar `train_configurable.py --preset quick`
2. **Amanhã**: Download manual Juliet + OWASP
3. **Dia 3**: `python scripts/download_datasets.py` para Defects4J
4. **Semana 1**: Treinar 3 baselines (Juliet, Defects4J, EvoSuite)
5. **Semana 2**: Comparar resultados e otimizar
6. **Semana 3**: Treino combinado com early stopping
7. **Semana 4**: Validação em ferramentas reais

---

## 📞 Questões Frequentes

**P: Posso usar só JUnit como antes?**
A: Sim! `python scripts/train_configurable.py --dataset combined` carrega JUnit automaticamente

**P: Quanto tempo levam os treinos?**
A: 
- Juliet (100 ep): ~10 min (CPU)
- Defects4J (100 ep): ~15 min (CPU)
- Combinado (200 ep): ~30 min (CPU)
- Com GPU: 2-3x mais rápido

**P: Como saber se está overfitting?**
A: Monitorizar early stopping metric (diversity/wasserstein). Se desce durante treino mas sobe na validação = overfitting

**P: Preciso de GPU?**
A: Não obrigatório, mas recomendado. CPU funciona mas é ~10x mais lento

---

## 🎉 Resumo

Implementamos uma **infraestrutura profissional** para treinar WGAN com múltiplos datasets, métricas avançadas e early stopping automático. Tudo é configurável via CLI ou código, com suporte a presets e comparação de experimentos.

**Status**: ✅ Pronto para usar
**Próxima ação**: Testar `--preset quick`, depois preparar Juliet/OWASP

