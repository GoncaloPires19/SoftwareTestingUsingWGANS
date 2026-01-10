# Project Structure Guide

Organização do repositório "Software Testing Using WGAN-GP"

## 📁 Estrutura do Projeto

```
SoftwareTestingUsingWGANS/
├── 📄 README.md                    # Documentação principal do projeto
├── 📄 pom.xml                      # Configuração Maven (Java)
├── 📄 setup.py                     # Configuração Python
├── 📄 requirements.txt             # Dependências Python
│
├── 📁 docs/                        # Documentação do projeto
│   ├── CLEANUP_SUMMARY.md
│   ├── EVOSUITE_GUIDE.md
│   ├── FINAL_VALIDATION.md
│   ├── PHASE2_REPORT.md
│   ├── PROJECT_VISUAL_MAP.md
│   ├── TESTING_GUIDE.md
│   └── TRAINING_RESULTS.md
│
├── 📁 src/                         # Código Java (Maven)
│   ├── main/java/dataloader/      # Utilitários Java
│   │   ├── MathOperations.java    # Classe utilitária (12 métodos)
│   │   └── ...
│   └── test/java/                 # 193 testes JUnit
│       ├── ExampleTests.java      # 34 testes
│       ├── AdvancedTests.java     # 18 testes
│       ├── AdvancedValidationTests.java  # 14 testes
│       ├── RobustTests.java       # 36 testes (8 nested classes)
│       ├── MathOperationsTests.java  # 45 testes
│       └── ExtendedTests.java     # 46 testes
│
├── 📁 python/                      # Código Python (WGAN-GP)
│   ├── model.py                    # Arquitetura dos modelos (Generator, Critic)
│   ├── utils.py                    # Gradient penalty e utilitários
│   ├── junit_datasets.py           # Dataset loader e normalização
│   │
│   ├── 📁 scripts/                 # Scripts de treinamento e análise
│   │   ├── train_simple.py         # WGAN training (básico)
│   │   ├── train_phase2.py         # WGAN training (Phase 2)
│   │   ├── train_robust.py         # WGAN training robusto (final)
│   │   ├── clean_data.py           # Limpeza de dados CSV
│   │   ├── generate_phase2_tests.py # Geração de testes sintéticos
│   │   └── experimental_analysis.py # Análise Train vs Test metrics
│   │
│   ├── 📁 data/                    # Dados de entrada
│   │   ├── junit_data.csv          # Dados brutos dos testes
│   │   └── junit_data_cleaned.csv  # Dados limpos (202 amostras)
│   │
│   ├── 📁 models/                  # Modelos treinados (final)
│   │   ├── generator_final.pth     # Generator final (50 épocas)
│   │   └── critic_final.pth        # Critic final
│   │
│   └── 📁 outputs/                 # Resultados e outputs
│       ├── synthetic_tests.csv     # Testes sintéticos (Phase 2)
│       ├── synthetic_tests_robust.csv  # Testes sintéticos (final)
│       ├── training_history.csv    # Histórico de treino
│       ├── training_curves.png     # Gráficos de treino
│       └── experimental/           # Análise experimental
│           ├── train_metrics.csv   # Métricas de treino
│           ├── test_metrics.csv    # Métricas de teste
│           ├── analysis.json       # Relatório JSON
│           ├── train_vs_test.png   # Gráfico treino vs teste
│           ├── quality_metrics.png # Gráfico qualidade
│           └── generalization_gap.png  # Gráfico gap
│
├── 📁 .scripts/                    # Scripts auxiliares
│   ├── run_evosuite.bat            # Script EvoSuite (Windows)
│   ├── run_evosuite.ps1            # Script EvoSuite (PowerShell)
│   └── evosuite_wgan_pipeline.ps1  # Pipeline integrado
│
├── 📁 target/                      # Build output (Maven)
│   ├── classes/
│   ├── test-classes/
│   ├── generated-sources/
│   ├── generated-test-sources/
│   ├── surefire-reports/
│   └── maven-status/
│
├── 📁 .git/                        # Repositório Git
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .vscode/                     # VS Code settings
└── 📁 __pycache__/                 # Cache Python
```

## 🎯 Informações Importantes

### Testes JUnit (193 total)
- **ExampleTests**: 34 testes básicos
- **AdvancedTests**: 18 testes avançados
- **AdvancedValidationTests**: 14 testes de validação
- **ExtendedTests**: 46 testes de boundary cases
- **MathOperationsTests**: 45 testes dedicados
- **RobustTests**: 36 testes profissionais (8 categorias)

### Modelo WGAN-GP
- **Generator**: 158,979 parâmetros (100 → 256 → 512 → 3)
- **Critic**: 133,633 parâmetros (3 → 512 → 256 → 1)
- **Treinamento**: 50 épocas em 161 amostras
- **Batch size**: 32
- **Learning rate**: 0.0002
- **Lambda GP**: 10

### Resultados
- ✅ 193/193 testes JUnit: **PASSING**
- ✅ 500 testes sintéticos gerados
- ✅ Risco de overfitting: **BAIXO** (gap: 0.164)
- ✅ Qualidade sintética: **74.91%**

## 🚀 Como Usar

### 1. Setup Inicial
```bash
cd SoftwareTestingUsingWGANS
pip install -r requirements.txt
mvn clean install
```

### 2. Treinar o Modelo
```bash
python python/scripts/train_robust.py
```

### 3. Análise Experimental
```bash
python python/scripts/experimental_analysis.py
```

### 4. Executar Testes
```bash
mvn clean test
```

## 📊 Documentação
Consulte a pasta `docs/` para guias e relatórios detalhados.
