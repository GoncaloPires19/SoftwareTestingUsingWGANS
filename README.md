# Software Testing Using WGAN-GP

## Visao Geral
Este repositorio implementa um pipeline hibrido Java + Python para geracao de dados sinteticos de testes com WGAN-GP. O objetivo central e melhorar a robustez do processo de teste com base em analise empirica reprodutivel.

## Estado Atual (abril 2026)
- Suite Java ativa: 229 metodos de teste anotados com @Test no pacote dataloader.
- Mutation testing agregado: 94.12% (PIT, alvo dataloader.*).
- Dataset principal para treino: python/data/junit_data_cleaned_strict.csv com 398 linhas.
- Experimentos adicionais: exp_b_high_quality.csv com 79 linhas e exp_c_low_redundancy.csv com 398 linhas.

## Estrutura do Projeto
- src/main/java: codigo Java de suporte.
- src/test/java: suites JUnit e instrumentacao de logging.
- python: modelos WGAN-GP, carga de dados, treino, geracao e avaliacao.
- docs: documentacao tecnica em portugues academico.
- outputs: artefactos de treino, avaliacao e mutation testing.

## Pipeline Recomendado

### 1) Preparacao do ambiente
```bash
pip install -r requirements.txt
```

### 2) Geracao de dados via testes Java
```bash
mvn clean test
```

### 3) Validacao de dados antes do treino
```bash
python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100
```

### 4) Treino WGAN-GP
```bash
python python/scripts/train_basic_profiles.py --profile balanced --epochs 50 --data-path python/data/junit_data_cleaned_strict.csv
```

### 5) Geracao sintetica
```bash
python python/scripts/generate_synthetic.py --num-samples 1000 --output outputs/synthetic_samples.csv
```

### 6) Analise experimental
```bash
python python/scripts/experimental_analysis.py \
  --training-history outputs/<run_name>/training_history.csv \
  --real-data python/data/junit_data_cleaned_strict.csv \
  --synthetic-data outputs/synthetic_samples.csv \
  --output-dir outputs/experimental/<run_name>
```

## Mutation Testing
Para executar avaliacao PIT por suite e agregado:
```bash
python python/scripts/pit_experiment.py
```

Artefactos principais:
- outputs/mutation_experiments/mutation_results.csv
- outputs/mutation_experiments/MUTATION_REPORT.md
- outputs/mutation_experiments/with_without_evosuite_comparison.json

## Documentacao Principal
- docs/QUICK_START.md
- docs/ARCHITECTURE.md
- docs/ESSENTIAL_5_FILES.md
- docs/ADVANCED_TRAINING_GUIDE.md
- docs/DATA_SCHEMA.md

## Nota Metodologica
As variaveis historicamente nomeadas como methods, branches e statements sao mantidas por compatibilidade de pipeline, mas devem ser interpretadas segundo o contrato de dados vigente em docs/DATA_SCHEMA.md.
