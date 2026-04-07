# Guia de Arranque Rapido

Ultima atualizacao: abril de 2026

## 1. Preparacao
```bash
pip install -r requirements.txt
```

## 2. Verificacao inicial dos dados
```bash
python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100
```

## 3. Pipeline completo
```bash
# 1) Gerar dados via testes Java
mvn clean test

# 2) Treinar modelo WGAN-GP
python python/scripts/train_basic_profiles.py --profile balanced --epochs 50 --data-path python/data/junit_data_cleaned_strict.csv

# 3) Gerar dados sinteticos
python python/scripts/generate_synthetic.py --num-samples 1000 --output outputs/synthetic_samples.csv

# 4) Avaliar resultados
python python/scripts/experimental_analysis.py \
  --training-history outputs/<run_name>/training_history.csv \
  --real-data python/data/junit_data_cleaned_strict.csv \
  --synthetic-data outputs/synthetic_samples.csv \
  --output-dir outputs/experimental/<run_name>
```

Notas:
- Substituir <run_name> pelo nome da pasta de run criada em outputs/.
- Confirmar parametros disponiveis com: python python/scripts/experimental_analysis.py --help

## 4. Mutation testing
```bash
python python/scripts/pit_experiment.py
```

Artefactos principais:
- outputs/mutation_experiments/mutation_results.csv
- outputs/mutation_experiments/MUTATION_REPORT.md
- outputs/mutation_experiments/with_without_evosuite_comparison.json

## 5. Documentacao relacionada
- README.md
- docs/ARCHITECTURE.md
- docs/DATA_SCHEMA.md
- docs/ESSENTIAL_5_FILES.md
- docs/ADVANCED_TRAINING_GUIDE.md
