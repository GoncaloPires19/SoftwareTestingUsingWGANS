# Software Testing Using WGAN-GP

## Visao geral

Pipeline hibrido Java + Python para avaliar e melhorar qualidade de testes de
software combinando tres contribuicoes:

1. **WGAN-GP** para geracao de perfis estruturais sinteticos e treino de um
   preditor de testabilidade.
2. **Selecao guiada de classes-alvo** para o PIT (mutation testing) baseada
   no modelo treinado em perfis WGAN-aware.
3. **Curadoria do conjunto de mutators** para eliminar mutadores conhecidamente
   fracos que introduzem ruido no score.

## Resultados finais

Sobre os 111 projetos do benchmark **SF110**:

| Metrica | Baseline | v8 curated | Merged final |
|---|---|---|---|
| Projetos com sucesso | 67 | 69 | **80** |
| Weighted mutation score | 16.73% | **49.74%** | 45.20% |
| Weighted test strength | 63.25% | **76.63%** | 73.39% |
| Projetos com score >= 70% | 4 | 13 | 14 |

O **objetivo de >70% em weighted test strength** foi atingido pela primeira
vez na campanha `v8_curated` (76.63%).

## Estrutura do repositorio

```
.
├── README.md
├── pom.xml                  Maven build (Java)
├── requirements.txt         Dependencias Python
├── src/                     MathOperations (caso controlado)
│   ├── main/java/dataloader/
│   └── test/java/dataloader/
├── python/
│   ├── config.py            Config centralizada do treino
│   ├── model.py             Generator e Critic WGAN-GP
│   ├── utils.py             Funcoes helper (gradient penalty, feature matching)
│   ├── sf110_lib.py         Modulo partilhado do pipeline SF110
│   ├── junit_datasets.py    Dataset PyTorch para JUnit metrics
│   ├── data/                Dataset canonico e SF110 extraido
│   ├── scripts/             17 scripts do pipeline canonico
│   │   ├── archive/         Experimentos anteriores (24 scripts arquivados)
│   │   └── tools/           Utilitarios
│   └── tests/               Testes pytest (64 testes)
├── docs/                    Documentacao academica
├── outputs/
│   ├── sf110_v8_curated_results.pdf   Relatorio v8
│   ├── sf110_wgan_results.pdf         Relatorio consolidado
│   └── mutation_experiments/          Datasets, modelo e results
└── target/                  Maven build output (gitignored)
```

## Documentacao

- **[docs/SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md)** - cronologia e estado final
- **[docs/SESSION_LOG.md](docs/SESSION_LOG.md)** - log detalhado
- **[docs/WGAN_TESTABILITY_METHODOLOGY.md](docs/WGAN_TESTABILITY_METHODOLOGY.md)** - metodologia academica
- **[docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)** - contrato dos datasets WGAN
- **[docs/sf110_v8_results.tex](docs/sf110_v8_results.tex)** - relatorio LaTeX final
- **[docs/mutation_testing_results.tex](docs/mutation_testing_results.tex)** - relatorio LaTeX caso `MathOperations`
- **[docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)** - registo da limpeza consolidada

## Pipeline reproduzivel

```bash
# 1. Ambiente
pip install -r requirements.txt

# 2. Testes Java + mutation testing no caso controlado
mvn clean test

# 3. Extracao SF110 completa (long-path safe no Windows)
python python/scripts/extract_sf110_full.py

# 4. Baseline (sem guidance)
python python/scripts/run_sf110_mutation_testing.py \
    --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
    --test-mode both --output-root outputs/mutation_experiments/sf110_full111 \
    --pit-version 1.7.3 --retry-oom --retry-compile

# 5. Analise por classe + dual metric
python python/scripts/analyze_class_level.py \
    --results-dir outputs/mutation_experiments/sf110_full111 \
    --export outputs/mutation_experiments/class_level_dataset_111.csv

# 6. Metricas estruturais das classes
python python/scripts/extract_class_metrics.py \
    --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
    --output outputs/mutation_experiments/class_structural_metrics_111.csv

# 7. Dataset de bridge (estrutura -> testabilidade)
python python/scripts/build_bridge_dataset.py

# 8. Preditor cross-project (RandomForest, GroupKFold, AUC ~ 0.75)
python python/scripts/train_testability_predictor.py --model rf \
    --model-out outputs/mutation_experiments/testability_model_111.json

# 9. Whitelist por projeto (selector v3: blacklist + incerteza)
python python/scripts/guided_class_selector.py \
    --output-dir outputs/mutation_experiments/guided_targets_v3 \
    --drop-threshold 0.2 --preserve-coverage 0.3 \
    --uncertain-threshold 0.5 --uncertain-coverage 0.1

# 10. Run v8_curated (selector + mutator set curated)
python python/scripts/run_sf110_mutation_testing.py \
    --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
    --test-mode both --output-root outputs/mutation_experiments/sf110_v8_curated \
    --guided-targets-dir outputs/mutation_experiments/guided_targets_v3 \
    --mutators "CONDITIONALS_BOUNDARY,NEGATE_CONDITIONALS,MATH,INCREMENTS,INVERT_NEGS,RETURN_VALS,EMPTY_RETURNS,FALSE_RETURNS,TRUE_RETURNS,NULL_RETURNS,PRIMITIVE_RETURNS" \
    --pit-version 1.7.3 --retry-oom --retry-compile

# 11. Merge best-per-project (baseline + v3 + v5_evo + v8)
python python/scripts/merge_runs.py \
    --runs outputs/mutation_experiments/sf110_full111 \
           outputs/mutation_experiments/sf110_wgan_guided_v3 \
           outputs/mutation_experiments/sf110_wgan_guided_v5_evo \
           outputs/mutation_experiments/sf110_v8_curated \
    --output outputs/mutation_experiments/sf110_final_v8

# 12. Gerar PDF
python python/scripts/generate_results_pdf.py \
    --baseline outputs/mutation_experiments/sf110_full111 \
    --guided outputs/mutation_experiments/sf110_final_v8 \
    --output outputs/sf110_wgan_results.pdf
```

## Testes

```bash
# Java (mvn)
mvn test              # 340 testes

# Python (pytest)
python -m pytest python/tests/ -q  # 64 testes
```

## Nota metodologica

As variaveis historicamente nomeadas como `methods`, `branches` e `statements`
sao mantidas por compatibilidade de pipeline, mas devem ser interpretadas
segundo o contrato de dados em [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)
como tuplos numericos derivados dos testes, nao como metricas formais de
cobertura de codigo de producao.
