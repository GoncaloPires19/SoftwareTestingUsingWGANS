# Testability predictor data (line 1: SF110 and Defects4J)

The datasets behind the testability predictor and its cross-benchmark
validation. Each row is one Java class: the 15 structural features extracted
from its source, and the outcome of mutation testing its EvoSuite-generated
suite with PIT.

| file | content |
|---|---|
| `bridge_dataset_111.csv` | Predictor dataset: 1 638 SF110 classes from 70 projects, structural features joined with PIT outcomes; `testable` is the label |
| `class_level_dataset_111.csv` | PIT outcome per SF110 class (mutations, coverage, mutation score, test strength) |
| `bridge_dataset_d4j.csv` | The same features and outcomes for 249 Defects4J classes (17 projects), mutated with Major |
| `coverage_dataset_d4j.csv` | JaCoCo line, branch and method coverage of the Defects4J classes |
| `predictor_baselines.json` | Cross-project AUC of the model ladder: majority, decision stump, logistic regression, random forest |
| `testability_model_111.json` | Summary of the random forest trained on all SF110 classes |
| `bootstrap_score_ci.json` | Bootstrap 95 % intervals for the SF110 campaign scores (baseline vs curated mutators) |

## Reproducing the numbers

```bash
# Cross-project AUC on SF110 (GroupKFold, k = 5): 0.752
python python/scripts/train_testability_predictor.py --model rf \
    --bridge results/predictor/bridge_dataset_111.csv \
    --model-out outputs/mutation_experiments/testability_model_111.json

# Transfer to Defects4J without retraining: AUC 0.360
python python/scripts/score_defects4j_with_sf110_model.py \
    --bridge results/predictor/bridge_dataset_d4j.csv \
    --model outputs/mutation_experiments/testability_model_111.pkl

# Model ladder (majority, stump, logistic regression, random forest)
python python/scripts/predictor_baselines.py --bridge results/predictor/bridge_dataset_111.csv

# WGAN-GP augmentation of the predictor, per fold and per class label
python python/scripts/wgan_augmented_predictor.py \
    --bridge results/predictor/bridge_dataset_111.csv --seeds 5
```

The first two commands were re-run from these files before publication and
reproduce 0.752 and 0.360. The early augmentation result on 27 projects
(AUC 0.622 → 0.648 with one seed, +0.002 ± 0.013 over five) used a smaller,
earlier extraction of 532 classes that was superseded by the full dataset and
is not included; its procedure is described in
[`docs/WGAN_TESTABILITY_METHODOLOGY.md`](../../docs/WGAN_TESTABILITY_METHODOLOGY.md).

Producing these files from scratch requires the SF110 distribution and,
for Defects4J, a Unix environment; see
[`docs/DEFECTS4J_CROSS_BENCHMARK.md`](../../docs/DEFECTS4J_CROSS_BENCHMARK.md).
