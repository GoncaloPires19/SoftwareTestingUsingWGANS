# Software Testing Using WGAN-GP

[![CI](https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS/actions/workflows/ci-smoke.yml/badge.svg)](https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS/actions/workflows/ci-smoke.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Java 11+](https://img.shields.io/badge/java-11%2B-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*Versão portuguesa: [README.pt.md](README.pt.md).*

Code, data and measurements of an MSc dissertation (Universidade de Évora, 2026):
*Geração Inteligente de Dados de Teste de Software Utilizando Redes Adversariais Generativas (GANs)*
(*Intelligent Generation of Software Test Data Using Generative Adversarial Networks*).

Can a generative adversarial network learn the distribution of the inputs
that a project's existing unit tests use, and generate new inputs that detect
more faults? We answer with mutation testing (PIT) and coverage (JaCoCo), and
against a ladder of increasingly demanding baselines that include what a
practitioner would have without training anything: resampling the existing
inputs, or fitting a classical density estimate to them.

**In short.** One WGAN-GP core is used in two roles. Synthesizing *coverage
profiles* to augment a testability predictor does not help; collecting more
real data does. Synthesizing the *inputs* that tests execute does help: the
generated inputs beat uniform sampling in every seed of both targets, and beat
resampling of the corpus when training inputs are scarce. What limits the
gains is the input encoding rather than the generator.

## The two roles

```
                    ┌───────────────────────┐
 coverage profiles ─▶      WGAN-GP core     ├─▶ synthetic profiles ─▶ testability predictor
 (SF110 classes)    │ critic + grad. penalty│
 input corpus      ─▶                       ├─▶ inputs ─▶ oracle on original code ─▶ JUnit + PIT
 (existing tests)   └───────────────────────┘
```

1. **Coverage profiles → testability predictor.** The network synthesizes
   structural profiles of SF110 classes (111 projects) to augment a random
   forest that predicts, from 15 structural features, whether generated tests
   will reach a useful mutation score. Validated across projects
   (`GroupKFold`), and transferred to Defects4J without retraining.
2. **Test inputs → JUnit suites measured by mutation.** The network generates
   concrete input values for a class under test; a regression oracle runs each
   input once on the original code and writes the result as a literal
   expected value; the emitted suite is measured with JaCoCo and PIT against
   `RANDOM`, `CORPUS` (resampling) and `KDE`, with the human-written and
   EvoSuite suites as reference points. Two targets: `MathOperations`
   (12-dimensional encoding, 114 mutants) and `ArithmeticUtils` from Apache
   Commons Math 3.6.1 (2-dimensional encoding, 210 mutants).

## Main results

### Role 1: testability predictor

| setting | AUC |
|---|---|
| SF110, 1 638 classes from 70 projects, real data only | **0.752** |
| + synthetic profiles, five seeds (on an earlier 27-project extraction) | +0.002 ± 0.013 |
| Defects4J, 249 classes, no retraining | 0.360 |

The generator learns the profiles faithfully (W1 = 0.056, no mode collapse),
but a description of a test does not teach the predictor anything the real
data had not. Growing the dataset from 532 to 1 638 real classes raised AUC
from 0.63 to 0.752, about sixty times the mean synthetic gain.

### Role 2: generated test inputs

Means over 12 seeds (11 on the second target), full corpus. Line / branch
coverage (JaCoCo) and mutation score (PIT), in %:

| variant | `MathOperations` line / branch / mut. | `ArithmeticUtils` line / branch / mut. |
|---|---|---|
| `MANUAL` | 99.1 / 97.9 / 92.1 | — |
| `CORPUS` | 90.8 / 90.3 / **82.2** | 65.3 / 68.5 / **63.6** |
| `WGAN` | 89.6 / 87.9 / 78.7 | 56.9 / 60.1 / 57.3 |
| `KDE` | 89.0 / 86.6 / 76.5 | 63.4 / 66.7 / 62.3 |
| `RANDOM` | 86.9 / 82.7 / 71.6 | 50.0 / 50.0 / 47.6 |
| `EVOSUITE` | 73.4 / 67.0 / 61.4 | — |

- **Over uniform sampling**, the WGAN-GP raises the mutation score by
  **+7.02 pp** (12/12 seeds, Holm-corrected p = 0.0020) and **+9.70 pp**
  (11/11, p = 0.0029).
- **With the full corpus**, resampling the corpus is better on both targets.
- **With scarce data the ranking inverts.** With 18 training inputs, the
  WGAN-GP beats resampling by +3.73 pp and +4.56 pp. Resampling 18 inputs
  yields 17 distinct values in 200 draws; the generator yields 176, close to
  the real distribution. Against KDE it wins only with the 12-dimensional
  encoding; in two dimensions, 18 points are enough for a stable density
  estimate and KDE wins.
- **The limit is the encoding, not the generator.** On the first target the
  WGAN-GP and resampling share the same ceiling (94 of 114 mutants in every
  mutator class); the gap is one of consistency. On the second, changing only
  the input normalization (signed logarithm) moved the score by **+16.7 pp**,
  against +2.85 pp for seven times longer training and +0.37 pp (not
  significant) for conditioning the generator.

## Where each result comes from

Every number above is recomputed from versioned measurements, without
repeating any experiment:

| result | data | command |
|---|---|---|
| Role 1: predictor AUC, model ladder | `results/predictor/` | see [`results/predictor/README.md`](results/predictor/README.md) |
| Role 2, target 1: ladder, coverage, paired tests | `results/task5/results_114_e2000.csv` | `python python/scripts/analyze_task5_results.py --results results/task5/results_114_e2000.csv` |
| Role 2, target 1: ceiling and consistency | `results/task5/results_114_e2000_by_mutator.csv` | `python python/scripts/analyze_task5_ceiling.py --by-mutator results/task5/results_114_e2000_by_mutator.csv` |
| Role 2, target 1: low-data regime | `results/task5/results_114_frac*.csv` | `python python/scripts/analyze_task5_lowdata.py --full results/task5/results_114_e2000.csv --fracs "0.15:results/task5/results_114_frac15.csv,0.30:results/task5/results_114_frac30_full.csv,0.60:results/task5/results_114_frac60.csv"` |
| Role 2, target 1: training length, conditioning | `results/task5/results_114_e300_full.csv`, `results_114_cond.csv` | `python python/scripts/analyze_task5_conditional.py` (see [`results/README.md`](results/README.md)) |
| Role 2, target 2: ladder and low-data regime | `results/task5b/results_arith_final.csv`, `results/task5b/frac069/` | `python python/scripts/analyze_task5_results.py --results results/task5b/results_arith_final.csv` |
| Role 2, target 2: effect of the encoding | `results/task5b/gate_linear.csv`, `gate_log.csv` | direct reading (one paired run) |

The full index, with the dissertation table each file supports, is in
[`results/README.md`](results/README.md).

## Repository layout

```
.
├── pom.xml                      Maven: JUnit 5, JaCoCo, PIT 1.15.2, EvoSuite
├── requirements.txt             Python dependencies (PyTorch, scikit-learn, SciPy)
├── src/main/java/dataloader/    Classes under test: MathOperations, ArithmeticUtils (Apache 2.0, see NOTICE)
├── src/test/java/
│   ├── dataloader/              Human-written and EvoSuite suites; generated suites are rewritten per seed (git-ignored)
│   ├── tools/                   Oracle runners, outside the mutated package on purpose
│   └── corpus/                  Runtime instrumentation that records the second target's corpus
├── corpus_sources/              Untouched copies of the original Commons Math class and suite (for diff)
├── python/
│   ├── model.py, utils.py       WGAN-GP generator, critic and training loop
│   ├── config.py, junit_datasets.py, sf110_lib.py
│   ├── scripts/                 Pipelines and analyses (below); archive/ keeps superseded scripts
│   └── tests/                   pytest
├── results/                     Final measurements, indexed by table
└── docs/                        Methodology and experiment notes (Portuguese)
```

Main scripts in `python/scripts/`:

| role | script | purpose |
|---|---|---|
| 1 | `extract_sf110_full.py`, `run_sf110_mutation_testing.py`, `extract_class_metrics.py` | SF110 extraction, PIT campaigns, structural features |
| 1 | `build_bridge_dataset.py`, `train_testability_predictor.py`, `predictor_baselines.py` | predictor dataset, predictor, model ladder |
| 1 | `wgan_augmented_predictor.py`, `fidelity_check.py` | synthetic augmentation per fold, hold-out fidelity |
| 1 | `run_defects4j_mutation_testing.py`, `score_defects4j_with_sf110_model.py` | cross-benchmark validation |
| 2 | `extract_test_input_corpus.py` | input corpus (from literals, or `--from-runtime-csv` for instrumentation) |
| 2 | `generate_test_inputs_wgan.py` | training and generation; builds the four variants |
| 2 | `emit_junit_from_oracle.py` | oracle → JUnit 5 suite with literal assertions |
| 2 | `run_task5_experiment.py`, `task5b_resume.py` | one full measurement per seed (JaCoCo + PIT); resumable runner |
| 2 | `analyze_task5_results.py`, `analyze_task5_ceiling.py`, `analyze_task5_conditional.py`, `analyze_task5_lowdata.py` | paired comparisons with Holm; ceiling; ablations; low-data regime |

## Reproducing

```bash
pip install -r requirements.txt
mvn clean test                       # compiles, runs the suites and PIT on the controlled case
python -m pytest python/tests/ -q    # Python tests
```

Java 11 or later, Maven 3, Python 3.11. All experiments ran on CPU.

One measurement of role 2 (training, oracle, suite emission, JaCoCo and PIT)
takes about 20 minutes per seed on a laptop:

```bash
# target 1, 12 seeds, 2000 epochs
python python/scripts/run_task5_experiment.py --target mathops --base-seed 42 --seeds 12 --epochs 2000

# target 2, resumable runner (survives restarts, merges at the end)
python python/scripts/task5b_resume.py

# low-data regime: every variant restricted to the same subset
python python/scripts/run_task5_experiment.py --target mathops --corpus-frac 0.15 --base-seed 42 --seeds 12 --epochs 2000
```

Every run starts with `mvn clean` and aborts if the number of mutants changes:
an IDE language server compiling into `target/` with a different JDK once
changed the mutant population from 114 to 118 (see
[`docs/TAREFA5_GAN_TEST_DATA.md`](docs/TAREFA5_GAN_TEST_DATA.md), §4.2).

Role 1 from scratch requires the SF110 distribution and takes hours; the step
by step procedure is in
[`docs/WGAN_TESTABILITY_METHODOLOGY.md`](docs/WGAN_TESTABILITY_METHODOLOGY.md).
Defects4J requires a Unix environment (it ran on WSL2), see
[`docs/DEFECTS4J_CROSS_BENCHMARK.md`](docs/DEFECTS4J_CROSS_BENCHMARK.md).

## Dissertation version

The tag [`tese-submetida`](https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS/tree/tese-submetida) marks the state of the
repository when the dissertation was submitted (14 October 2026). Later
commits only add documentation and data files; no measurement changed.

## License

MIT (see [`LICENSE`](LICENSE)), except `ArithmeticUtils` and the copies in
`corpus_sources/`, which come from Apache Commons Math under the Apache
License 2.0 (see [`NOTICE`](NOTICE)).
