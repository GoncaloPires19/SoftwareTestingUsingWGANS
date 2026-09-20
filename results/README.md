# Resultados medidos (E3: geração de dados de teste)

Os ficheiros desta pasta são as medições que sustentam o Capítulo 6 da
dissertação. São produto das campanhas descritas em
`docs/TAREFA5_GAN_TEST_DATA.md`; cada linha é uma medição (braço, *seed*)
com cobertura JaCoCo, contagens do PIT (`killed`, `survived`, `no_coverage`,
`timed_out`) e métricas de geração (`w1`, `n_unique`, `nn_dist_median`,
`n_exact_copies`). Os ficheiros `*_by_mutator.csv` desagregam as mortes por
classe de mutador, para a análise por teto.

Todas as medições do primeiro alvo foram feitas contra 114 mutações e as do
segundo contra 210; os analisadores abortam se encontrarem populações
diferentes no mesmo ficheiro.

## Primeiro alvo — `dataloader.MathOperations` (`task5/`)

| tabela / figura da tese | ficheiro | comando |
|---|---|---|
| 6.10 resumo, 6.11 escada, 6.12 alcance | `results_114_e2000.csv` | `python python/scripts/analyze_task5_results.py --results results/task5/results_114_e2000.csv` |
| 6.13 teto por mutador, 6.14 consistência | `results_114_e2000_by_mutator.csv` | `python python/scripts/analyze_task5_ceiling.py --by-mutator results/task5/results_114_e2000_by_mutator.csv` |
| 6.15 duração do treino (300 vs 2000 épocas) | `results_114_e300_full.csv` + `results_114_e2000.csv` | `python python/scripts/analyze_task5_conditional.py --baseline results/task5/results_114_e300_full.csv --conditional results/task5/results_114_e2000.csv --label "300 vs 2000 epocas"` |
| 6.16 condicionamento em `arr_len` | `results_114_cond.csv` | `python python/scripts/analyze_task5_conditional.py --baseline results/task5/results_114_e2000.csv --conditional results/task5/results_114_cond.csv --label "condicionamento"` |
| 6.17–6.20 e Fig. 6.1–6.2 regime de poucos dados | `results_114_frac15.csv`, `results_114_frac30_full.csv`, `results_114_frac60.csv` | `python python/scripts/analyze_task5_lowdata.py --full results/task5/results_114_e2000.csv --fracs "0.15:results/task5/results_114_frac15.csv,0.30:results/task5/results_114_frac30_full.csv,0.60:results/task5/results_114_frac60.csv"` |
| varrimento de orçamento (não reportado) | `budget_sweep.csv` | — |
| *corpus* de treino (121 entradas) | `input_corpus.csv` | regenerável com `extract_test_input_corpus.py` |

Os braços `MANUAL` e `EVOSUITE` estão em `results_114_e2000.csv` com
`seed = -`, porque não variam com a *seed*.

Nota sobre os testes estatísticos: o Wilcoxon corre sobre os valores tal
como estão no CSV (fracções com quatro casas). Multiplicar por 100 antes do
teste altera o último bit de diferenças iguais, desfaz empates de ordens e
muda o *p* exacto — foi um defeito real de uma versão anterior de
`analyze_task5_results.py`, e é a razão de os três analisadores usarem a
mesma convenção.

## Segundo alvo — `ArithmeticUtils` do Commons Math (`task5b/`)

| tabela / figura da tese | ficheiro | comando |
|---|---|---|
| 6.21 efeito da codificação | `gate_linear.csv`, `gate_log.csv` | leitura directa (uma *seed*, dois escalonamentos) |
| 6.22 escada, 6.23 Holm | `results_arith_final.csv` (11 *seeds*) | `python python/scripts/analyze_task5_results.py --results results/task5b/results_arith_final.csv` |
| 6.24 teto por mutador, Fig. 6.4 | `seeds/seed_*_by_mutator.csv` | `python python/scripts/analyze_task5_ceiling.py` sobre a concatenação dos ficheiros por *seed* |
| 6.25–6.27 regime de poucos dados (18 entradas) | `frac069/results_arith_final.csv` (12 *seeds*) | `python python/scripts/analyze_task5_results.py --results results/task5b/frac069/results_arith_final.csv` |
| *corpus* de treino (260 entradas) | `input_corpus.csv`, `corpus_raw.csv` | **não regenerável bit a bit**: a *suite* original usa `RandomDataGenerator`; o *corpus* é versionado como artefacto |

As pastas `seeds/` guardam a medição de cada *seed* tal como foi escrita
pelo executor retomável (`task5b_resume.py`), antes da fusão.

## Reprodução completa

Repetir uma medição do zero (treino da rede, oráculo, emissão da *suite*
JUnit, JaCoCo e PIT) demora cerca de 20 minutos por *seed* num portátil, e
exige `mvn clean` antes de cada execução — ver `docs/TAREFA5_GAN_TEST_DATA.md`,
secção 4.2, para a razão.

```
python python/scripts/run_task5_experiment.py --target mathops --base-seed 42 --seeds 12 --epochs 2000
python python/scripts/run_task5_experiment.py --target arith   --base-seed 42 --seeds 12 --epochs 2000
```
