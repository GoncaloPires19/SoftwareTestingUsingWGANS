# Tabela Comparativa: Baseline vs WGAN v3

Gerado automaticamente em 2026-05-09.

## Configurações

| Configuração | Baseline | WGAN v3 (melhorado) |
|---|---|---|
| Profile | `optimized` | `improved` |
| Epochs | 50 | 200 |
| Batch size | 32 | 16 |
| Learning rate | 5e-5 | 3e-5 |
| Lambda GP | 10.0 | 8.0 |
| N_critic | 10 | 7 |
| Log1p transform | ❌ Não | ✅ Sim |
| Feature matching loss | ❌ Não | ✅ Sim (weight=0.5) |
| He/Kaiming init | ❌ Não | ✅ Sim |
| Dataset | SF110 + H2 (504 amostras) | SF110 + H2 (504 amostras) |

## Métricas de Qualidade (best epoch)

| Métrica | Baseline | WGAN v3 | Melhoria |
|---|---|---|---|
| `max_statements` (sintético) | 12.5 | 645.4 | **+5063%** |
| `std_distance_z` statements | 0.604 | 0.235 | **-61%** (mais próximo do real) |
| `robust_v2` (best epoch) | 56.43 | **80.55** (ep.57) | **+42.8%** |
| `coverage_max` (best epoch) | — | 63.7% (ep.57) / 94.7% (ep.6) | — |
| `cov_p99` (best robust_v2 epoch) | 39.67% | 38.88% | ≈ igual |
| `cov_p95` (best robust_v2 epoch) | — | 20.0% | — |
| Overall Quality Score (experimental_analysis) | — | **88.6%** ("Excellent") | — |

## Análise por Feature (experimental_analysis.py — WGAN v3)

| Feature | Coverage | Distribution Match | Mean Match | Feature Score |
|---|---|---|---|---|
| methods | 100% | 90.9% | 77.4% | **89.6%** |
| branches | 100% | 81.2% | 89.8% | **89.4%** |
| statements | 100% | 72.9% | 92.4% | **86.9%** |

## Suite Java (mvn clean test)

| Classe | Testes | Falhas | Erros |
|---|---|---|---|
| AdvancedTests | 20 | 0 | 0 |
| AdvancedValidationTests | 17 | 0 | 0 |
| ExampleTests | 38 | 0 | 0 |
| ExtendedTests | 49 | 0 | 0 |
| MathOperationsEvoSuiteTests | 12 | 0 | 0 |
| MathOperationsTests | 47 | 0 | 0 |
| RobustTests (subtests) | 38 | 0 | 0 |
| TestDataLoaderEvoSuiteTests | 8 | 0 | 0 |
| **TOTAL** | **229** | **0** | **0** |

Score de mutação (PIT): **94.12%** (registado em MUTATION_REPORT.md)

## SF110 Mutation Testing

| Estado | Detalhes |
|---|---|
| Projetos processados | 28/111 |
| Checkpoint | `outputs/mutation_experiments/sf110_full_both1/checkpoint.json` |
| Run a decorrer | Terminal 31a29f84 |

## Ficheiros Gerados

| Ficheiro | Descrição |
|---|---|
| `outputs/wgan_v3_improved_20260509_183926/synthetic_samples_best.csv` | 1000 amostras sintéticas (best robust_v2, ep.57) |
| `outputs/wgan_v3_improved_20260509_183926/training_history.csv` | Histórico completo 200 epochs |
| `outputs/experimental/wgan_v3_vs_baseline/experimental_analysis_report.json` | Análise detalhada |
| `outputs/experimental/wgan_v3_vs_baseline/training_analysis.png` | Visualização curvas de treino |

## Conclusão

O WGAN v3 com **log1p transform + feature matching loss + He/Kaiming init** resolve o problema central diagnosticado:

> O baseline sem log1p produzia `max_statements=12.5` (1.25% do real máx=1000) devido a skewness=7.0 que comprimia 90% dos valores próximos de -1 após normalização min-max.

Com v3: `max_statements=645.4` (64.5% do máximo real), `std_distance_z=0.235` (vs 0.604), `robust_v2=80.55` (vs 56.43), qualidade global **88.6%**.
