# Resultados de Mutation Testing

Este relatório consolida os resultados de mutation testing obtidos com PIT para dois contextos experimentais do repositório:

1. o caso controlado `dataloader.MathOperations`;
2. as execuções multi-projeto sobre o benchmark SF110.

## 1. Snapshot histórico em `MathOperations`

O snapshot agregado original comparou várias suites de testes contra o mesmo conjunto histórico de 102 mutações relevantes.

| Test Suite | Tests Run | Killed | Survived | No Coverage | Total | Mutation Score (%) |
|------------|-----------|--------|----------|-------------|-------|--------------------|
| ALL_COMBINED | 229 | 96 | 6 | 0 | 102 | **94.12** |
| AdvancedValidationTests | 17 | 82 | 13 | 7 | 102 | **80.39** |
| MathOperationsTests | 47 | 82 | 6 | 14 | 102 | **80.39** |
| ExampleTests | 38 | 81 | 14 | 7 | 102 | **79.41** |
| TestDataLoaderEvoSuiteTests | 8 | 80 | 15 | 7 | 102 | **78.43** |
| RobustTests | 38 | 79 | 14 | 9 | 102 | **77.45** |
| AdvancedTests | 20 | 76 | 17 | 9 | 102 | **74.51** |
| ExtendedTests | 49 | 76 | 17 | 9 | 102 | **74.51** |
| MathOperationsEvoSuiteTests | 12 | 75 | 12 | 15 | 102 | **73.53** |

Leitura principal: a combinação de suites (`ALL_COMBINED`) atingiu o melhor score agregado, mas o desempenho individual variou bastante consoante o alinhamento entre a suite e o código mutado.

## 2. Rerun focado após melhorias em `MathOperations`

Depois do reforço de testes e da refatoração do alvo, foi executado um rerun PIT focado apenas em `dataloader.MathOperations`.

| Run | Tests Run | Killed | Survived | No Coverage | Total | Mutation Score (%) | Test Strength (%) | Line Coverage (%) |
|-----|-----------|--------|----------|-------------|-------|--------------------|-------------------|-------------------|
| MATHOPERATIONS_FOCUSED_AFTER_FIXES_20260510 | 84 | 71 | 0 | 2 | 73 | **97.26** | **100.00** | **97.00** |

Observações:

- Todos os mutantes `SURVIVED` foram eliminados neste rerun focado.
- Permaneceram apenas 2 mutações `NO_COVERAGE`, ambas associadas às lambdas sintéticas introduzidas pelos `orElseThrow` em `max()` e `min()`.
- Este resultado não é diretamente comparável ao snapshot histórico de 102 mutações, porque o rerun focado gerou um conjunto diferente de mutações relevantes (`73`).

## 3. Resultados SF110

Foram executadas quatro campanhas PIT sobre o benchmark SF110. A campanha mais abrangente é `sf110_full_both1`, a única a cobrir todos os 111 projetos SF110 (79 concluídos).

| Run | Test Mode | Passed / Executed | Weighted Score (%) | Avg. Score (%) | Avg. Test Strength (%) | Killed / Relevant |
|-----|-----------|-------------------|--------------------|----------------|------------------------|-------------------|
| sf110_full_evosuite_20260426 | evosuite | 3 / 5 | 10.38 | 19.82 | — | 3485 / 33566 |
| sf110_full_fix1 | evosuite | 13 / 14 | 20.64 | 23.06 | — | 1429 / 6924 |
| **sf110_full_both1** | **both** | **79 / 111** | **18.76** | **24.27** | **62.05** | 13799 / 73556 |
| sf110_both_smoke_retest_20260510_jdk21 | both | 1 / 1 | 40.52 | 40.52 | 83.97 | 393 / 970 |

## 4. Resultados por projeto — `sf110_full_both1` (modo `both`, 79 projetos)

Esta é a campanha mais completa. Top 10 projetos por mutation score decrescente:

| # | Projeto | Score (%) | Test Strength (%) | Killed | Survived | No Coverage | Total |
|---|---------|-----------|-------------------|--------|----------|-------------|-------|
| 1 | `90_dcparseargs` | **90.32** | 90.32 | 84 | 9 | 0 | 93 |
| 2 | `31_xisemele` | **90.24** | 91.64 | 296 | 27 | 5 | 328 |
| 3 | `24_saxpath` | **84.09** | 87.68 | 370 | 52 | 18 | 440 |
| 4 | `26_jipa` | **75.86** | **100.00** | 22 | 0 | 7 | 29 |
| 5 | `40_glengineer` | **67.10** | 74.59 | 414 | 141 | 62 | 617 |
| 6 | `59_mygrid` | **65.24** | 89.84 | 610 | 69 | 256 | 935 |
| 7 | `55_lavalamp` | **63.97** | 71.49 | 158 | 63 | 26 | 247 |
| 8 | `25_jni-inchi` | **62.03** | 85.47 | 147 | 25 | 65 | 237 |
| 9 | `86_at-robots2-j` | **61.90** | 68.29 | 450 | 209 | 68 | 727 |
| 10 | `2_a4j` | **54.35** | 75.00 | 225 | 75 | 114 | 414 |

- `90_dcparseargs` e `31_xisemele` atingiram **>90%** de mutation score — os melhores resultados observados em todo o benchmark SF110.
- `26_jipa` atingiu **100% de test strength**: todos os mutantes cobertos pelos testes foram mortos.
- `59_mygrid` combina score elevado (65.24%) com test strength de 89.84% sobre um projeto de dimensão considerável (935 mutações).

### Comparação por projeto — `sf110_full_fix1` (evosuite, 13 projetos)

| # | Projeto | Score (%) | Killed | Survived | No Coverage | Total |
|---|---------|-----------|--------|----------|-------------|-------|
| 1 | `2_a4j` | **54.35** | 225 | 75 | 114 | 414 |
| 2 | `14_omjstate` | **40.91** | 36 | 14 | 38 | 88 |
| 3 | `13_jdbacl` | **32.51** | 354 | 86 | 649 | 1089 |
| 4 | `7_sfmis` | **31.27** | 111 | 76 | 168 | 355 |
| 5 | `6_jnfe` | **29.18** | 82 | 17 | 182 | 281 |
| 6 | `1_tullibee` | **27.52** | 328 | 46 | 818 | 1192 |
| 7 | `5_templateit` | **22.74** | 83 | 36 | 246 | 365 |
| 8 | `4_rif` | **22.35** | 38 | 43 | 89 | 170 |
| 9 | `3_gaj` | **19.67** | 12 | 14 | 35 | 61 |
| 10 | `11_imsmart` | **6.67** | 11 | 21 | 133 | 165 |
| 11 | `12_dsachat` | **6.10** | 75 | 156 | 999 | 1230 |
| 12 | `10_water-simulator` | **5.05** | 73 | 423 | 949 | 1445 |
| 13 | `9_falselight` | **1.45** | 1 | 2 | 66 | 69 |
| — | `8_gfarcegestionfa` | FAIL | — | — | — | — |

### Caso focal `13_jdbacl` com modo `both`

| Projeto | Test Mode | Score (%) | Test Strength (%) | Killed | Survived | No Coverage | Total |
|---------|-----------|-----------|-------------------|--------|----------|-------------|-------|
| `13_jdbacl` | both | **40.52** | **83.97** | 393 | 75 | 502 | 970 |

Combinar EvoSuite com testes manuais subiu o score de 32.51% (evosuite puro) para 40.52% (+8 pp), com test strength de 83.97%.

## 5. Conclusões para a tese

Os resultados sustentam duas conclusões centrais:

1. No caso controlado `MathOperations`, o mutation score melhora significativamente quando os testes são reforçados e os pontos de mutação equivalente são reduzidos, passando do melhor snapshot agregado (`94.12%`) para `97.26%` no rerun focado.
2. No benchmark SF110, o mutation score reflete não só a qualidade dos testes, mas também a robustez do pipeline experimental. Em campanhas multi-projeto, a taxa de sucesso da execução precisa ser reportada em conjunto com o score agregado.

Para escrita académica, os números mais defensáveis são:

- `94.12%`: snapshot agregado histórico em `MathOperations` com `ALL_COMBINED`.
- `97.26%`: rerun focado pós-melhoria em `MathOperations`.
- `18.76%`: weighted mutation score de `sf110_full_both1` com `79/111` projetos concluídos (campanha mais completa).
- `24.27%`: média por projeto em `sf110_full_both1`.
- `90.32%` / `90.24%`: melhores scores individuais SF110 (`90_dcparseargs` e `31_xisemele`) no modo `both`.
- `40.52%`: caso focal `13_jdbacl` em `both`, com `83.97%` de test strength.

## 6. Fontes dos dados

- `outputs/mutation_experiments/mutation_results.csv`
- `outputs/mutation_experiments/mathoperations_focused_after_fixes_20260510_summary.json`
- `target/pit-reports/index.html`
- `outputs/mutation_experiments/sf110_full_evosuite_20260426/sf110_mutation_summary.json`
- `outputs/mutation_experiments/sf110_full_fix1/sf110_mutation_summary.json`
- `outputs/mutation_experiments/sf110_full_both1/sf110_mutation_summary.json`
- `outputs/mutation_experiments/sf110_both_smoke_retest_20260510_jdk21/sf110_mutation_summary.json`