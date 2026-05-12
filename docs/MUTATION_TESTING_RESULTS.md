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

Para avaliar o pipeline fora do caso controlado, foram executadas campanhas PIT sobre projetos do benchmark SF110.

| Run | Test Mode | Projects Passed / Executed | Weighted Mutation Score (%) | Avg. Project Score (%) | Killed / Relevant | Observações |
|-----|-----------|----------------------------|-----------------------------|------------------------|-------------------|-------------|
| sf110_full_evosuite_20260426 | evosuite | 3 / 5 | **10.38** | 19.82 | 3485 / 33566 | Execução inicial em projetos grandes; falhas em `102_squirrel-sql` e `104_vuze` |
| sf110_full_fix1 | evosuite | 13 / 14 | **20.64** | 23.06 | 1429 / 6924 | Rerun após correções do runner; falha isolada em `8_gfarcegestionfa` |
| sf110_both_smoke_retest_20260510_jdk21 | both | 1 / 1 | **40.52** | 40.52 | 393 / 970 | Caso focal em `13_jdbacl`, com **83.97** de test strength |

Leituras principais:

- O ganho de `10.38%` para `20.64%` entre os cenários `evosuite` sugere que a estabilização do runner teve impacto material nos resultados observados.
- O caso focal `13_jdbacl` mostra que, quando a execução PIT é estável e a combinação de suites funciona, o score pode subir para `40.52%` com `83.97%` de test strength.
- Em SF110, os resultados devem ser sempre acompanhados pelo número de projetos efetivamente concluídos, porque falhas de build, classpath e geração de `mutations.csv` distorcem fortemente a agregação.

## 4. Casos representativos em SF110

Top cases:

- `2_a4j`: `54.35%` de mutation score em `sf110_full_fix1`.
- `13_jdbacl`: `40.52%` em `sf110_both_smoke_retest_20260510_jdk21`.
- `101_netweaver`: `37.60%` em `sf110_full_evosuite_20260426`.

Worst cases:

- `9_falselight`: `1.45%` em `sf110_full_fix1`.
- `103_sweethome3d`: `3.70%` em `sf110_full_evosuite_20260426`.
- `11_imsmart`: `6.67%` em `sf110_full_fix1`.

## 5. Conclusões para a tese

Os resultados sustentam duas conclusões centrais:

1. No caso controlado `MathOperations`, o mutation score melhora significativamente quando os testes são reforçados e os pontos de mutação equivalente são reduzidos, passando do melhor snapshot agregado (`94.12%`) para `97.26%` no rerun focado.
2. No benchmark SF110, o mutation score reflete não só a qualidade dos testes, mas também a robustez do pipeline experimental. Em campanhas multi-projeto, a taxa de sucesso da execução precisa ser reportada em conjunto com o score agregado.

Para escrita académica, os números mais defensáveis são:

- `94.12%`: snapshot agregado histórico em `MathOperations` com `ALL_COMBINED`.
- `97.26%`: rerun focado pós-melhoria em `MathOperations`.
- `20.64%`: weighted mutation score do rerun `sf110_full_fix1` com `13/14` projetos concluídos.
- `40.52%`: caso focal `13_jdbacl` em `both`, com `83.97%` de test strength.

## 6. Fontes dos dados

- `outputs/mutation_experiments/mutation_results.csv`
- `outputs/mutation_experiments/mathoperations_focused_after_fixes_20260510_summary.json`
- `target/pit-reports/index.html`
- `outputs/mutation_experiments/sf110_full_evosuite_20260426/sf110_mutation_summary.json`
- `outputs/mutation_experiments/sf110_full_fix1/sf110_mutation_summary.json`
- `outputs/mutation_experiments/sf110_both_smoke_retest_20260510_jdk21/sf110_mutation_summary.json`