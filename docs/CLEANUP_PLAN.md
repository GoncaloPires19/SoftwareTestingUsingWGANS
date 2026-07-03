# Plano de Limpeza Consolidada do Repositório

Este documento inventaria TUDO o que está no repositório e propõe uma
limpeza estruturada. A ideia é ter um repositório limpo, coerente com o
estado actual do trabalho (pipeline v8, resultado final consolidado) e sem
dumps de sessões antigas.

## Secção 1 - Ficheiros na raiz

### 1.1 A ELIMINAR

Estes markdowns são dumps de sessões antigas de refactoring, redundantes
com a documentação actual em `docs/`:

- `FINAL_OPTIMIZATION_REPORT.md`
- `IMPLEMENTATION_REPORT.md`
- `OPTIMIZATION_STRUCTURE.md`
- `POST_IMPLEMENTATION_CHECKLIST.md`
- `QUICK_START_AFTER_OPTIMIZATION.md`
- `REFACTORING_LOG.md`
- `validate_refactoring.py` (script de validação do refactor pré-actual)
- `run_all.bat` (não reflecte o pipeline v8)

Directorias vazias na raiz (não deviam existir):

- `sf110_opt_weak_1780748769/`
- `sf110_opt_weak_1780754299/`
- `sf110_opt_weak_1780754317/`
- `sf110_opt_weak_1780754426/`

### 1.2 A MANTER

- `README.md` (74 linhas, ainda útil, precisa de update)
- `.gitignore`
- `pom.xml` (Maven config)
- `requirements.txt`
- `src/` (Java: MathOperations, testes)
- `target/` (Maven build output, gitignored)

## Secção 2 - `docs/`

### 2.1 A ELIMINAR

- `MUTATION_TESTING_RESULTS.md` (superado por `SESSION_SUMMARY.md` e resultados v8)
- `MUTATION_TESTING_RESULTS.pdf` (idem)
- `REPO_COMPLETE_ANALYSIS_GITHUB_PLAN.md` (documento de análise antigo)
- `ADVANCED_TRAINING_GUIDE.md` (guia de treino WGAN antigo, não alinhado com v8)
- `QUICK_START.md` (superseded por SESSION_SUMMARY §5)
- `ESSENTIAL_5_FILES.md` (superseded pelo pipeline canónico de 17 scripts)
- `ARCHITECTURE.md` (desactualizado: fala em treino WGAN sem mencionar
  o pipeline de mutation testing)
- `CLEANUP_PLAN.md` VERSÃO ANTIGA (este documento substitui)

### 2.2 A MANTER

- `README.md` (a criar/reescrever para reflectir estado actual)
- `SESSION_SUMMARY.md` (cronologia final com resultados v8)
- `SESSION_LOG.md` (log detalhado da sessão)
- `WGAN_TESTABILITY_METHODOLOGY.md` (metodologia académica)
- `DATA_SCHEMA.md` (contrato de dados WGAN)
- `mutation_testing_results.tex` (LaTeX do caso MathOperations, histórico)
- `sf110_v8_results.tex` (LaTeX final v8)
- `CLEANUP_PLAN.md` (este documento, após aprovação vira registo)

## Secção 3 - `outputs/`

### 3.1 A ELIMINAR

Runs de treino WGAN antigas (39 MB cada):

- `balanced_20260601_200846/` (vazio, 0 bytes)
- `balanced_20260601_201119/` (39 MB, run de treino antiga)
- `balanced_20260601_201234/` (39 MB, run de treino antiga)
- `test_melhorias/` (15 MB, smoke test)
- `test_v2_melhorias/` (12 MB, smoke test)
- `training_run.log`
- `sf110_extraction.log`, `sf110_extraction2.log`, `sf110_extraction3.log`
  (logs de extração antiga)
- `synthetic_samples.csv`, `synthetic_samples_1000.csv`,
  `synthetic_7features_20260607_110940.csv` (amostras geradas em runs antigas)
- `models/anti_collapse/`, `models/ensemble_anti_collapse/` (modelos WGAN antigos
  não usados no pipeline actual)
- `COMPARISON_TABLE.md`, `FINAL_RESULTS_TABLE.md` (superados por SESSION_SUMMARY
  e pelos PDFs)

### 3.2 A MANTER

- `sf110_v8_curated_results.pdf` (relatório principal)
- `sf110_wgan_results.pdf` (relatório consolidado)
- `mutation_experiments/` (dados essenciais do pipeline v8, ver Secção 4)

## Secção 4 - `outputs/mutation_experiments/` (já limpo antes)

Estado actual (13 items após a primeira limpeza + `sf110_v8_curated` novo):

### 4.1 A MANTER

- `sf110_full111/` - baseline (67 PASS, 16.73%)
- `sf110_wgan_guided_v3/` - selecção guiada melhor (69 PASS, 38.55%)
- `sf110_wgan_guided_v5_evo/` - cobertura máxima (80 PASS)
- `sf110_v8_curated/` - final (69 PASS, 49.74%, strength 76.63%)
- `sf110_final_v8/` - merged consolidado (80 PASS, 45.20%)
- `bridge_dataset_111.csv`
- `class_level_dataset_111.csv`
- `class_structural_metrics_111.csv`
- `guided_targets_v3/` (whitelist do preditor)
- `testability_model_111.json`, `testability_model_111.pkl`
- `sf110_full111_run.log`

### 4.2 A ELIMINAR

- `sf110_best_merged_final/` (substituído por `sf110_final_v8/`)
- `sf110_v8_curated_run.log` (log temporário, pode ir para archive se preciso)
- `sf110_dd_guided_run/` (dir bloqueada anteriormente, tentar novamente)

## Secção 5 - `python/`

### 5.1 A ELIMINAR

- `python/models/` (vazio)
- `python/scripts/outputs/` (vazio, artefacto de run antigo)
- `python/scripts/tools/monitor_training.py` (não usado)
- `python/data/junit_data_cleaned_v2.csv`, `python/data/junit_data_cleaned_v2_norm_stats.json`
  (superseded por `_strict`)
- `python/data/exp_b_high_quality.csv`, `python/data/exp_c_low_redundancy.csv`
  (datasets experimentais não usados no pipeline final)
- `python/data/junit_data.csv` (raw, superseded por `_cleaned_strict`)
- `python/data/junit_data_cleaned_norm_stats.json` (não usado)
- `python/data/archive/` (arquivo antigo)
- `python/data/processed/` sub-files experimentais não usados no pipeline v8

### 5.2 A MANTER

**Scripts (17, já limpos):**
- `run_sf110_mutation_testing.py`, `extract_sf110_full.py`,
  `extract_class_metrics.py`, `analyze_class_level.py`,
  `build_bridge_dataset.py`, `train_testability_predictor.py`,
  `guided_class_selector.py`, `merge_runs.py`, `analyze_sf110_results.py`,
  `compare_baseline_vs_guided.py`, `generate_results_pdf.py`,
  `diagnose_wgan.py`, `train_optimized.py`, `generate_synthetic.py`,
  `validate_before_training.py`, `pit_experiment.py`,
  `prepare_sf110_dataset.py`
- `python/scripts/archive/` (24 scripts arquivados anteriormente)
- `python/scripts/tools/cleanup.py` (utilitário)

**Módulos:**
- `config.py`, `model.py`, `utils.py`, `sf110_lib.py`, `junit_datasets.py`
- `data_validation.py`, `evaluation.py`, `metrics.py`
- `training_profiles_advanced.py` (usado por train_optimized)
- `test_system.py`

**Testes:**
- `python/tests/test_modules.py`, `python/tests/test_sf110_lib.py`

**Dados:**
- `python/data/junit_data_cleaned_strict.csv` (dataset canónico)
- `python/data/junit_data_cleaned_strict_norm_stats.json`
- `python/data/raw/evosuite/sf110/` (SF110 extraído, gitignored via LFS)

## Secção 6 - Directorias transversais

### 6.1 A MANTER

- `data/` (paralelo a python/data, com processed/ raw/) - VERIFICAR se é usado.
  Se vazio ou duplicado com python/data, ELIMINAR.
- `logs/` (vazio na raiz, ELIMINAR se não for usado)
- `models/` (vazio na raiz, ELIMINAR)

## Impacto estimado

Antes:
- Root: 14 items (8 MDs, 4 dirs vazias, 6 súbidos)
- docs/: 14 items
- outputs/: 18 items (105 MB fora de mutation_experiments)
- python/scripts/: 17 essenciais + archive

Depois:
- Root: 5 items (README, .gitignore, pom.xml, requirements.txt, run_all.bat opcional)
- docs/: 8 items focados
- outputs/: 3 items (2 PDFs + mutation_experiments/)
- python/: mesmos scripts, sem dumps de data

Espaço recuperado: ~105 MB (balanced runs + test dirs + logs antigos).

## Passos após aprovação

1. Executar limpeza fase 1 (raiz)
2. Executar limpeza fase 2 (docs/)
3. Executar limpeza fase 3 (outputs/ não-mutation)
4. Executar limpeza fase 4 (python/data cleanup)
5. Verificar tudo (testes Python 64/64, scripts import OK)
6. Atualizar docs finais:
   - `README.md` na raiz (nova versão, refletir v8)
   - `SESSION_SUMMARY.md` (adicionar v8 resultados finais)
   - `WGAN_TESTABILITY_METHODOLOGY.md` (adicionar secção v8)
7. Este `CLEANUP_PLAN.md` vira registo histórico do que foi limpo
