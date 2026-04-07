# Analise Completa do Repositorio e Plano GitHub

## 1. Objetivo
Consolidar uma analise tecnica completa do estado atual do repositorio, definir claramente o que queremos e o que nao queremos manter no projeto, e converter isso num plano de execucao pronto para GitHub.

## 2. Snapshot Atual
- Estado do projeto: prototipo de investigacao avancado, com pipeline Java + Python funcional e resultados experimentais consistentes.
- Forca atual: existe infraestrutura de repeticao de experiencias com seeds, comparacao estatistica e historico de runs experimentais.
- Risco principal: drift entre documentacao, contratos declarados de dados, e comportamento real da implementacao.

## 3. Arquitetura e Fluxos
### 3.1 Fluxo Java
- Suites JUnit geram dados via logger para CSV.
- O pipeline de mutation testing usa PIT para medir eficacia real dos testes.

### 3.2 Fluxo Python
- Carregamento de datasets com normalizacao para intervalo [-1, 1].
- Treino WGAN-GP com perfis de treino e selecao de checkpoints.
- Geracao sintetica e analise experimental posterior.

### 3.3 Fronteira de Integracao
- O handoff Java -> Python funciona, mas ha ambiguidade semantica nas features registadas em alguns pontos.

## 4. Achados Principais (Resumo)
### Criticos
1. Semantica das features inconsistente entre o que a documentacao declara e o que parte dos testes realmente regista.

### Altos
1. Comandos de documentacao desatualizados face as CLIs atuais.
2. Quick Start referencia ficheiros/componentes que nao existem.
3. Inconsistencia no termo de gradient penalty entre scripts de treino.

### Medios
1. Threshold minimo de amostras desalinhado entre docs e validadores.
2. Contratos de output CSV variam entre artefactos historicos e defaults atuais.
3. Comparacao com/sem EvoSuite incompleta em pelo menos um artefacto agregado.

## 5. Matriz de Decisao
## 5.1 Queremos
- Repeticao de experiencias com seeds e analise estatistica robusta.
- Normalizacao/denormalizacao com stats persistidas e auditaveis.
- Selecao de checkpoints por metricas robustas e nao apenas por loss local.
- Mutation testing como gate de qualidade funcional (nao apenas cobertura de linhas).
- Documentacao executavel e alinhada com os scripts reais.
- Contrato unico e versionado para schema de dados e outputs.

## 5.2 Nao Queremos
- Semantica ambigua de features em dataset experimental.
- Docs com comandos quebrados ou referencias obsoletas.
- Duplicacao de sources de dados sem source of truth.
- Comparacoes incompletas em artefactos de avaliacao.
- Divergencia de formulas de treino entre scripts equivalentes.
- Crescimento de debito tecnico sem backlog priorizado.

## 6. Backlog GitHub Proposto
## 6.1 Issue P0 - Alinhar ontologia de features
- Titulo sugerido: P0 - Alinhar semantica de features Java/Python e regenerar datasets
- Labels: research-validity, data-contract, priority:P0
- Descricao:
  - Definir semanticamente cada feature (nome, unidade, origem).
  - Ajustar extracao Java e validacao Python para o mesmo contrato.
  - Regenerar datasets de referencia e rerun baseline.
- Definition of Done:
  - Documento de contrato atualizado.
  - Dataset regenerado com hash/version.
  - Resultado baseline repetido e anexado.

## 6.2 Issue P0 - Corrigir docs executaveis
- Titulo sugerido: P0 - Sincronizar README/ARCHITECTURE/QUICK_START com CLIs reais
- Labels: docs, reproducibility, priority:P0
- Descricao:
  - Rever comandos e argumentos obrigatorios.
  - Remover referencias a ficheiros inexistentes.
  - Validar todos os comandos num ambiente limpo.
- Definition of Done:
  - Checklist de comandos executados com sucesso.

## 6.3 PR P0 - Corrigir gradient penalty no treino otimizado
- Titulo sugerido: P0 - Fix formula de gradient penalty no train_optimized
- Labels: training, bugfix, priority:P0
- Checklist:
  - Corrigir formula para evitar escala duplicada de lambda.
  - Rodar smoke training de 5 epocas.
  - Comparar curvas com baseline e anexar evidencia.

## 6.4 Issue P1 - Unificar politica de min_samples
- Titulo sugerido: P1 - Unificar thresholds de min_samples por perfil
- Labels: validation, config, priority:P1
- Definition of Done:
  - Valor(es) canonicos definidos por perfil e refletidos em docs + validadores.

## 6.5 Issue P1 - Versionar schema de artefactos
- Titulo sugerido: P1 - Versionar schema de CSV sintetico e validar carregamento
- Labels: data-contract, tooling, priority:P1
- Definition of Done:
  - Schema versionado.
  - Validacoes com erro explicito para formatos invalidos.

## 6.6 Issue P1 - Fechar comparacao EvoSuite
- Titulo sugerido: P1 - Reexecutar comparacao com/sem EvoSuite com desenho pareado
- Labels: mutation-testing, evosuite, priority:P1
- Definition of Done:
  - JSON final com ambos os modos, metadados de execucao e delta interpretavel.

## 6.7 Issue P2 - Reduzir debito de qualidade estatica
- Titulo sugerido: P2 - Refatoracao incremental scripts core
- Labels: refactor, maintainability, priority:P2
- Definition of Done:
  - Reducao mensuravel de problemas criticos de complexidade.

## 7. Plano de Execucao em PRs
1. PR-1 (P0 docs): sincronizar documentacao operacional.
2. PR-2 (P0 features): contrato de features + extracao/validacao + dataset baseline.
3. PR-3 (P0 training fix): ajuste gradient penalty + smoke comparativo.
4. PR-4 (P1 data contracts): schema versioning + validadores.
5. PR-5 (P1 mutation study): comparacao EvoSuite pareada final.
6. PR-6 (P2 quality): refatoracao incremental orientada a risco.

## 8. O que adicionar agora ao GitHub
- Este documento em docs.
- Criacao das issues P0/P1/P2 acima com labels.
- Abertura imediata da PR-1 para docs executaveis.

## 9. Validacao Final Recomendada
1. mvn clean test
2. python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100
3. python python/scripts/train_basic_profiles.py --profile balanced --epochs 5 --data-path python/data/junit_data_cleaned_strict.csv
4. python python/scripts/generate_synthetic.py --num-samples 1000 --output outputs/smoke_synthetic.csv
5. python python/scripts/experimental_analysis.py --training-history <run>/training_history.csv --real-data python/data/junit_data_cleaned_strict.csv --synthetic-data outputs/smoke_synthetic.csv --output-dir outputs/experimental/smoke

---
Documento preparado para servir como base de backlog e governance tecnica no GitHub.