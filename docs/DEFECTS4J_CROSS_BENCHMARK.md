# Validação Cross-Benchmark no Defects4J e Medição de Cobertura Real

> Agosto de 2026 — complementa [WGAN_TESTABILITY_METHODOLOGY.md](WGAN_TESTABILITY_METHODOLOGY.md)
> e o relatório LaTeX da campanha `v8_curated`.

## 1. Motivação

O preditor de testabilidade (`testability_model_111.pkl`) foi treinado e validado
**apenas dentro do SF110**, com *leave-projects-out* cross-validation (AUC = 0,752 em
1 638 classes de 70 projectos). Essa validação prova que o modelo não memoriza
projectos específicos — mas **não** prova que o sinal estrutural existe fora do SF110.

Para testar generalização a sério é preciso um benchmark genuinamente diferente. O
**Defects4J** (17 projectos maduros, bugs reais, suites de teste escritas por
programadores — não EvoSuite) é o candidato natural.

Objectivo: aplicar o modelo SF110 ao Defects4J **sem qualquer retreino** e medir se o
sinal se mantém.

---

## 2. Infraestrutura

O `defects4j` é Perl + Ant, exclusivo de Unix — não corre nativamente em Windows. Todo
o pipeline D4J corre em **WSL2**. O pipeline SF110, por contraste, sempre foi
Windows-nativo (Maven + PIT), e assim se manteve.

### 2.1 Problemas encontrados (e causas reais)

Os runs longos morriam repetidamente. Foram identificadas **três causas distintas**,
diagnosticadas por esta ordem:

| # | Causa | Diagnóstico | Solução |
|---|---|---|---|
| 1 | PC adormece (5 min AC / 3 min bateria) | `uptime` da VM a "0 min" após cada hiato | Keep-awake temporário (`SetThreadExecutionState`), reversível, sem alterar o *power plan* |
| 2 | `systemd` com `Linger=no` | `loginctl show-user` — o `systemd-logind` mata todos os processos do utilizador ao fechar a sessão de login, incluindo o servidor `tmux` | `sudo loginctl enable-linger pires` |
| 3 | **OOM kills** (causa dominante) | `dmesg`/`journalctl -k`: `Out of memory: Killed process (java)`, JVMs a 3,4–4,5 GB cada, VM com 7,4 GB total | `--workers 1` |

A causa 3 era a dominante: com `--workers 2`, duas JVMs do Major em simultâneo
esgotavam a memória, e o *OOM killer* arrastava consigo a *user slice* inteira —
o que explicava o desaparecimento total do `tmux`, e não apenas a falha de um bug.

**Configuração final estável:** `tmux` + `enable-linger` + `--workers 1` + keep-awake.

### 2.2 Bugs corrigidos em `run_defects4j_mutation_testing.py`

1. **`TypeError: can't concat str to bytes`** — em timeout, `subprocess.TimeoutExpired`
   podia trazer `stdout`/`stderr` como `bytes` mesmo com `text=True`.
2. **Timeout não era aplicado de facto** — `subprocess.run(timeout=...)` só matava o
   processo imediato; os filhos (ant, JVM) continuavam. Bugs do Closure corriam
   3 060 s com timeout de 900 s. Corrigido com `start_new_session=True` + `os.killpg`.
3. **Timeouts classificados como `compile_error`** — a verificação da substring
   `"compil"` corria **antes** da verificação de `code == 124`; um processo morto a
   meio da compilação tinha "Compiling..." no output residual. Reordenado.

---

## 3. Resultados: o preditor não generaliza

### 3.1 Dataset

`run_defects4j_mutation_testing.py` (Major, via `defects4j mutation`), scope
`--max-bugs-per-project 15`:

| | Valor |
|---|---|
| Bugs tentados | 246 |
| Classes extraídas | **249** |
| Projectos | **17 / 17** |
| Falhas | 36 (30 `compile_error`, 6 `timeout`) |
| Classes `testable = 1` | **242 (97,2 %)** |

> O Mockito, que falhava sistematicamente com `--workers 2`, passou a funcionar com
> `--workers 1` — as falhas de build eram artefacto do OOM, não do ambiente.

### 3.2 Modelo SF110 aplicado ao D4J (zero retreino)

`score_defects4j_with_sf110_model.py`:

| Métrica | SF110 (treino) | **Defects4J** |
|---|---|---|
| AUC cross-project | **0,752** | **0,360** |
| Accuracy | — | 0,197 |
| Precision | — | 0,957 |
| Recall | — | 0,182 |

O AUC de 0,360 é **pior que aleatório** (0,5). Com a amostra parcial de 168 classes o
valor era 0,399 — piorou com mais dados, o que exclui ruído amostral.

### 3.3 Alternativa: regressão contínua

Hipótese: talvez o problema fosse apenas o desequilíbrio do label binário (97 %
positivos). Testou-se regredir directamente um alvo contínuo
(`train_testability_regressor.py`, com `sf110_lib.load_bridge(..., target=...)`):

| Benchmark | Alvo | R² *out-of-fold* | Spearman |
|---|---|---|---|
| SF110 (1 638 classes, 70 proj.) | `mutation_score` | **+0,124** | **+0,341** |
| Defects4J (249 classes, 17 proj.) | `mutation_score` | **−0,371** | −0,094 |
| Defects4J (168 classes, 16 proj.) | `coverage` | −0,483 | −0,078 |

R² negativo significa **pior que prever sempre a média**. A hipótese fica refutada:
não era o formato do label — o sinal genuinamente não existe no D4J.

Confirma-o também a *feature importance*: no SF110 domina `has_io` (0,271), que separa
classes de rede/IO do resto; no D4J `has_io` é quase irrelevante (0,032) e dominam
métricas de tamanho (`loc`, `methods`).

### 3.4 Augmentação WGAN

`wgan_augmented_predictor.py --seeds 5` (WGAN retreinada no próprio D4J, por fold e
por classe, sempre só com projectos de treino):

| | Amostra parcial (168) | **Completo (249)** |
|---|---|---|
| AUC baseline | 0,490 ± 0,041 | **0,532 ± 0,023** |
| AUC WGAN-augmentado | 0,473 ± 0,032 | **0,505 ± 0,016** |
| Delta | −0,016 ± 0,041 | **−0,027 ± 0,027** |
| Seeds com ganho | 2 / 5 | **1 / 5** |
| Fidelidade (W1) | 0,086 | 0,067 |

Com mais dados o efeito tornou-se **consistentemente negativo** (desvio reduzido a
metade), não melhor. A fidelidade dos sintéticos é boa (W1 ≈ 0,067) — a WGAN gera
perfis realistas; simplesmente não há sinal de base para amplificar.

---

## 4. Cobertura real (line / branch / method)

O PIT **não reporta** *branch* nem *method coverage* — apenas *line coverage* e
*mutation coverage*. Foram criados dois harnesses para medir cobertura a sério.

### 4.1 SF110 — JaCoCo

`run_sf110_jacoco_coverage.py` reutiliza os projectos Maven **já construídos** pelo run
PIT (`sf110_full111/workdir/*/`, com `pom.xml` e `target/` intactos) e corre
`mvn jacoco:prepare-agent test jacoco:report`. Corre **nativamente em Windows** — sem
nenhum dos problemas de infraestrutura do WSL.

67 / 108 projectos passaram (coincide com o run PIT original), 9 259 classes.

### 4.2 Defects4J — Cobertura

`run_defects4j_coverage.py` corre `defects4j coverage` por bug e parseia o
`coverage.xml`. Muito mais leve que o mutation testing (executa a suite uma vez, sem
gerar mutantes): **completou 246/246 bugs com apenas 3 falhas**, sem qualquer
problema de OOM.

### 4.3 Comparação

| Métrica | SF110 (JaCoCo, 9 259 classes) | **Defects4J** (Cobertura, 300 classes) |
|---|---|---|
| Line coverage | 21,8 % | **82,2 %** |
| Branch coverage | 18,2 % | **75,3 %** |
| Method coverage | 23,4 % | **84,5 %** |

> A *line coverage* extraída dos relatórios PIT (`extract_pit_line_coverage.py`) dá
> 30,5 % global, superior aos 21,8 % do JaCoCo. Não é contradição: o PIT reporta só as
> *target classes* seleccionadas para mutação (subconjunto curado), enquanto o JaCoCo
> instrumenta todas as classes no *classpath*.

---

## 5. Conclusão

Três métodos independentes convergem na mesma conclusão:

1. Classificação binária: AUC 0,752 → **0,360**
2. Regressão contínua: R² +0,124 → **−0,371**
3. Augmentação WGAN: delta **−0,027 ± 0,027** (1/5 seeds)

**A causa é estrutural, não uma falha de implementação.** No Defects4J, cada "bug" é
por construção uma classe que já tem um teste de regressão real associado à correcção —
97 % das classes são testáveis, e a cobertura real confirma-o (82 % / 75 % / 84 %). A
heterogeneidade de que o preditor depende no SF110 — código de UI/rede inerentemente
não-testável *versus* código de lógica testável — **não existe** neste benchmark.

Este é um **resultado negativo válido**: delimita o domínio de aplicabilidade da
abordagem. O preditor é útil onde há classes genuinamente não-testáveis a filtrar
(SF110 + EvoSuite); é inútil onde já todas o são (Defects4J).

### Nota importante sobre comparações

Os números deste documento **não são comparáveis** aos da campanha `v8_curated`
(*score* 49,74 % / *strength* 76,63 %) do relatório LaTeX. São afirmações distintas:

- `v8_curated` é um resultado de **engenharia dentro do SF110** — selecção guiada +
  curadoria de mutators. Mantém-se válido e não é contrariado por nada aqui.
- Este documento mede **generalização externa** e **cobertura crua**, sem selecção
  guiada nem curadoria de mutators aplicadas.

*Mutation score* / *test strength* e *coverage* são métricas diferentes: a primeira
ignora, por construção, o código não coberto.

---

## 6. Reprodutibilidade

Todos os scripts aceitam `--resume` (checkpoint JSON) e `--max-bugs-per-project`.

```bash
# --- Defects4J (WSL obrigatório) ---
export PATH=$PATH:$HOME/tools/defects4j/framework/bin
cd "/mnt/c/.../SoftwareTestingUsingWGANS"

# Mutation testing (Major)   — lento: ~horas, exige --workers 1 (OOM)
tmux new-session -d -s d4j "python3 -u python/scripts/run_defects4j_mutation_testing.py \
    --workers 1 --max-bugs-per-project 15 --timeout-seconds 700 --resume"

# Coverage (Cobertura)       — rápido: ~2h para 246 bugs
python3 -u python/scripts/run_defects4j_coverage.py --workers 1 --max-bugs-per-project 15 --resume

# --- SF110 (Windows nativo) ---
python python/scripts/run_sf110_jacoco_coverage.py --workers 2 --resume
python python/scripts/extract_pit_line_coverage.py      # só parsing, sem re-run

# --- Análise ---
python python/scripts/score_defects4j_with_sf110_model.py
python python/scripts/train_testability_regressor.py \
    --bridge outputs/mutation_experiments/defects4j/bridge_dataset_d4j.csv --target mutation_score
python python/scripts/wgan_augmented_predictor.py \
    --bridge outputs/mutation_experiments/defects4j/bridge_dataset_d4j.csv --seeds 5
```

Pré-requisito WSL (uma vez): `sudo loginctl enable-linger $USER`.

### Datasets gerados

Em `outputs/mutation_experiments/` (não versionados — ver `.gitignore`):

| Ficheiro | Conteúdo |
|---|---|
| `defects4j/bridge_dataset_d4j.csv` | 249 classes, schema idêntico a `bridge_dataset_111.csv` |
| `defects4j/coverage_dataset_d4j.csv` | 300 classes, line/branch/method |
| `sf110_jacoco_coverage.csv` | 9 259 classes, line/branch/method |
| `sf110_line_coverage.csv` | 70 projectos, agregado dos relatórios PIT |
