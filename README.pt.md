# Software Testing Using WGAN-GP

*English version: [README.md](README.md).*

Código, dados e resultados da dissertação
*Geração Inteligente de Dados de Teste de Software Utilizando Redes Adversariais Generativas (GANs)*
(Mestrado em Engenharia Informática, Universidade de Évora, 2026).

A pergunta do trabalho é se uma rede adversarial (WGAN-GP) consegue aprender
a distribuição das entradas usadas pelas *suites* de teste que um projeto já
tem, e produzir entradas novas que cubram mais código e detetem mais
defeitos. A resposta é medida por cobertura de linha e de ramo (JaCoCo) e por
*mutation score* (PIT), sendo esta a métrica principal porque a cobertura
satura antes, e contra uma escada de bases que inclui o que um profissional
teria sem treinar nada: reamostrar o *corpus* existente ou ajustar-lhe um
estimador de densidade clássico.

## As duas linhas do trabalho

O mesmo núcleo adversarial (`python/model.py`, `python/utils.py`) é usado em
dois pontos distintos do processo de teste:

1. **Perfis de cobertura → preditor de testabilidade (E1, E2, E4).** A rede
   sintetiza perfis estruturais de *suites* do *benchmark* SF110 (111
   projetos) para aumentar o treino de um preditor de testabilidade, validado
   *leave-projects-out*; o preditor guia a seleção de classes para o PIT e a
   curadoria do conjunto de mutadores. Validação externa no Defects4J.
2. **Entradas de teste → *suites* JUnit medidas por mutação (E3).** A rede
   gera valores de entrada concretos para uma classe sob teste; um oráculo de
   regressão fixa o resultado esperado como literal; a *suite* emitida é
   medida com JaCoCo e PIT contra `RANDOM`, `CORPUS` (reamostragem), `KDE` e
   as *suites* manual e EvoSuite. Dois alvos: `MathOperations` (12 dimensões)
   e `ArithmeticUtils` do Commons Math (2 dimensões).

## Resultados

### Linha 1 — SF110 (E2) e Defects4J (E4)

| Métrica | *Baseline* | `v8_curated` | *Merged* |
|---|---|---|---|
| Projetos com sucesso | 67 | 69 | **80** |
| *Weighted mutation score* | 16,73 % | **49,74 %** | 45,20 % |
| *Weighted test strength* | 63,25 % | **76,63 %** | 73,39 % |
| Projetos com *score* ≥ 70 % | 4 | 13 | 14 |

- Preditor de testabilidade: AUC *cross-project* **0,752** no SF110 (1 638
  classes, 70 projetos), contra 0,745 de uma regressão logística sobre as
  mesmas 15 *features*.
- Aumentação sintética do preditor: **não robusta** — +0,026 de AUC numa
  *seed*, +0,002 ± 0,013 em cinco. O que resolveu a escassez foi recolher mais
  dados reais (532 → 1 638 classes).
- Defects4J, sem retreino: AUC **0,360**. O sinal que o preditor aprende só
  existe em *benchmarks* estruturalmente heterogéneos.

### Linha 2 — geração de entradas de teste (E3)

Médias sobre 12 *seeds* (11 no segundo alvo), corpus completo, 114 mutações
no alvo 1 e 210 no alvo 2. Cobertura de linha e de ramo por JaCoCo, *mutation
score* por PIT:

| braço | `MathOperations` linha / ramo / *mut.* | `ArithmeticUtils` linha / ramo / *mut.* |
|---|---|---|
| `MANUAL` | 99,1 / 97,9 / 92,1 % | — |
| `CORPUS` | 90,8 / 90,3 / **82,2 %** | 65,3 / 68,5 / **63,6 %** |
| `WGAN` | 89,6 / 87,9 / 78,7 % | 56,9 / 60,1 / 57,3 % |
| `KDE` | 89,0 / 86,6 / 76,5 % | 63,4 / 66,7 / 62,3 % |
| `RANDOM` | 86,9 / 82,7 / 71,6 % | 50,0 / 50,0 / 47,6 % |
| `EVOSUITE` | 73,4 / 67,0 / 61,4 % | — |

- Sobre amostragem uniforme a rede ganha em *mutation score* **+7,02 pp**
  (12/12 *seeds*, Holm 0,0020) e **+9,70 pp** (11/11, Holm 0,0029), e em
  cobertura de ramo **+5,23 pp** e **+10,12 pp** (também unânimes e
  significativas). Contra a reamostragem do *corpus* completo, perde nas três
  métricas.
- **Regime de poucos dados**: com 18 entradas de treino a relação inverte-se
  nos dois alvos — a rede bate a reamostragem por +3,73 pp e +4,56 pp de
  *mutation score*. No alvo 1 a inversão dá-se sem diferença de cobertura
  (todos os braços do corpus entre 88 e 91 % de linha em todas as frações): a
  vantagem é de valores, não de alcance. No alvo 2 estende-se à cobertura de
  ramo (+4,17 pp, Holm 0,013). Sobre o KDE só ganha em 12 dimensões (o
  estimador fica mal condicionado); em 2 dimensões o KDE ganha à rede. A
  contribuição adversarial é condicional ao regime de dados e à
  dimensionalidade da codificação.
- **O limite é a codificação, não o gerador**: mudar só a normalização
  (escalonamento logarítmico com sinal) vale **+16,7 pp** no segundo alvo;
  prolongar o treino por um fator de sete vale 2,85 pp; condicionar o gerador
  vale +0,37 pp, sem significância.

Todas as medições que sustentam estes números estão em [`results/`](results/),
com o comando que reproduz cada tabela; os dados do preditor de
testabilidade estão em [`results/predictor/`](results/predictor/).

A *tag* [`tese-submetida`](https://github.com/GoncaloPires19/SoftwareTestingUsingWGANS/tree/tese-submetida)
marca o estado do repositório à data da submissão da dissertação (14 de
outubro de 2026). Os *commits* posteriores só acrescentam documentação e
ficheiros de dados; nenhuma medição mudou.

## Estrutura do repositório

```
.
├── README.md, LICENSE, NOTICE
├── pom.xml                      Maven: JUnit 5, JaCoCo, PIT 1.15.2, EvoSuite
├── requirements.txt             Dependências Python (PyTorch, scikit-learn, SciPy)
├── src/main/java/dataloader/    Classes sob teste: MathOperations, ArithmeticUtils (Apache 2.0, ver NOTICE)
├── src/test/java/
│   ├── dataloader/              Suites manual e EvoSuite; as suites geradas são reescritas por seed (gitignored)
│   ├── tools/                   OracleRunner / ArithmeticOracleRunner (fora do pacote mutado, de propósito)
│   └── corpus/                  Instrumentação em execução para extrair o corpus do segundo alvo
├── corpus_sources/              Cópias intactas da classe e suite originais do Commons Math (para diff)
├── python/
│   ├── config.py, model.py, utils.py, junit_datasets.py, sf110_lib.py
│   ├── data/                    Dataset de perfis de cobertura (733 registos brutos, 398 após limpeza estrita)
│   ├── scripts/                 Pipeline (ver abaixo); archive/ guarda scripts substituídos
│   └── tests/                   pytest
├── results/                     Medições finais de E3 (task5/, task5b/) e dados do preditor (predictor/)
├── docs/                        Metodologia e documentação das experiências
└── outputs/                     Execuções (gitignored, excepto os dois relatórios PDF)
```

Scripts principais de `python/scripts/`:

| linha | script | papel |
|---|---|---|
| 1 | `train_optimized.py`, `wgan_multi_seed.py` | treino da WGAN-GP sobre perfis; estudo de robustez (E1) |
| 1 | `extract_sf110_full.py`, `run_sf110_mutation_testing.py` | extração do SF110 e campanhas PIT |
| 1 | `build_bridge_dataset.py`, `train_testability_predictor.py`, `predictor_baselines.py` | preditor de testabilidade e escada de modelos |
| 1 | `wgan_augmented_predictor.py`, `fidelity_check.py`, `bootstrap_score_ci.py` | aumentação sintética, fidelidade *hold-out*, IC por *bootstrap* |
| 1 | `guided_class_selector.py`, `merge_runs.py`, `generate_results_pdf.py` | seleção guiada, fusão de campanhas, relatório |
| 1 | `run_defects4j_mutation_testing.py`, `run_defects4j_coverage.py`, `score_defects4j_with_sf110_model.py` | validação *cross-benchmark* (E4) |
| 2 | `extract_test_input_corpus.py` | *corpus* de entradas (literais; ou `--from-runtime-csv` para instrumentação) |
| 2 | `generate_test_inputs_wgan.py` | treino e geração; constrói os quatro braços (`WGAN`, `RANDOM`, `CORPUS`, `KDE`) |
| 2 | `emit_junit_from_oracle.py` | oráculo → *suite* JUnit 5 com asserções literais |
| 2 | `run_task5_experiment.py`, `task5b_resume.py` | uma medição completa por *seed* (JaCoCo + PIT); executor retomável |
| 2 | `analyze_task5_results.py`, `analyze_task5_ceiling.py`, `analyze_task5_conditional.py`, `analyze_task5_lowdata.py` | comparações emparelhadas com Holm; teto; ablações; poucos dados |

## Reprodução

### Ambiente

```bash
pip install -r requirements.txt
mvn clean test          # compila, corre as suites e o PIT sobre o caso controlado
```

Java 11 (`--release 11`), Maven 3, Python 3.11. Todas as experiências correram
em CPU.

### Linha 2 — uma medição de E3

```bash
# alvo 1, 12 seeds, 2000 épocas (~20 min por seed)
python python/scripts/run_task5_experiment.py --target mathops --base-seed 42 --seeds 12 --epochs 2000 \
    --out outputs/task5/results.csv

# alvo 2, executor retomável (sobrevive a reinícios; funde no fim)
python python/scripts/task5b_resume.py

# regime de poucos dados (todos os braços restritos ao mesmo subconjunto)
python python/scripts/run_task5_experiment.py --target mathops --corpus-frac 0.15 --base-seed 42 --seeds 12 --epochs 2000
```

Cada execução começa por `mvn clean`: o PIT tem de mutar o *bytecode* que a
*build* produz, e um servidor de linguagem Java aberto no editor pode
compilar para o mesmo diretório com outro JDK e alterar a população de
mutantes (ver `docs/TAREFA5_GAN_TEST_DATA.md`, §4.2). Os executores abortam
se a população mudar a meio.

As análises correm sobre `results/` sem repetir nenhuma medição — ver
[`results/README.md`](results/README.md).

### Linha 1 — campanha SF110

```bash
python python/scripts/extract_sf110_full.py
python python/scripts/run_sf110_mutation_testing.py \
    --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
    --test-mode both --output-root outputs/mutation_experiments/sf110_full111 \
    --pit-version 1.7.3 --retry-oom --retry-compile
python python/scripts/analyze_class_level.py --results-dir outputs/mutation_experiments/sf110_full111 \
    --export outputs/mutation_experiments/class_level_dataset_111.csv
python python/scripts/extract_class_metrics.py \
    --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
    --output outputs/mutation_experiments/class_structural_metrics_111.csv
python python/scripts/build_bridge_dataset.py
python python/scripts/train_testability_predictor.py --model rf \
    --model-out outputs/mutation_experiments/testability_model_111.json
python python/scripts/guided_class_selector.py --output-dir outputs/mutation_experiments/guided_targets_v3 \
    --drop-threshold 0.2 --preserve-coverage 0.3 --uncertain-threshold 0.5 --uncertain-coverage 0.1
python python/scripts/run_sf110_mutation_testing.py \
    --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
    --test-mode both --output-root outputs/mutation_experiments/sf110_v8_curated \
    --guided-targets-dir outputs/mutation_experiments/guided_targets_v3 \
    --mutators "CONDITIONALS_BOUNDARY,NEGATE_CONDITIONALS,MATH,INCREMENTS,INVERT_NEGS,RETURN_VALS,EMPTY_RETURNS,FALSE_RETURNS,TRUE_RETURNS,NULL_RETURNS,PRIMITIVE_RETURNS" \
    --pit-version 1.7.3 --retry-oom --retry-compile
```

A campanha completa sobre os 111 projetos demora horas e exige o SF110
extraído; o procedimento passo a passo está em
`docs/WGAN_TESTABILITY_METHODOLOGY.md`. O Defects4J exige um ambiente Unix
(correu em WSL2) — ver `docs/DEFECTS4J_CROSS_BENCHMARK.md`.

## Testes

```bash
mvn test                            # Java
python -m pytest python/tests/ -q   # Python
```

## Documentação

- [`docs/TAREFA5_GAN_TEST_DATA.md`](docs/TAREFA5_GAN_TEST_DATA.md) — desenho e resultados de E3, incluindo as duas armadilhas de integridade
- [`docs/WGAN_TESTABILITY_METHODOLOGY.md`](docs/WGAN_TESTABILITY_METHODOLOGY.md) — metodologia da linha 1
- [`docs/DEFECTS4J_CROSS_BENCHMARK.md`](docs/DEFECTS4J_CROSS_BENCHMARK.md) — validação externa e infraestrutura
- [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) — contrato dos datasets de perfis

Nota sobre os perfis de cobertura: as colunas `methods`, `branches` e
`statements` do dataset de perfis são tuplos numéricos derivados das *suites*,
interpretados segundo `docs/DATA_SCHEMA.md`, e não métricas formais de
cobertura do código de produção.

## Licença

MIT (ver `LICENSE`), excepto a classe `ArithmeticUtils` e as cópias em
`corpus_sources/`, que são do Apache Commons Math sob Apache License 2.0
(ver `NOTICE`).
