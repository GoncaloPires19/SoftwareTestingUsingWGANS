# Tarefa 5 — Impacto dos Dados de Teste Gerados pela WGAN-GP

> Agosto de 2026 — responde ao **Objetivo 3** do plano de dissertação:
> *"Investigar o impacto dos conjuntos de testes gerados pela GAN na cobertura
> do código e deteção de bugs"*.

## 1. Enquadramento e mudança face ao trabalho anterior

Até aqui, a WGAN do projeto gerava **métricas agregadas** de testes (`methods`,
`branches`, `statements`). Essas métricas descrevem testes, mas não *são* testes:
não há caminho causal entre gerar um vetor de métricas e matar um mutante. Por
isso nenhuma das experiências anteriores conseguia atribuir ganhos de mutation
score à WGAN-GP (ver [DEFECTS4J_CROSS_BENCHMARK.md](DEFECTS4J_CROSS_BENCHMARK.md)).

A Tarefa 5 fecha essa lacuna: a WGAN passa a gerar **inputs de teste concretos**,
que são compilados em testes JUnit executáveis e medidos. É o que o plano pedia
desde o início ("gerar dados de teste ... eliminando a necessidade de criar dados
de teste de forma manual").

## 2. Desenho experimental

### 2.1 Pipeline

| Passo | Script | O que faz |
|---|---|---|
| 1 | `extract_test_input_corpus.py` | Extrai inputs dos testes JUnit existentes (233 call sites → **121 inputs únicos**) |
| 2 | `generate_test_inputs_wgan.py` | WGAN-GP treina no corpus; gera inputs sintéticos + baseline aleatório |
| 3 | `src/test/java/tools/OracleRunner.java` | Corre a implementação **original** e regista o resultado (oráculo de regressão) |
| 4 | `emit_junit_from_oracle.py` | Emite JUnit 5 com o valor esperado como **literal** |
| 5 | `run_task5_experiment.py` | Mede JaCoCo + PIT nos 4 arms × N seeds |
| 6 | `analyze_task5_results.py` | Wilcoxon emparelhado + deltas por seed |

### 2.2 Codificação dos inputs

Vetor de 12 dimensões, para que um único gerador sirva todas as famílias de
assinatura (descodificado por família a jusante):

```
[ arr_len, v0..v7, s0, s1, s2 ]
```

`arr_len = -1` codifica um array `null`. Famílias: `ARRAY`, `ARRAY_INT`, `INT1`,
`INT2`, `INT3`. O método `isPalindrome(String)` fica fora do âmbito (não numérico).

### 2.3 A escada de baselines

| Arm | Origem dos testes |
|---|---|
| `MANUAL` | Suites escritas à mão já no repositório |
| `EVOSUITE` | Suite derivada do EvoSuite já no repositório |
| `RANDOM` | Inputs uniformes **no mesmo intervalo por dimensão** do corpus real |
| `CORPUS` | Reamostragem com reposição do próprio corpus de treino |
| `KDE` | Densidade por kernels gaussianos, bandwidth por validação cruzada |
| `WGAN` | Inputs gerados pela WGAN-GP |

O arm `RANDOM` é o que dá validade científica à comparação. Tem o mesmo número de
testes, a mesma codificação, o mesmo oráculo e os mesmos intervalos de valores
que o `WGAN` — a **única** diferença é a origem dos inputs. Qualquer vantagem da
WGAN vem, portanto, da distribuição aprendida, e não de amostrar um intervalo mais
favorável.

Os outros dois arms gerados existem porque vencer só o `RANDOM` diria pouco:

- **`CORPUS`** responde à objeção cética direta: bastaria reutilizar os 121 inputs
  humanos de que a rede foi treinada? É o adversário mais duro precisamente por
  não gerar nada de novo. O seu teto são 121 inputs distintos, por maior que seja
  o orçamento.
- **`KDE`** responde à outra: é preciso uma rede adversarial, ou um modelo de
  densidade clássico com os mesmos dados chega ao mesmo sítio? A bandwidth é
  escolhida por validação cruzada e não fixada — dar ao rival um parâmetro
  arbitrário viciaria a comparação a favor da WGAN, que recebe milhares de
  épocas de ajuste.

### 2.4 Dois detalhes que determinam a validade

**O valor esperado tem de ser um literal.** Um teste da forma
`assertEquals(MathOperations.sum(x), MathOperations.sum(x))` sofreria a mutação
dos dois lados e nunca falharia — mataria zero mutantes. Por isso o passo 3 corre
a implementação original e "cozinha" o resultado como constante em tempo de
geração.

**O oráculo não pode ser mutado.** O `OracleRunner` está deliberadamente no
pacote `tools` (fontes de teste), e não em `dataloader`: a configuração do PIT
muta `dataloader.*`, pelo que colocá-lo lá poluiria o mutation score.

## 3. Resultados (12 seeds, 2000 épocas, 114 mutações)

Classe-alvo: `dataloader.MathOperations`. 200 inputs por arm → 2 800 testes.
**114 mutações por arm** — ver secção 4 sobre porque este número importa.

| Arm | Linha | Ramo | Método | **Mutation score** | Test strength |
|---|---|---|---|---|---|
| `MANUAL` | 99,1 % | 97,9 % | 100,0 % | **92,1 %** | 92,1 % |
| `CORPUS` | 90,8 ± 0,3 % | 90,3 ± 0,3 % | 93,3 % | **82,2 ± 0,5 %** | 91,1 ± 0,4 % |
| **`WGAN`** | 89,6 ± 0,7 % | 87,9 ± 1,5 % | 93,3 % | **78,7 ± 2,7 %** | 88,7 ± 2,3 % |
| `KDE` | 89,0 ± 0,7 % | 86,6 ± 0,8 % | 93,3 % | 76,5 ± 1,8 % | 87,0 ± 2,1 % |
| `RANDOM` | 86,9 ± 1,6 % | 82,7 ± 1,9 % | 93,3 % | 71,6 ± 2,3 % | 85,4 ± 1,9 % |
| `EVOSUITE` | 73,4 % | 67,0 % | 73,3 % | 61,4 % | 83,3 % |

> A cobertura de método está saturada por construção: dos 15 métodos, 14 aceitam
> inteiros/arrays e um aceita `String`. A codificação de 12 dimensões não
> representa strings, pelo que todos os arms gerados ficam presos em
> 14/15 = 93,3 %. A métrica não discrimina neste desenho.

### 3.1 Comparações emparelhadas (p corrigido por Holm-Bonferroni)

| Contra | Métrica | Delta | Seeds a favor | p corrigido |
|---|---|---|---|---|
| `RANDOM` | **Mutation score** | **+7,02 pp** | **12/12** | **0,0020** |
| `RANDOM` | **Cobertura de ramo** | **+5,23 pp** | **12/12** | **0,0020** |
| `RANDOM` | **Cobertura de linha** | **+2,68 pp** | **11/12** | **0,0020** |
| `RANDOM` | **Test strength** | **+3,25 pp** | **9/12** | **0,0093** |
| `KDE` | Mutation score | +2,19 pp | 8/12 | 0,1289 (n.s.) |
| `KDE` | Cobertura de ramo | +1,33 pp | 8/12 | 0,1289 (n.s.) |
| `CORPUS` | Mutation score | **−3,59 pp** | 1/12 | **0,0039** |
| `CORPUS` | Cobertura de ramo | **−2,40 pp** | 0/12 | **0,0039** |

Contra amostragem uniforme a vantagem é inequívoca. Contra o KDE tem o sinal
certo mas não resiste à correção, o que **delimita a contribuição adversarial
com precisão**: neste domínio é medível contra amostragem uniforme e
indistinguível de um modelo de densidade sem adversário.

Contra o CORPUS a WGAN perde — e a secção 3.3 mostra que essa derrota não
significa o que aparenta.

### 3.2 Onde está a vantagem: alcance e (agora) asserções

| Arm | Mortas | Sobreviventes | **Sem cobertura** |
|---|---|---|---|
| `MANUAL` | 105,0 | 9,0 | **0,0** |
| `CORPUS` | 93,8 | 9,2 | 11,1 |
| `WGAN` | 89,7 | 11,4 | **12,9** |
| `KDE` | 87,2 | 13,1 | 13,8 |
| `RANDOM` | 81,7 | 13,9 | 18,4 |
| `EVOSUITE` | 70,0 | 14,0 | 30,0 |

Das 8,0 mutações extra que a WGAN mata face ao RANDOM, **5,5 vêm de alcance** e
**2,5 de qualidade das asserções**. A 300 épocas esta segunda parcela era 1,2 e
não atingia significância; com 2000 passa a atingi-la (test strength p = 0,0093).
Corrige a conclusão anterior deste documento, que afirmava que a vantagem *não*
estava nas asserções.

O corpus de treino vem de testes escritos por pessoas, que concentram valores em
fronteiras: 0, 1, negativos, arrays vazios, nulls. São exatamente os valores que
ativam ramos condicionais e que a amostragem uniforme acerta com baixa
probabilidade.

### 3.3 Capacidade ou consistência? O teto

Tomando o **melhor** que cada arm atinge por classe de mutator ao longo dos 12
seeds — o seu teto — em vez da média (`analyze_task5_ceiling.py`):

| Mutator | `WGAN` | `CORPUS` | `KDE` | `RANDOM` | `MANUAL` | total |
|---|---|---|---|---|---|---|
| NegateConditionals | 40 | 40 | 40 | 40 | 44 | 44 |
| Math | 17 | 17 | 17 | 17 | 18 | 21 |
| ConditionalsBoundary | 11 | 11 | 12 | 9 | 12 | 18 |
| PrimitiveReturns | 14 | 14 | 14 | 11 | 14 | 14 |
| BooleanTrueReturnVals | 5 | 5 | 5 | 5 | 6 | 6 |
| Increments | 1 | 1 | 1 | 1 | 3 | 3 |
| BooleanFalseReturnVals | 2 | 2 | 2 | 2 | 4 | 4 |
| NullReturnVals | 2 | 2 | 2 | 2 | 2 | 2 |
| VoidMethodCall | 2 | 2 | 2 | 2 | 2 | 2 |
| **TETO** | **94** | **94** | **95** | 89 | **105** | 114 |

**A WGAN e o CORPUS têm o mesmo teto, em todas as nove classes.** Alcançam o
mesmo conjunto de mutações. A diferença é consistência:

| Arm | Média | Teto | Fração do teto |
|---|---|---|---|
| `MANUAL` | 92,1 % | 92,1 % | 100,0 % |
| `CORPUS` | 82,2 % | 82,5 % | **99,7 %** |
| `WGAN` | 78,7 % | 82,5 % | 95,4 % |
| `KDE` | 76,5 % | 83,3 % | 91,8 % |
| `RANDOM` | 71,6 % | 78,1 % | 91,8 % |

O CORPUS ganha porque reproduz inputs que já se sabe funcionarem, acertando no
teto quase sempre. A WGAN é o arm mais consistente **entre os que produzem
inputs novos**. A distinção não é retórica: o CORPUS não é um gerador, não produz
nada que não tenha visto, e está por construção impedido de ser aplicado onde
ainda não existe uma suite de onde extrair corpus — que é exatamente a situação
em que gerar dados de teste tem utilidade.

Duas consequências:

1. **Afinar a rede rende no máximo +3,8 pp** (até ao seu próprio teto), e isso
   seria empatar com o CORPUS, nunca ultrapassá-lo.
2. **O teto é da codificação, não do gerador.** A suite humana mata 105; nenhum
   método derivado do corpus passa de 95. São **10 mutações perdidas na
   representação**, concentradas em `NegateConditionals` (40 vs 44),
   `Increments` (1 vs 3) e `BooleanFalseReturnVals` (2 vs 4). Recuperá-las exige
   mudar a codificação, não treinar melhor.

> **Limitação do método.** O teto é somado por *classe* de mutator, não por
> mutação individual (o relatório agregado do PIT não guarda o identificador de
> cada mutação). Dois seeds que matem mutações distintas da mesma classe contam
> como uma só. É portanto um limite superior à granularidade da classe. A
> igualdade WGAN/CORPUS é robusta a isto por se verificar em todas as classes
> em separado.

### 3.4 Ablações

Um fator de cada vez, tudo o resto igual, incluindo as seeds. Os arms `RANDOM`,
`CORPUS` e `KDE` não dependem dos fatores manipulados e servem de controlo de
regressão: em todas as ablações ficaram **bit a bit idênticos**, o que é o que
distingue "melhorei a rede" de "mudei a medição".

**Duração do treino (300 → 2000 épocas):**

| Arm | 300 ép. | 2000 ép. | Delta | Seeds a favor |
|---|---|---|---|---|
| `RANDOM` | 72,16 % | 72,16 % | 0,00 pp | controlo |
| `CORPUS` | 82,81 % | 82,81 % | 0,00 pp | controlo |
| `KDE` | 78,09 % | 78,09 % | 0,00 pp | controlo |
| **`WGAN`** | 77,12 % | **79,78 %** | **+2,66 pp** | 6/7 |

A configuração original estava **subtreinada por um fator ~7**. p = 0,109, mas
com 7 seeds e uma derrota o mínimo possível é 0,078 — é limite de amostra, não
ausência de efeito. W1 desce de 0,101 para 0,057. Sem memorização: 197,9 inputs
únicos em 200, zero cópias exatas.

**Feature matching:** varrimento sobre {0; 0,3; 1,0} a 800 épocas move o W1
menos de 0,001. O termo combate colapso de variância, e neste corpus a variância
não estava a colapsar. Foi decisivo em E1, é inerte aqui.

### 3.5 Referência face às suites existentes

Não é comparação controlada (tamanhos e processos de geração diferentes), mas
situa a WGAN:

- **vs EvoSuite: +17,3 pp** de mutation score e **+20,9 pp** de cobertura de
  ramo. A diferença está no alcance: o EvoSuite deixa 30 mutações sem execução,
  a WGAN deixa 12,9. Em test strength a distância é +5,4 pp.
- **vs Manual: −13,4 pp** de mutation score. Os testes humanos matam 105 das 114.
  Das ~15 mutações que separam a média da WGAN da suite humana, **11 estão acima
  do seu teto** e são inalcançáveis com esta codificação; só 4 são margem de
  melhoria da rede.

## 4. Duas armadilhas que custaram correções

### 4.1 Poder estatístico

O primeiro run usou 5 seeds e deu **p = 0,125**, o que parecia indicar ausência de
efeito. Era falta de poder: **com n = 5, o menor p-valor que o teste de Wilcoxon
emparelhado consegue produzir é 0,0625** — seria matematicamente impossível
atingir p < 0,05, mesmo com a WGAN a ganhar 5/5. Com 12 seeds o mesmo efeito dá
p = 0,0005.

Qualquer replicação deve usar **≥ 12 seeds**.

### 4.2 Populações de mutações diferentes

O mutation score é `killed / total`. Durante semanas, execuções desta experiência
alternaram entre **114** e **118** mutações sem explicação aparente. O
`MathOperations.java` não mudou uma linha nesse período.

**Causa:** o language server de Java do VS Code (`redhat.java`) compila para
`target/classes` — o mesmo diretório de onde o PIT lê o bytecode a mutar — usando
um JDK 21 próprio, enquanto o `pom.xml` compila para Java 11 com JDK 17. O Maven
decide recompilar por data de ficheiro: encontrava os `.class` do IDE mais
recentes que os `.java` e saltava a compilação, deixando o PIT a mutar bytecode
que o Maven nunca produziu. Bytecode diferente, número de pontos mutáveis
diferente.

**Qual é o número legítimo:** 114. O critério é a reprodutibilidade — é o que
qualquer pessoa obtém ao clonar o repositório e correr `mvn clean test`. As
diferenças concentram-se em `Increments` (3 vs 6) e `BooleanFalseReturnVals`
(4 vs 5), exatamente onde compiladores diferentes geram instruções diferentes.

**Custo do erro:** a ablação das épocas apareceu como +3,11 pp com p = 0,0068 e,
com os seeds filtrados, ficou +2,66 pp com p = 0,109 — a conclusão de
significância inverteu-se. Chegou a haver uma versão deste documento a afirmar
que 118 era o número correto.

Corrigido em quatro sítios:

- `run_task5_experiment.py` corre `mvn clean test-compile` antes da primeira
  medição (o `test-compile` não é opcional: o `OracleRunner` é invocado a partir
  de `target/test-classes` antes de qualquer compilação, e um `clean` sozinho
  deixa o passo seguinte sem classes)
- levanta `MutantPopulationChanged` se o total mudar a meio de um run (levantar e
  não avisar: um aviso no topo de um log de 2h é um aviso que ninguém lê)
- `analyze_task5_conditional.py` exclui automaticamente seeds cujas populações
  não coincidam entre as duas execuções comparadas
- `.vscode/settings.json` desliga `java.autobuild.enabled`, deixando a escrita em
  `target/` exclusivamente para o Maven

**Ao correr experiências longas**, confirmar que a extensão Java está desativada.
Uma recompilação a meio aborta o run (por desenho — melhor abortar do que
misturar denominadores).

## 5. Ameaças à validade

1. **Uma única classe-alvo.** É a limitação principal. Procurou-se uma segunda
   classe de código real (Commons Math `ArithmeticUtils` e `FastMath`, Commons
   Lang `IEEE754rUtils`, `Conversion`, `NumberUtils`), mas nenhuma satisfaz
   simultaneamente os quatro requisitos: pequena (viável para PIT multi-seed),
   auto-contida, com assinatura compatível com a codificação, e com suite de
   testes suficiente para extrair corpus. O `IEEE754rUtils` chegou perto, mas o
   seu teste tem apenas 22 call sites (≈10 inputs extraíveis) e a sua semântica é
   dominada por `NaN`/`Infinity`, que um gerador contínuo não representa bem.
   Os resultados valem para uma classe de utilitários numéricos; a generalização
   para outros perfis de código não está demonstrada.

2. **Oráculo de regressão.** Os valores esperados vêm da implementação original,
   pelo que a suite não deteta defeitos já presentes nessa implementação — mede
   capacidade de detetar *regressões*. É a mesma limitação do EvoSuite, e é a
   prática corrente na literatura de geração automática de testes.

3. **Corpus pequeno.** A WGAN treina em 121 inputs. Fidelidade medida:
   W1 ≈ 0,057 no espaço normalizado, após o prolongamento do treino (secção 3.4).

4. **Cobertura de método saturada por construção.** Todos os arms gerados ficam
   presos em 93,3 % (14/15 métodos) porque a codificação de 12 dimensões
   não representa `String`, e um dos métodos aceita `String`. A métrica não
   discrimina neste desenho e não deve ser lida como evidência.

4b. **O W1 guia mal a afinação fina.** É a média das distâncias 1-D por marginal,
   logo é cego ao acoplamento entre dimensões. Entre arms discrimina bem
   (Spearman −0,86 com o mutation score, p = 3e−15, sobre 48 medições); *dentro*
   do arm WGAN, entre seeds, cai para −0,26 (p = 0,41). Serve para separar
   métodos, não para escolher entre execuções do mesmo método — que é o regime
   em que seria mais útil. Passou a registar-se também a energy distance, que é
   sensível à distribuição conjunta, mas o problema de fundo mantém-se: a
   fidelidade distribucional é mau substituto da utilidade quando a diferença
   entre candidatos é pequena.

4c. **O teto limita a leitura de tudo o resto.** Nenhum método derivado do
   corpus passa de 95 das 114 mutações (secção 3.3). Todas as comparações entre
   geradores decorrem numa janela estreita: do piso do RANDOM (71,6 %) ao teto
   comum (83,3 %) vão **11,7 pp**. Os deltas reportados são reais, mas devem ser
   lidos a essa escala e não à dos 100 % teóricos — os +7,02 pp da WGAN sobre o
   RANDOM são ~60 % do espaço disponível.

5. **Bug de medição já corrigido (registo histórico).** A primeira leitura de
   cobertura dava 100 % nos quatro arms. Causa: o agente do JaCoCo faz *append*
   em `target/jacoco.exec` por omissão, e o `_measure` apagava o relatório mas
   não o `.exec`, pelo que cada arm reportava a união de todos os anteriores.
   Corrigido em `run_task5_experiment.py` e `run_task5_budget_sweep.py` (apaga o
   `.exec` + `-Djacoco.append=false`). Os valores da secção 3 vêm já do run
   corrigido. **O PIT nunca foi afetado** (cada execução é independente).
   Lição: uma métrica saturada a 100 % em todos os arms é suspeita, não é
   ausência de efeito.

6. **Budget sweep por repetir.** O `run_task5_budget_sweep.py` varia o orçamento
   (3–200 inputs) para localizar o ponto onde a distribuição aprendida começa a
   importar. A execução existente (`outputs/task5/budget_sweep.csv`, 7 ago)
   sofria do bug do `jacoco.exec`, pelo que as suas colunas de cobertura são
   inválidas; as de mutação continuam válidas. Falta re-correr.

## 6. Reprodutibilidade

```bash
# 1. Corpus a partir dos testes existentes
python python/scripts/extract_test_input_corpus.py

# 2-5. Experiência principal (12 seeds, 2000 épocas)
python python/scripts/run_task5_experiment.py --seeds 12 --epochs 2000 \
    --out outputs/task5/results_e2000.csv

# 6. Análise estatística (escada + Holm)
python python/scripts/analyze_task5_results.py --results outputs/task5/results_e2000.csv

# 7. Decomposição teto/consistência — o resultado da secção 3.3
python python/scripts/analyze_task5_ceiling.py \
    --by-mutator outputs/task5/results_e2000_by_mutator.csv

# Ablação: um fator de cada vez, com verificação automática dos controlos
python python/scripts/run_task5_experiment.py --seeds 12 --epochs 2000 --conditional \
    --skip-fixed --out outputs/task5/results_e2000_cond.csv
python python/scripts/analyze_task5_conditional.py \
    --baseline outputs/task5/results_e2000.csv \
    --conditional outputs/task5/results_e2000_cond.csv \
    --label "condicionamento em arr_len"

# Regime de poucos dados: o único cenário onde o CORPUS perde o seu teto
python python/scripts/run_task5_experiment.py --seeds 12 --epochs 2000 \
    --corpus-frac 0.15 --skip-fixed --out outputs/task5/results_frac15.csv

# Complementar: cobertura/deteção vs orçamento de testes
python python/scripts/run_task5_budget_sweep.py --seeds 5 --budgets 3,5,10,20,50,200
```

Pré-requisito: `mvn -q -B test-compile` funcional (JDK 11+, Maven).

### Ficheiros gerados

| Ficheiro | Conteúdo |
|---|---|
| `outputs/task5/input_corpus.csv` | 121 inputs extraídos dos testes existentes |
| `outputs/task5/inputs_{gan,random,corpus,kde}.csv` | Inputs gerados por arm |
| `outputs/task5/oracle_{gan,random,corpus,kde}.csv` | Resultados da implementação original |
| `outputs/task5/generation_metrics.json` | W1, energy distance, novidade, violações estruturais |
| `outputs/task5/results_114_e2000.csv` | Medições principais (12 seeds, 114 mutações) |
| `outputs/task5/results_114_e2000_by_mutator.csv` | Discriminação por classe de mutator (base do teto) |
| `outputs/task5/budget_sweep.csv` | Varrimento de orçamento |
| `src/test/java/dataloader/MathOperations{Gan,Random,Corpus,Kde}Tests.java` | Suites geradas (auto-geradas, não editar) |

> **Nota histórica.** Qualquer ficheiro sem `114` no nome pode conter medições
> feitas contra bytecode do IDE (secção 4.2) e **não deve ser usado**. Os
> resultados deste documento vêm de `results_114_*.csv`, todos verificados a 114.
