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

### 3.6 Regime de poucos dados: onde gerar bate repetir

As secções anteriores deixam a WGAN numa posição desconfortável: melhor que
amostragem uniforme, indistinguível do KDE, pior que reamostrar o corpus. Se
reamostrar produz melhores dados de teste, reamostrar é o que se deve fazer.

Só que há uma assimetria estrutural que a comparação com o corpus completo não
consegue exibir. **A reamostragem só devolve entradas que já viu**; a rede
aprende uma densidade e interpola. Com 121 entradas a diferença não se manifesta,
porque 121 chegam. A questão é o que acontece quando não chegam — que é o cenário
que justifica gerar dados de teste, porque se já existisse uma suite extensa de
onde extrair corpus o problema estaria em boa medida resolvido.

O `--corpus-frac` restringe **todos** os arms ao mesmo subconjunto, incluindo o
CORPUS. Restringir só a rede dar-lhe-ia uma desvantagem de informação.

| Fração | Entradas | `WGAN` | `CORPUS` | `KDE` | `RANDOM` |
|---|---|---|---|---|---|
| 0,15 | 18 | **79,8 %** | 76,1 % | 78,1 % | 75,2 % |
| 1,00 | 121 | 78,7 % | **82,2 %** | 76,5 % | 71,6 % |

**A ordenação inverte-se.** Com o corpus completo a reamostragem vence por 3,6 pp;
com 18 entradas perde por 3,7 pp.

#### Comparações emparelhadas a 15 % (12 seeds, Holm-Bonferroni)

| Base | Delta | Seeds a favor | p corrigido |
|---|---|---|---|
| `RANDOM` | +4,61 pp | 11/12 | **0,0103** |
| `CORPUS` | +3,73 pp | 9/12 | **0,0312** |
| `KDE` | +1,68 pp | 7/12 | **0,0312** |

É o **único ponto de toda a Tarefa 5 em que a WGAN supera as três bases em
simultâneo**, com significância que sobrevive à correção para comparações
múltiplas. Com o corpus completo era indistinguível do KDE e perdia para o
CORPUS; com 15 % separa-se de ambas.

#### O mecanismo, medido e não inferido

| Arm | Distintas (121) | W1 (121) | Distintas (18) | W1 (18) |
|---|---|---|---|---|
| `WGAN` | 198 | 0,057 | **176** | **0,088** |
| `CORPUS` | 72 | 0,017 | **17** | 0,018 |
| `KDE` | 200 | 0,090 | 189 | **0,167** |
| `RANDOM` | 200 | 0,468 | 200 | 0,498 |

Cada arm falha de uma maneira diferente:

- **`CORPUS`** conserva fidelidade perfeita e **colapsa em diversidade**: de 72
  para 17 entradas distintas em 200 sorteios. Não pode ser de outra forma — não
  emite nada que não tenha visto.
- **`KDE`** conserva diversidade e a sua **fidelidade degrada-se quase para o
  dobro** (0,090 → 0,167): ajustar kernels gaussianos a 18 pontos em 12 dimensões
  é mal condicionado.
- **`RANDOM`** conserva diversidade total e nunca teve fidelidade.
- **`WGAN`** é o único que conserva as duas: perde 11 % de diversidade e o W1
  sobe 0,031.

A leitura é mais estreita do que "GANs geram bons dados de teste", e por isso mais
defensável: **a utilidade exige diversidade e fidelidade em simultâneo**. Com
dados abundantes vários métodos satisfazem ambas e a arquitetura torna-se
indiferente; com poucos, os métodos simples sacrificam uma delas, e é aí que a
capacidade adversarial passa a ser mensurável.

> **Ressalva metodológica.** A análise de teto (secção 3.3) **não se aplica** com
> `corpus_frac < 1`. O teto é o máximo por classe de mutator ao longo dos seeds,
> e cada seed sorteia um subconjunto diferente, pelo que a união acaba por cobrir
> grande parte do corpus. O sintoma que denuncia o problema é o teto do RANDOM
> *subir* quando o corpus encolhe, o que é impossível como facto sobre
> capacidade. A limitação do CORPUS lê-se na diversidade por seed, que é uma
> medida por execução e não sofre desse confundidor.

## 3.7 Réplica num segundo alvo: `ArithmeticUtils`

Alvo: `org.apache.commons.math3.util.ArithmeticUtils` (tag `MATH_3_6_1`), copiado
literalmente com **uma linha alterada** (o package), verificável por diff contra
`corpus_sources/`. Perfil deliberadamente diferente do primeiro: dominado por
verificação de overflow e exceções, e não por operações sobre arrays.

**Escada (11 seeds, 210 mutações, população estável em todas as medições):**

| braço | mutation score | vs WGAN | seeds | p (Holm, 3 comparações) |
|---|---|---|---|---|
| CORPUS | 63,59 ± 1,51 % | −6,28 pp | 0/11 | 0,0029 |
| KDE | 62,25 ± 1,33 % | −4,94 pp | 0/11 | 0,0029 |
| **WGAN** | **57,32 ± 0,83 %** | — | — | — |
| RANDOM | 47,62 ± **0,00** % | **+9,70 pp** | **11/11** | **0,0029** |

O resultado central **replica com margem maior** (+9,70 contra +7,02 pp), unânime
nos dois alvos. O `RANDOM` deu exactamente 47,62 % nas 11 seeds, sem variação:
amostrar uniformemente em ±2³¹ transborda quase sempre e exercita sempre os mesmos
caminhos. Neste alvo o piso é uma constante, não uma distribuição.

**A seed 49 foi excluída** por falha de infraestrutura na medição (timeout, duas
tentativas), não pelo valor — que nunca chegou a existir. A n=11 o p mínimo é
0,00098.

### Três limitações do método que o segundo alvo expôs

1. **A extracção por literais não funciona em suites reais.** Sobre a suite do
   `ArithmeticUtils` captou 8 chamadas distintas em 154. Substituída por
   instrumentação em execução — um delegador fino fora do pacote que o PIT muta —
   passou a captar 274.

2. **Sobrecargas `long` partilham o nome com as `int`** e o PIT conta-as na mesma
   classe: 68 % das mutações ficavam sem cobertura e o CORPUS media 29,1 %.
   Invocando ambas as versões, desce para 25 % e o CORPUS sobe para 66,7 %.

3. **Min-max destrói distribuições de cauda pesada.** O corpus vai de −2³¹ a
   2³¹−1 mas as medianas dos absolutos são 1 e 50: 95 % fica a menos de 0,01 da
   mediana depois de normalizar. Sintoma contra-intuitivo: **treinar mais
   piorava** (47,6 % a 60 épocas → 40,9 % a 2000). Corrigido com
   `sign(x)·log1p(|x|)` antes da normalização (`log_scale` por alvo):

   | braço | linear | logarítmica | delta |
   |---|---|---|---|
   | **WGAN** | 40,9 % | **57,6 %** | **+16,7 pp** |
   | CORPUS | 63,3 % | 63,3 % | 0,00 (controlo) |
   | RANDOM | 47,6 % | 47,6 % | 0,00 (controlo) |

   É a maior variação de toda a Tarefa 5 — seis vezes o que as épocas renderam.

### O que NÃO se transfere: o teto

| mutator | WGAN | CORPUS | KDE | RANDOM | total |
|---|---|---|---|---|---|
| NegateConditionals | 56 | 59 | 59 | 47 | 75 |
| Math | 36 | 39 | 39 | 33 | 53 |
| **ConditionalsBoundary** | **7** | **12** | 11 | 0 | **32** |
| PrimitiveReturns | 16 | 18 | 18 | 11 | 28 |
| InvertNegs | 8 | 10 | 10 | 8 | 11 |
| **Teto** | **124** | **139** | **138** | **100** | **210** |

No primeiro alvo WGAN e CORPUS tinham o **mesmo** teto nas nove classes, e
concluiu-se que a derrota era de consistência. **Aqui é falso**: 15 mutações de
diferença, com fracções do teto equivalentes (97,1 / 96,1 / 94,7 %). Neste alvo a
derrota é de **capacidade**.

O défice concentra-se em `ConditionalsBoundary` — fronteiras de overflow,
exercitadas exactamente em 2³¹−1. O corpus tem esses literais; um gerador
contínuo acerta na ordem de grandeza sem aterrar no ponto. Nenhum hiperparâmetro
corrige isto: aponta para hibridização com análise de valores de fronteira.

O KDE também inverte (era +2,19 pp, agora −4,94), o que é coerente com a curva da
secção 3.6: 260 entradas de treino aqui contra 121 lá, e acima de ~70 a vantagem
adversarial desaparece. É evidência independente, noutra classe, de uma hipótese
formulada a partir do primeiro alvo.

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

1. **Dois alvos, ambos utilitários numéricos.** Uma versão anterior deste
   documento dava o alvo único como limitação principal e registava que nenhuma
   segunda classe servia (Commons Math `ArithmeticUtils` e `FastMath`, Commons
   Lang `IEEE754rUtils`, `Conversion`, `NumberUtils`). **O diagnóstico estava
   errado**: o requisito que falhava era o do corpus, e a causa era o extractor
   por literais, não as classes. Com instrumentação em execução (secção 3.7) o
   mesmo `ArithmeticUtils` passou a servir. A limitação que subsiste é mais
   estreita: ambos os alvos são utilitários numéricos com métodos estáticos e sem
   estado. Código com estado, hierarquias de objetos ou cadeias de carateres
   continua por demonstrar, e a codificação não o representa. O `IEEE754rUtils`
   continua fora de alcance, por a sua semântica ser dominada por
   `NaN`/`Infinity`.

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

# Segundo alvo: escada completa, retomável (uma seed por ficheiro)
python python/scripts/task5b_resume.py            # corre o que falta
python python/scripts/task5b_resume.py --status   # só relata

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
| `outputs/task5b/input_corpus.csv` | 274 chamadas do segundo alvo, por instrumentação |
| `outputs/task5b/seeds/seed_NN.csv` | Uma seed por ficheiro (permite retomar) |
| `outputs/task5b/results_arith_final.csv` | Escada do segundo alvo, 11 seeds, 210 mutações |
| `outputs/task5b/gate_linear.csv` / `gate_log.csv` | Efeito isolado da codificação (+16,7 pp) |

> **Nota histórica.** Qualquer ficheiro sem `114` no nome pode conter medições
> feitas contra bytecode do IDE (secção 4.2) e **não deve ser usado**. Os
> resultados deste documento vêm de `results_114_*.csv`, todos verificados a 114.
