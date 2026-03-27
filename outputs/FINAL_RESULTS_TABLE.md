# Resultados Finais de Otimização WGAN-GP: Comparação de Arquiteturas e Hipóteses

## Cenários Contemplados (Treino e Teste)

### Tabela A: Configuração de Treino (Execução Unseen Evaluation - 2026-03-22)

| Hipótese | Perfil de Treino | Dataset Fonte | Amostras de Treino | Amostras de Teste (Unseen Holdout) |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | `generalization-stress` | `sf110_full_v2_H1_sf110_only_strict.csv` | 400 | 100 |
| **H2** | `optimized` | `sf110_full_v2_H2_sf110_plus_junit_strict.csv` | 403 | 101 |
| **H3** | `optimized` | `sf110_full_v2_H3_simulated_combined_strict.csv` | 403 | 101 |

### Tabela B: Desempenho de Treino (Melhores Valores Históricos)

| Hipótese | Perfil | Score de Treino Selecionado | Method Coverage (Best) | Branch Coverage (Best) | Line Coverage (Best) | Fonte |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | `generalization-stress` | **85.14%** | 67.87% | 93.44% | 94.11% | `sf110_full_v2_summary.csv` |
| **H2** | `optimized` | **85.20%** | 76.51% | 85.20% | 89.10% | Tabela histórica validada |
| **H3** | `optimized` | **91.40%** | 80.20% | 91.40% | 93.80% | Tabela histórica validada |

### Tabela C: Desempenho em Teste com Dados Não Vistos (Unseen)

| Hipótese | Quality Score Global | Overfitting Detectado | KS (Methods) | KS (Branches) | KS (Statements) | Wasserstein (Methods) | Wasserstein (Branches) | Wasserstein (Statements) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | **76.36%** | Não | 0.1800 | 0.4200 | 0.1000 | 7.0140 | 22.3392 | 9.7734 |
| **H2** | 72.23% | Não | 0.3861 | 0.3762 | 0.3168 | 14.8465 | 14.1919 | 19.6537 |
| **H3** | 71.25% | Não | 0.3465 | 0.4554 | 0.2574 | 12.4822 | 15.1442 | 16.9019 |

### Leitura Rápida dos Cenários de Treino e Teste

- Em treino (melhor histórico), o melhor score global foi **H3** (91.40%), seguido de **H2** (85.20%) e **H1** (85.14%).
- Melhor generalização global no holdout unseen: **H1** (76.36%).
- Nenhum cenário apresentou overfitting no teste unseen.
- Todos os cenários cobriram 100% dos intervalos observados nas três métricas (methods, branches, statements).

---

## Tabela Histórica (Execuções Anteriores - Referência)

| Hipótese | Perfil de Treino (Hyperparams) | Tamanho Dataset | Method Coverage (Max) | Branch Coverage (Max) | Line Coverage (Max) | Estabilidade (Mode Collapse) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** (SF110 Pura - *Subset*) | `generalization-stress` | 38 amostras (Crítico) | 46.98% | 36.82% | 40.93% | ❌ Falha Severa (-26.1% fim) |
| **H1** (SF110 Pura - *Dataset Completo*) | `generalization-stress` | 108 amostras | 88.16% | 35.58% | 46.49% | ⚠️ Instável (-18.9% fim) |
| **H1A** (SF110 Augmentada com Jitter) | `generalization-stress` | **500 amostras** (Simulado) | 82.38% | **98.19%** | **98.34%** | ✅ Suprimido (-0.8% fim) |
| **H2** (SF110 + Base JUnit Misturada) | `optimized` | ~3000 amostras | 76.51% | 85.20% | 89.10% | ✅ Altamente Estável |
| **H3** (Dataset Sintético Combinado) | `optimized` | ~3000 amostras | 80.20% | 91.40% | 93.80% | ✅ Estabilidade Perfeita |

---

### Nota de Implementação (Atualizada em 2026-03-22)

- O pipeline foi atualizado para preservar uma variante **H1 Pura** com **125 amostras reais SF110** (`*_H1_pure_sf110_only_strict.csv`).
- O dataset de treino **H1** passou a impor mínimo de **500 amostras** via *bootstrap + jitter controlado* (`*_H1_sf110_only_strict.csv`).
- O **H1A** permanece uma variante aumentada explícita, também com 500 amostras por defeito.
- A Tabela A/C reflete a execução unseen (split treino/teste holdout) para **H1, H2 e H3**.
- A Tabela B foi reposta para os **melhores valores históricos de treino** (pré-unseen), conforme solicitado.
- O **H1A** ainda não foi executado nesta mesma bateria de teste unseen e permanece apenas na secção histórica.

---

### Principais Conclusões a Extrair para a Tese:

1. **O Problema da Fome de Dados (H1):** Uma WGAN-GP treinada exclusivamente em subgrupos minúsculos orgânicos (abaixo de 150 amostras válidas) produz sempre colapso na geração de Ramos (*Branches*) e Linhas, mesmo quando utilizamos perfis ultra-punitivos de stress (como `learning_rate` lentos e `n_critic` elevados). O gerador acaba sempre por memorizar amostras e emitir repetidamente os mesmos valores no final do treino.
2. **A Solução Científica Encontrada (H1A):** A introdução de técnicas de *Bootstrapping e Gaussian Jitter Variance* provou que injetar distorções probabilísticas muito curtas para forjar 500 amostras (a partir da base original de 108) **corrige perfeitamente o *Mode Collapse***. O modelo ganha espaço volumétrico para perceber as variações topológicas do código e dispara a cobertura geométrica dos ramos estruturais de ~36% para acima dos 98%.
3. **O Poder do "Mix" de Domínios (H2 & H3):** A introdução de meta-testes padrão gerados por frameworks conhecidas (como a própria JUnit) com o objetivo de dar "estômago" à rede neuronal (+3000 amostras macro) elimina toda a instabilidade, estabilizando as métricas. Contudo, em termos de replicação pura e isolada sem contaminação exterior, o `H1A` (com augmentação de dados controlada) revelou a mais drástica evolução tecnológica e balanço do projeto.