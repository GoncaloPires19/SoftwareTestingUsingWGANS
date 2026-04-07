# Arquitetura do Sistema

## 1. Objetivo Arquitetural
O sistema foi desenhado para integrar uma camada de testes Java com uma camada de modelacao generativa em Python. O objetivo e produzir amostras sinteticas consistentes com o comportamento observado no conjunto real e avaliadas por metodos experimentais reproduziveis.

## 2. Camadas do Pipeline

### 2.1 Camada Java (extracao)
- As suites JUnit executam cenarios funcionais e registam tuplos numericos de tres dimensoes.
- O logger grava os dados em CSV para consumo pela camada Python.
- O mutation testing com PIT mede eficacia das suites por classe e no agregado.

### 2.2 Camada de dados (Python)
- Carregamento de CSV sem cabecalho com tres colunas.
- Validacao de integridade (limites, nulos, tipos).
- Normalizacao para intervalo [-1, 1] com estatisticas persistidas em ficheiro JSON.

### 2.3 Camada de treino WGAN-GP
- Modelo gerador: vetor latente -> features sinteticas.
- Modelo critico: score de realismo das amostras.
- Treino por perfis (balanced, optimized, experimental, aggressive, conservative, generalization-stress).
- Registo de checkpoints e historico de treino por run.

### 2.4 Camada de geracao
- Carrega checkpoint selecionado.
- Gera N amostras sinteticas.
- Denormaliza para escala original.
- Exporta CSV em schema canonico de tres colunas (sem cabecalho por omissao).

### 2.5 Camada de analise experimental
- Compara distribuicoes real vs sintetica.
- Avalia convergencia e riscos de sobreajuste.
- Produz relatorios e figuras em outputs/experimental.

## 3. Fluxo Operacional Recomendado
```bash
# 1) Gerar dados pela camada Java
mvn clean test

# 2) Validar dataset antes do treino
python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100

# 3) Treinar perfil
python python/scripts/train_basic_profiles.py --profile balanced --epochs 50 --data-path python/data/junit_data_cleaned_strict.csv

# 4) Gerar amostras
python python/scripts/generate_synthetic.py --num-samples 1000 --output outputs/synthetic_samples.csv

# 5) Executar analise
python python/scripts/experimental_analysis.py \
  --training-history outputs/<run_name>/training_history.csv \
  --real-data python/data/junit_data_cleaned_strict.csv \
  --synthetic-data outputs/synthetic_samples.csv \
  --output-dir outputs/experimental/<run_name>
```

## 4. Decisoes de Projeto
- O schema de tres colunas e mantido por compatibilidade de historico experimental.
- A interpretacao semantica das features e definida no contrato de dados vigente.
- O treino otimizado deve usar gradient penalty sem escala duplicada de lambda.
- Toda a documentacao operacional deve referenciar a CLI real dos scripts.

## 5. Criticos de Qualidade
- Reprodutibilidade: seeds e configuracoes devem ser guardadas por run.
- Rastreabilidade: cada grafico e relatorio deve apontar para os ficheiros de entrada usados.
- Consistencia documental: comandos, caminhos e parametros devem refletir o estado atual do codigo.

## 6. Artefactos Esperados
- outputs/<run_name>/models/*.pth
- outputs/<run_name>/training_history.csv
- outputs/<run_name>/synthetic_samples*.csv
- outputs/experimental/<run_name>/*
- outputs/mutation_experiments/mutation_results.csv
