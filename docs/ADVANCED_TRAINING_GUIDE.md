# Guia Avancado de Treino WGAN-GP

## 1. Enquadramento
Este guia descreve a execucao de experiencias de treino em perfis controlados, com foco em estabilidade, rastreabilidade e comparabilidade estatistica.

## 2. Perfis Disponiveis
Os perfis estao definidos em python/scripts/train_basic_profiles.py.

### 2.1 aggressive
- objetivo: rapidez de treino;
- risco: maior instabilidade.

### 2.2 experimental
- objetivo: configuracao hibrida para exploracao.

### 2.3 conservative
- objetivo: maxima estabilidade;
- custo: maior tempo de treino.

### 2.4 balanced
- objetivo: compromisso entre custo e qualidade;
- uso recomendado para baseline operacional.

### 2.5 optimized
- objetivo: reduzir risco de mode collapse com maior rigor no critic.

### 2.6 generalization-stress
- objetivo: forcar generalizacao em cenarios de baixa disponibilidade de dados.

## 3. Procedimento de Treino Recomendado
```bash
# 1) Validar dados
python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100

# 2) Treinar perfil balanced
python python/scripts/train_basic_profiles.py --profile balanced --epochs 50 --data-path python/data/junit_data_cleaned_strict.csv

# 3) Treinar perfil optimized (comparativo)
python python/scripts/train_basic_profiles.py --profile optimized --epochs 50 --data-path python/data/junit_data_cleaned_strict.csv
```

## 4. Reproducibilidade
Recomenda-se guardar por run:
- seed;
- perfil e hiperparametros;
- dataset e ficheiro de normalizacao;
- checkpoints e metrica de selecao do melhor modelo;
- relatorio de analise experimental.

## 5. Geracao e Avaliacao Pos-Treino
```bash
python python/scripts/generate_synthetic.py --num-samples 1000 --output outputs/synthetic_samples.csv
python python/scripts/experimental_analysis.py \
  --training-history outputs/<run_name>/training_history.csv \
  --real-data python/data/junit_data_cleaned_strict.csv \
  --synthetic-data outputs/synthetic_samples.csv \
  --output-dir outputs/experimental/<run_name>
```

## 6. Diagnostico
Sinais comuns de problema:
- perda do critic sem estabilidade e sem melhoria de diversidade;
- amostras sinteticas com baixa variacao;
- desvio sistematico da distribuicao real.

Acoes recomendadas:
- ajustar perfil de treino;
- rever dataset e normalizacao;
- repetir run com seed controlada para confirmacao.

## 7. Integracao com Mutation Testing
A avaliacao generativa deve ser analisada em paralelo com resultados PIT para garantir que melhoria estatistica nao se traduz em degradacao funcional da qualidade de testes.
