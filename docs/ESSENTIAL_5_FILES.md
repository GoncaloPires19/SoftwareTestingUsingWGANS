# Cinco Ficheiros Essenciais

## Objetivo
Este guia identifica os cinco ficheiros com maior impacto funcional no pipeline atual. A leitura sequencial destes ficheiros permite compreender a maior parte do comportamento do sistema.

## 1. python/config.py
Funcao principal:
- centralizar configuracoes de dados, modelo, treino e saida.

Porque e essencial:
- define valores padrao de reproducibilidade;
- concentra escolhas de hiperparametros;
- evita dispersao de configuracao pelo projeto.

## 2. python/junit_datasets.py
Funcao principal:
- carregar CSV de entrada;
- validar e normalizar dados para treino;
- fornecer objetos de dataset para o treino.

Porque e essencial:
- garante coerencia entre distribuicao real e distribuicao sintetica;
- controla a normalizacao/denormalizacao usada ao longo do pipeline.

## 3. python/scripts/train_basic_profiles.py
Funcao principal:
- executar o treino WGAN-GP por perfil.

Perfis suportados atualmente:
- aggressive
- experimental
- conservative
- balanced
- optimized
- generalization-stress

Porque e essencial:
- implementa o ciclo de treino;
- guarda checkpoints e historico;
- controla a selecao de melhor modelo por metrica.

## 4. python/scripts/generate_synthetic.py
Funcao principal:
- carregar modelo treinado;
- gerar amostras sinteticas;
- denormalizar e exportar CSV.

Porque e essencial:
- transforma o resultado do treino em artefacto utilizavel;
- aplica o contrato final de output (3 colunas por omissao).

## 5. python/scripts/experimental_analysis.py
Funcao principal:
- comparar dados reais e sinteticos;
- avaliar convergencia e consistencia estatistica;
- produzir relatorios e figuras.

Porque e essencial:
- fornece evidencia quantitativa para validacao da qualidade;
- suporta decisao sobre aceitacao de uma run.

## Sequencia de Leitura Recomendada
1. config.py
2. junit_datasets.py
3. train_basic_profiles.py
4. generate_synthetic.py
5. experimental_analysis.py

## Comandos Minimos de Verificacao
```bash
mvn clean test
python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100
python python/scripts/train_basic_profiles.py --profile balanced --epochs 5 --data-path python/data/junit_data_cleaned_strict.csv
python python/scripts/generate_synthetic.py --num-samples 200 --output outputs/smoke_synthetic.csv
python python/scripts/experimental_analysis.py --training-history outputs/<run_name>/training_history.csv --real-data python/data/junit_data_cleaned_strict.csv --synthetic-data outputs/smoke_synthetic.csv --output-dir outputs/experimental/<run_name>
```
