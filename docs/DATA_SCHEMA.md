# Esquema de Dados e Regras de Validacao

## 1. Objetivo
Este documento define o contrato de dados utilizado entre a camada Java e a camada Python no pipeline WGAN-GP.

## 2. Formato CSV Canonico
- Ficheiro sem cabecalho.
- Tres colunas numericas por linha.
- Separador virgula.

Estrutura:
1. Coluna 0: methods (nome legado)
2. Coluna 1: branches (nome legado)
3. Coluna 2: statements (nome legado)

Nota metodologica:
- Estes nomes sao preservados por compatibilidade historica.
- A semantica deve ser lida como tuplo numerico de tres features derivadas dos testes.
- Nao devem ser interpretadas automaticamente como metricas formais de cobertura de codigo de producao.

## 3. Regras de Validacao
### Regras obrigatorias
- Valores numericos finitos (sem NaN e sem infinito).
- Valores nao negativos.
- Limite superior operacional: 1000.
- Tres colunas por linha.

### Regras recomendadas
- Tamanho minimo operacional para treino: 100 amostras.
- Tamanho recomendado para maior estabilidade estatistica: 500 amostras ou mais.

## 4. Estado Atual dos Datasets (abril 2026)
- python/data/junit_data.csv: 733 linhas.
- python/data/junit_data_cleaned_strict.csv: 398 linhas.
- python/data/exp_b_high_quality.csv: 79 linhas.
- python/data/exp_c_low_redundancy.csv: 398 linhas.

## 5. Normalizacao
- Escala de treino: [-1, 1].
- Estatisticas de normalizacao devem ser guardadas e reutilizadas na denormalizacao.
- O script de geracao deve usar ficheiro de stats coerente com o dataset usado no treino.

## 6. Comando de Validacao Recomendado
```bash
python python/scripts/validate_before_training.py --data-path python/data/junit_data_cleaned_strict.csv --min-samples 100
```

## 7. Riscos Comuns
- Misturar datasets de origem diferente sem registo de versao.
- Treinar com stats de normalizacao de outro dataset.
- Assumir semantica de cobertura formal para colunas legadas sem prova instrumental.

## 8. Recomendacao de Governanca
Cada run experimental deve guardar:
- dataset utilizado;
- ficheiro de normalizacao;
- configuracao de treino;
- artefactos de geracao;
- relatorio de analise.
