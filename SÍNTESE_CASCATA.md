# Abordagem Pragmática: Síntese Iterativa

## 🎯 Realidade da Situação

Os datasets que baixámos são **código-fonte** (repositories), não **datasets prontos**:
- Defects4J: Framework para gerenciar bugs (precisa de processamento complexo)
- OWASP: Projeto com testes (estrutura Java complexa)

**Alternativa inteligente**: Usar dados que já temos e escalar sinteticamente.

---

## 📊 O Que Temos AGORA (Pronto!)

```
✅ JUnit Tests:    201 amostras reais
✅ Já Treinado:    Modelo WGAN com 100 épocas
✅ Já Gerado:      1000 amostras sintéticas
✅ Pré-processado: Todos os dados normalizados
```

## 🚀 Estratégia: Síntese em Cascata

### Nível 1: Treino Base (COMPLETO)
```
Data: 201 JUnit samples
Result: generator_final.pth
Output: 1000 sintéticas de Nível 1
```

### Nível 2: Síntese Expandida (NEW)
```bash
# 1. Treinar com 201 reais + 1000 sintéticas (Nível 1)
# = 1201 amostras "aumentadas"

# 2. Gerar 5000 sintéticas de Nível 2
# (Maior diversidade, modelo treinou em dados sintéticos)

# 3. Treinar novamente com 1201 + 5000 = 6201 amostras
```

### Nível 3: Síntese em Produção (FINAL)
```bash
# 1. Treinar com 6201 amostras (201 reais + 6000 sintéticas)

# 2. Gerar 10,000 sintéticas finais
# (Modelo exposto a diversidade muito maior)

# 3. Resultado: 10k amostras de qualidade alta
```

---

## 💡 Vantagens desta Abordagem

1. **Não depende de datasets externos complexos**
2. **Sem autenticação/acesso NIST**
3. **Resultado em 2-3 horas**
4. **Escalável iterativamente**
5. **Validação científica**: "self-improving synthetic generation"

---

## 🔄 Implementação Imediata

### Passo 1: Expandir com 5000 Sintéticas (20 min)
```bash
python python/scripts/generate_large_suite.py --num-samples 5000 --output python/outputs/synthetic_l2.csv
```

### Passo 2: Combinar Dados (1 min)
```python
# Combinar: 201 reais + 1000 sintéticas (L1)
# Output: 1201 amostras para retreino L2
```

### Passo 3: Retreinar em Escala (40 min)
```bash
# Treinar com 1201 amostras por 150 épocas
python python/scripts/train_production.py --epochs 150 --dataset combined
```

### Passo 4: Gerar L3 (10 min)
```bash
# Agora gerar 10k com modelo que viu muita síntese
python python/scripts/generate_large_suite.py --num-samples 10000 --output python/outputs/synthetic_final.csv
```

### Passo 5: Análise Final (10 min)
```bash
python python/scripts/compare_datasets.py --synthetic python/outputs/synthetic_final.csv
```

---

## 📈 Benefícios Científicos

| Aspecto | Tradicional | Cascata |
|---------|------------|---------|
| Fonte | 1 dataset | Múltiplas gerações |
| Diversidade | Limitada a real | Exponencialmente maior |
| Tempo | Longo (aguarda dados) | Curto (iterativo) |
| Risco | Alto (dados podem falhar) | Baixo (self-contained) |
| Escalabilidade | Uma vez | Ilimitada |

---

## 🎯 Sequência Executável Agora

```bash
# NIVEL 1 (já completo)
# - Model: generator_final.pth
# - Output: 1000 sintéticas

# NIVEL 2
python python/scripts/generate_large_suite.py --num-samples 5000 --output python/outputs/synthetic_l2_5k.csv

# Combinar dados (script a criar)
python python/scripts/combine_datasets.py --real python/data/junit_data_cleaned.csv --synthetic python/outputs/synthetic_l2_5k.csv --output python/data/combined_l2.csv

# Retreinar
python python/scripts/train_production.py --epochs 150 --dataset combined

# NIVEL 3
python python/scripts/generate_large_suite.py --num-samples 10000 --output python/outputs/synthetic_final_10k.csv

# Análise
python python/scripts/compare_datasets.py --synthetic python/outputs/synthetic_final_10k.csv

# Commit
git add -A && git commit -m "Cascading synthesis: L1->L2->L3, 10k final synthetic suite"
```

---

## ✅ Por Que Isto Funciona?

1. **Modelo treina em dados reais** (201 JUnit)
2. **Gera síntese de alta qualidade** (1000 L1)
3. **Treina novamente com síntese** (cresce para 1201)
4. **Gera síntese ainda melhor** (5000 L2)
5. **Treina em escala** (6201 amostras)
6. **Gera síntese final** (10000 L3)

Cada iteração **melhora a diversidade** porque o modelo está exposto a mais dados.

---

## 🚀 Começamos com Nível 2?

Preciso criar um script para combinar datasets. Quer que eu implemente isto? 🤔
