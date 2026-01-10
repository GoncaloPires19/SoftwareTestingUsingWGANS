# Estratégia Otimizada: Treino com Dados Disponíveis

## 📊 Dados Disponíveis AGORA

```
✅ JUnit (Java):    201 samples
✅ Defects4J:      ~395+ real bugs
✅ OWASP:          100+ test cases
────────────────────────────
TOTAL:            ~700 samples
```

## 🎯 Plano Ajustado (Realista)

### Fase 1: Processamento & Treino com Dados Atuais (2 horas)

```bash
# 1. Processar Defects4J + OWASP
python python/scripts/process_datasets.py --defects4j --owasp
# Tempo: 10-15 min
# Output: ~500 amostras normalizadas

# 2. Treino em Produção (maior epochs com menos dados)
python python/scripts/train_production.py --epochs 150 --patience 25
# Tempo: 20-30 min (com 500 amostras)
# Foco: Qualidade com dataset pequeno

# 3. Gerar Amostras Sintéticas
python python/scripts/generate_large_suite.py --num-samples 5000
# Tempo: 5-10 min
# Output: 5000 sintéticas com boa diversidade

# 4. Análise de Qualidade
python python/scripts/compare_datasets.py
# Tempo: 5-10 min
# Output: Relatório de similaridade
```

**Tempo Fase 1**: ~1.5-2 horas

### Fase 2: Escalar com Juliet (quando disponível)

Três caminhos possíveis:

**A) Download Manual de Juliet**
```bash
# Se conseguires acesso NIST ou outro mirror
# 1. Download & extract para data/raw/juliet/
# 2. Reprocessar com Juliet incluído
# 3. Retreinar com 300+ épocas
# Tempo adicional: 2-3 horas
```

**B) Synthetic Escalation (mais inteligente)**
```bash
# Usar as 5000 sintéticas atuais como dados de treino
# Gerar 50,000 sintéticas de segunda geração
# Criar feedback loop de melhoria
# Tempo: 2-3 horas
```

**C) Hybrid (recomendado)**
```bash
# 1. Fase 1 completa (5000 sintéticas)
# 2. Combinar: 700 reais + 5000 sintéticas = 5700 "treino"
# 3. Retreinar com 5700 amostras (150 épocas)
# 4. Gerar 10,000 sintéticas finais
# Tempo: 2-3 horas
```

---

## 🚀 Recomendação Imediata: Começar Fase 1

### Porque isto é Smart:

1. ✅ **Pronto AGORA** - Não espera por Juliet
2. ✅ **Resultados tangíveis** - 5000 amostras em 2h
3. ✅ **Escalável** - Fase 2 é just a repetição
4. ✅ **Validação** - Prova o pipeline completo
5. ✅ **Iteração** - Refinável com feedback

### Expectativa de Qualidade:

| Dataset | Épocas | Amostras | Quality Score |
|---------|--------|----------|----------------|
| Atual (700) | 150 | 5,000 | ~70-75/100 |
| + Juliet (60k) | 300 | 10,000 | >85/100 |
| + Hybrid (5.7k) | 150 | 10,000 | ~75-80/100 |

---

## 💭 Melhor Caso de Uso

**Se Juliet fica indisponível:**
- Já temos sistema completo e validado
- 5000 sintéticas usáveis para produção
- Pode-se treinar continuamente com novos dados Java

**Se Juliet fica disponível depois:**
- Integra-se sem problema
- Retreina com 60k amostras
- Escala para 10k+ sintéticas

---

## 🔄 Comando Único para Começar

```bash
# Tudo em sequência:
python python/scripts/process_datasets.py --defects4j --owasp && \
python python/scripts/train_production.py --epochs 150 --patience 25 && \
python python/scripts/generate_large_suite.py --num-samples 5000 && \
python python/scripts/compare_datasets.py && \
git add -A && \
git commit -m "Phase 1: Production training with available datasets (700 samples, 5000 synthetic)"
```

---

## ✅ Próximos Passos

1. ✅ Confirmar Defects4J & OWASP completados
2. ✅ Executar pipeline Fase 1
3. ✅ Gerar relatório de qualidade
4. ✅ Commit & validar
5. ⏳ Explorar opções para Juliet (Phase 2)

**Começamos?** 🚀
