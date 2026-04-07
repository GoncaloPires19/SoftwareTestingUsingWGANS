#  Estrutura Redefinida do Projeto

## Visão Geral da Hierarquia Atual (Pós-Otimização)

```
SoftwareTestingUsingWGANS/
│
├── .github/                   # Configuração do GitHub
├── docs/                      # Documentação
│   ├── ADVANCED_TRAINING_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── DATA_SCHEMA.md
│   └── QUICK_START.md
│
├── java/  (struct. virtual)   # Lógica Java (via src/ + pom.xml)
│   ├── pom.xml                # Build config Maven
│   └── src/                   # Testes unitários & extração de métricas
│
├── python/                    # Pipeline WGAN-GP
│   ├── src/                   # Módulos core
│   │   ├── model.py           # WGAN Generator/Critic
│   │   ├── junit_datasets.py  # Data loading & normalization
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── utils.py           # Utilities & gradient penalty
│   │   ├── config.py          # Hyperparameters
│   │   ├── evaluation.py      # Validation logic
│   │   └── data_validation.py # Data checks
│   │
│   ├── scripts/               # Pipelines executáveis
│   │   ├── train_optimized.py
│   │   ├── train_basic_profiles.py
│   │   ├── train_continuation.py
│   │   ├── run_sf110_hypotheses.py
│   │   ├── generate_synthetic.py
│   │   ├── experimental_analysis.py
│   │   ├── run_unseen_hypothesis_experiments.py
│   │   ├── prepare_sf110_dataset.py
│   │   ├── validate_before_training.py
│   │   │
│   │   └── tools/             # ⭐ Novo: Utilitários de suporte
│   │       ├── cleanup.py         # Limpeza de experimentos
│   │       └── monitor_training.py # Monitor real-time
│   │
│   ├── tests/                 # Testes pytest (se existentes)
│   ├── notebooks/             # Jupyter exploração (opcional)
│   └── data/                  # Dependência local
│       ├── raw/
│       └── processed/
│
├── data/                      # Potencial consolidação futura
│   ├── raw/
│   └── processed/
│
├── outputs/                   # Resultados de execuções
│   ├── sf110_hypotheses/      # Batch de H1/H2/H3 experiments
│   ├── experimental/          # Análises comparativas
│   ├── consistency_check/     # Validação de consistência
│   └── FINAL_RESULTS_TABLE.md # Tabela consolidada de resultados
│
├── models/                    # Pesos treinados (pode estar vazio)
│
├── logs/                      # ⭐ Novo: Centralizar logs aqui
│   └── (criada no próximo passo)
│
├── pom.xml                    # Build Maven parent
├── requirements.txt           #  Atualizado com scipy
├── .gitignore                 #  Melhorado
├── README.md                  # Este ficheiro
├── REFACTORING_LOG.md         # Log de mudanças (novo)
└── .venv/                     # Virtualenv (ignorado)
```

---

## 🔄 Mudanças Realizadas

###  Automáticas (Fase 1 Completa)

1. **Limpeza da Raiz**
   - Removidos 6 ficheiros de log temporários
   - A raiz agora contém apenas ficheiros essenciais

2. **Gestão de Dependências**
   -  Adicionado `scipy>=1.9.0` ao `requirements.txt`
   -  Atualizado `.gitignore` com regras `*.err`, `*.out`, `*.txt`

3. **Organização de Utilitários**
   -  `cleanup.py` → `python/scripts/tools/`
   -  `monitor_training.py` → `python/scripts/tools/`
   - Estes scripts continuam funcionando normalmente quando executados a partir da raiz

---

##  Como Usar a Estrutura Refatorada

### Instalar Dependências
```bash
pip install -r requirements.txt
```

### Executar Treino
```bash
# Exemplo: H1/H2/H3 com dados SF110
python python/scripts/run_sf110_hypotheses.py --hypotheses H1 H2 H3 --epochs 150

# Com arquivo explícito
python python/scripts/run_sf110_hypotheses.py --hypotheses H1 --profile generalization-stress
```

### Usar Ferramentas de Suporte
```bash
# Monitorizar treino em tempo real
python python/scripts/tools/monitor_training.py

# Limpar experimentos completados
python python/scripts/tools/cleanup.py
```

### Executar Testes Java
```bash
cd .
mvn clean test
```

---

## 📋 Próximas Otimizações Recomendadas (Opcional)

Embora a Fase 1 esteja completa e segura, as seguintes melhorias podem ser feitas se desejado:

### Fase 2 (Complexidade: Alta)
- [ ] Consolidar `python/data/` com `data/` na raiz
- [ ] Centralizar todos os outputs em `outputs/`
- [ ] Criar `scripts/` na raiz para bash/bats genéricos

### Fase 3 (Complexidade: Baixa)
- [ ] Criar arquivo para exps antigas em `outputs/archive/`
- [ ] Documentar pipelines no README principal
- [ ] Adicionar CI/CD template

---

##  Configuração Segura para Consolidação Futura

Se decidir fazer a consolidação de dados, siga este procedimento:

1. **Criar branch provisória:**
   ```bash
   git checkout -b refactor/consolidate-structure
   ```

2. **Executar script de teste:**
   ```bash
   python python/scripts/validate_before_training.py
   ```

3. **Validar que tudo funciona:**
   ```bash
   mvn clean test  # Java tests
   python -m pytest python/tests/  # Python tests (if any)
   ```

4. **Merge apenas após validação completa**

---

## 📞 Notas Técnicas

- **Compatibilidade:** Todos os scripts continuam funcionáveis a partir da raiz
- **Git**: Ficheiros de log agora ignorados corretamente
- **Dependências**: Nenhuma funcionalidade foi removida
- **Reprodutibilidade**: Ambiente Python agora completo com scipy

---

**Última Atualização:** 2026-03-23
**Estado:** Fase 1 Concluída 
