# Log de Refatoração e Otimização do Repositório
**Data:** 2026-03-23
**Objetivos:** Consolidar estrutura, remover redundâncias, melhorar organização

## ✅ Fase 1: Limpeza e Consolidação de Dependências (Concluída)

### 1.1 Ficheiros Removidos
Os seguintes ficheiros de log/erro temporários foram removidos da raiz:
- ❌ `err.txt`
- ❌ `err_final.txt`
- ❌ `h1_large.err`
- ❌ `logs_final.txt`
- ❌ `log_full.txt`
- ❌ `run_output.txt`

**Razão:** Ficheiros gerados durante execução de scripts, não devem estar na raiz nem em controlo de versão.

### 1.2 Ficheiros Movidos
- ✅ `cleanup.py` → `python/scripts/tools/cleanup.py`
- ✅ `monitor_training.py` → `python/scripts/tools/monitor_training.py`

**Razão:** Utilitários de suporte e monitorização devem estar organizados numa pasta dedicada.

### 1.3 Atualização do `.gitignore`
Adicionadas regras para evitar tracking de ficheiros temporários:
```gitignore
*.err
*.out
err_*.txt
log_*.txt
run_output.txt
training_out.txt
```

### 1.4 Atualização do `requirements.txt`
Adicionada dependência em falta:
- ✅ `scipy>=1.9.0` (era importada mas não declarada)

---

## 📋 Fase 2: Preparação para Refatoração de Estrutura (Próximas Etapas)

### 2.1 Estrutura Proposta (Não Implementada)
A refatoração de dados e outputs será mais complexa e requer:

#### Opção A: Consolidação Total (Recomendada)
```
SoftwareTestingUsingWGANS/
├── data/                  # ÚNICA fonte de dados
│   ├── raw/              # Dados orgânicos (ignorado em git)
│   └── processed/        # Datasets limpos (ignorado em git)
├── outputs/              # Resultados únicos
│   ├── current/          # Experiências ativas
│   └── archive/          # Histórico
└── logs/                 # Novo: direcionar todos os logs aqui
```

#### Opção B: Manter Híbrida (Compatibilidade Máxima)
Manter `python/data/` e `python/models/` como estão, apenas reduzindo lixo.

### 2.2 Impacto de Imports Hardcoded
Os seguintes ficheiros contêm caminhos hardcoded e precisariam de atualização:
- `python/scripts/run_sf110_hypotheses.py`
- `python/scripts/train_basic_profiles.py`
- `python/scripts/generate_synthetic.py`
- `python/scripts/experimental_analysis.py`
- `python/config.py`
- E mais 8+ ficheiros

---

## 🎯 Próximas Recomendações (Opcional)

1. **Consolidação de Dados** (Complexidade: Alta)
   - Requer atualização de ~15+ ficheiros Python
   - Teste abrangente necessário
   - Tempo estimado: 2-3 horas

2. **Criação de `scripts/` na Raiz** (Complexidade: Baixa)
   - Mover `run_all.bat` para `scripts/run_all.ps1`
   - Documentar como executar os pipelines

3. **Reorganização de `/docs`** (Complexidade: Baixa)
   - Consolidar documentação técnica
   - Criar índice central

4. **Automação de Testes** (Complexidade: Média)
   - Criar script de validação pós-refactoring
   - Confirmar que `mvn test` funciona
   - Confirmar que `python scripts/...` funcionam

---

## ⚠️ Notas de Cautela

- **Não foram removidas pastas duplicadas** de `python/scripts/outputs/`, `python/scripts/logs/`, `python/scripts/models/` porque são usadas ativamente pelos scripts
- **A consolidação de dados** poderia quebrar o fluxo de trabalho atual se não forem atualizados todos os imports
- **Recomendação:** Criar branch separada (`refactor/consolidate-structure`) para testes antes de merge

---

## 📊 Estado Atual do Repositório (Após Fase 1)
- ✅ Raiz limpa de ficheiros temporários
- ✅ Dependências corretas em `requirements.txt`
- ✅ `.gitignore` melhorado com regras de segurança
- ✅ Utilitários organizados em `python/scripts/tools/`
- ⏳ Estrutura de dados/outputs ainda duplicada (estratégia híbrida)
