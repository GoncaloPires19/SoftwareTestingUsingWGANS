"""
Diagnóstico da WGAN - analisa runs existentes e identifica problemas.
Uso: python python/scripts/diagnose_wgan.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

HIST_PATH = Path("outputs/sf110_mixed_h2_optimized_20260426_110943/training_history.csv")
CFG_PATH  = Path("outputs/sf110_mixed_h2_optimized_20260426_110943/config.json")
SAMPLES_PATH = Path("outputs/sf110_mixed_h2_optimized_20260426_110943/synthetic_samples_best.csv")

DATASET_CSV  = Path("python/data/processed/sf110_full_v2_H2_sf110_plus_junit_strict.csv")
NORM_STATS   = Path("python/data/processed/sf110_full_v2_H2_sf110_plus_junit_strict_norm_stats.json")

SEP = "=" * 70

def load_data():
    df_hist = pd.read_csv(HIST_PATH)
    with open(CFG_PATH) as f:
        cfg = json.load(f)
    df_real = pd.read_csv(DATASET_CSV, header=None)
    df_real.columns = ["methods", "branches", "statements"]
    with open(NORM_STATS) as f:
        ns = json.load(f)
    df_synth = pd.read_csv(SAMPLES_PATH, header=None) if SAMPLES_PATH.exists() else None
    if df_synth is not None:
        df_synth.columns = ["methods", "branches", "statements"]
    return df_hist, cfg, df_real, ns, df_synth


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_issue(tag, msg):
    print(f"  [{'PROBLEMA':^9}] {tag}: {msg}")

def print_ok(tag, msg):
    print(f"  [{'OK':^9}] {tag}: {msg}")

def print_warn(tag, msg):
    print(f"  [{'AVISO':^9}] {tag}: {msg}")


# ── análises ─────────────────────────────────────────────────────────────────

def analyse_dataset(df_real, ns):
    section("1. DATASET – distribuição dos dados reais")
    n = len(df_real)
    print(f"  Amostras: {n}")
    if n < 200:
        print_issue("tamanho", f"apenas {n} amostras — WGAN precisa de pelo menos 200-500 para estabilidade")
    else:
        print_ok("tamanho", f"{n} amostras")

    feat_names = ["methods", "branches", "statements"]
    for i, col in enumerate(feat_names):
        s = df_real[col]
        skew = s.skew()
        kurt = s.kurtosis()
        rng  = s.max() - s.min()
        print(f"\n  {col}:")
        print(f"    min={s.min():.0f}  q50={s.quantile(.5):.1f}  q90={s.quantile(.9):.1f}  q99={s.quantile(.99):.1f}  max={s.max():.0f}")
        print(f"    mean={s.mean():.1f}  std={s.std():.1f}  skew={skew:.2f}  kurtosis={kurt:.2f}")
        if abs(skew) > 2.0:
            print_issue("skewness", f"{col} muito assimétrico (skew={skew:.2f}) → outliers comprimem a normalização [-1,1]")
        if s.max() / (s.mean() + 1e-8) > 20:
            print_warn("outliers", f"max/mean = {s.max()/s.mean():.1f}x — considerar log-transform")


def analyse_training(df_hist, cfg):
    section("2. TREINO – diagnóstico da convergência")

    n_epochs = len(df_hist)
    print(f"  Epochs executados: {n_epochs}  (config: {cfg.get('epochs', '?')})")
    if n_epochs < cfg.get("epochs", 100):
        print_warn("epochs", "treino parou antes do fim — verificar early stopping ou crash")

    # Diversity collapse
    div_last = df_hist["diversity"].iloc[-1]
    div_min  = df_hist["diversity"].min()
    if div_last < 0.80:
        print_issue("diversity", f"diversity no final = {div_last:.3f} < 0.80 → provável mode collapse parcial")
    elif div_last < 0.90:
        print_warn("diversity", f"diversity = {div_last:.3f} — a cair, monitorizar")
    else:
        print_ok("diversity", f"{div_last:.3f}")

    # Tendência de diversity
    d_start = df_hist["diversity"].iloc[:5].mean()
    d_end   = df_hist["diversity"].iloc[-5:].mean()
    if d_end < d_start * 0.85:
        print_issue("diversity_trend", f"caiu de {d_start:.3f} para {d_end:.3f} — modo a colapsar com mais treino")

    # coverage_p99 — deve crescer, não cair
    cov_max_epoch = int(df_hist["coverage_p99"].idxmax()) + 1
    cov_max_val   = df_hist["coverage_p99"].max()
    cov_final     = df_hist["coverage_p99"].iloc[-1]
    if cov_final < cov_max_val * 0.85:
        print_issue("coverage_p99", f"máximo={cov_max_val:.1f}% @ epoch {cov_max_epoch}, final={cov_final:.1f}% — degradou com mais treino (overfitting)")
    else:
        print_ok("coverage_p99", f"final={cov_final:.1f}% (max={cov_max_val:.1f}%)")

    # std_distance_z — quanto o std sintético diverge do real
    std_z_final = df_hist["std_distance_z"].iloc[-1]
    std_z_min   = df_hist["std_distance_z"].min()
    if std_z_final > 0.5:
        print_issue("std_mismatch", f"std_distance_z={std_z_final:.3f} — variância das features sintéticas muito diferente das reais")
    elif std_z_final > 0.3:
        print_warn("std_mismatch", f"std_distance_z={std_z_final:.3f} — aceitável mas melhorável")
    else:
        print_ok("std_mismatch", f"{std_z_final:.3f}")

    # mean_distance_z
    mean_z_final = df_hist["mean_distance_z"].iloc[-1]
    if mean_z_final > 0.3:
        print_warn("mean_mismatch", f"mean_distance_z={mean_z_final:.3f} — média das features afastada da real")
    else:
        print_ok("mean_mismatch", f"{mean_z_final:.3f}")

    # Convergência nas últimas 10 epochs
    last10 = df_hist["robust_v2_score"].tail(10).values
    var10  = float(np.var(last10))
    still_growing = last10[-1] > last10[-5] * 1.02
    if still_growing:
        print_warn("convergencia", f"robust_v2 ainda a subir nas últimas 10 epochs ({last10[-5]:.1f} → {last10[-1]:.1f}) — treino insuficiente")
    elif var10 > 20:
        print_warn("convergencia", f"variância alta nas últimas 10 epochs ({var10:.1f}) — oscilando, instável")
    else:
        print_ok("convergencia", f"estável (var={var10:.1f})")

    # G_loss trend — deve estabilizar, não divergir
    g_last = df_hist["gen_loss"].iloc[-1]
    g_mid  = df_hist["gen_loss"].iloc[len(df_hist)//2]
    if abs(g_last) > abs(g_mid) * 3:
        print_warn("gen_loss", f"generator loss a crescer ({g_mid:.3f} → {g_last:.3f}) — possível instabilidade")


def analyse_architecture():
    section("3. ARQUITECTURA – análise do modelo")
    print("  Generator: z(100) → 256 (LeakyReLU) → 512 (LeakyReLU) → 3 (Tanh)")
    print("  Critic:    3 → 512 (LeakyReLU) → 256 (LeakyReLU) → 1 (Linear)")
    print()
    print_warn("capacidade", "apenas 2 hidden layers — para dados de baixa dimensão (3 features) é suficiente, mas pouco expressivo para caudas longas")
    print_warn("batch_norm", "Batch Norm desativado no Generator (batch_norm_g=True em config mas não implementado em model.py)")
    print_warn("tanh_output", "Tanh obriga output a [-1,1] — se dados têm caudas muito pesadas após normalização, o Tanh corta os extremos")
    print_warn("weight_init", "Sem inicialização explícita de pesos (Xavier/He) — pode atrasar convergência")


def analyse_synthetic_vs_real(df_real, df_synth):
    if df_synth is None:
        print("\n  [AVISO] synthetic_samples_best.csv não encontrado — saltar comparação")
        return
    section("4. SINTÉTICOS vs REAIS – qualidade das amostras geradas")
    feat_names = ["methods", "branches", "statements"]
    for col in feat_names:
        r = df_real[col]
        s = df_synth[col]
        mean_err = abs(s.mean() - r.mean()) / (r.mean() + 1e-8) * 100
        std_err  = abs(s.std()  - r.std())  / (r.std()  + 1e-8) * 100
        q99_err  = abs(s.quantile(.99) - r.quantile(.99)) / (r.quantile(.99) + 1e-8) * 100
        max_cover = min(s.max() / (r.max() + 1e-8), 1.0) * 100

        print(f"\n  {col}:")
        print(f"    REAL:  mean={r.mean():.1f}  std={r.std():.1f}  q99={r.quantile(.99):.1f}  max={r.max():.0f}")
        print(f"    SYNTH: mean={s.mean():.1f}  std={s.std():.1f}  q99={s.quantile(.99):.1f}  max={s.max():.0f}")
        print(f"    Δmean={mean_err:.1f}%  Δstd={std_err:.1f}%  Δq99={q99_err:.1f}%  max_coverage={max_cover:.1f}%")

        if mean_err > 20:
            print_issue("mean_drift", f"{col}: média sintética desvia {mean_err:.1f}% da real")
        if std_err > 30:
            print_issue("std_drift", f"{col}: std sintético desvia {std_err:.1f}% do real — falta de variância (mode collapse parcial)")
        if max_cover < 70:
            print_issue("max_coverage", f"{col}: sintéticos só chegam a {max_cover:.1f}% do máximo real — não geram casos extremos")


def suggest_improvements(df_hist, df_real, df_synth):
    section("5. SUGESTÕES DE MELHORIA")

    n = len(df_real)
    div_last = df_hist["diversity"].iloc[-1]
    std_z = df_hist["std_distance_z"].iloc[-1]
    cov_max = df_hist["coverage_p99"].max()
    cov_final = df_hist["coverage_p99"].iloc[-1]
    still_growing = df_hist["robust_v2_score"].iloc[-1] > df_hist["robust_v2_score"].iloc[-5] * 1.02

    suggestions = []

    if still_growing:
        suggestions.append((
            "A) Mais epochs",
            "robust_v2 ainda a crescer — aumentar de 50 para 150-200 epochs",
            "  → train_basic_profiles.py --profile optimized --epochs 200"
        ))

    if div_last < 0.85:
        suggestions.append((
            "B) Combater mode collapse",
            f"diversity={div_last:.3f} — o generator está a produzir amostras semelhantes",
            "  → Aumentar n_critic (10→15), reduzir lambda_gp (10→7), usar minibatch std (feature no Critic)"
        ))

    if std_z > 0.5:
        suggestions.append((
            "C) Corrigir variância das features",
            f"std_distance_z={std_z:.3f} — sintéticos têm variância muito diferente dos reais",
            "  → Adicionar feature matching loss: penalizar |std(fake) - std(real)| durante treino do G"
        ))

    if cov_final < cov_max * 0.85:
        suggestions.append((
            "D) Evitar regressão com mais treino",
            f"coverage_p99 degradou de {cov_max:.1f}% para {cov_final:.1f}% — overfitting do Critic",
            "  → Usar o modelo salvo no melhor epoch (robust_v2), não o final; ou adicionar weight decay no Critic"
        ))

    if df_real is not None and df_real["statements"].skew() > 2:
        suggestions.append((
            "E) Log-transform dos dados",
            f"statements skew={df_real['statements'].skew():.2f} — distribuição muito assimétrica",
            "  → Aplicar log1p antes de normalizar: reduz o espaço para o G aprender, melhora a cauda"
        ))

    if n < 300:
        suggestions.append((
            "F) Mais dados de treino",
            f"apenas {n} amostras — dataset pequeno causa instabilidade",
            "  → Usar todos os resultados SF110 (111 projetos × N testes) como dados de treino"
        ))

    suggestions.append((
        "G) Weight initialization",
        "Adicionar inicialização He/Xavier ao Generator e Critic",
        "  → Em model.py: apply(weights_init) após __init__"
    ))

    suggestions.append((
        "H) Spectral Normalization no Critic",
        "Alternativa ao gradient penalty — mais estável para datasets pequenos",
        "  → torch.nn.utils.spectral_norm() em cada Linear do Critic"
    ))

    for letter, problem, fix in suggestions:
        print(f"\n  {letter}")
        print(f"    Problema: {problem}")
        print(f"    Fix:      {fix}")

    print(f"\n\n{'='*70}")
    print("  PRIORIDADE RECOMENDADA:")
    print("  1. [A] Mais epochs (50→200) — impacto imediato, custo zero")
    print("  2. [E] Log-transform — melhora qualitativamente a aprendizagem das caudas")
    print("  3. [C] Feature matching loss — força o G a replicar média+std")
    print("  4. [B] Anti-mode collapse: n_critic=15, minibatch std")
    print("  5. [D] Usar melhor checkpoint, não o final")
    print(f"{'='*70}\n")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("  WGAN DIAGNÓSTICO — análise da run sf110_mixed_h2_optimized_20260426")
    print(SEP)

    df_hist, cfg, df_real, ns, df_synth = load_data()

    analyse_dataset(df_real, ns)
    analyse_training(df_hist, cfg)
    analyse_architecture()
    analyse_synthetic_vs_real(df_real, df_synth)
    suggest_improvements(df_hist, df_real, df_synth)
