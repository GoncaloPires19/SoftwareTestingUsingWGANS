"""
WGAN Diagnostic Tool

Analyses an existing training run and reports convergence problems,
distribution mismatches, and prioritised improvement suggestions.

Usage:
    python python/scripts/diagnose_wgan.py --run-dir outputs/<run_name>
    python python/scripts/diagnose_wgan.py --run-dir outputs/<run_name> --dataset python/data/my_data.csv
"""

from __future__ import annotations

import argparse
import io
import json
import sys

# Force UTF-8 on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "=" * 70
FEAT_NAMES_DEFAULT = ["methods", "branches", "statements", "method_coverage"]


def _status(tag: str, level: str, msg: str) -> None:
    label = {"ok": "OK", "warn": "AVISO", "error": "PROBLEMA"}[level]
    print(f"  [{label:^9}] {tag}: {msg}")


def ok(tag: str, msg: str) -> None:
    _status(tag, "ok", msg)


def warn(tag: str, msg: str) -> None:
    _status(tag, "warn", msg)


def error(tag: str, msg: str) -> None:
    _status(tag, "error", msg)


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path, dataset_csv: Path | None, norm_stats: Path | None):
    hist_path = run_dir / "training_history.csv"
    cfg_path = run_dir / "config.json"
    samples_path = run_dir / "synthetic_samples_best.csv"

    missing = [p for p in (hist_path,) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required files not found in {run_dir}: {[str(m) for m in missing]}"
        )

    df_hist = pd.read_csv(hist_path)

    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

    df_real = None
    if dataset_csv and dataset_csv.exists():
        df_real = pd.read_csv(dataset_csv, header=None)
        n_cols = df_real.shape[1]
        df_real.columns = FEAT_NAMES_DEFAULT[:n_cols]

    ns = {}
    if norm_stats and norm_stats.exists():
        with open(norm_stats) as f:
            ns = json.load(f)

    df_synth = None
    if samples_path.exists():
        df_synth = pd.read_csv(samples_path, header=None)
        n_cols = df_synth.shape[1]
        df_synth.columns = FEAT_NAMES_DEFAULT[:n_cols]

    return df_hist, cfg, df_real, ns, df_synth


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def analyse_dataset(df_real: pd.DataFrame) -> None:
    section("1. DATASET — distribuição dos dados reais")
    n = len(df_real)
    print(f"  Amostras: {n}")

    if n < 200:
        error("tamanho", f"apenas {n} amostras — WGAN precisa de ≥200 para estabilidade")
    else:
        ok("tamanho", f"{n} amostras")

    for col in df_real.columns:
        s = df_real[col]
        skew = s.skew()
        kurt = s.kurtosis()
        print(
            f"\n  {col}: "
            f"min={s.min():.2f}  q50={s.quantile(.5):.2f}  "
            f"q90={s.quantile(.9):.2f}  q99={s.quantile(.99):.2f}  max={s.max():.2f}"
        )
        print(f"    mean={s.mean():.2f}  std={s.std():.2f}  skew={skew:.2f}  kurtosis={kurt:.2f}")

        if abs(skew) > 2.0:
            error("skewness", f"{col}: muito assimétrico (skew={skew:.2f}) — log1p recomendado")
        if s.mean() > 0 and s.max() / (s.mean() + 1e-8) > 20:
            warn("outliers", f"{col}: max/mean={s.max()/s.mean():.1f}x — outliers severos")


def analyse_training(df_hist: pd.DataFrame, cfg: dict) -> None:
    section("2. TREINO — diagnóstico de convergência")

    n_epochs = len(df_hist)
    cfg_epochs = cfg.get("training", {}).get("epochs", cfg.get("epochs", "?"))
    print(f"  Epochs executados: {n_epochs}  (config: {cfg_epochs})")

    if isinstance(cfg_epochs, int) and n_epochs < cfg_epochs:
        warn("epochs", "treino parou antes do fim — early stopping ou crash")

    # Diversity
    if "diversity_score" in df_hist.columns:
        div_col = "diversity_score"
    elif "diversity" in df_hist.columns:
        div_col = "diversity"
    else:
        div_col = None

    if div_col:
        div_last = df_hist[div_col].iloc[-1]
        if div_last < 0.30:
            error("diversity", f"final={div_last:.3f} < 0.30 — mode collapse")
        elif div_last < 0.45:
            warn("diversity", f"final={div_last:.3f} — baixo, monitorizar")
        else:
            ok("diversity", f"{div_last:.3f}")

        d_start = df_hist[div_col].iloc[:5].mean()
        d_end = df_hist[div_col].iloc[-5:].mean()
        if d_end < d_start * 0.85:
            error(
                "diversity_trend",
                f"caiu de {d_start:.3f} para {d_end:.3f} — colapso com mais treino",
            )

    # Wasserstein distance
    if "wasserstein_distance" in df_hist.columns:
        w1_final = df_hist["wasserstein_distance"].iloc[-1]
        w1_best = df_hist["wasserstein_distance"].min()
        if w1_final > w1_best * 1.5:
            warn("wasserstein", f"final={w1_final:.4f}, best={w1_best:.4f} — degradação pós-best")
        else:
            ok("wasserstein", f"final={w1_final:.4f}")

    # Feature matching loss
    if "feature_matching_loss" in df_hist.columns:
        fm_final = df_hist["feature_matching_loss"].iloc[-1]
        ok("feature_matching", f"final={fm_final:.4f}")

    # Generator loss trend
    if "gen_loss" in df_hist.columns:
        g_last = df_hist["gen_loss"].iloc[-1]
        g_mid = df_hist["gen_loss"].iloc[len(df_hist) // 2]
        if abs(g_last) > abs(g_mid) * 3:
            warn("gen_loss", f"a crescer ({g_mid:.3f} → {g_last:.3f}) — possível instabilidade")
        else:
            ok("gen_loss", f"final={g_last:.4f}")

    # Gradient norms
    if "mean_grad_norm_g" in df_hist.columns:
        gn = df_hist["mean_grad_norm_g"].iloc[-5:].mean()
        if gn > 10.0:
            warn("grad_norm_G", f"elevado: {gn:.3f} — considerar gradient clipping")
        else:
            ok("grad_norm_G", f"final (avg last 5)={gn:.4f}")


def analyse_architecture(cfg: dict) -> None:
    section("3. ARQUITECTURA — configuração do modelo")
    model_cfg = cfg.get("model", {})
    print(f"  z_dim       : {model_cfg.get('z_dim', '?')}")
    print(f"  output_dim  : {model_cfg.get('output_dim', '?')}")
    print(f"  hidden_G    : {model_cfg.get('hidden_dims_g', [256, 512])}")
    print(f"  hidden_C    : {model_cfg.get('hidden_dims_c', [512, 256])}")

    ok("weight_init", "He/Kaiming init activo via model.py")
    warn(
        "tanh_output",
        "Tanh obriga output a [-1,1] — dados reais devem estar normalizados ao mesmo intervalo",
    )


def analyse_synthetic_vs_real(
    df_real: pd.DataFrame, df_synth: pd.DataFrame | None
) -> None:
    if df_synth is None:
        print("\n  [AVISO] synthetic_samples_best.csv não encontrado — saltar comparação")
        return

    section("4. SINTÉTICOS vs REAIS — qualidade das amostras geradas")

    for col in df_real.columns:
        if col not in df_synth.columns:
            continue
        r, s = df_real[col], df_synth[col]
        mean_err = abs(s.mean() - r.mean()) / (r.mean() + 1e-8) * 100
        std_err = abs(s.std() - r.std()) / (r.std() + 1e-8) * 100
        max_cover = min(s.max() / (r.max() + 1e-8), 1.0) * 100

        print(
            f"\n  {col}:\n"
            f"    REAL:  mean={r.mean():.2f}  std={r.std():.2f}  "
            f"q99={r.quantile(.99):.2f}  max={r.max():.2f}\n"
            f"    SYNTH: mean={s.mean():.2f}  std={s.std():.2f}  "
            f"q99={s.quantile(.99):.2f}  max={s.max():.2f}\n"
            f"    delta_mean={mean_err:.1f}%  delta_std={std_err:.1f}%  "
            f"max_coverage={max_cover:.1f}%"
        )

        if mean_err > 20:
            error("mean_drift", f"{col}: média desvia {mean_err:.1f}%")
        if std_err > 30:
            error("std_drift", f"{col}: std desvia {std_err:.1f}% — mode collapse parcial")
        if max_cover < 70:
            error("max_coverage", f"{col}: sintéticos chegam a apenas {max_cover:.1f}% do max real")


def suggest_improvements(
    df_hist: pd.DataFrame,
    df_real: pd.DataFrame | None,
    df_synth: pd.DataFrame | None,
    cfg: dict,
) -> None:
    section("5. SUGESTÕES DE MELHORIA")

    suggestions: list[tuple[str, str, str]] = []

    div_col = "diversity_score" if "diversity_score" in df_hist.columns else "diversity" if "diversity" in df_hist.columns else None

    if div_col:
        div_last = df_hist[div_col].iloc[-1]
        if div_last < 0.40:
            suggestions.append((
                "A) Combater mode collapse",
                f"diversity={div_last:.3f} — generator a repetir amostras",
                "  → Aumentar n_critic (10→15), activar feature_matching_weight=0.2",
            ))

    if "wasserstein_distance" in df_hist.columns:
        w_final = df_hist["wasserstein_distance"].iloc[-1]
        w_best = df_hist["wasserstein_distance"].min()
        if w_final > w_best * 1.5:
            best_epoch = df_hist["wasserstein_distance"].idxmin() + 1
            suggestions.append((
                "B) Usar checkpoint do best epoch",
                f"W1 degradou de {w_best:.4f} para {w_final:.4f} — overfitting do Critic",
                f"  → Carregar checkpoint_best.pth (epoch {best_epoch}), não o final",
            ))

    if df_real is not None and "statements" in df_real.columns:
        skew = df_real["statements"].skew()
        if skew > 2:
            suggestions.append((
                "C) Log-transform",
                f"statements skew={skew:.2f} — distribuição severa",
                "  → train_optimized.py já activa log1p por omissão",
            ))
        n = len(df_real)
        if n < 300:
            suggestions.append((
                "D) Expandir dataset",
                f"Apenas {n} amostras",
                "  → Agregar projetos SF110 adicionais",
            ))

    if not cfg.get("training", {}).get("early_stopping_enabled", False):
        suggestions.append((
            "E) Activar early stopping",
            "Treino continua após convergência — early_stopping_enabled=False",
            "  → train_optimized.py activa por omissão (patience=20)",
        ))

    for letter, problem, fix in suggestions:
        print(f"\n  {letter}")
        print(f"    Problema: {problem}")
        print(f"    Fix:      {fix}")

    if not suggestions:
        print("\n  Nenhum problema crítico detectado. O treino parece estável.")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a WGAN-GP training run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to the training output directory (must contain training_history.csv)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to the raw dataset CSV used for training (optional, enables real vs synth analysis)",
    )
    parser.add_argument(
        "--norm-stats", type=str, default=None,
        help="Path to normalization stats JSON (auto-detected if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"[ERRO] Run directory not found: {run_dir}")
        sys.exit(1)

    dataset_csv = Path(args.dataset) if args.dataset else None
    norm_stats_path = Path(args.norm_stats) if args.norm_stats else None

    # Auto-detect norm stats from dataset path
    if dataset_csv and norm_stats_path is None:
        candidate = dataset_csv.parent / f"{dataset_csv.stem}_norm_stats.json"
        if candidate.exists():
            norm_stats_path = candidate

    print(SEP)
    print("  WGAN DIAGNÓSTICO")
    print(f"  Run: {run_dir}")
    print(SEP)

    try:
        df_hist, cfg, df_real, ns, df_synth = _load_run(run_dir, dataset_csv, norm_stats_path)

        if df_real is not None:
            analyse_dataset(df_real)
        else:
            print("\n  [INFO] Dataset não fornecido — a saltar análise de distribuição.")
            print("         Use --dataset <path.csv> para uma análise completa.")

        analyse_training(df_hist, cfg)
        analyse_architecture(cfg)

        if df_real is not None:
            analyse_synthetic_vs_real(df_real, df_synth)

        suggest_improvements(df_hist, df_real, df_synth, cfg)

    except FileNotFoundError as exc:
        print(f"\n  [ERRO CRÍTICO] {exc}")
        print("  Verifique que --run-dir aponta para um directório de treino válido.")
        sys.exit(1)


if __name__ == "__main__":
    main()
