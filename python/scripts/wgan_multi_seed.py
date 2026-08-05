#!/usr/bin/env python3
"""
WGAN Multi-Seed Robustness Study.

Executa `train_optimized.py` para 3 seeds x 2 batch sizes (32 e 64), agrega as
metricas de fidelidade (W1, diversity, composite score) e escolhe a melhor
configuracao. Reporta media e desvio padrao para provar estabilidade.

Uso:
    python python/scripts/wgan_multi_seed.py --epochs 150 --output-root outputs/wgan_multi_seed
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def _run_one(
    output_dir: Path, data_path: str, epochs: int, batch_size: int,
    seed: int, fm_weight: float, learning_rate: float,
) -> dict:
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "train_optimized.py"),
        "--data", data_path,
        "--output-dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--fm-weight", str(fm_weight),
        "--learning-rate", str(learning_rate),
        "--seed", str(seed),
        "--device", "cpu",
    ]
    print(f"\n[RUN] batch={batch_size} seed={seed} -> {output_dir.name}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  FALHA (exit={proc.returncode}): {proc.stderr[-400:]}")
        return {"failed": True}

    hist_path = output_dir / "training_history.csv"
    if not hist_path.exists():
        return {"failed": True}

    rows = list(csv.DictReader(open(hist_path)))
    if not rows:
        return {"failed": True}

    best = max(rows, key=lambda r: float(r["composite_score"]))
    final = rows[-1]
    return {
        "failed": False,
        "epochs_run": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_composite": float(best["composite_score"]),
        "best_diversity": float(best["diversity_score"]),
        "best_w1": float(best["wasserstein_distance"]),
        "best_gen_loss": float(best["gen_loss"]),
        "best_critic_loss": float(best["critic_loss"]),
        "best_gp": float(best["gp_loss"]),
        "best_fm": float(best["feature_matching_loss"]),
        "final_composite": float(final["composite_score"]),
        "final_diversity": float(final["diversity_score"]),
        "final_w1": float(final["wasserstein_distance"]),
    }


def _agg(vals: list[float]) -> tuple[float, float]:
    return (mean(vals), pstdev(vals)) if len(vals) > 1 else (vals[0], 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="WGAN multi-seed robustness study")
    parser.add_argument("--data", type=str, default="python/data/junit_data_cleaned_strict.csv")
    parser.add_argument("--output-root", type=str, default="outputs/wgan_multi_seed")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--fm-weight", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64])
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    results: dict[int, list[dict]] = defaultdict(list)
    for batch_size in args.batch_sizes:
        for seed in args.seeds:
            run_dir = root / f"bs{batch_size}_seed{seed}"
            info = _run_one(
                run_dir, args.data, args.epochs, batch_size, seed,
                args.fm_weight, args.learning_rate,
            )
            info.update({"batch_size": batch_size, "seed": seed, "run_dir": str(run_dir)})
            results[batch_size].append(info)

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("  MULTI-SEED SUMMARY")
    print("=" * 78)
    print(f"\n  Data: {args.data}")
    print(f"  Epochs cap: {args.epochs}  fm_weight={args.fm_weight}  lr={args.learning_rate}")
    print(f"  Seeds: {args.seeds}\n")

    summary = {}
    for batch_size in args.batch_sizes:
        rows = [r for r in results[batch_size] if not r.get("failed")]
        if not rows:
            continue
        composite = [r["best_composite"] for r in rows]
        div = [r["best_diversity"] for r in rows]
        w1 = [r["best_w1"] for r in rows]

        m_comp, s_comp = _agg(composite)
        m_div, s_div = _agg(div)
        m_w1, s_w1 = _agg(w1)

        print(f"  batch_size={batch_size} (n={len(rows)}):")
        print(f"    composite: {m_comp:.4f} +/- {s_comp:.4f}")
        print(f"    diversity: {m_div:.4f} +/- {s_div:.4f}")
        print(f"    W1:        {m_w1:.4f} +/- {s_w1:.4f}")
        print(f"    seeds:     " + ", ".join(f"s{r['seed']}:c{r['best_composite']:.3f}" for r in rows))
        print()

        summary[str(batch_size)] = {
            "n_seeds": len(rows),
            "composite_mean": m_comp, "composite_std": s_comp,
            "diversity_mean": m_div, "diversity_std": s_div,
            "w1_mean": m_w1, "w1_std": s_w1,
            "individual_runs": rows,
        }

    # ---- Pick best config ----
    if summary:
        best_bs = max(summary, key=lambda k: summary[k]["composite_mean"])
        print(f"  >>> MELHOR CONFIGURACAO: batch_size={best_bs}")
        print(f"      composite={summary[best_bs]['composite_mean']:.4f} +/- {summary[best_bs]['composite_std']:.4f}")

        best_run = max(summary[best_bs]["individual_runs"], key=lambda r: r["best_composite"])
        print(f"      Melhor run individual: {best_run['run_dir']} (seed {best_run['seed']}, composite {best_run['best_composite']:.4f})")

    # ---- Save summary ----
    with open(root / "multi_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Sumario guardado em: {root / 'multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
