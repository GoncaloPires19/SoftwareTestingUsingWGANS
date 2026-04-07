#!/usr/bin/env python3
"""Monitor WGAN training progress in real-time."""

import argparse
import time
from pathlib import Path

import pandas as pd


HISTORY_FILE_NAME = "training_history.csv"


def _resolve_output_dir(explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir)

    outputs_root = Path("outputs")
    if not outputs_root.exists():
        raise FileNotFoundError("outputs directory does not exist")

    candidate_dirs = [
        d for d in outputs_root.iterdir()
        if d.is_dir() and (d / HISTORY_FILE_NAME).exists()
    ]
    if not candidate_dirs:
        raise FileNotFoundError(f"No run directory with {HISTORY_FILE_NAME} found under outputs/")

    return max(candidate_dirs, key=lambda p: (p / HISTORY_FILE_NAME).stat().st_mtime)


def _print_latest_metrics(latest: pd.Series, expected_epochs: int | None) -> int:
    epoch = int(latest.get("epoch", 0))

    print(f"\n📊 Epoch {epoch}", end="")
    if expected_epochs:
        print(f"/{expected_epochs}", end="")
    print(f"  | G: {latest.get('gen_loss', float('nan')):7.4f} | C: {latest.get('critic_loss', float('nan')):7.4f}")
    print(
        f"   Coverage: {latest.get('coverage', float('nan')):.2f}% | "
        f"Robust V2: {latest.get('robust_v2_score', float('nan')):.2f}"
    )
    print(
        f"   Diversity: {latest.get('diversity', float('nan')):.3f} | "
        f"Tail penalty: {latest.get('tail_excess_penalty', float('nan')):.4f}"
    )

    return epoch


def _should_stop(expected_epochs: int | None, epoch: int, once: bool) -> bool:
    if once:
        return True
    if expected_epochs and epoch >= expected_epochs:
        print("\n✅ Training reached expected epoch target.")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor training progress from training_history.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Run output directory. If omitted, the latest run under outputs/ is used")
    parser.add_argument("--expected-epochs", type=int, default=None,
                        help="Optional stop condition when current epoch reaches this value")
    parser.add_argument("--interval-seconds", type=int, default=30,
                        help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true",
                        help="Print latest status once and exit")

    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    history_path = output_dir / HISTORY_FILE_NAME

    print("⏱️  Real-time Training Monitor")
    print("=" * 70)
    print(f"Run directory: {output_dir}")

    while True:
        epoch = 0
        if history_path.exists():
            df = pd.read_csv(history_path)
            if not df.empty:
                epoch = _print_latest_metrics(df.iloc[-1], args.expected_epochs)

        if _should_stop(args.expected_epochs, epoch, args.once):
            break

        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    main()
