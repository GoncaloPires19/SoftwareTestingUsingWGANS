#!/usr/bin/env python3
"""
Extract Line Coverage from the SF110 PIT HTML reports (data already on disk).

PIT's `pit-reports/<project>/index.html` has a "Project Summary" table with
Number of Classes, Line Coverage, Mutation Coverage and Test Strength (each as
"X% covered/total"). We already use Mutation Coverage / Test Strength from the
CSV export; Line Coverage was never aggregated. This just parses the existing
HTML reports — no PIT re-run needed.

Note: PIT does not report branch coverage or method coverage by default (only
line coverage + mutation coverage/test strength) — those would need a
separate JaCoCo run.

Usage:
    python python/scripts/extract_pit_line_coverage.py \
        --workdir outputs/mutation_experiments/sf110_full111/workdir \
        --out outputs/mutation_experiments/sf110_line_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ROW_RE = re.compile(
    r"<h3>Project Summary</h3>.*?<tbody>\s*<tr>\s*"
    r"<td>(\d+)</td>\s*"
    r'<td>\d+% <div class="coverage_bar">.*?<div class="coverage_legend">(\d+)/(\d+)</div>.*?</td>\s*'
    r'<td>\d+% <div class="coverage_bar">.*?<div class="coverage_legend">(\d+)/(\d+)</div>.*?</td>\s*'
    r'<td>\d+% <div class="coverage_bar">.*?<div class="coverage_legend">(\d+)/(\d+)</div>.*?</td>',
    re.DOTALL,
)

SEP = "=" * 72


def _parse_report(path: Path) -> dict | None:
    html = path.read_text(encoding="utf-8", errors="replace")
    m = _ROW_RE.search(html)
    if not m:
        return None
    n_classes, line_cov, line_tot, mut_cov, mut_tot, ts_cov, ts_tot = (int(g) for g in m.groups())
    return {
        "n_classes": n_classes,
        "line_covered": line_cov, "line_total": line_tot,
        "mutation_covered": mut_cov, "mutation_total": mut_tot,
        "strength_covered": ts_cov, "strength_total": ts_tot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Line Coverage from existing SF110 PIT HTML reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workdir", type=str, default="outputs/mutation_experiments/sf110_full111/workdir")
    parser.add_argument("--out", type=str, default="outputs/mutation_experiments/sf110_line_coverage.csv")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    reports = sorted(workdir.glob("*/pit-reports/index.html"))
    print(f"Relatorios PIT encontrados: {len(reports)}")

    rows = []
    for report in reports:
        project = report.parent.parent.name
        parsed = _parse_report(report)
        if parsed is None:
            print(f"  [AVISO] Nao consegui extrair Project Summary de {project}")
            continue
        parsed["project"] = project
        rows.append(parsed)

    if not rows:
        print("[ERRO] Nenhum relatorio valido encontrado.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["project", "n_classes", "line_covered", "line_total",
                  "mutation_covered", "mutation_total", "strength_covered", "strength_total"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_line_cov = sum(r["line_covered"] for r in rows)
    total_line_tot = sum(r["line_total"] for r in rows)
    total_mut_cov = sum(r["mutation_covered"] for r in rows)
    total_mut_tot = sum(r["mutation_total"] for r in rows)
    total_ts_cov = sum(r["strength_covered"] for r in rows)
    total_ts_tot = sum(r["strength_total"] for r in rows)

    per_proj_line = [r["line_covered"] / r["line_total"] * 100 for r in rows if r["line_total"] > 0]
    per_proj_mut = [r["mutation_covered"] / r["mutation_total"] * 100 for r in rows if r["mutation_total"] > 0]

    print(f"\n{SEP}")
    print("  SF110 — LINE COVERAGE (extraido dos relatorios PIT existentes)")
    print(SEP)
    print(f"\n  Projetos com relatorio valido: {len(rows)}")
    print(f"\n  {'Metrica':<20}{'GLOBAL (ponderado)':>22}{'MEDIA POR-PROJETO':>22}")
    print(f"  {'-'*64}")
    print(f"  {'Line Coverage':<20}{total_line_cov/total_line_tot*100:>21.1f}%{sum(per_proj_line)/len(per_proj_line):>21.1f}%")
    print(f"  {'Mutation Coverage':<20}{total_mut_cov/total_mut_tot*100:>21.1f}%{sum(per_proj_mut)/len(per_proj_mut):>21.1f}%")
    print(f"  {'Test Strength':<20}{total_ts_cov/total_ts_tot*100:>21.1f}%")
    print(f"\n  Nota: PIT nao reporta branch coverage nem method coverage por omissao —")
    print(f"  so line coverage + mutation coverage/test strength.")
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
