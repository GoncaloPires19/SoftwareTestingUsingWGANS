#!/usr/bin/env python3
"""
Mutation Testing (PIT) Experiment Runner
----------------------------------------
This script runs Mutation Testing on the Java classes iteratively,
testing how each individual Test Class performs in terms of Mutation Score.
It generates a final summary Report comparing the quality of each test suite.

This serves as concrete proof of the relationships between:
[Test Count] vs [Mutation Score] 

Usage:
  python python/scripts/pit_experiment.py
"""

import os
import subprocess
import glob
from pathlib import Path
from typing import Optional
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns


PIT_STATUS_FALLBACK_COLUMN = 5
PIT_STATUS_COLUMN_CANDIDATES = ["status", "result", "mutation_status"]

def get_test_classes():
    """Finds all Test classes in the dataloader package."""
    test_dir = Path("src/test/java/dataloader")
    test_files = test_dir.glob("*.java")
    return [t.stem for t in test_files]


def _find_latest_mutations_csv() -> Path | None:
    """Return most recent mutations.csv from PIT output directories."""
    direct_csv = Path("target/pit-reports/mutations.csv")
    timestamped_csv = list(Path("target/pit-reports").glob("*/mutations.csv"))

    candidates = []
    if direct_csv.exists():
        candidates.append(direct_csv)
    candidates.extend([p for p in timestamped_csv if p.exists()])

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_status_series(csv_path: Path) -> pd.Series:
    """Load PIT status column from CSV that may or may not contain header row."""
    df_header = pd.read_csv(csv_path)
    lowered = {str(col).strip().lower(): col for col in df_header.columns}

    for candidate in PIT_STATUS_COLUMN_CANDIDATES:
        if candidate in lowered:
            return df_header[lowered[candidate]].astype(str).str.strip().str.upper()

    df_no_header = pd.read_csv(csv_path, header=None)
    return df_no_header[PIT_STATUS_FALLBACK_COLUMN].astype(str).str.strip().str.upper()

def run_pitest(test_class: Optional[str] = None):
    """
    Runs Maven PIT plugin. 
    If test_class is None, runs against ALL tests.
    """
    print(f"\n🚀 Running Mutation Testing for: {test_class if test_class else 'ALL CLASSES'}")
    
    cmd = ["mvn.cmd" if os.name == "nt" else "mvn", "clean", "test", "org.pitest:pitest-maven:mutationCoverage"]
    
    if test_class:
        # Include nested classes (e.g., RobustTests$ArithmeticTests) via wildcard suffix.
        cmd.append(f"-DtargetTests=dataloader.{test_class}*")
        
    # Execute the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stdout_tail = (result.stdout or "").splitlines()[-20:]
        stderr_tail = (result.stderr or "").splitlines()[-20:]
        print(f"⚠️  Maven failed for {test_class if test_class else 'ALL CLASSES'} (exit={result.returncode})")
        if stdout_tail:
            print("--- Maven stdout (tail) ---")
            print("\n".join(stdout_tail))
        if stderr_tail:
            print("--- Maven stderr (tail) ---")
            print("\n".join(stderr_tail))
        
    return parse_pitest_results()

def parse_pitest_results():
    """Parses target/pit-reports/mutations.csv and returns mutation stats."""
    csv_path = _find_latest_mutations_csv()
    if csv_path is None:
        return {"killed": 0, "survived": 0, "no_coverage": 0, "total": 0, "score": 0.0}

    try:
        status = _extract_status_series(csv_path)

        killed = int((status == "KILLED").sum())
        survived = int((status == "SURVIVED").sum())
        no_coverage = int((status == "NO_COVERAGE").sum())
        time_out = int((status == "TIMED_OUT").sum())
        memory_error = int((status == "MEMORY_ERROR").sum())
        non_viable = int((status == "NON_VIABLE").sum())
        
        killed_total = killed + time_out
        total_mutations = int(len(status))
        total_relevant = total_mutations - non_viable - memory_error
        
        score = (killed_total / total_relevant * 100) if total_relevant > 0 else 0.0
        
        return {
            "killed": killed_total,
            "survived": survived,
            "no_coverage": no_coverage,
            "total": total_relevant,
            "score": round(score, 2)
        }
    except Exception as e:
        print(f"Error parsing PIT CSV: {e}")
        return {"killed": 0, "survived": 0, "no_coverage": 0, "total": 0, "score": 0.0}

def get_test_count(test_class: Optional[str] = None):
    """Reads Surefire XML reports to count tests run."""
    total_tests = 0
    if test_class:
        # Include nested class reports, e.g., RobustTests$ArithmeticTests
        pattern = f"target/surefire-reports/TEST-dataloader.{test_class}*.xml"
    else:
        pattern = "target/surefire-reports/TEST-*.xml"
    
    for xml_file in glob.glob(pattern):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            total_tests += int(root.attrib.get('tests', 0))
        except Exception:
            pass
    return total_tests

def generate_report(results: list):
    """Generates a Markdown report and plots."""
    out_dir = Path("outputs/mutation_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "mutation_results.csv", index=False)
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=df.sort_values('score', ascending=False), x='score', y='test_suite', palette='viridis')
    plt.title('Mutation Score by Test Suite')
    plt.xlabel('Mutation Score (%)')
    plt.ylabel('Test Suite')
    
    # Add count annotations
    for i, p in enumerate(ax.patches):
        score = p.get_width()
        tests = df.sort_values('score', ascending=False).iloc[i]['tests_run']
        ax.annotate(f"{score:.1f}% ({tests} tests)", 
                    (score + 1, p.get_y() + p.get_height() / 2.), 
                    va='center')
        
    plt.tight_layout()
    plt.savefig(out_dir / "mutation_scores.png", dpi=150)
    plt.close()
    
    # Scatter plot: Test Count vs Mutation Score
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='tests_run', y='score', hue='test_suite', s=100)
    plt.title('Number of Tests vs. Mutation Score')
    plt.xlabel('Number of Tests Run')
    plt.ylabel('Mutation Score (%)')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "mutation_vs_count.png", dpi=150)
    plt.close()
    
    # Markdown Report
    with open(out_dir / "MUTATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Parâmetros de Testes de Mutação (PIT)\n\n")
        f.write("Esta experiência avaliou o código (`MathOperations`) contra as várias \n")
        f.write("classes de testes disponíveis para ver como o **Mutation Score** reage.\n\n")
        
        f.write("## 1. Resultados\n\n")
        f.write("| Test Suite | Tests Run | Mutations Killed | Survived | No Coverage | Mutation Score (%) |\n")
        f.write("|------------|-----------|------------------|----------|-------------|--------------------|\n")
        for r in sorted(results, key=lambda x: x['score'], reverse=True):
            f.write(f"| {r['test_suite']} | {r['tests_run']} | {r['killed']} | {r['survived']} | {r['no_coverage']} | **{r['score']}%** |\n")
            
        f.write("\n## 2. Gráficos Gerados\n")
        f.write("- `mutation_scores.png`: Comparação direta dos scores\n")
        f.write("- `mutation_vs_count.png`: Correlação entre Número de Testes e Score de Mutação.\n")
        
        f.write("\n## 3. Conclusão para a Tese\n")
        f.write("Os resultados mostram que o score de mutação depende fortemente do alinhamento entre a suite e o código alvo mutado.\n")
        f.write("O número de testes, isoladamente, não explica o desempenho sem controlar tipo de teste e cobertura efetiva.\n")
        f.write("Para afirmar retornos decrescentes com rigor, é necessário repetir o estudo com subconjuntos balanceados e mesmo alvo funcional.\n")

    print(f"\n✅ Mutation Experiment finished! Results in {out_dir}/MUTATION_REPORT.md")


def main():
    test_classes = get_test_classes()
    results = []
    
    for t_class in test_classes:
        stats = run_pitest(t_class)
        tests_run = get_test_count(t_class)
        
        results.append({
            "test_suite": t_class,
            "tests_run": tests_run,
            **stats
        })
        
    # Baseline with ALL test classes combined
    stats_all = run_pitest(None)
    tests_all = get_test_count(None)
    results.append({
        "test_suite": "ALL_COMBINED",
        "tests_run": tests_all,
        **stats_all
    })
    
    generate_report(results)

if __name__ == "__main__":
    main()
