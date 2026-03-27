"""
Prepare SF110 data into the project's strict 3-feature schema:
(methods, branches, statements)

Pipeline:
1) Download SF110 assets page and linked archives/tables
2) Extract archives
3) Discover tabular files
4) Map candidate columns to methods/branches/statements
5) Clean using strict rules (non-negative, <= max_value)
6) Save no-header CSV ready for JUnitDataset training
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd

SF110_URL = "https://www.evosuite.org/experimental-data/sf110/"
DOWNLOAD_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".csv", ".tsv", ".txt", ".gz")

METHOD_KEYS = [
    "methods", "method", "method_count", "n_methods", "covered_methods", "method_coverage"
]
BRANCH_KEYS = [
    "branches", "branch", "branch_count", "n_branches", "covered_branches", "branch_coverage"
]
STATEMENT_KEYS = [
    "statements", "statement", "statement_count", "n_statements", "covered_statements", "line_coverage"
]


@dataclass
class PrepResult:
    strict_csv: Path
    full_with_meta_csv: Path
    report_json: Path
    project_count: int
    sample_count: int


def _read_html(url: str) -> str:
    with urlopen(url, timeout=60) as resp:
        content = resp.read()
    return content.decode("utf-8", errors="ignore")


def _extract_links(html: str, base_url: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    links = []
    for href in hrefs:
        full = urljoin(base_url, href)
        if any(full.lower().endswith(ext) for ext in DOWNLOAD_EXTENSIONS):
            links.append(full)
    seen = set()
    unique = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique


def _safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = re.sub(r"\W+", "_", url)
    return name


def _download_assets(links: List[str], download_dir: Path) -> List[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for link in links:
        name = _safe_filename_from_url(link)
        target = download_dir / name
        if not target.exists():
            urlretrieve(link, target)
        downloaded.append(target)

    return downloaded


def _extract_archives(files: List[Path], extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        lower = file_path.name.lower()
        try:
            if lower.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(extract_dir / file_path.stem)
            elif lower.endswith((".tar", ".tar.gz", ".tgz")):
                with tarfile.open(file_path, "r:*") as tf:
                    tf.extractall(extract_dir / file_path.stem)
        except Exception:
            continue


def _discover_tabular_files(base_dirs: List[Path]) -> List[Path]:
    candidates = []
    patterns = ["**/*.csv", "**/*.tsv", "**/*.txt"]
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(base_dir.glob(pattern))
    return sorted(set(candidates))


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower())


def _find_feature_column(columns: List[str], keys: List[str]) -> Optional[str]:
    norm = {_normalize_col_name(c): c for c in columns}
    for key in keys:
        key_norm = _normalize_col_name(key)
        if key_norm in norm:
            return norm[key_norm]

    for c in columns:
        c_norm = _normalize_col_name(c)
        if any(k in c_norm for k in keys):
            return c
    return None


def _project_from_path(path: Path) -> str:
    joined = str(path).replace("\\", "/")
    m = re.search(r"(\d{1,3}_[^/]+)", joined)
    if m:
        return m.group(1)
    return path.stem


def _load_candidate_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    for sep in [",", "\t", ";", None]:
        try:
            if sep is None:
                df = pd.read_csv(file_path, header=None, engine="python", sep=r"\s+")
            else:
                df = pd.read_csv(file_path, engine="python", sep=sep)

            if df.empty:
                continue
            return df
        except Exception:
            continue
    return None


def _extract_features_from_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    columns = list(df.columns)
    normalized_to_original = {_normalize_col_name(c): c for c in columns}

    # Prefer EvoSuite canonical metrics when they are available.
    # This avoids noisy mappings from unrelated CSV files.
    evo_required = ["total_methods", "total_branches", "total_goals"]
    if all(col in normalized_to_original for col in evo_required):
        out = pd.DataFrame({
            "methods": pd.to_numeric(df[normalized_to_original["total_methods"]], errors="coerce"),
            "branches": pd.to_numeric(df[normalized_to_original["total_branches"]], errors="coerce"),
            "statements": pd.to_numeric(df[normalized_to_original["total_goals"]], errors="coerce"),
        })
        return out

    methods_col = _find_feature_column(columns, METHOD_KEYS)
    branches_col = _find_feature_column(columns, BRANCH_KEYS)
    statements_col = _find_feature_column(columns, STATEMENT_KEYS)

    if methods_col is not None and branches_col is not None and statements_col is not None:
        out = pd.DataFrame({
            "methods": pd.to_numeric(df[methods_col], errors="coerce"),
            "branches": pd.to_numeric(df[branches_col], errors="coerce"),
            "statements": pd.to_numeric(df[statements_col], errors="coerce"),
        })
        return out

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
    if len(numeric_cols) >= 3:
        out = pd.DataFrame({
            "methods": numeric_df[numeric_cols[0]],
            "branches": numeric_df[numeric_cols[1]],
            "statements": numeric_df[numeric_cols[2]],
        })
        return out

    return None


def _clean_strict(df: pd.DataFrame, max_value: int) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    for c in ["methods", "branches", "statements"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df[["methods", "branches", "statements"]] >= 0).all(axis=1)]
    df = df[(df[["methods", "branches", "statements"]] <= max_value).all(axis=1)]
    # Keep project diversity by deduplicating per-project when metadata exists.
    if "project" in df.columns:
        subset = ["project", "methods", "branches", "statements"]
    else:
        subset = ["methods", "branches", "statements"]
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    return df


def prepare_sf110_dataset(
    sf110_url: str,
    subset_projects: int,
    output_name: str,
    force_download: bool,
    max_value: int,
) -> PrepResult:
    root = Path("python/data")
    raw_root = root / "raw" / "evosuite" / "sf110"
    downloads_dir = raw_root / "downloads"
    extracted_dir = raw_root / "extracted"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    html = _read_html(sf110_url)
    links = _extract_links(html, sf110_url)

    if force_download:
        for p in downloads_dir.glob("*"):
            if p.is_file():
                p.unlink()

    downloaded = _download_assets(links, downloads_dir)
    _extract_archives(downloaded, extracted_dir)

    candidates = _discover_tabular_files([downloads_dir, extracted_dir])
    statistics_candidates = [
        c for c in candidates
        if c.name.lower() == "statistics.csv" and "evosuite-report" in str(c).replace("\\", "/").lower()
    ]
    using_statistics_only = False
    if statistics_candidates:
        candidates = sorted(statistics_candidates)
        using_statistics_only = True

    extracted_frames: List[pd.DataFrame] = []
    for candidate in candidates:
        df_raw = _load_candidate_dataframe(candidate)
        if df_raw is None:
            continue
        df_features = _extract_features_from_df(df_raw)
        if df_features is None:
            continue
        df_features["project"] = _project_from_path(candidate)
        df_features["source_file"] = str(candidate).replace("\\", "/")
        extracted_frames.append(df_features)

    if not extracted_frames:
        raise RuntimeError(
            "Could not extract any usable SF110 tabular data with methods/branches/statements. "
            "Check downloaded files under python/data/raw/evosuite/sf110."
        )

    merged = pd.concat(extracted_frames, ignore_index=True)
    merged = _clean_strict(merged, max_value=max_value)

    projects = sorted(merged["project"].unique().tolist())
    if subset_projects > 0:
        selected = set(projects[:subset_projects])
        merged = merged[merged["project"].isin(selected)].copy()

    full_with_meta_csv = processed_dir / f"{output_name}_with_meta.csv"
    strict_csv = processed_dir / f"{output_name}_strict.csv"
    report_json = processed_dir / f"{output_name}_report.json"

    merged.to_csv(full_with_meta_csv, index=False)
    merged[["methods", "branches", "statements"]].to_csv(strict_csv, index=False, header=False)

    report = {
        "sf110_url": sf110_url,
        "links_found": len(links),
        "assets_downloaded": len(downloaded),
        "candidate_files": len(candidates),
        "using_statistics_only": using_statistics_only,
        "samples_after_cleaning": int(len(merged)),
        "projects_after_filter": int(merged["project"].nunique()),
        "subset_projects": subset_projects,
        "strict_csv": str(strict_csv).replace("\\", "/"),
        "full_with_meta_csv": str(full_with_meta_csv).replace("\\", "/"),
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return PrepResult(
        strict_csv=strict_csv,
        full_with_meta_csv=full_with_meta_csv,
        report_json=report_json,
        project_count=int(merged["project"].nunique()),
        sample_count=int(len(merged)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SF110 dataset for WGAN training")
    parser.add_argument("--url", type=str, default=SF110_URL, help="SF110 experimental data URL")
    parser.add_argument("--subset-projects", type=int, default=10, help="Use first N projects (0 = all)")
    parser.add_argument("--output-name", type=str, default="sf110_subset", help="Output filename prefix")
    parser.add_argument("--max-value", type=int, default=1000, help="Strict max value per feature")
    parser.add_argument("--force-download", action="store_true", help="Re-download remote assets")

    args = parser.parse_args()

    result = prepare_sf110_dataset(
        sf110_url=args.url,
        subset_projects=args.subset_projects,
        output_name=args.output_name,
        force_download=args.force_download,
        max_value=args.max_value,
    )

    print("=" * 70)
    print("✅ SF110 dataset prepared")
    print(f"   Strict CSV: {result.strict_csv}")
    print(f"   Full CSV:   {result.full_with_meta_csv}")
    print(f"   Report:     {result.report_json}")
    print(f"   Projects:   {result.project_count}")
    print(f"   Samples:    {result.sample_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
