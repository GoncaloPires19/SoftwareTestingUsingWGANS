"""
Dataset acquisition and normalization script.
Downloads and processes:
- Juliet Java benchmarks
- OWASP Benchmark
- Defects4J repository
- Generates EvoSuite tests
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Optional
import tempfile
import shutil


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and cache datasets"""
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_juliet(self, force: bool = False) -> Path:
        """
        Download Juliet Java Test Suite
        Free download from: https://samate.nist.gov/SARD/test-suites/81
        
        Returns path to extracted directory
        """
        target = self.data_dir / "juliet"
        
        if target.exists() and not force:
            logger.info(f"✅ Juliet already downloaded at {target}")
            return target
        
        logger.info("📥 Juliet requires manual download from:")
        logger.info("   https://samate.nist.gov/SARD/test-suites/81")
        logger.info(f"   Extract to: {target}")
        logger.info("   Expected structure: Juliet/*/Java/CWE-*/src/")
        
        return target
    
    def download_owasp(self, force: bool = False) -> Path:
        """
        Download OWASP Benchmark
        """
        target = self.data_dir / "owasp"
        
        if target.exists() and not force:
            logger.info(f"✅ OWASP already downloaded at {target}")
            return target
        
        logger.info("📥 OWASP Benchmark requires manual download from:")
        logger.info("   https://github.com/OWASP/benchmark")
        logger.info(f"   Extract to: {target}")
        
        return target
    
    def download_defects4j(self, force: bool = False) -> Path:
        """
        Download Defects4J repository
        """
        target = self.data_dir / "defects4j"
        
        if target.exists() and not force:
            logger.info(f"✅ Defects4J already cloned at {target}")
            return target
        
        logger.info("📥 Cloning Defects4J...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/rjust/defects4j.git",
                str(target)
            ], check=True, capture_output=True)
            logger.info(f"✅ Defects4J cloned to {target}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone Defects4J: {e}")
        
        return target
    
    def generate_evosuite(
        self,
        java_class_path: str,
        output_dir: Path,
        time_budget: int = 60,
    ) -> Path:
        """
        Generate tests with EvoSuite
        Requires: EvoSuite JAR in PATH or installed
        
        Args:
            java_class_path: Classpath to target Java classes
            output_dir: Where to save generated tests
            time_budget: Time per class in seconds
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧬 Generating tests with EvoSuite (budget: {time_budget}s per class)...")
        logger.info("   Note: Requires 'evosuite' command in PATH")
        logger.info(f"   Output: {output_dir}")
        
        # Command template
        cmd = f"evosuite -projectCP {java_class_path} -base_dir {output_dir} -timeInMinutesPerClass {time_budget//60}"
        logger.info(f"   Command: {cmd}")
        
        return output_dir


class DatasetNormalizer:
    """Normalize datasets to common format"""
    
    COMMON_SCHEMA = {
        "test_name": str,
        "test_code": str,
        "source_file": str,
        "source": str,  # juliet, owasp, defects4j, evosuite
        "cwe": Optional[str],  # CWE-ID if applicable
        "bug_type": Optional[str],
        "methods": int,
        "branches": int,
        "statements": int,
        "has_vulnerability": bool,
    }
    
    @staticmethod
    def normalize_juliet(juliet_dir: Path) -> pd.DataFrame:
        """
        Parse Juliet directory structure and extract test info
        Structure: Juliet/*/Java/CWE-*/src/*.java
        """
        logger.info("🔄 Processing Juliet dataset...")
        
        records = []
        java_files = list(juliet_dir.glob("**/CWE-*/**/src/*.java"))
        logger.info(f"   Found {len(java_files)} Java files")
        
        for java_file in java_files[:100]:  # Limit for demo
            try:
                # Extract CWE from path
                parts = java_file.parts
                cwe = None
                for part in parts:
                    if part.startswith("CWE-"):
                        cwe = part
                        break
                
                # Read file
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                # Simple metrics (count keywords)
                methods = code.count("public") + code.count("private") + code.count("protected")
                branches = code.count("if") + code.count("else") + code.count("switch")
                statements = code.count(";")
                
                records.append({
                    "test_name": java_file.stem,
                    "test_code": code[:500],  # Truncate for storage
                    "source_file": str(java_file),
                    "source": "juliet",
                    "cwe": cwe,
                    "bug_type": cwe.replace("CWE-", "") if cwe else None,
                    "methods": methods,
                    "branches": branches,
                    "statements": statements,
                    "has_vulnerability": True,
                })
            except Exception as e:
                logger.debug(f"Failed to process {java_file}: {e}")
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Processed {len(df)} Juliet samples")
        return df
    
    @staticmethod
    def normalize_owasp(owasp_dir: Path) -> pd.DataFrame:
        """Parse OWASP Benchmark Java test sources"""
        logger.info("🔄 Processing OWASP Benchmark dataset...")
        
        records = []
        java_files = list(owasp_dir.glob("**/BenchmarkTest*.java"))
        logger.info(f"   Found {len(java_files)} test files")
        
        for java_file in java_files[:100]:  # Limit for demo
            try:
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                # Metrics
                methods = code.count("public") + code.count("private")
                branches = code.count("if") + code.count("else")
                statements = code.count(";")
                
                # Determine if vulnerable (heuristic: contains vulnerable patterns)
                has_vuln = any(pattern in code for pattern in [
                    "PreparedStatement", "encrypt", "hash", "sql"
                ])
                
                records.append({
                    "test_name": java_file.stem,
                    "test_code": code[:500],
                    "source_file": str(java_file),
                    "source": "owasp",
                    "cwe": None,
                    "bug_type": None,
                    "methods": methods,
                    "branches": branches,
                    "statements": statements,
                    "has_vulnerability": has_vuln,
                })
            except Exception as e:
                logger.debug(f"Failed to process {java_file}: {e}")
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Processed {len(df)} OWASP samples")
        return df
    
    @staticmethod
    def normalize_defects4j(defects4j_dir: Path, limit: int = 100) -> pd.DataFrame:
        """
        Extract test files from Defects4J bug repositories
        """
        logger.info("🔄 Processing Defects4J bugs...")
        
        records = []
        
        # Defects4J structure: defects4j/framework/projects/{project}/
        projects_dir = defects4j_dir / "framework" / "projects"
        
        if not projects_dir.exists():
            logger.warning(f"Defects4J projects directory not found at {projects_dir}")
            return pd.DataFrame(columns=list(DatasetNormalizer.COMMON_SCHEMA.keys()))
        
        test_files = list(projects_dir.glob("**/src/test/**/*.java"))
        logger.info(f"   Found {len(test_files)} test files")
        
        for java_file in test_files[:limit]:
            try:
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                metrics = {
                    "methods": code.count("public") + code.count("private"),
                    "branches": code.count("if") + code.count("else"),
                    "statements": code.count(";"),
                }
                
                records.append({
                    "test_name": java_file.stem,
                    "test_code": code[:500],
                    "source_file": str(java_file),
                    "source": "defects4j",
                    "cwe": None,
                    "bug_type": "real_bug",
                    "methods": metrics["methods"],
                    "branches": metrics["branches"],
                    "statements": metrics["statements"],
                    "has_vulnerability": False,  # Real tests usually passing
                })
            except Exception as e:
                logger.debug(f"Failed to process {java_file}: {e}")
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Processed {len(df)} Defects4J samples")
        return df
    
    @staticmethod
    def normalize_evosuite(evosuite_dir: Path, limit: int = 100) -> pd.DataFrame:
        """Process EvoSuite generated tests"""
        logger.info("🔄 Processing EvoSuite tests...")
        
        records = []
        java_files = list(evosuite_dir.glob("**/*_ESTest.java"))
        logger.info(f"   Found {len(java_files)} ESTest files")
        
        for java_file in java_files[:limit]:
            try:
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                records.append({
                    "test_name": java_file.stem,
                    "test_code": code[:500],
                    "source_file": str(java_file),
                    "source": "evosuite",
                    "cwe": None,
                    "bug_type": None,
                    "methods": code.count("public") + code.count("private"),
                    "branches": code.count("if") + code.count("else"),
                    "statements": code.count(";"),
                    "has_vulnerability": False,
                })
            except Exception as e:
                logger.debug(f"Failed to process {java_file}: {e}")
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Processed {len(df)} EvoSuite samples")
        return df


def prepare_datasets(
    data_dir: Path = Path("data"),
    download_force: bool = False,
    process_juliet: bool = True,
    process_owasp: bool = True,
    process_defects4j: bool = True,
    process_evosuite: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Master function to download and prepare all datasets
    """
    
    print("\n" + "="*70)
    print("🔧 DATASET ACQUISITION & NORMALIZATION")
    print("="*70)
    
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = DatasetDownloader(raw_dir)
    normalizer = DatasetNormalizer()
    
    datasets = {}
    
    # Juliet
    if process_juliet:
        print("\n📥 JULIET:")
        juliet_path = downloader.download_juliet(force=download_force)
        if juliet_path.exists():
            try:
                datasets["juliet"] = normalizer.normalize_juliet(juliet_path)
            except Exception as e:
                logger.error(f"Failed to process Juliet: {e}")
    
    # OWASP
    if process_owasp:
        print("\n📥 OWASP BENCHMARK:")
        owasp_path = downloader.download_owasp(force=download_force)
        if owasp_path.exists():
            try:
                datasets["owasp"] = normalizer.normalize_owasp(owasp_path)
            except Exception as e:
                logger.error(f"Failed to process OWASP: {e}")
    
    # Defects4J
    if process_defects4j:
        print("\n📥 DEFECTS4J:")
        d4j_path = downloader.download_defects4j(force=download_force)
        if d4j_path.exists():
            try:
                datasets["defects4j"] = normalizer.normalize_defects4j(d4j_path)
            except Exception as e:
                logger.error(f"Failed to process Defects4J: {e}")
    
    # EvoSuite (requires manual setup)
    if process_evosuite:
        print("\n🧬 EVOSUITE:")
        print("   Requires: evosuite command installed")
        print("   Generate with: python scripts/generate_evosuite.py")
    
    # Summary
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    
    total_samples = 0
    for name, df in datasets.items():
        count = len(df)
        total_samples += count
        print(f"  {name.upper()}: {count} samples")
    
    print(f"  TOTAL: {total_samples} samples")
    
    # Save summaries
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in datasets.items():
        output_path = processed_dir / f"{name}_normalized.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved {name} to {output_path}")
    
    return datasets


if __name__ == "__main__":
    datasets = prepare_datasets(
        download_force=False,
        process_juliet=False,  # Manual download required
        process_owasp=False,   # Manual download required
        process_defects4j=True,
        process_evosuite=False,
    )
    
    print("\n✅ Dataset preparation complete!")
