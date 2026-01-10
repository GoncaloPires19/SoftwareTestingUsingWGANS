"""
Dataset processing and normalization for WGAN training.
Converts Juliet, OWASP, Defects4J into normalized CSV format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import ast
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Process and normalize public datasets"""
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/raw")
    
    def process_juliet(self) -> pd.DataFrame:
        """
        Process Juliet Java Test Suite
        Extracts features from Java source files
        """
        logger.info("🔄 Processing Juliet dataset...")
        
        juliet_dir = self.data_dir / "juliet" / "Juliet"
        if not juliet_dir.exists():
            logger.warning(f"Juliet not found at {juliet_dir}")
            return pd.DataFrame()
        
        samples = []
        cwe_dirs = list(juliet_dir.glob("CWE-*/src"))
        logger.info(f"Found {len(cwe_dirs)} CWE directories")
        
        for cwe_dir in cwe_dirs:
            java_files = list(cwe_dir.glob("*.java"))
            
            for java_file in java_files:
                try:
                    features = self._extract_java_features(java_file)
                    if features:
                        features['source'] = 'juliet'
                        features['cwe'] = cwe_dir.parent.name
                        samples.append(features)
                except Exception as e:
                    logger.debug(f"Error processing {java_file}: {e}")
        
        df = pd.DataFrame(samples)
        logger.info(f"✅ Juliet: {len(df)} samples processed")
        
        if len(df) > 0:
            df.to_csv(self.output_dir / "juliet_java.csv", index=False)
            logger.info(f"   Saved to juliet_java.csv")
        
        return df
    
    def process_owasp(self) -> pd.DataFrame:
        """
        Process OWASP Benchmark test cases
        """
        logger.info("🔄 Processing OWASP Benchmark...")
        
        owasp_dir = self.data_dir / "owasp" / "src" / "main" / "java" / "org" / "owasp" / "benchmark"
        if not owasp_dir.exists():
            logger.warning(f"OWASP not found at {owasp_dir}")
            return pd.DataFrame()
        
        samples = []
        java_files = list(owasp_dir.glob("testcases/*/*.java"))
        logger.info(f"Found {len(java_files)} test files")
        
        for java_file in java_files:
            try:
                features = self._extract_java_features(java_file)
                if features:
                    features['source'] = 'owasp'
                    features['test_type'] = java_file.parent.name
                    samples.append(features)
            except Exception as e:
                logger.debug(f"Error processing {java_file}: {e}")
        
        df = pd.DataFrame(samples)
        logger.info(f"✅ OWASP: {len(df)} samples processed")
        
        if len(df) > 0:
            df.to_csv(self.output_dir / "owasp_benchmark.csv", index=False)
            logger.info(f"   Saved to owasp_benchmark.csv")
        
        return df
    
    def process_defects4j(self) -> pd.DataFrame:
        """
        Process Defects4J bug repository
        """
        logger.info("🔄 Processing Defects4J...")
        
        defects4j_dir = self.data_dir / "defects4j"
        if not defects4j_dir.exists():
            logger.warning(f"Defects4J not found at {defects4j_dir}")
            return pd.DataFrame()
        
        samples = []
        
        # Look for test directories in each project
        project_dirs = list(defects4j_dir.glob("projects/*/"))
        logger.info(f"Found {len(project_dirs)} Defects4J projects")
        
        for project_dir in project_dirs:
            test_dirs = list(project_dir.glob("**/src/test/**/*.java"))
            
            for test_file in test_dirs:
                try:
                    features = self._extract_java_features(test_file)
                    if features:
                        features['source'] = 'defects4j'
                        features['project'] = project_dir.name
                        samples.append(features)
                except Exception as e:
                    logger.debug(f"Error processing {test_file}: {e}")
        
        df = pd.DataFrame(samples)
        logger.info(f"✅ Defects4J: {len(df)} samples processed")
        
        if len(df) > 0:
            df.to_csv(self.output_dir / "defects4j_tests.csv", index=False)
            logger.info(f"   Saved to defects4j_tests.csv")
        
        return df
    
    @staticmethod
    def _extract_java_features(java_file: Path) -> Dict:
        """
        Extract test complexity metrics from Java source
        Returns dict with: num_methods, num_branches, num_statements
        """
        try:
            with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Count methods (rough estimate)
            num_methods = len(re.findall(r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', content))
            
            # Count branches (if, else, switch, etc.)
            num_branches = (
                content.count('if(') + content.count('if (') +
                content.count('else') + content.count('switch') +
                content.count('case ') + content.count('for') +
                content.count('while')
            )
            
            # Count statements (lines with semicolons, rough)
            num_statements = content.count(';')
            
            # Only return if has meaningful content
            if num_methods > 0 or num_branches > 0 or num_statements > 0:
                return {
                    'methods': float(num_methods),
                    'branches': float(num_branches),
                    'statements': float(num_statements),
                    'file': str(java_file.name)
                }
        except Exception as e:
            logger.debug(f"Error extracting features from {java_file}: {e}")
        
        return None
    
    def combine_all(self, juliet_df: pd.DataFrame, owasp_df: pd.DataFrame, 
                   defects4j_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all datasets into single normalized DataFrame
        """
        logger.info("📦 Combining all datasets...")
        
        dfs = []
        
        # Normalize each dataset to same columns
        for name, df in [('Juliet', juliet_df), ('OWASP', owasp_df), ('Defects4J', defects4j_df)]:
            if len(df) > 0:
                # Keep only numeric columns we need
                df_norm = df[['methods', 'branches', 'statements', 'source']].copy()
                logger.info(f"  {name}: {len(df_norm)} samples")
                dfs.append(df_norm)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            
            # Normalize to reasonable ranges
            combined['methods'] = combined['methods'].clip(0, 1000)
            combined['branches'] = combined['branches'].clip(0, 2000)
            combined['statements'] = combined['statements'].clip(0, 5000)
            
            combined.to_csv(self.output_dir / "combined_datasets.csv", index=False)
            logger.info(f"✅ Combined: {len(combined)} total samples")
            logger.info(f"   Saved to combined_datasets.csv")
            
            return combined
        
        return pd.DataFrame()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process public datasets")
    parser.add_argument("--juliet", action="store_true", help="Process Juliet")
    parser.add_argument("--owasp", action="store_true", help="Process OWASP")
    parser.add_argument("--defects4j", action="store_true", help="Process Defects4J")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    
    args = parser.parse_args()
    
    processor = DatasetProcessor()
    
    juliet_df = processor.process_juliet() if (args.all or args.juliet) else pd.DataFrame()
    owasp_df = processor.process_owasp() if (args.all or args.owasp) else pd.DataFrame()
    defects4j_df = processor.process_defects4j() if (args.all or args.defects4j) else pd.DataFrame()
    
    if args.all or any([args.juliet, args.owasp, args.defects4j]):
        processor.combine_all(juliet_df, owasp_df, defects4j_df)
    
    logger.info("\n✅ Dataset processing complete!")


if __name__ == "__main__":
    main()
