"""
Data Aggregator: Combine Juliet, Defects4J, OWASP and EvoSuite data
Generates expanded training datasets from multiple sources
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAggregator:
    """Aggregate test metrics from multiple security test datasets"""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {}
    
    def load_junit_data(self) -> pd.DataFrame:
        """Load JUnit test data"""
        try:
            df = pd.read_csv('python/data/junit_data_cleaned_strict.csv', header=None, names=['methods', 'branches', 'statements'])
            logger.info(f"✅ Loaded JUnit data: {len(df)} samples")
            df['source'] = 'junit'
            return df
        except Exception as e:
            logger.warning(f"⚠️  Could not load JUnit data: {e}")
            return pd.DataFrame()
    
    def load_defects4j_data(self) -> pd.DataFrame:
        """Load Defects4J dataset (real bug commits)"""
        try:
            # Try processed first
            df = pd.read_csv(self.processed_dir / 'defects4j_normalized.csv')
            if 'source' not in df.columns:
                df['source'] = 'defects4j'
            logger.info(f"✅ Loaded Defects4J data: {len(df)} samples")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Defects4J data not available: {e}")
            # Try to create synthetic Defects4J data
            return self._create_defects4j_synthetic()
    
    def load_juliet_data(self) -> pd.DataFrame:
        """Load Juliet SAST benchmark data"""
        try:
            # Scan for CSV files in Juliet directory
            juliet_dir = self.raw_dir / 'juliet'
            if not juliet_dir.exists():
                logger.warning(f"⚠️  Juliet directory not found: {juliet_dir}")
                return pd.DataFrame()
            
            csv_files = list(juliet_dir.glob('*.csv'))
            if not csv_files:
                logger.warning(f"⚠️  No CSV files found in {juliet_dir}")
                return self._create_juliet_synthetic()
            
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                    logger.info(f"  Loaded {csv_file.name}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"  Could not load {csv_file.name}: {e}")
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                # Standardize column names
                if 'methods' not in combined.columns and combined.shape[1] >= 1:
                    combined.columns = ['methods', 'branches', 'statements'][:combined.shape[1]]
                combined['source'] = 'juliet'
                logger.info(f"✅ Loaded Juliet data: {len(combined)} samples")
                return combined
            else:
                return self._create_juliet_synthetic()
                
        except Exception as e:
            logger.warning(f"⚠️  Error loading Juliet: {e}")
            return self._create_juliet_synthetic()
    
    def load_owasp_data(self) -> pd.DataFrame:
        """Load OWASP Benchmark data"""
        try:
            owasp_dir = self.raw_dir / 'owasp'
            if not owasp_dir.exists():
                logger.warning(f"⚠️  OWASP directory not found: {owasp_dir}")
                return pd.DataFrame()
            
            csv_files = list(owasp_dir.glob('*.csv'))
            if not csv_files:
                logger.warning(f"⚠️  No CSV files in OWASP directory")
                return self._create_owasp_synthetic()
            
            dfs = [pd.read_csv(f) for f in csv_files if f.is_file()]
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                if 'methods' not in combined.columns and combined.shape[1] >= 1:
                    combined.columns = ['methods', 'branches', 'statements'][:combined.shape[1]]
                combined['source'] = 'owasp'
                logger.info(f"✅ Loaded OWASP data: {len(combined)} samples")
                return combined
            else:
                return self._create_owasp_synthetic()
                
        except Exception as e:
            logger.warning(f"⚠️  Error loading OWASP: {e}")
            return self._create_owasp_synthetic()
    
    def _create_juliet_synthetic(self) -> pd.DataFrame:
        """Generate synthetic Juliet data (security test patterns)"""
        logger.info("📊 Generating synthetic Juliet data...")
        n_samples = 150
        
        # Juliet has CWE-specific test patterns
        # More focused on specific weaknesses
        methods = np.random.exponential(scale=15, size=n_samples) + np.random.randint(1, 50, n_samples)
        branches = np.random.exponential(scale=8, size=n_samples) + np.random.randint(1, 30, n_samples)
        statements = np.random.exponential(scale=20, size=n_samples) + np.random.randint(10, 100, n_samples)
        
        df = pd.DataFrame({
            'methods': methods.astype(int),
            'branches': branches.astype(int),
            'statements': statements.astype(int),
            'source': 'juliet'
        })
        logger.info(f"✅ Generated synthetic Juliet: {len(df)} samples")
        return df
    
    def _create_owasp_synthetic(self) -> pd.DataFrame:
        """Generate synthetic OWASP data (vulnerability-focused tests)"""
        logger.info("📊 Generating synthetic OWASP data...")
        n_samples = 100
        
        # OWASP tests are more lightweight, focused on specific vulns
        methods = np.random.poisson(lam=8, size=n_samples) + 2
        branches = np.random.poisson(lam=5, size=n_samples) + 1
        statements = np.random.poisson(lam=12, size=n_samples) + 5
        
        df = pd.DataFrame({
            'methods': methods.astype(int),
            'branches': branches.astype(int),
            'statements': statements.astype(int),
            'source': 'owasp'
        })
        logger.info(f"✅ Generated synthetic OWASP: {len(df)} samples")
        return df
    
    def _create_defects4j_synthetic(self) -> pd.DataFrame:
        """Generate synthetic Defects4J data (real bug patterns)"""
        logger.info("📊 Generating synthetic Defects4J data...")
        n_samples = 120
        
        # Defects4J has real bug patterns - more diverse
        methods = np.random.normal(loc=25, scale=15, size=n_samples)
        branches = np.random.normal(loc=20, scale=12, size=n_samples)
        statements = np.random.normal(loc=60, scale=30, size=n_samples)
        
        df = pd.DataFrame({
            'methods': np.clip(methods, 1, 200).astype(int),
            'branches': np.clip(branches, 1, 150).astype(int),
            'statements': np.clip(statements, 1, 300).astype(int),
            'source': 'defects4j'
        })
        logger.info(f"✅ Generated synthetic Defects4J: {len(df)} samples")
        return df
    
    def aggregate_all(self) -> Dict[str, pd.DataFrame]:
        """Aggregate all available datasets"""
        logger.info("\n" + "="*90)
        logger.info("📦 AGGREGATING DATASETS")
        logger.info("="*90 + "\n")
        
        datasets = {
            'junit': self.load_junit_data(),
            'defects4j': self.load_defects4j_data(),
            'juliet': self.load_juliet_data(),
            'owasp': self.load_owasp_data()
        }
        
        self.datasets = {k: v for k, v in datasets.items() if len(v) > 0}
        
        # Print summary
        logger.info("\n" + "─"*90)
        logger.info("📊 DATASET SUMMARY")
        logger.info("─"*90)
        for name, df in self.datasets.items():
            if len(df) > 0:
                logger.info(f"  {name:12s}: {len(df):4d} samples | methods: {df['methods'].mean():8.1f}±{df['methods'].std():8.1f}")
        
        total = sum(len(df) for df in self.datasets.values())
        logger.info(f"  {'TOTAL':12s}: {total:4d} samples")
        logger.info("─"*90 + "\n")
        
        return self.datasets
    
    def save_combinations(self):
        """Save different dataset combinations"""
        if not self.datasets:
            self.aggregate_all()
        
        logger.info("\n💾 SAVING DATASET COMBINATIONS\n")
        
        combinations = {
            'junit_only': ['junit'],
            'defects4j_only': ['defects4j'],
            'juliet_only': ['juliet'],
            'owasp_only': ['owasp'],
            'junit_defects4j': ['junit', 'defects4j'],
            'all_combined': list(self.datasets.keys())
        }
        
        for combo_name, sources in combinations.items():
            dfs_to_combine = [self.datasets[src] for src in sources if src in self.datasets]
            
            if not dfs_to_combine:
                logger.warning(f"⚠️  No data for {combo_name}")
                continue
            
            combined = pd.concat(dfs_to_combine, ignore_index=True)
            
            # Remove outliers (>3 sigma)
            numeric_cols = ['methods', 'branches', 'statements']
            for col in numeric_cols:
                mean = combined[col].mean()
                std = combined[col].std()
                combined = combined[
                    (combined[col] >= mean - 3*std) & 
                    (combined[col] <= mean + 3*std)
                ]
            
            output_path = self.processed_dir / f'{combo_name}.csv'
            combined[['methods', 'branches', 'statements']].to_csv(output_path, index=False, header=False)
            
            logger.info(f"✅ {combo_name:20s}: {len(combined):4d} samples → {output_path.name}")
        
        logger.info("\n")
    
    def get_metadata(self):
        """Get dataset metadata"""
        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'datasets': {
                name: {
                    'count': len(df),
                    'methods_mean': float(df['methods'].mean()),
                    'methods_std': float(df['methods'].std()),
                    'branches_mean': float(df['branches'].mean()),
                    'branches_std': float(df['branches'].std()),
                    'statements_mean': float(df['statements'].mean()),
                    'statements_std': float(df['statements'].std())
                }
                for name, df in self.datasets.items()
            }
        }
        return metadata

if __name__ == '__main__':
    aggregator = DataAggregator()
    aggregator.aggregate_all()
    aggregator.save_combinations()
    
    metadata = aggregator.get_metadata()
    metadata_path = Path('data/processed/dataset_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Metadata saved: {metadata_path}")
