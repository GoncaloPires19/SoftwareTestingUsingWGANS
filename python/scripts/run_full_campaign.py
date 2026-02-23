#!/usr/bin/env python
"""
Master Script: Orchestrate Full Training Campaign Workflow
1. Aggregate data from multiple sources
2. Run training campaign with multiple configurations 
3. Analyze and compare results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
from datetime import datetime

# Import our campaign modules
from data_aggregator import DataAggregator
from training_campaign import TrainingCampaign
from campaign_analyzer import CampaignAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CampaignOrchestrator:
    """Orchestrate the complete training campaign workflow"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.logs = []
    
    def print_banner(self, title: str):
        """Print formatted banner"""
        print("\n" + "█"*100)
        print("█" + " "*98 + "█")
        print("█" + " " + title.center(96) + " " + "█")
        print("█" + " "*98 + "█")
        print("█"*100 + "\n")
    
    def phase_1_aggregate_data(self):
        """Phase 1: Aggregate data from multiple sources"""
        self.print_banner("PHASE 1: DATA AGGREGATION")
        
        try:
            logger.info("🔄 Starting data aggregation process...\n")
            
            aggregator = DataAggregator()
            aggregator.aggregate_all()
            aggregator.save_combinations()
            
            self.logs.append({
                'phase': 'Data Aggregation',
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("\n✅ Phase 1 Complete: Data aggregation successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 1 Failed: {e}", exc_info=True)
            self.logs.append({
                'phase': 'Data Aggregation',
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def phase_2_training_campaign(self):
        """Phase 2: Execute training campaign"""
        self.print_banner("PHASE 2: TRAINING CAMPAIGN")
        
        try:
            logger.info("🚀 Starting training campaign with multiple configurations...\n")
            
            campaign = TrainingCampaign()
            campaign.run_campaign()
            
            self.logs.append({
                'phase': 'Training Campaign',
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("\n✅ Phase 2 Complete: Training campaign finished")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 2 Failed: {e}", exc_info=True)
            self.logs.append({
                'phase': 'Training Campaign',
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def phase_3_analyze_results(self):
        """Phase 3: Analyze and compare results"""
        self.print_banner("PHASE 3: RESULTS ANALYSIS")
        
        try:
            logger.info("📊 Starting results analysis...\n")
            
            analyzer = CampaignAnalyzer()
            analyzer.run_analysis()
            
            self.logs.append({
                'phase': 'Results Analysis',
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("\n✅ Phase 3 Complete: Results analysis finished")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 3 Failed: {e}", exc_info=True)
            self.logs.append({
                'phase': 'Results Analysis',
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def print_summary(self):
        """Print campaign summary"""
        elapsed = datetime.now() - self.start_time
        
        print("\n" + "="*100)
        print("🎯 TRAINING CAMPAIGN SUMMARY")
        print("="*100 + "\n")
        
        for log in self.logs:
            status_emoji = "✅" if log['status'] == 'SUCCESS' else "❌"
            print(f"  {status_emoji} {log['phase']:25s} → {log['status']}")
        
        print(f"\n⏱️  Total Execution Time: {elapsed}")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "="*100)
        print("📂 OUTPUT LOCATIONS:")
        print("="*100)
        print(f"  📊 Datasets:        data/processed/")
        print(f"  📈 Campaign Results: outputs/campaign/campaign_results.csv")
        print(f"  📝 Analysis:         outputs/campaign/analysis_report.json")
        print(f"  📉 Plots:            outputs/campaign/campaign_comparison.png")
        print(f"  🎯 Metadata:         data/processed/dataset_metadata.json")
        print("\n" + "="*100 + "\n")
    
    def run(self):
        """Execute full orchestration"""
        self.print_banner("WGAN TRAINING CAMPAIGN ORCHESTRATOR")
        
        logger.info("📋 Campaign Stages:")
        logger.info("  1. Aggregate data from Juliet, Defects4J, OWASP, EvoSuite")
        logger.info("  2. Execute training campaign with 8 different configurations")
        logger.info("  3. Analyze results and generate insights")
        logger.info("  4. Generate comparative visualizations")
        
        # Phase 1
        if not self.phase_1_aggregate_data():
            logger.warning("⚠️  Phase 1 failed but continuing anyway...")
        
        # Phase 2
        if not self.phase_2_training_campaign():
            logger.error("❌ Phase 2 failed - cannot continue to Phase 3")
            self.print_summary()
            return False
        
        # Phase 3
        if not self.phase_3_analyze_results():
            logger.error("❌ Phase 3 failed")
            self.print_summary()
            return False
        
        self.print_summary()
        return True

if __name__ == '__main__':
    orchestrator = CampaignOrchestrator()
    success = orchestrator.run()
    sys.exit(0 if success else 1)
