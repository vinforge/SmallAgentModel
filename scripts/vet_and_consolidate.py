#!/usr/bin/env python3
"""
Vet and Consolidate Script - Complete pipeline for vetting and knowledge consolidation
"""

import sys
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def vet_quarantined_content(quarantine_dir: str = "quarantine", 
                           approved_dir: str = "approved") -> dict:
    """Vet quarantined content and move approved items."""
    try:
        from web_retrieval.content_evaluator import ContentEvaluator
        
        quarantine_path = Path(quarantine_dir)
        approved_path = Path(approved_dir)
        
        # Create approved directory
        approved_path.mkdir(exist_ok=True)
        
        if not quarantine_path.exists():
            return {'success': True, 'message': 'No quarantine directory found', 'vetted_count': 0}
        
        # Get quarantined files
        quarantined_files = list(quarantine_path.glob('*.json'))
        
        # Filter out metadata files
        content_files = [f for f in quarantined_files if not f.name.endswith('_metadata.json')]
        
        if not content_files:
            return {'success': True, 'message': 'No content files to vet', 'vetted_count': 0}
        
        evaluator = ContentEvaluator()
        vetted_count = 0
        approved_count = 0
        rejected_count = 0
        
        for file_path in content_files:
            try:
                # Load quarantined content
                with open(file_path, 'r', encoding='utf-8') as f:
                    quarantined_data = json.load(f)
                
                # Extract content for evaluation based on file structure
                content_for_evaluation = extract_content_for_evaluation(quarantined_data)
                
                if not content_for_evaluation:
                    logger.warning(f"No evaluable content found in {file_path}")
                    continue
                
                # Evaluate content using the correct method
                evaluation = evaluator.evaluate_quarantined_file(str(file_path))
                
                # Check if content passes vetting
                recommendation = evaluation.get('recommendation', {})
                action = recommendation.get('action', 'FAIL')
                
                if action in ['PASS', 'REVIEW']:
                    # Move to approved directory
                    approved_file = approved_path / file_path.name
                    
                    # Add evaluation metadata
                    approved_data = {
                        **quarantined_data,
                        'vetting_result': evaluation,
                        'approved_at': datetime.now().isoformat(),
                        'vetting_action': action
                    }
                    
                    with open(approved_file, 'w', encoding='utf-8') as f:
                        json.dump(approved_data, f, indent=2, ensure_ascii=False)
                    
                    # Remove from quarantine
                    file_path.unlink()
                    
                    approved_count += 1
                    logger.info(f"Approved: {file_path.name} (Action: {action})")
                else:
                    rejected_count += 1
                    logger.info(f"Rejected: {file_path.name} (Action: {action})")
                
                vetted_count += 1
                
            except Exception as e:
                logger.error(f"Error vetting {file_path}: {e}")
                continue
        
        return {
            'success': True,
            'vetted_count': vetted_count,
            'approved_count': approved_count,
            'rejected_count': rejected_count,
            'total_files': len(content_files)
        }
        
    except Exception as e:
        logger.error(f"Vetting process failed: {e}")
        return {'success': False, 'error': str(e)}

def extract_content_for_evaluation(quarantined_data: dict) -> dict:
    """Extract content from quarantined data for evaluation."""
    try:
        # Handle intelligent web system format
        if 'result' in quarantined_data and 'data' in quarantined_data['result']:
            return quarantined_data['result']['data']
        
        # Handle direct articles format
        elif 'articles' in quarantined_data:
            return quarantined_data
        
        # Handle scraped data format
        elif 'scraped_data' in quarantined_data:
            return quarantined_data['scraped_data']
        
        # Handle direct content format
        elif 'content' in quarantined_data:
            return quarantined_data
        
        # Fallback - return the whole data
        else:
            return quarantined_data
            
    except Exception as e:
        logger.error(f"Error extracting content for evaluation: {e}")
        return {}

def consolidate_approved_content(approved_dir: str = "approved") -> dict:
    """Consolidate approved content into knowledge base."""
    try:
        from knowledge_consolidation.consolidation_manager import ConsolidationManager
        
        manager = ConsolidationManager()
        result = manager.consolidate_approved_content(approved_dir)
        
        return result
        
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main function for vetting and consolidation."""
    parser = argparse.ArgumentParser(description='Vet quarantined content and consolidate into knowledge base')
    parser.add_argument('--quarantine-dir', default='quarantine', help='Quarantine directory')
    parser.add_argument('--approved-dir', default='approved', help='Approved content directory')
    parser.add_argument('--vet-only', action='store_true', help='Only vet content, do not consolidate')
    parser.add_argument('--consolidate-only', action='store_true', help='Only consolidate, do not vet')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    results = {}
    
    # Step 1: Vet quarantined content (unless consolidate-only)
    if not args.consolidate_only:
        logger.info("Starting content vetting process...")
        vetting_result = vet_quarantined_content(args.quarantine_dir, args.approved_dir)
        results['vetting'] = vetting_result
        
        if vetting_result['success']:
            print(f"Vetting completed: {vetting_result['approved_count']} approved, "
                  f"{vetting_result['rejected_count']} rejected out of {vetting_result['total_files']} files")
        else:
            print(f"Vetting failed: {vetting_result.get('error')}")
            return 1
    
    # Step 2: Consolidate approved content (unless vet-only)
    if not args.vet_only:
        logger.info("Starting knowledge consolidation process...")
        consolidation_result = consolidate_approved_content(args.approved_dir)
        results['consolidation'] = consolidation_result
        
        if consolidation_result['success']:
            integrated_count = consolidation_result.get('total_items_integrated', 0)
            processed_count = consolidation_result.get('total_items_processed', 0)
            print(f"Consolidation completed: {integrated_count} items integrated out of {processed_count} processed")
        else:
            print(f"Consolidation failed: {consolidation_result.get('error')}")
            return 1
    
    # Summary
    if not args.quiet:
        print("\n=== Process Summary ===")
        if 'vetting' in results:
            vetting = results['vetting']
            print(f"Vetting: {vetting['approved_count']} approved, {vetting['rejected_count']} rejected")
        
        if 'consolidation' in results:
            consolidation = results['consolidation']
            print(f"Consolidation: {consolidation.get('total_items_integrated', 0)} items integrated")
        
        print(f"Completed at: {datetime.now().isoformat()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
