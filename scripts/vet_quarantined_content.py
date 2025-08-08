#!/usr/bin/env python3
"""
Automated Content Vetting Tool - Phase 7.2

This script processes quarantined web content through SAM's automated
vetting engine, analyzing content for security risks, bias, and credibility.

Features:
- Single file or batch processing
- Comprehensive security analysis
- Risk assessment and recommendations
- Detailed reporting and statistics
- Archive management

Usage:
    python scripts/vet_quarantined_content.py
    python scripts/vet_quarantined_content.py --file specific_file.json
    python scripts/vet_quarantined_content.py --batch --threshold 0.8
    python scripts/vet_quarantined_content.py --status
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from web_retrieval.content_evaluator import ContentEvaluator
    from web_retrieval.vetting_pipeline import VettingPipeline
except ImportError as e:
    print(f"❌ Error importing vetting modules: {e}")
    print("Make sure you're running from the SAM project root directory.")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def setup_directories() -> tuple:
    """Setup required directories for vetting process."""
    quarantine_dir = project_root / "quarantine"
    vetted_dir = project_root / "vetted"
    archive_dir = project_root / "archive"
    
    # Create directories if they don't exist
    vetted_dir.mkdir(exist_ok=True)
    archive_dir.mkdir(exist_ok=True)
    
    return quarantine_dir, vetted_dir, archive_dir


def display_file_result(result: dict, verbose: bool = False):
    """Display results for a single file processing."""
    
    if result['status'] == 'success':
        print(f"✅ {Path(result['original_file']).name}")
        print(f"   📊 Overall Score: {result['overall_score']:.3f}")
        print(f"   ⚖️ Recommendation: {result['recommendation']}")
        print(f"   🎯 Confidence: {result['confidence']:.1%}")
        print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")
        print(f"   📄 Vetted File: {Path(result['vetted_file']).name}")
        
        if verbose:
            print(f"   📦 Archived: {Path(result['archived_file']).name}")
    
    else:
        print(f"❌ {Path(result['original_file']).name}")
        print(f"   🚫 Error: {result['error']}")
        print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")


def display_batch_summary(batch_result: dict):
    """Display comprehensive batch processing summary."""
    
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    
    # Overall statistics
    print(f"📁 Total Files: {batch_result['total_files']}")
    print(f"✅ Successful: {batch_result['successful']}")
    print(f"❌ Failed: {batch_result['failed']}")
    print(f"⏱️ Total Time: {batch_result['total_processing_time']:.2f}s")
    print(f"📈 Average Time: {batch_result['average_processing_time']:.2f}s per file")
    
    # Recommendation breakdown
    recommendations = batch_result['recommendations']
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   🟢 PASS: {recommendations['PASS']} files")
    print(f"   🟡 REVIEW: {recommendations['REVIEW']} files")
    print(f"   🔴 FAIL: {recommendations['FAIL']} files")
    
    # Calculate percentages
    total_successful = batch_result['successful']
    if total_successful > 0:
        pass_rate = (recommendations['PASS'] / total_successful) * 100
        review_rate = (recommendations['REVIEW'] / total_successful) * 100
        fail_rate = (recommendations['FAIL'] / total_successful) * 100
        
        print(f"\n📊 SUCCESS RATES:")
        print(f"   🟢 Pass Rate: {pass_rate:.1f}%")
        print(f"   🟡 Review Rate: {review_rate:.1f}%")
        print(f"   🔴 Fail Rate: {fail_rate:.1f}%")


def display_detailed_analysis(enriched_data: dict):
    """Display detailed analysis results for a file."""
    
    vetting = enriched_data.get('vetting_results', {})
    scores = vetting.get('scores', {})
    
    print(f"\n🔍 DETAILED ANALYSIS:")
    print(f"   📊 Credibility: {scores.get('credibility', 0):.3f}")
    print(f"   🎯 Persuasion: {scores.get('persuasion', 0):.3f}")
    print(f"   🔮 Speculation: {scores.get('speculation', 0):.3f}")
    print(f"   🛡️ Purity: {scores.get('purity', 0):.3f}")
    print(f"   🏆 Source Reputation: {scores.get('source_reputation', 0):.3f}")
    
    # Risk factors
    risk_factors = vetting.get('risk_assessment', {}).get('risk_factors', [])
    if risk_factors:
        print(f"\n⚠️ RISK FACTORS:")
        for risk in risk_factors:
            severity_emoji = {
                'critical': '🚨',
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(risk.get('severity', 'medium'), '⚠️')
            
            print(f"   {severity_emoji} {risk.get('description', 'Unknown risk')}")
            print(f"      Dimension: {risk.get('dimension', 'unknown')}")
            print(f"      Score: {risk.get('score', 0):.3f}")
    
    # Sanitization results
    sanitization = vetting.get('sanitization', {})
    if sanitization:
        print(f"\n🧹 SANITIZATION:")
        print(f"   🗑️ Removed Elements: {len(sanitization.get('removed_elements', []))}")
        print(f"   🚨 Suspicious Patterns: {len(sanitization.get('suspicious_patterns', []))}")
        print(f"   🛡️ Purity Score: {sanitization.get('purity_score', 0):.3f}")


def show_system_status(pipeline: VettingPipeline):
    """Display system status and statistics."""
    
    print("🔧 SYSTEM STATUS")
    print("="*50)
    
    # Pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"📊 Files Processed: {stats['files_processed']}")
    print(f"✅ Files Passed: {stats['files_passed']}")
    print(f"🟡 Files for Review: {stats['files_review']}")
    print(f"❌ Files Failed: {stats['files_failed']}")
    print(f"🚫 Errors: {stats['errors']}")
    
    if stats['files_processed'] > 0:
        print(f"📈 Pass Rate: {stats['pass_rate']:.1%}")
        print(f"⏱️ Avg Processing Time: {stats['average_processing_time']:.2f}s")
    
    # Quarantine summary
    quarantine_summary = pipeline.get_quarantine_summary()
    print(f"\n📁 QUARANTINE STATUS:")
    print(f"   📄 Total Files: {quarantine_summary['total_files']}")
    print(f"   🔄 Ready for Processing: {quarantine_summary['ready_for_processing']}")
    
    # Evaluator configuration
    evaluator_config = stats['evaluator_config']
    print(f"\n⚙️ EVALUATOR CONFIG:")
    print(f"   📋 Profile: {evaluator_config['vetting_profile']}")
    print(f"   🎯 Threshold: {evaluator_config['safety_threshold']}")
    print(f"   📊 Analysis Mode: {evaluator_config['analysis_mode']}")
    print(f"   🧠 Dimension Prober: {'✅' if evaluator_config['dimension_prober_available'] else '❌'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Automated content vetting for quarantined web content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in quarantine
  python scripts/vet_quarantined_content.py --batch
  
  # Process specific file
  python scripts/vet_quarantined_content.py --file example.json
  
  # Show system status
  python scripts/vet_quarantined_content.py --status
  
  # Batch process with custom threshold
  python scripts/vet_quarantined_content.py --batch --threshold 0.8
        """
    )
    
    # Main operation modes
    parser.add_argument('--file', help='Process specific file in quarantine')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all files in quarantine')
    parser.add_argument('--status', action='store_true',
                       help='Show system status and statistics')
    
    # Configuration options
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Safety threshold for recommendations (default: 0.7)')
    parser.add_argument('--profile', default='web_vetting',
                       help='Vetting profile to use (default: web_vetting)')
    parser.add_argument('--mode', default='balanced',
                       choices=['security_focus', 'quality_focus', 'balanced'],
                       help='Analysis mode (default: balanced)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed analysis results')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (errors only)')
    
    # Maintenance options
    parser.add_argument('--cleanup', type=int, metavar='DAYS',
                       help='Clean up archive files older than DAYS')
    
    args = parser.parse_args()
    
    # Setup logging
    if not args.quiet:
        logger = setup_logging(args.verbose)
    
    # Validate arguments
    if not any([args.file, args.batch, args.status, args.cleanup]):
        print("❓ Please specify an operation: --file, --batch, --status, or --cleanup")
        parser.print_help()
        sys.exit(1)
    
    # Setup directories
    quarantine_dir, vetted_dir, archive_dir = setup_directories()
    
    # Initialize vetting system
    if not args.quiet:
        print("🔧 Initializing automated vetting engine...")
    
    try:
        evaluator = ContentEvaluator(
            vetting_profile=args.profile,
            safety_threshold=args.threshold,
            analysis_mode=args.mode
        )
        
        pipeline = VettingPipeline(
            content_evaluator=evaluator,
            quarantine_dir=str(quarantine_dir),
            vetted_dir=str(vetted_dir),
            archive_dir=str(archive_dir)
        )
        
        if not args.quiet:
            print("✅ Vetting engine ready")
            
    except Exception as e:
        print(f"❌ Failed to initialize vetting engine: {e}")
        sys.exit(1)
    
    # Execute requested operation
    try:
        if args.status:
            show_system_status(pipeline)
            
        elif args.cleanup:
            if not args.quiet:
                print(f"🧹 Cleaning up archive files older than {args.cleanup} days...")
            
            cleanup_result = pipeline.cleanup_old_archives(args.cleanup)
            print(f"✅ Cleaned up {cleanup_result['files_removed']} old files")
            
        elif args.file:
            # Process single file
            filepath = quarantine_dir / args.file
            
            if not filepath.exists():
                print(f"❌ File not found: {filepath}")
                sys.exit(1)
            
            if not args.quiet:
                print(f"🔄 Processing: {args.file}")
            
            result = pipeline.process_quarantined_file(filepath)
            
            if not args.quiet:
                display_file_result(result, args.verbose)
                
                if args.detailed and result['status'] == 'success':
                    # Load enriched data for detailed display
                    with open(result['vetted_file'], 'r') as f:
                        enriched_data = json.load(f)
                    display_detailed_analysis(enriched_data)
            
        elif args.batch:
            # Process all files
            if not args.quiet:
                print("🔄 Starting batch processing...")
            
            batch_result = pipeline.process_all_quarantined_files()
            
            if batch_result['status'] == 'no_files':
                print("📭 No files found in quarantine directory")
                return
            
            # Display individual results if verbose
            if args.verbose and not args.quiet:
                print("\n📄 INDIVIDUAL RESULTS:")
                for result in batch_result['results']:
                    display_file_result(result, False)
                    print()
            
            # Display summary
            if not args.quiet:
                display_batch_summary(batch_result)
                
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
