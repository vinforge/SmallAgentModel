#!/usr/bin/env python3
"""
Setup A/B Test for Task 30 Conversational Coherence
===================================================

Creates an A/B test to compare single-stage vs two-stage response pipelines
for validating the conversational coherence improvements.

Usage: python scripts/setup_ab_test.py
"""

import sys
import os
from datetime import datetime, timedelta

# Add SAM modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam.evaluation.ab_testing import ABTestingFramework, ABTestConfig

def setup_coherence_ab_test():
    """Set up the main conversational coherence A/B test."""
    
    print("🧪 Setting up Task 30 Conversational Coherence A/B Test")
    print("=" * 60)
    
    # Initialize A/B testing framework
    ab_framework = ABTestingFramework()
    
    # Create test configuration
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)  # 30-day test
    
    test_config = ABTestConfig(
        test_id="task30_coherence_001",
        test_name="Task 30 Conversational Coherence Test",
        description="Compare single-stage vs two-stage response pipeline for conversational coherence",
        control_pipeline="single_stage",
        treatment_pipeline="two_stage", 
        traffic_split=0.5,  # 50/50 split
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        success_metrics=[
            "response_quality",
            "coherence_score", 
            "persona_consistency",
            "user_satisfaction",
            "response_time"
        ],
        enabled=True
    )
    
    # Create the test
    success = ab_framework.create_test(test_config)
    
    if success:
        print("✅ A/B Test Created Successfully!")
        print(f"   Test ID: {test_config.test_id}")
        print(f"   Test Name: {test_config.test_name}")
        print(f"   Control: {test_config.control_pipeline}")
        print(f"   Treatment: {test_config.treatment_pipeline}")
        print(f"   Traffic Split: {test_config.traffic_split * 100}% to treatment")
        print(f"   Duration: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Status: {'Active' if test_config.enabled else 'Inactive'}")
        
        print("\n📊 Success Metrics:")
        for metric in test_config.success_metrics:
            print(f"   • {metric}")
        
        print("\n🎯 Test Objectives:")
        print("   • Validate conversational coherence improvements")
        print("   • Compare single-stage vs two-stage pipeline performance")
        print("   • Measure persona consistency across sessions")
        print("   • Assess user satisfaction with responses")
        print("   • Monitor response generation time")
        
        print("\n🔬 How it works:")
        print("   • Users are consistently assigned to control or treatment")
        print("   • Control users get single-stage responses")
        print("   • Treatment users get two-stage responses with persona alignment")
        print("   • All interactions are logged for analysis")
        print("   • LLM-as-a-Judge evaluates response quality")
        
        print("\n📈 Expected Outcomes:")
        print("   • Treatment group should show higher coherence scores")
        print("   • Better persona consistency in treatment responses")
        print("   • Improved user satisfaction with personalized responses")
        print("   • Validation of Task 30 implementation success")
        
        return True
    else:
        print("❌ Failed to create A/B test")
        return False

def show_test_status():
    """Show status of all active tests."""
    
    print("\n📊 Current A/B Test Status")
    print("=" * 40)
    
    ab_framework = ABTestingFramework()
    summaries = ab_framework.get_all_test_summaries()
    
    if not summaries:
        print("No active A/B tests found.")
        return
    
    for test_id, summary in summaries.items():
        config = summary['test_config']
        print(f"\n🧪 Test: {config['test_name']}")
        print(f"   ID: {test_id}")
        print(f"   Status: {summary['test_status']}")
        print(f"   Control: {config['control_pipeline']}")
        print(f"   Treatment: {config['treatment_pipeline']}")
        print(f"   Traffic Split: {config['traffic_split'] * 100}%")
        
        print(f"\n📈 Results Summary:")
        print(f"   Total Results: {summary['total_results']}")
        print(f"   Control Count: {summary['control_count']}")
        print(f"   Treatment Count: {summary['treatment_count']}")
        
        if summary['total_results'] > 0:
            print(f"   Control Avg Time: {summary['control_avg_response_time_ms']:.0f}ms")
            print(f"   Treatment Avg Time: {summary['treatment_avg_response_time_ms']:.0f}ms")
            print(f"   Performance Change: {summary['performance_improvement']:.1f}%")
            print(f"   Statistical Significance: {'Yes' if summary['is_statistically_significant'] else 'No'}")

def main():
    """Main function."""
    
    print("🚀 Task 30 Phase 3: A/B Testing Setup")
    print("=" * 50)
    
    # Setup the main A/B test
    success = setup_coherence_ab_test()
    
    if success:
        # Show current status
        show_test_status()
        
        print("\n🎉 A/B Test Setup Complete!")
        print("\n📝 Next Steps:")
        print("   1. Start SAM with the secure interface (port 8502)")
        print("   2. Begin normal conversations with SAM")
        print("   3. Users will be automatically assigned to control/treatment")
        print("   4. Monitor results in the A/B testing dashboard")
        print("   5. Analyze results after sufficient sample size")
        
        print("\n🔍 Monitoring:")
        print("   • Check ab_tests/ directory for result files")
        print("   • Use get_test_summary() for real-time statistics")
        print("   • LLM evaluations will be logged automatically")
        print("   • Response caching will improve performance")
        
        print("\n✨ Expected Benefits:")
        print("   • Quantitative validation of Task 30 improvements")
        print("   • Data-driven insights into conversational coherence")
        print("   • Performance optimization through caching")
        print("   • Scientific validation of persona consistency")
        
    else:
        print("\n❌ A/B Test setup failed. Please check the logs.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
