#!/usr/bin/env python3
"""
Phase 3: Final Report Generation & Analysis
Comprehensive analysis of A/B test results with statistical validation
"""

import sys
import json
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HypothesisResult:
    """Result of hypothesis testing."""
    hypothesis: str
    supported: bool
    confidence: str
    evidence: List[str]
    metrics: Dict[str, Any]

class ABTestAnalyzer:
    """Comprehensive analyzer for A/B test results."""
    
    def __init__(self, enhanced_results_file: Path):
        self.results_file = enhanced_results_file
        
        # Load enhanced results
        with open(enhanced_results_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded {len(self.results)} enhanced test results")
        
        # Group results by test arm
        self.arm_results = self._group_by_arm()
    
    def _group_by_arm(self) -> Dict[str, List[Dict]]:
        """Group results by test arm."""
        arms = {'A': [], 'B': [], 'C': []}
        
        for result in self.results:
            arm = result['test_arm']
            if arm in arms:
                arms[arm].append(result)
        
        logger.info(f"Results per arm: A={len(arms['A'])}, B={len(arms['B'])}, C={len(arms['C'])}")
        return arms
    
    def calculate_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each arm."""
        metrics = {}
        
        for arm, results in self.arm_results.items():
            # Filter out error results
            valid_results = [r for r in results if not r.get('error')]
            
            if not valid_results:
                metrics[arm] = {'count': 0}
                continue
            
            latencies = [r['end_to_end_latency_ms'] for r in valid_results]
            tokens = [r['total_tokens_generated'] for r in valid_results]
            
            # Quality scores (if available)
            quality_overall = [r.get('quality_overall', 0) for r in valid_results if r.get('quality_overall')]
            quality_conciseness = [r.get('quality_conciseness', 0) for r in valid_results if r.get('quality_conciseness')]
            
            metrics[arm] = {
                'count': len(valid_results),
                'avg_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'avg_tokens': statistics.mean(tokens),
                'median_tokens': statistics.median(tokens),
                'avg_quality_overall': statistics.mean(quality_overall) if quality_overall else 0,
                'avg_quality_conciseness': statistics.mean(quality_conciseness) if quality_conciseness else 0,
                'error_rate': (len(results) - len(valid_results)) / len(results) if results else 0
            }
        
        return metrics
    
    def analyze_control_decisions(self) -> Dict[str, Any]:
        """Analyze active control decisions for Arm C."""
        arm_c_results = self.arm_results['C']
        
        halt_reasons = {}
        control_decisions = {}
        
        for result in arm_c_results:
            # Count halt reasons
            halt_reason = result.get('tpv_halt_reason')
            if halt_reason:
                halt_reasons[halt_reason] = halt_reasons.get(halt_reason, 0) + 1
            
            # Count control decisions
            control_decision = result.get('control_decision', 'CONTINUE')
            control_decisions[control_decision] = control_decisions.get(control_decision, 0) + 1
        
        total_c = len(arm_c_results)
        
        return {
            'total_tests': total_c,
            'halt_reasons': halt_reasons,
            'control_decisions': control_decisions,
            'intervention_rate': (total_c - control_decisions.get('CONTINUE', 0)) / total_c if total_c > 0 else 0
        }
    
    def test_efficiency_hypothesis(self, metrics: Dict[str, Dict[str, float]]) -> HypothesisResult:
        """Test the efficiency hypothesis: Active Control reduces latency and tokens without quality loss."""
        
        baseline = metrics.get('A', {})
        active = metrics.get('C', {})
        
        if not baseline or not active:
            return HypothesisResult(
                "Efficiency Hypothesis",
                False,
                "Low",
                ["Insufficient data for comparison"],
                {}
            )
        
        # Calculate improvements
        latency_improvement = (baseline['avg_latency_ms'] - active['avg_latency_ms']) / baseline['avg_latency_ms']
        token_reduction = (baseline['avg_tokens'] - active['avg_tokens']) / baseline['avg_tokens']
        quality_change = active['avg_quality_overall'] - baseline['avg_quality_overall']
        
        evidence = []
        supported = True
        
        # Latency analysis
        if latency_improvement > 0.05:  # 5% improvement threshold
            evidence.append(f"Latency improved by {latency_improvement:.1%} ({baseline['avg_latency_ms']:.1f}ms → {active['avg_latency_ms']:.1f}ms)")
        elif latency_improvement < -0.1:  # 10% degradation threshold
            evidence.append(f"Latency degraded by {abs(latency_improvement):.1%}")
            supported = False
        else:
            evidence.append(f"Latency change minimal: {latency_improvement:.1%}")
        
        # Token analysis
        if token_reduction > 0.05:  # 5% reduction threshold
            evidence.append(f"Token usage reduced by {token_reduction:.1%} ({baseline['avg_tokens']:.1f} → {active['avg_tokens']:.1f})")
        elif token_reduction < -0.1:  # 10% increase threshold
            evidence.append(f"Token usage increased by {abs(token_reduction):.1%}")
            supported = False
        else:
            evidence.append(f"Token usage change minimal: {token_reduction:.1%}")
        
        # Quality analysis
        if quality_change >= -0.2:  # Allow small quality decrease
            evidence.append(f"Quality maintained (change: {quality_change:+.1f}/5)")
        else:
            evidence.append(f"Quality significantly decreased: {quality_change:+.1f}/5")
            supported = False
        
        confidence = "High" if supported and (latency_improvement > 0.1 or token_reduction > 0.1) else "Medium" if supported else "Low"
        
        return HypothesisResult(
            "Efficiency Hypothesis",
            supported,
            confidence,
            evidence,
            {
                'latency_improvement': latency_improvement,
                'token_reduction': token_reduction,
                'quality_change': quality_change
            }
        )
    
    def test_quality_hypothesis(self, metrics: Dict[str, Dict[str, float]]) -> HypothesisResult:
        """Test the quality hypothesis: Active Control improves answer quality."""
        
        baseline = metrics.get('A', {})
        active = metrics.get('C', {})
        
        if not baseline or not active:
            return HypothesisResult(
                "Quality Hypothesis",
                False,
                "Low",
                ["Insufficient data for comparison"],
                {}
            )
        
        # Quality improvements
        overall_improvement = active['avg_quality_overall'] - baseline['avg_quality_overall']
        conciseness_improvement = active['avg_quality_conciseness'] - baseline['avg_quality_conciseness']
        
        evidence = []
        supported = True
        
        # Overall quality
        if overall_improvement >= 0.3:  # 0.3 point improvement threshold
            evidence.append(f"Overall quality improved by {overall_improvement:+.2f}/5")
        elif overall_improvement <= -0.3:
            evidence.append(f"Overall quality decreased by {abs(overall_improvement):.2f}/5")
            supported = False
        else:
            evidence.append(f"Overall quality change minimal: {overall_improvement:+.2f}/5")
        
        # Conciseness
        if conciseness_improvement >= 0.3:
            evidence.append(f"Conciseness improved by {conciseness_improvement:+.2f}/5")
        elif conciseness_improvement <= -0.3:
            evidence.append(f"Conciseness decreased by {abs(conciseness_improvement):.2f}/5")
            supported = False
        else:
            evidence.append(f"Conciseness change minimal: {conciseness_improvement:+.2f}/5")
        
        confidence = "High" if supported and overall_improvement > 0.5 else "Medium" if supported else "Low"
        
        return HypothesisResult(
            "Quality Hypothesis",
            supported,
            confidence,
            evidence,
            {
                'overall_improvement': overall_improvement,
                'conciseness_improvement': conciseness_improvement
            }
        )
    
    def test_user_experience_hypothesis(self, metrics: Dict[str, Dict[str, float]], control_analysis: Dict[str, Any]) -> HypothesisResult:
        """Test the user experience hypothesis: Faster, more concise answers improve UX."""
        
        baseline = metrics.get('A', {})
        active = metrics.get('C', {})
        
        if not baseline or not active:
            return HypothesisResult(
                "User Experience Hypothesis",
                False,
                "Low",
                ["Insufficient data for comparison"],
                {}
            )
        
        # UX factors
        latency_improvement = (baseline['avg_latency_ms'] - active['avg_latency_ms']) / baseline['avg_latency_ms']
        conciseness_improvement = active['avg_quality_conciseness'] - baseline['avg_quality_conciseness']
        intervention_rate = control_analysis['intervention_rate']
        
        evidence = []
        supported = True
        
        # Speed factor
        if latency_improvement > 0.1:
            evidence.append(f"Faster responses improve perceived performance ({latency_improvement:.1%} faster)")
        elif latency_improvement < -0.05:
            evidence.append(f"Slower responses may hurt user experience ({abs(latency_improvement):.1%} slower)")
            supported = False
        
        # Conciseness factor
        if conciseness_improvement > 0.2:
            evidence.append(f"More concise responses improve readability (+{conciseness_improvement:.2f}/5)")
        elif conciseness_improvement < -0.2:
            evidence.append(f"Less concise responses may hurt readability ({conciseness_improvement:.2f}/5)")
            supported = False
        
        # Transparency factor
        if intervention_rate > 0.1:
            evidence.append(f"Active control provides transparency ({intervention_rate:.1%} intervention rate)")
        else:
            evidence.append(f"Low intervention rate ({intervention_rate:.1%}) - limited transparency benefit")
        
        confidence = "High" if supported and latency_improvement > 0.15 else "Medium" if supported else "Low"
        
        return HypothesisResult(
            "User Experience Hypothesis",
            supported,
            confidence,
            evidence,
            {
                'latency_improvement': latency_improvement,
                'conciseness_improvement': conciseness_improvement,
                'intervention_rate': intervention_rate
            }
        )
    
    def generate_final_report(self) -> str:
        """Generate comprehensive final report."""
        logger.info("📊 Generating Final A/B Test Report")
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        control_analysis = self.analyze_control_decisions()
        
        # Test hypotheses
        efficiency_result = self.test_efficiency_hypothesis(metrics)
        quality_result = self.test_quality_hypothesis(metrics)
        ux_result = self.test_user_experience_hypothesis(metrics, control_analysis)
        
        # Generate report
        report = f"""# Phase 3: A/B Testing Final Report
## SAM Active Reasoning Control Validation

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Test Results:** {len(self.results)} total test executions

---

## Executive Summary

This report presents the results of comprehensive A/B testing to validate SAM's Active Reasoning Control system. We tested three configurations:

- **Arm A (Baseline)**: No TPV - Control group
- **Arm B (Monitoring)**: Phase 1 passive monitoring
- **Arm C (Active Control)**: Phase 2 active reasoning control

## Performance Metrics

### Arm A (Baseline)
- **Tests:** {metrics.get('A', {}).get('count', 0)}
- **Avg Latency:** {metrics.get('A', {}).get('avg_latency_ms', 0):.1f}ms
- **Avg Tokens:** {metrics.get('A', {}).get('avg_tokens', 0):.1f}
- **Avg Quality:** {metrics.get('A', {}).get('avg_quality_overall', 0):.2f}/5
- **Error Rate:** {metrics.get('A', {}).get('error_rate', 0):.1%}

### Arm B (Monitoring)
- **Tests:** {metrics.get('B', {}).get('count', 0)}
- **Avg Latency:** {metrics.get('B', {}).get('avg_latency_ms', 0):.1f}ms
- **Avg Tokens:** {metrics.get('B', {}).get('avg_tokens', 0):.1f}
- **Avg Quality:** {metrics.get('B', {}).get('avg_quality_overall', 0):.2f}/5
- **Error Rate:** {metrics.get('B', {}).get('error_rate', 0):.1%}

### Arm C (Active Control)
- **Tests:** {metrics.get('C', {}).get('count', 0)}
- **Avg Latency:** {metrics.get('C', {}).get('avg_latency_ms', 0):.1f}ms
- **Avg Tokens:** {metrics.get('C', {}).get('avg_tokens', 0):.1f}
- **Avg Quality:** {metrics.get('C', {}).get('avg_quality_overall', 0):.2f}/5
- **Error Rate:** {metrics.get('C', {}).get('error_rate', 0):.1%}
- **Intervention Rate:** {control_analysis['intervention_rate']:.1%}

## Active Control Analysis

**Total Interventions:** {control_analysis['total_tests'] - control_analysis['control_decisions'].get('CONTINUE', 0)}/{control_analysis['total_tests']}

**Control Decisions:**
{chr(10).join([f"- {decision}: {count}" for decision, count in control_analysis['control_decisions'].items()])}

**Halt Reasons:**
{chr(10).join([f"- {reason}: {count}" for reason, count in control_analysis['halt_reasons'].items()]) if control_analysis['halt_reasons'] else "- No halts recorded"}

## Hypothesis Testing Results

### 1. {efficiency_result.hypothesis}
**Result:** {'✅ SUPPORTED' if efficiency_result.supported else '❌ NOT SUPPORTED'}
**Confidence:** {efficiency_result.confidence}

**Evidence:**
{chr(10).join([f"- {evidence}" for evidence in efficiency_result.evidence])}

### 2. {quality_result.hypothesis}
**Result:** {'✅ SUPPORTED' if quality_result.supported else '❌ NOT SUPPORTED'}
**Confidence:** {quality_result.confidence}

**Evidence:**
{chr(10).join([f"- {evidence}" for evidence in quality_result.evidence])}

### 3. {ux_result.hypothesis}
**Result:** {'✅ SUPPORTED' if ux_result.supported else '❌ NOT SUPPORTED'}
**Confidence:** {ux_result.confidence}

**Evidence:**
{chr(10).join([f"- {evidence}" for evidence in ux_result.evidence])}

## Final Recommendation

**Overall Assessment:** {self._get_overall_assessment([efficiency_result, quality_result, ux_result])}

**Go/No-Go Decision:** {self._get_go_no_go_decision([efficiency_result, quality_result, ux_result])}

---

*This report was generated automatically from A/B test data. For detailed analysis and raw data, see the accompanying JSON files.*
"""
        
        return report
    
    def _get_overall_assessment(self, results: List[HypothesisResult]) -> str:
        """Get overall assessment based on hypothesis results."""
        supported_count = sum(1 for r in results if r.supported)
        high_confidence_count = sum(1 for r in results if r.confidence == "High")
        
        if supported_count == 3 and high_confidence_count >= 2:
            return "STRONG SUCCESS - All hypotheses supported with high confidence"
        elif supported_count >= 2:
            return "SUCCESS - Majority of hypotheses supported"
        elif supported_count == 1:
            return "MIXED RESULTS - Limited support for hypotheses"
        else:
            return "UNSUCCESSFUL - Hypotheses not supported by data"
    
    def _get_go_no_go_decision(self, results: List[HypothesisResult]) -> str:
        """Get Go/No-Go decision based on results."""
        supported_count = sum(1 for r in results if r.supported)
        
        if supported_count >= 2:
            return "🟢 GO - Enable Active Control by default"
        elif supported_count == 1:
            return "🟡 CONDITIONAL GO - Enable with monitoring"
        else:
            return "🔴 NO-GO - Keep current system"

def main():
    """Main report generation function."""
    if len(sys.argv) != 2:
        logger.error("Usage: python generate_final_report.py <enhanced_results_file.json>")
        return 1
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return 1
    
    logger.info("🚀 Starting Final Report Generation")
    
    # Create analyzer
    analyzer = ABTestAnalyzer(results_file)
    
    # Generate report
    report = analyzer.generate_final_report()
    
    # Save report
    report_file = results_file.parent / f"final_report_{int(time.time())}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📁 Final report saved to: {report_file}")
    logger.info("🎉 Phase 3 A/B Testing Complete!")
    
    # Print summary to console
    print("\n" + "="*60)
    print("PHASE 3 A/B TESTING COMPLETE")
    print("="*60)
    print(f"📊 Report: {report_file}")
    print("🎯 Ready for Go/No-Go decision!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
