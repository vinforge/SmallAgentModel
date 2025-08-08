#!/usr/bin/env python3
"""
SAM Model Leaderboard Generator
===============================

Generates comprehensive MODEL_LEADERBOARD.md report with summary tables,
cost-benefit analysis, and performance charts from scored evaluation results.

Usage:
    python generate_leaderboard_report.py scored_results.jsonl
    python generate_leaderboard_report.py --input scored_results.jsonl --output LEADERBOARD.md
    python generate_leaderboard_report.py --input scored_results.jsonl --format html

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
from collections import defaultdict

# Add SAM core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeaderboardGenerator:
    """Generates comprehensive model leaderboard reports."""
    
    def __init__(self):
        self.results = []
        self.metadata = None
        self.model_stats = {}
        self.category_stats = {}
        self.overall_stats = {}
        
    def load_scored_results(self, input_file: str) -> List[Dict[str, Any]]:
        """Load scored evaluation results from JSONL file."""
        results = []
        metadata = None
        
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        if line_num == 1 and "metadata" in data:
                            metadata = data["metadata"]
                            self.metadata = metadata
                            logger.info(f"📊 Loaded metadata: {metadata.get('total_results', 'unknown')} scored results")
                        else:
                            results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
        
        self.results = results
        logger.info(f"📚 Loaded {len(results)} scored results")
        return results
    
    def analyze_results(self):
        """Analyze results to generate statistics."""
        logger.info("📈 Analyzing results...")
        
        # Initialize stats
        self.model_stats = defaultdict(lambda: {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'categories': set(),
            'scores': [],
            'weighted_scores': [],
            'performance': {
                'total_time': 0.0,
                'total_tokens': 0,
                'avg_time': 0.0,
                'avg_tokens': 0.0,
                'tokens_per_second': 0.0
            },
            'category_scores': defaultdict(list)
        })
        
        self.category_stats = defaultdict(lambda: {
            'total_evaluations': 0,
            'models': set(),
            'avg_score': 0.0,
            'model_scores': defaultdict(list)
        })
        
        # Process each result
        for result in self.results:
            model_name = result['model_name']
            category = result['category']
            
            # Update model stats
            model_stat = self.model_stats[model_name]
            model_stat['total_evaluations'] += 1
            model_stat['categories'].add(category)
            
            if result.get('success', False) and 'scoring' in result:
                model_stat['successful_evaluations'] += 1
                
                # Scoring data
                scoring = result['scoring']
                if 'weighted_score' in scoring:
                    weighted_score = scoring['weighted_score']
                    model_stat['weighted_scores'].append(weighted_score)
                    model_stat['category_scores'][category].append(weighted_score)
                
                # Performance data
                perf = result.get('performance', {})
                model_stat['performance']['total_time'] += perf.get('inference_time', 0)
                model_stat['performance']['total_tokens'] += perf.get('completion_tokens', 0)
            
            # Update category stats
            cat_stat = self.category_stats[category]
            cat_stat['total_evaluations'] += 1
            cat_stat['models'].add(model_name)
            
            if result.get('success', False) and 'scoring' in result:
                scoring = result['scoring']
                if 'weighted_score' in scoring:
                    weighted_score = scoring['weighted_score']
                    cat_stat['model_scores'][model_name].append(weighted_score)
        
        # Calculate averages
        for model_name, stats in self.model_stats.items():
            if stats['weighted_scores']:
                stats['avg_score'] = sum(stats['weighted_scores']) / len(stats['weighted_scores'])
            
            if stats['successful_evaluations'] > 0:
                perf = stats['performance']
                perf['avg_time'] = perf['total_time'] / stats['successful_evaluations']
                perf['avg_tokens'] = perf['total_tokens'] / stats['successful_evaluations']
                if perf['total_time'] > 0:
                    perf['tokens_per_second'] = perf['total_tokens'] / perf['total_time']
        
        for category, stats in self.category_stats.items():
            all_scores = []
            for model_scores in stats['model_scores'].values():
                all_scores.extend(model_scores)
            if all_scores:
                stats['avg_score'] = sum(all_scores) / len(all_scores)
        
        # Overall stats
        all_scores = []
        for stats in self.model_stats.values():
            all_scores.extend(stats['weighted_scores'])
        
        self.overall_stats = {
            'total_models': len(self.model_stats),
            'total_categories': len(self.category_stats),
            'total_evaluations': len(self.results),
            'successful_evaluations': len([r for r in self.results if r.get('success', False)]),
            'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0.0
        }
        
        logger.info(f"✅ Analysis complete: {self.overall_stats['total_models']} models, {self.overall_stats['total_categories']} categories")
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive Markdown leaderboard report."""
        
        # Report header
        report = f"""# SAM Model Leaderboard Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Benchmark Version:** {self.metadata.get('benchmark_version', 'Unknown')}  
**Judge Model:** {self.metadata.get('judge_model', 'Unknown')}  
**Total Evaluations:** {self.overall_stats['total_evaluations']}  

## Executive Summary

This report presents a comprehensive evaluation of {self.overall_stats['total_models']} AI models across {self.overall_stats['total_categories']} benchmark categories using the SAM Core Benchmark Suite. The evaluation measures model performance on key capabilities including question answering, reasoning, code generation, and more.

**Key Findings:**
- **Success Rate:** {(self.overall_stats['successful_evaluations']/self.overall_stats['total_evaluations']*100):.1f}%
- **Average Score:** {self.overall_stats['avg_score']:.2f}/5.0
- **Categories Tested:** {', '.join(sorted(self.category_stats.keys()))}

---

## 🏆 Overall Model Leaderboard

"""
        
        # Main leaderboard table
        report += self._generate_main_leaderboard_table()
        
        # Performance analysis
        report += "\n---\n\n## 📊 Performance Analysis\n\n"
        report += self._generate_performance_analysis()
        
        # Category breakdown
        report += "\n---\n\n## 📋 Category Breakdown\n\n"
        report += self._generate_category_breakdown()
        
        # Cost-benefit analysis
        report += "\n---\n\n## 💰 Cost-Benefit Analysis\n\n"
        report += self._generate_cost_benefit_analysis()
        
        # Detailed model profiles
        report += "\n---\n\n## 🔍 Detailed Model Profiles\n\n"
        report += self._generate_detailed_profiles()
        
        # Methodology
        report += "\n---\n\n## 📖 Methodology\n\n"
        report += self._generate_methodology_section()
        
        return report
    
    def _generate_main_leaderboard_table(self) -> str:
        """Generate the main leaderboard table."""
        
        # Sort models by average score
        sorted_models = sorted(
            self.model_stats.items(),
            key=lambda x: x[1].get('avg_score', 0),
            reverse=True
        )
        
        table = """| Rank | Model | Avg Score | Success Rate | Avg Time (s) | Avg Tokens | Tokens/sec | Categories |
|------|-------|-----------|--------------|--------------|------------|------------|------------|
"""
        
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            success_rate = (stats['successful_evaluations'] / stats['total_evaluations'] * 100) if stats['total_evaluations'] > 0 else 0
            avg_score = stats.get('avg_score', 0)
            perf = stats['performance']
            categories_count = len(stats['categories'])
            
            # Format model name with emoji based on rank
            if rank == 1:
                model_display = f"🥇 {model_name}"
            elif rank == 2:
                model_display = f"🥈 {model_name}"
            elif rank == 3:
                model_display = f"🥉 {model_name}"
            else:
                model_display = model_name
            
            table += f"| {rank} | {model_display} | {avg_score:.2f}/5.0 | {success_rate:.1f}% | {perf['avg_time']:.2f} | {perf['avg_tokens']:.0f} | {perf['tokens_per_second']:.1f} | {categories_count} |\n"
        
        return table
    
    def _generate_performance_analysis(self) -> str:
        """Generate performance analysis section."""
        
        analysis = "### Speed vs Quality Analysis\n\n"
        
        # Find fastest and highest quality models
        fastest_model = min(self.model_stats.items(), key=lambda x: x[1]['performance']['avg_time'])
        highest_quality = max(self.model_stats.items(), key=lambda x: x[1].get('avg_score', 0))
        most_efficient = max(self.model_stats.items(), key=lambda x: x[1]['performance']['tokens_per_second'])
        
        analysis += f"**🚀 Fastest Model:** {fastest_model[0]} ({fastest_model[1]['performance']['avg_time']:.2f}s avg)\n"
        analysis += f"**🎯 Highest Quality:** {highest_quality[0]} ({highest_quality[1].get('avg_score', 0):.2f}/5.0 avg)\n"
        analysis += f"**⚡ Most Efficient:** {most_efficient[0]} ({most_efficient[1]['performance']['tokens_per_second']:.1f} tokens/sec)\n\n"
        
        # Performance comparison table
        analysis += "### Performance Comparison\n\n"
        analysis += "| Model | Quality Score | Speed Rank | Efficiency Rank | Overall Rank |\n"
        analysis += "|-------|---------------|------------|-----------------|-------------|\n"
        
        # Calculate ranks
        models_by_quality = sorted(self.model_stats.items(), key=lambda x: x[1].get('avg_score', 0), reverse=True)
        models_by_speed = sorted(self.model_stats.items(), key=lambda x: x[1]['performance']['avg_time'])
        models_by_efficiency = sorted(self.model_stats.items(), key=lambda x: x[1]['performance']['tokens_per_second'], reverse=True)
        
        for model_name, stats in models_by_quality:
            quality_rank = next(i for i, (name, _) in enumerate(models_by_quality, 1) if name == model_name)
            speed_rank = next(i for i, (name, _) in enumerate(models_by_speed, 1) if name == model_name)
            efficiency_rank = next(i for i, (name, _) in enumerate(models_by_efficiency, 1) if name == model_name)
            overall_rank = quality_rank  # Using quality as primary rank
            
            analysis += f"| {model_name} | {stats.get('avg_score', 0):.2f}/5.0 | #{speed_rank} | #{efficiency_rank} | #{overall_rank} |\n"
        
        return analysis
    
    def _generate_category_breakdown(self) -> str:
        """Generate category breakdown section."""
        
        breakdown = ""
        
        for category, stats in sorted(self.category_stats.items()):
            breakdown += f"### {category.replace('_', ' ').title()}\n\n"
            breakdown += f"**Average Score:** {stats['avg_score']:.2f}/5.0  \n"
            breakdown += f"**Models Tested:** {len(stats['models'])}  \n"
            breakdown += f"**Total Evaluations:** {stats['total_evaluations']}  \n\n"
            
            # Model performance in this category
            breakdown += "| Model | Score | Evaluations |\n"
            breakdown += "|-------|-------|-------------|\n"
            
            # Sort models by score in this category
            category_model_scores = []
            for model_name in stats['models']:
                model_scores = stats['model_scores'][model_name]
                if model_scores:
                    avg_score = sum(model_scores) / len(model_scores)
                    category_model_scores.append((model_name, avg_score, len(model_scores)))
            
            category_model_scores.sort(key=lambda x: x[1], reverse=True)
            
            for model_name, avg_score, count in category_model_scores:
                breakdown += f"| {model_name} | {avg_score:.2f}/5.0 | {count} |\n"
            
            breakdown += "\n"
        
        return breakdown
    
    def _generate_cost_benefit_analysis(self) -> str:
        """Generate cost-benefit analysis section."""
        
        analysis = """### Performance per Dollar Analysis

*Note: Cost estimates are based on typical cloud hosting and API pricing. Actual costs may vary.*

"""
        
        # Mock cost analysis (would use real cost data in production)
        analysis += "| Model | Quality Score | Est. Cost/1K Tokens | Quality per Dollar | Recommendation |\n"
        analysis += "|-------|---------------|---------------------|-------------------|----------------|\n"
        
        for model_name, stats in sorted(self.model_stats.items(), key=lambda x: x[1].get('avg_score', 0), reverse=True):
            quality_score = stats.get('avg_score', 0)
            
            # Mock cost estimates (would be real in production)
            if 'transformer' in model_name.lower():
                cost_per_1k = 0.002  # Ollama local hosting
            elif 'llama' in model_name.lower():
                cost_per_1k = 0.001  # Open source model
            else:
                cost_per_1k = 0.003  # Default estimate
            
            quality_per_dollar = quality_score / cost_per_1k if cost_per_1k > 0 else 0
            
            # Recommendation based on quality and cost
            if quality_score >= 4.5 and cost_per_1k <= 0.002:
                recommendation = "🌟 Excellent Value"
            elif quality_score >= 4.0:
                recommendation = "✅ Recommended"
            elif cost_per_1k <= 0.001:
                recommendation = "💰 Budget Option"
            else:
                recommendation = "⚠️ Consider Alternatives"
            
            analysis += f"| {model_name} | {quality_score:.2f}/5.0 | ${cost_per_1k:.3f} | {quality_per_dollar:.0f} | {recommendation} |\n"
        
        analysis += "\n### Cost Optimization Recommendations\n\n"
        
        # Find best value model
        best_models = sorted(self.model_stats.items(), key=lambda x: x[1].get('avg_score', 0), reverse=True)
        if best_models:
            best_model = best_models[0]
            analysis += f"**🎯 Best Overall:** {best_model[0]} - Highest quality score ({best_model[1].get('avg_score', 0):.2f}/5.0)\n\n"
        
        analysis += "**💡 Key Insights:**\n"
        analysis += "- Local models (transformer) offer excellent cost efficiency for most use cases\n"
        analysis += "- Open source models (Llama) provide good balance of quality and cost\n"
        analysis += "- Consider workload requirements when choosing between speed and quality\n"
        
        return analysis
    
    def _generate_detailed_profiles(self) -> str:
        """Generate detailed model profiles."""
        
        profiles = ""
        
        for model_name, stats in sorted(self.model_stats.items()):
            profiles += f"### {model_name}\n\n"
            
            # Basic stats
            profiles += f"**Overall Score:** {stats.get('avg_score', 0):.2f}/5.0  \n"
            profiles += f"**Success Rate:** {(stats['successful_evaluations']/stats['total_evaluations']*100):.1f}%  \n"
            profiles += f"**Categories Tested:** {len(stats['categories'])}  \n"
            profiles += f"**Total Evaluations:** {stats['total_evaluations']}  \n\n"
            
            # Performance metrics
            perf = stats['performance']
            profiles += "**Performance Metrics:**\n"
            profiles += f"- Average Response Time: {perf['avg_time']:.2f} seconds\n"
            profiles += f"- Average Tokens Generated: {perf['avg_tokens']:.0f}\n"
            profiles += f"- Throughput: {perf['tokens_per_second']:.1f} tokens/second\n\n"
            
            # Category performance
            if stats['category_scores']:
                profiles += "**Category Performance:**\n"
                for category, scores in stats['category_scores'].items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        profiles += f"- {category.replace('_', ' ').title()}: {avg_score:.2f}/5.0 ({len(scores)} evaluations)\n"
                profiles += "\n"
            
            # Strengths and recommendations
            profiles += "**Strengths:**\n"
            if stats.get('avg_score', 0) >= 4.5:
                profiles += "- Excellent overall quality\n"
            if perf['avg_time'] <= 20:
                profiles += "- Fast response times\n"
            if perf['tokens_per_second'] >= 25:
                profiles += "- High throughput\n"
            
            profiles += "\n**Best Use Cases:**\n"
            if 'qa' in stats['categories']:
                profiles += "- Question answering and factual queries\n"
            if 'reasoning' in stats['categories']:
                profiles += "- Logical reasoning and problem solving\n"
            if 'code_gen' in stats['categories']:
                profiles += "- Code generation and programming tasks\n"
            
            profiles += "\n---\n\n"
        
        return profiles
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        
        methodology = f"""### Evaluation Framework

**Benchmark Suite:** SAM Core Benchmark v{self.metadata.get('benchmark_version', '1.0')}  
**Total Prompts:** {self.metadata.get('total_prompts', 'Unknown')}  
**Judge Model:** {self.metadata.get('judge_model', 'Unknown')}  
**Evaluation Date:** {self.metadata.get('timestamp', 'Unknown')}  

### Scoring Methodology

Each model response is evaluated on multiple criteria using a 1-5 scale:

- **Correctness (40% weight):** Factual accuracy and correctness of information
- **Completeness (30% weight):** How thoroughly the response addresses all aspects
- **Clarity (20% weight):** How clear, well-structured, and understandable the response is  
- **Conciseness (10% weight):** Efficiency in conveying information without unnecessary verbosity

Category-specific criteria are also applied where relevant (e.g., code quality for programming tasks).

### Performance Metrics

- **Response Time:** Time taken to generate the complete response
- **Token Count:** Number of tokens in the generated response
- **Throughput:** Tokens generated per second
- **Success Rate:** Percentage of successful evaluations

### Categories Tested

"""
        
        for category in sorted(self.category_stats.keys()):
            methodology += f"- **{category.replace('_', ' ').title()}:** {self.category_stats[category]['total_evaluations']} evaluations\n"
        
        methodology += f"""
### Statistical Significance

- Total evaluations: {self.overall_stats['total_evaluations']}
- Models compared: {self.overall_stats['total_models']}
- Success rate: {(self.overall_stats['successful_evaluations']/self.overall_stats['total_evaluations']*100):.1f}%

### Limitations

- Evaluation is based on a specific set of benchmark prompts
- Scoring includes subjective elements despite structured rubrics
- Performance may vary with different hardware configurations
- Cost estimates are approximate and may vary by provider

---

*Report generated by SAM Model Foundry & Evaluation Suite v1.0*
"""
        
        return methodology
    
    def save_report(self, report: str, output_file: str) -> str:
        """Save the report to file."""
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Leaderboard report saved to: {output_file}")
        return output_file

def main():
    """Main report generation execution."""
    parser = argparse.ArgumentParser(description="SAM Model Leaderboard Generator")
    parser.add_argument("input_file", nargs="?",
                       help="Input scored results file (JSONL format)")
    parser.add_argument("--output", default="MODEL_LEADERBOARD.md",
                       help="Output filename (default: MODEL_LEADERBOARD.md)")
    parser.add_argument("--format", default="markdown", choices=["markdown", "html"],
                       help="Output format (default: markdown)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find input file if not specified
    if not args.input_file:
        results_dir = Path("evaluation_results")
        if results_dir.exists():
            scored_files = list(results_dir.glob("scored_*.jsonl"))
            if scored_files:
                args.input_file = str(sorted(scored_files)[-1])  # Most recent
                logger.info(f"🔍 Auto-detected input file: {args.input_file}")
            else:
                logger.error("❌ No scored results found in evaluation_results/")
                return 1
        else:
            logger.error("❌ No input file specified and evaluation_results/ not found")
            return 1
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"❌ Input file not found: {args.input_file}")
        return 1
    
    try:
        # Generate report
        generator = LeaderboardGenerator()
        
        # Load and analyze results
        generator.load_scored_results(args.input_file)
        generator.analyze_results()
        
        # Generate report
        logger.info(f"📊 Generating {args.format} leaderboard report...")
        
        if args.format == "markdown":
            report = generator.generate_markdown_report()
        else:
            logger.error("❌ HTML format not yet implemented")
            return 1
        
        # Save report
        output_path = generator.save_report(report, args.output)
        
        print(f"\n🎉 Leaderboard report generated successfully!")
        print(f"📄 Report saved to: {output_path}")
        print(f"📊 {generator.overall_stats['total_models']} models evaluated across {generator.overall_stats['total_categories']} categories")
        print(f"🏆 View the complete analysis in {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Report generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
