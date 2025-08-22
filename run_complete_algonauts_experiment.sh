#!/bin/bash
"""
Complete Algonauts Experiment Runner
===================================

This script runs the complete Algonauts experiment from start to finish:
1. Runs needle-in-haystack benchmarks with cognitive tracing for both engines
2. Analyzes cognitive trajectories and generates metrics
3. Creates comprehensive analysis report
4. Provides instructions for interactive visualization

Usage:
    chmod +x run_complete_algonauts_experiment.sh
    ./run_complete_algonauts_experiment.sh

Author: SAM Development Team
Version: 1.0.0 - Complete Algonauts Pipeline
"""

set -e  # Exit on any error

# Configuration
EXPERIMENT_DIR="results/algonauts_experiment"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="algonauts_experiment_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    log "🧠 Starting Complete Algonauts Experiment"
    log "=========================================="
    
    # Create experiment directory
    mkdir -p "$EXPERIMENT_DIR"
    
    # Phase 1: Run Algonauts Experiment
    log "📋 Phase 1: Running Algonauts Experiment..."
    
    if python scripts/run_algonauts_experiment.py --output-dir "$EXPERIMENT_DIR"; then
        success "✅ Algonauts experiment completed successfully"
    else
        error "❌ Algonauts experiment failed"
        exit 1
    fi
    
    # Phase 2: Generate Comprehensive Analysis Report
    log "📊 Phase 2: Generating comprehensive analysis report..."
    
    if python analysis/algonauts_analysis_report.py \
        --experiment-dir "$EXPERIMENT_DIR" \
        --output "analysis/algonauts_comprehensive_report_${TIMESTAMP}.md"; then
        success "✅ Analysis report generated successfully"
    else
        warning "⚠️ Analysis report generation failed, but experiment data is available"
    fi
    
    # Phase 3: Summary and Next Steps
    log "🎉 Phase 3: Experiment Summary"
    log "=============================="
    
    # Display results summary
    if [ -f "$EXPERIMENT_DIR/trajectory_metrics.json" ]; then
        log "📊 Trajectory metrics calculated and saved"
        
        # Extract key metrics using Python
        python3 -c "
import json
import sys
try:
    with open('$EXPERIMENT_DIR/trajectory_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    if 'mistral' in metrics and 'jamba' in metrics:
        mistral = metrics['mistral']
        jamba = metrics['jamba']
        
        print('\\n🔍 KEY FINDINGS:')
        print(f'  Trajectory Length - Mistral: {mistral.get(\"avg_trajectory_length\", 0):.3f}, Jamba: {jamba.get(\"avg_trajectory_length\", 0):.3f}')
        print(f'  State Volatility - Mistral: {mistral.get(\"avg_state_volatility\", 0):.3f}, Jamba: {jamba.get(\"avg_state_volatility\", 0):.3f}')
        print(f'  RAG Influence - Mistral: {mistral.get(\"avg_rag_influence\", 0):.3f}, Jamba: {jamba.get(\"avg_rag_influence\", 0):.3f}')
        
        # Determine winners
        if mistral.get('avg_trajectory_length', 0) < jamba.get('avg_trajectory_length', 0):
            print('  🏆 Efficiency Winner: Mistral (shorter trajectory)')
        else:
            print('  🏆 Efficiency Winner: Jamba (shorter trajectory)')
            
        if mistral.get('avg_state_volatility', 0) < jamba.get('avg_state_volatility', 0):
            print('  🧘 Stability Winner: Mistral (lower volatility)')
        else:
            print('  🧘 Stability Winner: Jamba (lower volatility)')
    else:
        print('⚠️ Incomplete metrics data')
        
except Exception as e:
    print(f'❌ Error reading metrics: {e}')
"
    else
        warning "⚠️ Trajectory metrics file not found"
    fi
    
    # Display file locations
    log ""
    log "📁 Generated Files:"
    log "  📊 Experiment Results: $EXPERIMENT_DIR/"
    log "  📈 Trajectory Metrics: $EXPERIMENT_DIR/trajectory_metrics.json"
    log "  🧠 Mistral Traces: $EXPERIMENT_DIR/mistral_trace_logs/"
    log "  🧠 Jamba Traces: $EXPERIMENT_DIR/jamba_trace_logs/"
    log "  📄 Analysis Report: analysis/algonauts_comprehensive_report_${TIMESTAMP}.md"
    log "  📝 Execution Log: $LOG_FILE"
    
    # Next steps
    log ""
    log "🚀 Next Steps:"
    log "  1. Review the comprehensive analysis report"
    log "  2. Open SAM's Memory Control Center and navigate to 🔬 Algonauts tab"
    log "  3. Use 'Comparative Analysis' mode to visualize trajectories side-by-side"
    log "  4. Load trace files for interactive exploration"
    log "  5. Share findings with the research team"
    
    # Interactive visualization instructions
    log ""
    log "🎨 Interactive Visualization:"
    log "  To explore trajectories interactively:"
    log "  1. Start SAM's Streamlit interface"
    log "  2. Navigate to Memory Control Center > 🔬 Algonauts"
    log "  3. Select 'Comparative Analysis' mode"
    log "  4. Choose sessions from Mistral and Jamba trace logs"
    log "  5. Enable 'Overlay Trajectories' for side-by-side comparison"
    
    success "🎉 Complete Algonauts Experiment finished successfully!"
    log "Total execution time: $(date)"
}

# Error handling
trap 'error "❌ Script interrupted"; exit 1' INT TERM

# Check dependencies
check_dependencies() {
    log "🔧 Checking dependencies..."
    
    if ! command -v python &> /dev/null; then
        error "Python is required but not installed"
        exit 1
    fi
    
    if [ ! -f "scripts/run_algonauts_experiment.py" ]; then
        error "Algonauts experiment script not found"
        exit 1
    fi
    
    if [ ! -f "analysis/algonauts_analysis_report.py" ]; then
        error "Analysis report generator not found"
        exit 1
    fi
    
    success "✅ All dependencies found"
}

# Run the experiment
check_dependencies
main

# Final status
if [ $? -eq 0 ]; then
    success "🎊 Algonauts experiment pipeline completed successfully!"
    log "📧 Results ready for analysis and presentation"
else
    error "💥 Algonauts experiment pipeline failed"
    exit 1
fi
