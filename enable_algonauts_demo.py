#!/usr/bin/env python3
"""
Enable Algonauts Demo
====================

Creates sample reasoning traces with cognitive vectors to demonstrate
the Algonauts-style visualizations in the Dynamic Trace Explorer.

Author: SAM Development Team
Version: 1.0.0
"""

import sys
import numpy as np
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent / "SmallAgentModel-main"))

def create_sample_reasoning_traces():
    """Create sample reasoning traces with cognitive vectors for demo."""
    print("🧠 Creating Sample Reasoning Traces for Algonauts Demo")
    print("=" * 55)
    
    try:
        from sam.introspection.flight_recorder import (
            initialize_flight_recorder, TraceSession, ReasoningStep
        )
        
        # Initialize flight recorder
        recorder = initialize_flight_recorder(auto_save=False)
        
        # Create multiple sample sessions to demonstrate different reasoning patterns
        
        # Session 1: Data Analysis Query
        print("📊 Creating data analysis reasoning trace...")
        with TraceSession("Analyze the employee_data.csv file for salary trends") as session_id:
            
            # Memory retrieval with cognitive vector
            memory_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.MEMORY_RETRIEVAL,
                component="memory",
                operation="search_knowledge",
                input_data={"query": "employee data analysis", "file": "employee_data.csv"},
                output_data={"results_found": 12, "confidence": 0.89},
                cognitive_vector=memory_vector
            )
            
            # Context assembly
            context_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.CONTEXT_ASSEMBLY,
                component="context_assembler",
                operation="assemble_context",
                input_data={"memory_results": 12, "file_size": "18 rows"},
                output_data={"context_tokens": 1024, "relevance_score": 0.92},
                cognitive_vector=context_vector
            )
            
            # Agent Zero planning
            planning_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.AGENT_ZERO_PLANNING,
                component="agent_zero",
                operation="plan_data_analysis",
                input_data={"task": "salary trend analysis", "data_type": "CSV"},
                output_data={"plan_steps": 4, "tools_needed": ["code_interpreter", "visualization"]},
                cognitive_vector=planning_vector
            )
            
            # Code interpreter execution
            code_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.CODE_INTERPRETER,
                component="code_interpreter",
                operation="execute_analysis",
                input_data={
                    "code": "df = pd.read_csv('employee_data.csv'); df.groupby('department')['salary'].mean()",
                    "data_files": ["employee_data.csv"]
                },
                output_data={
                    "results": {"Engineering": 95000, "Sales": 80000, "HR": 61667, "Marketing": 94000},
                    "plot_generated": True,
                    "execution_time": 0.234
                },
                cognitive_vector=code_vector
            )
            
            # Model inference
            inference_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.MODEL_INFERENCE,
                component="model",
                operation="generate_insights",
                input_data={"analysis_results": "department salary averages calculated"},
                output_data={
                    "insights": "Engineering and Marketing have highest average salaries",
                    "confidence": 0.94,
                    "reasoning": "Clear departmental salary differences observed"
                },
                cognitive_vector=inference_vector
            )
            
            # Response generation
            response_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.RESPONSE_GENERATION,
                component="response_generator",
                operation="format_response",
                input_data={"insights": "departmental salary analysis"},
                output_data={
                    "response_length": 342,
                    "includes_visualization": True,
                    "confidence": 0.96
                },
                cognitive_vector=response_vector
            )
        
        print(f"✅ Data analysis trace created: {session_id}")
        
        # Session 2: Complex Problem Solving
        print("🧮 Creating complex problem solving trace...")
        with TraceSession("What are the key factors affecting employee satisfaction?") as session_id:
            
            # Multiple memory retrievals (showing iterative thinking)
            for i, topic in enumerate(["employee satisfaction", "workplace factors", "correlation analysis"]):
                vector = np.random.rand(128).tolist()
                recorder.log_step(
                    session_id=session_id,
                    step_type=ReasoningStep.MEMORY_RETRIEVAL,
                    component="memory",
                    operation=f"search_{topic.replace(' ', '_')}",
                    input_data={"query": topic, "iteration": i+1},
                    output_data={"results_found": np.random.randint(5, 15), "confidence": 0.7 + i*0.1},
                    cognitive_vector=vector
                )
            
            # Complex planning with multiple sub-steps
            planning_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.AGENT_ZERO_PLANNING,
                component="agent_zero",
                operation="plan_correlation_analysis",
                input_data={"variables": ["satisfaction_score", "salary", "experience_years", "projects_completed"]},
                output_data={"analysis_type": "correlation_matrix", "visualization_needed": True},
                cognitive_vector=planning_vector
            )
            
            # Code execution for correlation analysis
            code_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.CODE_INTERPRETER,
                component="code_interpreter",
                operation="correlation_analysis",
                input_data={"variables": 4, "method": "pearson"},
                output_data={
                    "correlations": {
                        "satisfaction_salary": 0.23,
                        "satisfaction_experience": 0.45,
                        "satisfaction_projects": 0.67
                    },
                    "strongest_correlation": "projects_completed"
                },
                cognitive_vector=code_vector
            )
            
            # Final inference
            final_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.MODEL_INFERENCE,
                component="model",
                operation="synthesize_findings",
                input_data={"correlation_results": "projects_completed shows strongest correlation"},
                output_data={
                    "key_insight": "Project involvement is the strongest predictor of satisfaction",
                    "confidence": 0.91
                },
                cognitive_vector=final_vector
            )
        
        print(f"✅ Problem solving trace created: {session_id}")
        
        # Session 3: Quick Query (showing different reasoning pattern)
        print("⚡ Creating quick query trace...")
        with TraceSession("What is the average salary in the dataset?") as session_id:
            
            # Simple memory retrieval
            memory_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.MEMORY_RETRIEVAL,
                component="memory",
                operation="quick_lookup",
                input_data={"query": "average salary calculation"},
                output_data={"method": "pandas.mean()", "confidence": 0.99},
                cognitive_vector=memory_vector
            )
            
            # Direct code execution
            code_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.CODE_INTERPRETER,
                component="code_interpreter",
                operation="calculate_mean",
                input_data={"column": "salary", "method": "df['salary'].mean()"},
                output_data={"result": 74333.33, "execution_time": 0.012},
                cognitive_vector=code_vector
            )
            
            # Quick response
            response_vector = np.random.rand(128).tolist()
            recorder.log_step(
                session_id=session_id,
                step_type=ReasoningStep.RESPONSE_GENERATION,
                component="response_generator",
                operation="format_simple_answer",
                input_data={"value": 74333.33},
                output_data={"response": "The average salary is $74,333.33", "length": 32},
                cognitive_vector=response_vector
            )
        
        print(f"✅ Quick query trace created: {session_id}")
        
        print("\n🎉 Sample traces created successfully!")
        print("📊 You now have 3 different reasoning patterns to explore:")
        print("   1. 📈 Complex data analysis (6 steps)")
        print("   2. 🧮 Multi-step problem solving (7 steps)")  
        print("   3. ⚡ Quick query (3 steps)")
        print("\n🌐 Open http://localhost:5003 to view the Algonauts visualizations!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample traces: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Create sample traces for Algonauts demo."""
    print("🚀 Algonauts Visualization Demo Setup")
    print("=" * 40)
    
    success = create_sample_reasoning_traces()
    
    if success:
        print("\n✅ Demo setup complete!")
        print("\n📋 Next steps:")
        print("1. Run: python launch_dynamic_trace_explorer.py")
        print("2. Open: http://localhost:5003")
        print("3. Select a session from the dropdown")
        print("4. View the cognitive trajectory in the right panel")
        print("\n🧠 The Algonauts visualization will show:")
        print("   • 2D projection of cognitive states")
        print("   • Color-coded reasoning steps")
        print("   • Interactive trajectory path")
        print("   • Hover details for each cognitive moment")
    else:
        print("\n❌ Demo setup failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
