#!/usr/bin/env python3
"""
Create Algonauts Demo Sessions
=============================

Creates sample cognitive trace sessions to demonstrate the Algonauts comparative
analysis feature. This generates sessions that simulate different architectural
patterns (Transformer vs Hybrid) for visualization testing.

Usage:
    python create_algonauts_demo_sessions.py

Author: SAM Development Team
Version: 1.0.0 - Algonauts Demo
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add SAM to path
sys.path.insert(0, str(Path(__file__).parent / "SmallAgentModel-main"))

def create_algonauts_demo_sessions():
    """Create demo sessions simulating Mistral vs Jamba cognitive patterns."""
    print("🧠 Creating Algonauts Demo Sessions")
    print("=" * 40)
    
    try:
        from sam.introspection.flight_recorder import (
            initialize_flight_recorder, get_flight_recorder, TraceLevel, 
            ReasoningStep, CognitiveVector
        )
        
        # Initialize flight recorder
        recorder = initialize_flight_recorder(
            trace_level=TraceLevel.DETAILED,
            max_sessions=50,
            auto_save=False,
            save_directory="demo_traces"
        )
        
        print("✅ Flight Recorder initialized")
        
        # Session 1: Simulated Mistral (Transformer) - More volatile, scattered pattern
        print("🔵 Creating Mistral-style session (Transformer pattern)...")
        
        session_id_mistral = recorder.start_session(
            query="Find the specific ship name mentioned in Chapter 51 of Moby Dick",
            session_id="mistral_demo_001"
        )
        
        # Simulate Transformer attention patterns - more volatile
        np.random.seed(42)  # For reproducible demo
        
        # Initial query processing - high variance
        vector_data1 = np.random.normal(0, 0.4, 128).tolist()

        recorder.log_step(
            session_id=session_id_mistral,
            step_type=ReasoningStep.QUERY_RECEIVED,
            component="attention_mechanism",
            operation="parse_query",
            input_data={"query": "ship name Chapter 51 Moby Dick"},
            output_data={"tokens_identified": 6, "attention_heads": 12},
            cognitive_vector=vector_data1
        )
        
        # Context processing - scattered attention
        vector_data2 = np.random.normal(0.2, 0.5, 128).tolist()  # Higher variance

        recorder.log_step(
            session_id=session_id_mistral,
            step_type=ReasoningStep.CONTEXT_ASSEMBLY,
            component="attention_mechanism",
            operation="process_long_context",
            input_data={"context_tokens": 125000, "chapter": 51},
            output_data={"attention_weights": "scattered", "confidence": 0.7},
            cognitive_vector=vector_data2
        )
        
        # Multiple reasoning steps - showing volatility
        for i in range(3):
            vector_data = np.random.normal(0.1 * i, 0.3, 128).tolist()

            recorder.log_step(
                session_id=session_id_mistral,
                step_type=ReasoningStep.MODEL_INFERENCE,
                component="attention_mechanism",
                operation=f"search_step_{i+1}",
                input_data={"search_progress": f"{(i+1)*25}%"},
                output_data={"found_references": i*2, "confidence": 0.6 + i*0.1},
                cognitive_vector=vector_data
            )
        
        # Final response - still somewhat scattered
        vector_data_final = np.random.normal(0.3, 0.25, 128).tolist()

        recorder.log_step(
            session_id=session_id_mistral,
            step_type=ReasoningStep.RESPONSE_GENERATION,
            component="attention_mechanism",
            operation="generate_response",
            input_data={"found_ship": "Bachelor", "confidence": 0.85},
            output_data={"response_length": 156, "accuracy": "high"},
            cognitive_vector=vector_data_final
        )
        
        recorder.end_session(session_id_mistral, "The ship mentioned in Chapter 51 is the Bachelor.")
        print(f"✅ Mistral session created: {session_id_mistral}")
        
        # Session 2: Simulated Jamba (Hybrid) - More stable, efficient pattern
        print("🔴 Creating Jamba-style session (Hybrid pattern)...")
        
        session_id_jamba = recorder.start_session(
            query="Find the specific ship name mentioned in Chapter 51 of Moby Dick",
            session_id="jamba_demo_001"
        )
        
        # Simulate Hybrid SSM patterns - more stable
        np.random.seed(123)  # Different seed for different pattern
        
        # Initial query processing - lower variance
        vector_data1_j = np.random.normal(0, 0.15, 128).tolist()  # Lower variance

        recorder.log_step(
            session_id=session_id_jamba,
            step_type=ReasoningStep.QUERY_RECEIVED,
            component="hybrid_ssm",
            operation="parse_query",
            input_data={"query": "ship name Chapter 51 Moby Dick"},
            output_data={"tokens_identified": 6, "ssm_state": "initialized"},
            cognitive_vector=vector_data1_j
        )
        
        # Context processing - stable SSM compression
        vector_data2_j = np.random.normal(0.1, 0.2, 128).tolist()  # Much lower variance

        recorder.log_step(
            session_id=session_id_jamba,
            step_type=ReasoningStep.CONTEXT_ASSEMBLY,
            component="hybrid_ssm",
            operation="compress_long_context",
            input_data={"context_tokens": 125000, "chapter": 51},
            output_data={"compressed_state": "stable", "confidence": 0.9},
            cognitive_vector=vector_data2_j
        )
        
        # Fewer, more direct reasoning steps
        for i in range(2):  # Fewer steps than Mistral
            vector_data = np.random.normal(0.15, 0.1, 128).tolist()  # Very stable

            recorder.log_step(
                session_id=session_id_jamba,
                step_type=ReasoningStep.MODEL_INFERENCE,
                component="hybrid_ssm",
                operation=f"focused_search_{i+1}",
                input_data={"search_progress": f"{(i+1)*50}%"},
                output_data={"found_references": (i+1)*3, "confidence": 0.85 + i*0.05},
                cognitive_vector=vector_data
            )
        
        # Final response - stable and confident
        vector_data_final_j = np.random.normal(0.2, 0.1, 128).tolist()  # Low variance

        recorder.log_step(
            session_id=session_id_jamba,
            step_type=ReasoningStep.RESPONSE_GENERATION,
            component="hybrid_ssm",
            operation="generate_response",
            input_data={"found_ship": "Bachelor", "confidence": 0.95},
            output_data={"response_length": 142, "accuracy": "high"},
            cognitive_vector=vector_data_final_j
        )
        
        recorder.end_session(session_id_jamba, "The ship mentioned in Chapter 51 is the Bachelor.")
        print(f"✅ Jamba session created: {session_id_jamba}")
        
        # Session 3: Another Mistral session for comparison
        print("🔵 Creating second Mistral session...")
        
        session_id_mistral2 = recorder.start_session(
            query="What is the voltage requirement for Model X-7429 in Section 12.4.7?",
            session_id="mistral_demo_002"
        )
        
        # Similar volatile pattern but different task
        np.random.seed(456)
        
        # More scattered search pattern
        for i in range(4):  # More steps, showing inefficiency
            vector_data = np.random.normal(0.2 * i, 0.35, 128).tolist()

            step_type = ReasoningStep.QUERY_RECEIVED if i == 0 else ReasoningStep.MODEL_INFERENCE
            recorder.log_step(
                session_id=session_id_mistral2,
                step_type=step_type,
                component="attention_mechanism",
                operation=f"search_technical_spec_{i+1}",
                input_data={"section": "12.4.7", "model": "X-7429"},
                output_data={"progress": f"{i*20}%", "confidence": 0.5 + i*0.1},
                cognitive_vector=vector_data
            )
        
        recorder.end_session(session_id_mistral2, "The voltage requirement is 24V DC.")
        print(f"✅ Second Mistral session created: {session_id_mistral2}")
        
        # Get session list to verify
        sessions = recorder.get_all_sessions()
        print(f"\n📊 Total sessions created: {len(sessions)}")
        for session in sessions:
            print(f"  - {session}")
        
        print("\n🎉 Demo sessions created successfully!")
        print("💡 You can now use the Algonauts Comparative Analysis feature")
        print("   1. Select 'Comparative Analysis' mode")
        print("   2. Choose a Mistral session for Model A")
        print("   3. Choose a Jamba session for Model B")
        print("   4. Enable 'Overlay Trajectories' to see the difference!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create demo sessions: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    create_algonauts_demo_sessions()
