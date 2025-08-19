#!/usr/bin/env python3
"""
Model Interface for SSRL Testing
=================================

Mock model interface for testing SSRL functionality without requiring
the full SAM model infrastructure.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import time
import random

logger = logging.getLogger(__name__)


class MockModelInterface:
    """Mock model interface for testing SSRL functionality."""
    
    def __init__(self):
        """Initialize mock model interface."""
        self.call_count = 0
        logger.info("Mock model interface initialized for SSRL testing")
    
    def generate_response(self, 
                         prompt: str, 
                         max_tokens: int = 2000,
                         temperature: float = 0.7,
                         system_message: str = None) -> str:
        """
        Generate a mock SSRL-structured response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (ignored in mock)
            temperature: Temperature setting (ignored in mock)
            system_message: System message (ignored in mock)
            
        Returns:
            Mock SSRL-structured response
        """
        self.call_count += 1
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Extract the user's question from the prompt
        user_question = "unknown question"
        if "User's question:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.strip().startswith("User's question:"):
                    user_question = line.replace("User's question:", "").strip()
                    break
        
        # Generate mock structured response based on question type
        if any(word in user_question.lower() for word in ['capital', 'france']):
            return self._generate_factual_response()
        elif any(word in user_question.lower() for word in ['quantum', 'computing']):
            return self._generate_complex_response()
        elif any(word in user_question.lower() for word in ['machine', 'learning', 'ai']):
            return self._generate_technical_response()
        else:
            return self._generate_general_response(user_question)
    
    def _generate_factual_response(self) -> str:
        """Generate mock response for factual questions."""
        return """<think>
This is a straightforward factual question about geography. I need to recall the capital of France, which is a well-established fact. This is basic geographical knowledge that I'm confident about.
</think>

<search>
Searching my knowledge about France: France is a country in Western Europe. Its capital and largest city is Paris, which has been the capital since the 12th century. Paris is located in the north-central part of France.
</search>

<information>
The capital of France is Paris. This is a definitive fact with no ambiguity. Paris has been the capital of France for centuries and is well-documented in geographical and historical sources.
</information>

<confidence>
0.95
</confidence>

<answer>
The capital of France is Paris. Paris has served as France's capital city since the 12th century and is located in the north-central part of the country.
</answer>"""
    
    def _generate_complex_response(self) -> str:
        """Generate mock response for complex topics."""
        return """<think>
This is a complex technical topic that requires explaining quantum computing concepts. I should break this down into understandable components while being honest about the complexity and my knowledge limitations.
</think>

<search>
Quantum computing involves quantum mechanical phenomena like superposition and entanglement. Key concepts include qubits (quantum bits), quantum gates, quantum algorithms, and quantum supremacy. Major applications include cryptography, optimization, and simulation.
</search>

<information>
Quantum computing is a fundamentally different approach to computation that leverages quantum mechanical properties. Unlike classical bits that are either 0 or 1, qubits can exist in superposition. This allows quantum computers to potentially solve certain problems exponentially faster than classical computers.
</information>

<confidence>
0.75
</confidence>

<answer>
Quantum computing is a revolutionary computing paradigm that uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, potentially enabling exponential speedups for specific types of problems like cryptography and optimization.
</answer>"""
    
    def _generate_technical_response(self) -> str:
        """Generate mock response for technical topics."""
        return """<think>
This question is about machine learning or AI, which is a broad technical field. I should provide a clear explanation while acknowledging the breadth of the topic and potential areas where my knowledge might be incomplete.
</think>

<search>
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Key types include supervised learning, unsupervised learning, and reinforcement learning. Common algorithms include neural networks, decision trees, and support vector machines.
</search>

<information>
Machine learning involves training algorithms on data to recognize patterns and make predictions or decisions. It's widely used in applications like image recognition, natural language processing, recommendation systems, and autonomous vehicles.
</information>

<confidence>
0.85
</confidence>

<answer>
Machine learning is a branch of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario. It involves training algorithms on datasets to recognize patterns, make predictions, and improve performance over time through experience.
</answer>"""
    
    def _generate_general_response(self, question: str) -> str:
        """Generate mock response for general questions."""
        confidence = random.uniform(0.6, 0.9)
        
        return f"""<think>
This is a general question: "{question}". I need to think through what I know about this topic and provide a helpful response while being honest about any limitations in my knowledge.
</think>

<search>
Searching my knowledge base for information related to this question. I'll try to find relevant facts, concepts, and examples that can help provide a comprehensive answer.
</search>

<information>
Based on my knowledge search, I can provide information about this topic, though I should note any areas where my knowledge might be incomplete or where the user might need more specific or recent information.
</information>

<confidence>
{confidence:.2f}
</confidence>

<answer>
I understand you're asking about "{question}". While I can provide some general information on this topic, I want to be transparent that my response is based on my training data and may not include the most recent developments. For the most current and comprehensive information, you might want to consult additional sources.
</answer>"""
    
    def get_stats(self) -> dict:
        """Get mock model statistics."""
        return {
            'total_calls': self.call_count,
            'model_type': 'mock_model_for_testing',
            'status': 'active'
        }


# Global mock instance
_mock_model_interface = None

def get_model_interface():
    """Get the mock model interface for testing."""
    global _mock_model_interface
    if _mock_model_interface is None:
        _mock_model_interface = MockModelInterface()
    return _mock_model_interface
