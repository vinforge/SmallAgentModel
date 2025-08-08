#!/usr/bin/env python3
"""
Simple Chat Launcher for SAM
Launch the enhanced chat interface with memory integration.

This launcher bypasses complex initialization for quick testing.
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_model():
    """Create a mock model for testing the chat interface."""
    class MockModel:
        def generate(self, prompt, **kwargs):
            return "I'm a mock SAM model. The enhanced chat interface is working! Try using memory commands like '!memhelp' or type 'memory' to open the Memory Control Center."
        
        def chat(self, prompt, **kwargs):
            return self.generate(prompt, **kwargs)
        
        def __call__(self, prompt):
            return self.generate(prompt)
    
    return MockModel()

def create_mock_vector_store():
    """Create a mock vector store for testing."""
    class MockVectorStore:
        def similarity_search(self, query, k=3):
            # Return mock documents
            class MockDoc:
                def __init__(self, content, metadata=None):
                    self.page_content = content
                    self.metadata = metadata or {}
            
            return [
                MockDoc("Mock document about AI and machine learning", {"source": "mock_ai.txt"}),
                MockDoc("Mock document about data science", {"source": "mock_data.txt"}),
                MockDoc("Mock document about programming", {"source": "mock_code.txt"})
            ]
    
    return MockVectorStore()

def launch_enhanced_chat():
    """Launch the enhanced chat interface with memory integration."""
    try:
        print("🧠 SAM Enhanced Chat Interface with Memory Integration")
        print("=" * 60)
        print("Features:")
        print("  💬 Interactive chat with SAM")
        print("  🧠 Memory commands (start with !)")
        print("  ⚙️ Memory Control Center access")
        print("  🔍 Source transparency")
        print("=" * 60)
        
        # Add current directory to Python path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Create mock components
        model = create_mock_model()
        vector_index = create_mock_vector_store()
        system_prompt = """You are SAM (Small Agent Model), a curious and analytical AI assistant. 
You help users explore and understand complex information with transparency and insight."""
        
        # Import and launch chat interface
        from ui.chat_ui import launch_chat_interface
        
        logger.info("Launching enhanced chat interface...")
        launch_chat_interface(model=model, vector_index=vector_index, system_prompt=system_prompt)
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Chat session ended by user")
        return True
    except Exception as e:
        logger.error(f"Error launching chat interface: {e}")
        print(f"❌ Error: {e}")
        return False

def main():
    """Main launcher function."""
    print("🚀 SAM Enhanced Chat Launcher")
    print("Testing memory integration and control center access")
    print()
    
    success = launch_enhanced_chat()
    
    if success:
        print("\n✅ Chat session completed successfully")
        return 0
    else:
        print("\n❌ Chat session failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
