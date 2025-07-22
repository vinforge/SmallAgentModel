#!/usr/bin/env python3
"""
Simple launcher for SAM's multimodal chat interface.
Focuses on the core multimodal functionality without complex dependencies.
"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_system_prompt():
    """Read system prompt from file or use default."""
    prompt_file = Path("curious_analyst_prompt.txt")
    
    if prompt_file.exists():
        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read system prompt file: {e}")
    
    # Default system prompt
    return """You are SAM (Small Agent Model), an intelligent assistant with advanced multimodal knowledge processing capabilities.

You can:
- Process and analyze multimodal documents (PDF, DOCX, Markdown, HTML, code files)
- Consolidate knowledge across different content types (text, code, tables, images)
- Perform semantic search across your knowledge base
- Provide enrichment scores and content analysis

You have access to:
- Multimodal document processing pipeline
- Vector-based semantic search
- Content type filtering and analysis
- Real-time document processing

Be helpful, accurate, and leverage your multimodal capabilities to provide comprehensive responses."""

def initialize_components():
    """Initialize the core components for multimodal chat."""
    try:
        # Initialize Ollama model
        from deepseek_enhanced_learning.model_loader import OllamaModel
        model = OllamaModel()
        logger.info("✅ Ollama model initialized")
        
        # Initialize vector store
        from utils.vector_manager import VectorManager
        vector_manager = VectorManager()
        logger.info("✅ Vector store initialized")
        
        return model, vector_manager
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

def main():
    """Main entry point for multimodal chat."""
    print("🚀 Starting SAM Multimodal Chat Interface...")
    print("=" * 60)
    
    try:
        # Initialize components
        model, vector_manager = initialize_components()
        
        # Read system prompt
        system_prompt = read_system_prompt()
        logger.info("✅ System prompt loaded")
        
        # Launch chat interface
        from ui.chat_ui import ChatInterface
        
        print("\n🎉 SAM Multimodal Chat Interface Ready!")
        print("=" * 60)
        print("Available multimodal commands:")
        print("  • process <file>     - Process a multimodal document")
        print("  • search <query>     - Search multimodal content")
        print("  • multimodal on/off  - Toggle multimodal features")
        print("  • status             - Show system status")
        print("  • help               - Show all commands")
        print("=" * 60)
        
        # Launch chat interface using the function
        from ui.chat_ui import launch_chat_interface

        launch_chat_interface(
            model=model,
            vector_index=vector_manager,
            system_prompt=system_prompt
        )
        
    except KeyboardInterrupt:
        print("\n👋 SAM shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
