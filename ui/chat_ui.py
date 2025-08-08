# ui/chat_ui.py

import logging
import sys
from typing import Optional, Any, List, Tuple
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class ChatInterface:
    """
    Enhanced chat interface for interacting with the SAM model.
    Includes source transparency and metadata display features.
    """

    def __init__(self, model: Any, vector_index: Any, system_prompt: str):
        """
        Initialize the chat interface.

        Args:
            model: The loaded language model
            vector_index: The vector store for knowledge retrieval
            system_prompt: The system prompt for the model
        """
        self.model = model
        self.vector_index = vector_index
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.session_start = datetime.now()

        # UI settings for transparency features
        self.show_sources = False
        self.show_metadata = False
        self.show_tags = False

        # Semantic search settings
        self.semantic_mode = True  # Default to semantic mode
        self.semantic_top_k = 5
        self.semantic_threshold = 0.1

        # Multimodal settings
        self.multimodal_mode = True  # Default to multimodal mode
        self.show_content_types = True

        # Tool-augmented reasoning settings
        self.tool_mode = True  # Enable tool-augmented reasoning
        self.show_reasoning = False  # Show reasoning traces
        self.enable_web_search = False  # Local web search toggle
        self.show_tool_usage = True  # Show tool usage in responses

        # Initialize knowledge database for metadata access
        try:
            from knowledge_navigator import KnowledgeDatabase
            self.knowledge_db = KnowledgeDatabase()
        except ImportError:
            logger.warning("Knowledge database not available - source transparency disabled")
            self.knowledge_db = None

        # Initialize semantic components
        self.query_router = None
        self._initialize_semantic_components()

        # Initialize retrieval logger
        try:
            from utils.retrieval_logger import get_retrieval_logger
            self.retrieval_logger = get_retrieval_logger()
        except ImportError:
            logger.warning("Retrieval logger not available")
            self.retrieval_logger = None

        # Initialize multimodal pipeline
        try:
            from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
            self.multimodal_pipeline = get_multimodal_pipeline()
        except ImportError:
            logger.warning("Multimodal pipeline not available")
            self.multimodal_pipeline = None

        # Initialize tool-augmented reasoning components
        try:
            from reasoning.self_decide_framework import get_self_decide_framework
            from reasoning.tool_selector import get_tool_selector
            from reasoning.tool_executor import get_tool_executor
            from reasoning.answer_synthesizer import get_answer_synthesizer

            self.self_decide = get_self_decide_framework()
            self.tool_selector = get_tool_selector(enable_web_search=self.enable_web_search)
            self.tool_executor = get_tool_executor()
            self.answer_synthesizer = get_answer_synthesizer(model=self.model)

            # Connect components
            self.self_decide.tool_selector = self.tool_selector
            self.self_decide.model = self.model
            self.self_decide.vector_manager = self.vector_index

        except ImportError as e:
            logger.warning(f"Tool-augmented reasoning not available: {e}")
            self.self_decide = None
            self.tool_selector = None
            self.tool_executor = None
            self.answer_synthesizer = None

        logger.info("Enhanced chat interface with semantic and multimodal capabilities initialized")

    def _initialize_semantic_components(self):
        """Initialize semantic search components."""
        try:
            from utils.vector_manager import VectorManager
            from utils.embedding_utils import get_embedding_manager
            from core.query_router import SemanticQueryRouter

            # Initialize components
            self.vector_manager = VectorManager()
            self.embedding_manager = get_embedding_manager()
            self.query_router = SemanticQueryRouter(
                vector_manager=self.vector_manager,
                embedding_manager=self.embedding_manager,
                config={
                    'default_top_k': self.semantic_top_k,
                    'score_threshold': self.semantic_threshold
                }
            )

            logger.info("Semantic components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing semantic components: {e}")
            self.query_router = None
    
    def _format_prompt(self, user_input: str, context: Optional[str] = None) -> str:
        """
        Format the prompt with system prompt, context, and user input.
        
        Args:
            user_input: The user's input message
            context: Optional context from vector retrieval
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [self.system_prompt]
        
        if context:
            prompt_parts.append(f"\n## Relevant Context:\n{context}")
        
        # Add conversation history (last 5 exchanges)
        if self.conversation_history:
            prompt_parts.append("\n## Recent Conversation:")
            for exchange in self.conversation_history[-5:]:
                prompt_parts.append(f"User: {exchange['user']}")
                prompt_parts.append(f"Assistant: {exchange['assistant']}")
        
        prompt_parts.append(f"\n## Current Query:\nUser: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _retrieve_context(self, query: str, top_k: int = 3) -> tuple[Optional[str], List[dict]]:
        """
        Retrieve relevant context from the vector store with metadata.

        Args:
            query: The user's query
            top_k: Number of top results to retrieve

        Returns:
            Tuple of (formatted context string, list of source metadata)
        """
        try:
            if self.vector_index is None:
                return None, []

            # Retrieve relevant documents
            results = self.vector_index.similarity_search(query, k=top_k)

            if not results:
                return None, []

            # Format context and collect metadata
            context_parts = []
            source_metadata = []

            for i, doc in enumerate(results, 1):
                context_parts.append(f"[{i}] {doc.page_content}")

                # Extract metadata if available
                metadata = getattr(doc, 'metadata', {})
                source_info = {
                    "source_id": i,
                    "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "metadata": metadata
                }

                # Try to get additional metadata from knowledge database
                if self.knowledge_db and 'source' in metadata:
                    try:
                        # This would need to be implemented based on your vector store structure
                        pass
                    except Exception:
                        pass

                source_metadata.append(source_info)

            context_string = "\n\n".join(context_parts)
            return context_string, source_metadata

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return None, []
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Ollama model.

        Args:
            prompt: The formatted prompt

        Returns:
            Generated response
        """
        try:
            # Use the enhanced model interface
            if hasattr(self.model, 'generate'):
                response = self.model.generate(prompt, temperature=0.7, max_tokens=800)
            elif hasattr(self.model, 'chat'):
                response = self.model.chat(prompt, temperature=0.7, max_tokens=800)
            elif callable(self.model):
                response = self.model(prompt)
            else:
                response = "Model interface not recognized. Please check Ollama configuration."

            # Clean up the response
            if isinstance(response, str):
                response = response.strip()
                # Remove any system prompt echoing
                if response.startswith("You are SAM"):
                    lines = response.split('\n')
                    # Find where the actual response starts
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith("You are") and not line.startswith("##"):
                            response = '\n'.join(lines[i:]).strip()
                            break
            else:
                response = str(response).strip()

            return response if response else "I'm ready to help! Please ask me a question."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request. Please check that Ollama is running with the DeepSeek-R1 model loaded."
    
    def _display_welcome(self):
        """Display welcome message and instructions."""
        print("\n" + "="*70)
        print("🤖 Welcome to SAM (Small Agent Model) Enhanced Chat Interface")
        print("="*70)
        print("SAM is your curious analytical assistant, ready to help you")
        print("explore and understand complex information.")
        print("\n🧠 Memory Control Center:")
        print("  - Access the Memory Control Center at: http://localhost:8501")
        print("  - Manage memories, visualize connections, and configure access")
        print("  - Type 'memory' to open the control center in your browser")
        print("\n📋 Commands:")
        print("  - Type your questions or requests normally")
        print("  - Type 'quit', 'exit', or 'bye' to end the session")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'help' to see this message again")
        print("\n🔍 Transparency Features:")
        print("  - Type 'sources on/off' to toggle source document display")
        print("  - Type 'metadata on/off' to toggle metadata display")
        print("  - Type 'tags on/off' to toggle tag display")
        print("  - Type 'status' to see current settings")
        print("\n🧠 Semantic Search Features:")
        print("  - Type 'semantics on/off' to toggle semantic retrieval")
        print("  - Type 'topk N' to set number of retrieved chunks (e.g., 'topk 7')")
        print("  - Type 'threshold N' to set similarity threshold (e.g., 'threshold 0.2')")
        print("\n📄 Multimodal Features:")
        print("  - Type 'multimodal on/off' to toggle multimodal content display")
        print("  - Type 'process <file>' to process a multimodal document")
        print("  - Type 'search <query> [type]' to search multimodal content")
        print("\n🔧 Tool-Augmented Reasoning:")
        print("  - Type 'tools on/off' to toggle tool-augmented reasoning")
        print("  - Type 'reasoning on/off' to toggle reasoning trace display")
        print("  - Type 'websearch on/off' to toggle local web search")
        print("  - Type 'tools list' to show available tools")
        print("\n💭 Memory Commands:")
        print("  - Type '!recall topic AI' to find AI-related memories")
        print("  - Type '!recall last 5' to get recent memories")
        print("  - Type '!searchmem tag:important' to search by tags")
        print("  - Type '!memstats' to view memory statistics")
        print("  - Type '!memhelp' for all memory commands")
        print("="*70 + "\n")
    
    def _display_help(self):
        """Display help information."""
        print("\n📚 SAM Chat Interface Help")
        print("-" * 30)
        print("SAM is designed to be curious and analytical. Try asking:")
        print("• Questions about complex topics")
        print("• Requests for analysis or synthesis")
        print("• Exploratory questions that invite deeper investigation")
        print("• Requests to connect different concepts or ideas")
        print("\nSAM will use any relevant knowledge from processed documents")
        print("to provide informed and contextual responses.\n")
    
    def _process_command(self, user_input: str) -> bool:
        """
        Process special commands including transparency controls.

        Args:
            user_input: The user's input

        Returns:
            True if command was processed, False otherwise
        """
        command = user_input.lower().strip()

        if command in ['quit', 'exit', 'bye']:
            print("\n👋 Thank you for chatting with SAM! Goodbye!")
            return True
        elif command == 'clear':
            self.conversation_history.clear()
            print("\n🧹 Conversation history cleared.")
            return False
        elif command == 'help':
            self._display_help()
            return False
        elif command == 'status':
            self._display_status()
            return False
        elif command == 'memory':
            self._open_memory_control_center()
            return False
        elif command.startswith('sources '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.show_sources = True
                print("✅ Source document display enabled")
            elif setting == 'off':
                self.show_sources = False
                print("❌ Source document display disabled")
            else:
                print("❓ Use 'sources on' or 'sources off'")
            return False
        elif command.startswith('metadata '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.show_metadata = True
                print("✅ Metadata display enabled")
            elif setting == 'off':
                self.show_metadata = False
                print("❌ Metadata display disabled")
            else:
                print("❓ Use 'metadata on' or 'metadata off'")
            return False
        elif command.startswith('tags '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.show_tags = True
                print("✅ Tag display enabled")
            elif setting == 'off':
                self.show_tags = False
                print("❌ Tag display disabled")
            else:
                print("❓ Use 'tags on' or 'tags off'")
            return False
        elif command.startswith('semantics '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.semantic_mode = True
                print("✅ Semantic retrieval enabled")
            elif setting == 'off':
                self.semantic_mode = False
                print("❌ Semantic retrieval disabled (direct mode)")
            else:
                print("❓ Use 'semantics on' or 'semantics off'")
            return False
        elif command.startswith('topk '):
            try:
                value = int(command.split(' ', 1)[1])
                if 1 <= value <= 20:
                    self.semantic_top_k = value
                    print(f"✅ Top-K set to {value}")
                else:
                    print("❓ Top-K must be between 1 and 20")
            except ValueError:
                print("❓ Use 'topk N' where N is a number (e.g., 'topk 7')")
            return False
        elif command.startswith('threshold '):
            try:
                value = float(command.split(' ', 1)[1])
                if 0.0 <= value <= 1.0:
                    self.semantic_threshold = value
                    print(f"✅ Similarity threshold set to {value}")
                else:
                    print("❓ Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("❓ Use 'threshold N' where N is a decimal (e.g., 'threshold 0.2')")
            return False
        elif command.startswith('multimodal '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.multimodal_mode = True
                print("✅ Multimodal content display enabled")
            elif setting == 'off':
                self.multimodal_mode = False
                print("❌ Multimodal content display disabled")
            else:
                print("❓ Use 'multimodal on' or 'multimodal off'")
            return False
        elif command.startswith('process '):
            file_path = command.split(' ', 1)[1].strip()
            return self._process_multimodal_document(file_path)
        elif command.startswith('search '):
            search_parts = command.split(' ', 2)
            if len(search_parts) >= 2:
                query = search_parts[1]
                content_type = search_parts[2] if len(search_parts) > 2 else None
                return self._search_multimodal_content(query, content_type)
            else:
                print("❓ Use 'search <query> [content_type]'")
            return False
        elif command.startswith('tools '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.tool_mode = True
                print("✅ Tool-augmented reasoning enabled")
            elif setting == 'off':
                self.tool_mode = False
                print("❌ Tool-augmented reasoning disabled")
            elif setting == 'list':
                return self._show_available_tools()
            else:
                print("❓ Use 'tools on/off/list'")
            return False
        elif command.startswith('reasoning '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.show_reasoning = True
                print("✅ Reasoning trace display enabled")
            elif setting == 'off':
                self.show_reasoning = False
                print("❌ Reasoning trace display disabled")
            else:
                print("❓ Use 'reasoning on/off'")
            return False
        elif command.startswith('websearch '):
            setting = command.split(' ', 1)[1]
            if setting == 'on':
                self.enable_web_search = True
                print("✅ Local web search enabled")
                # Reinitialize tool selector with web search
                if self.tool_selector:
                    from reasoning.tool_selector import get_tool_selector
                    self.tool_selector = get_tool_selector(enable_web_search=True)
            elif setting == 'off':
                self.enable_web_search = False
                print("❌ Local web search disabled")
                # Reinitialize tool selector without web search
                if self.tool_selector:
                    from reasoning.tool_selector import get_tool_selector
                    self.tool_selector = get_tool_selector(enable_web_search=False)
            else:
                print("❓ Use 'websearch on/off'")
            return False

        return False

    def _open_memory_control_center(self):
        """Open the Memory Control Center in the default browser."""
        try:
            import webbrowser
            import subprocess
            import time

            # Check if the memory control center is already running
            try:
                import requests
                response = requests.get("http://localhost:8501", timeout=2)
                if response.status_code == 200:
                    print("🧠 Opening Memory Control Center in your browser...")
                    webbrowser.open("http://localhost:8501")
                    print("✅ Memory Control Center opened at http://localhost:8501")
                    return
            except:
                pass

            # If not running, try to start it
            print("🚀 Starting Memory Control Center...")
            try:
                # Launch the memory UI in the background
                subprocess.Popen([
                    sys.executable, "launch_memory_ui.py"
                ], cwd=".", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Wait a moment for it to start
                print("⏳ Waiting for Memory Control Center to start...")
                time.sleep(3)

                # Try to open it
                webbrowser.open("http://localhost:8501")
                print("✅ Memory Control Center started and opened at http://localhost:8501")
                print("💡 You can now manage memories, visualize connections, and configure access")

            except Exception as e:
                print(f"❌ Could not start Memory Control Center automatically: {e}")
                print("💡 You can manually start it by running: python launch_memory_ui.py")
                print("🔗 Then visit: http://localhost:8501")

        except Exception as e:
            print(f"❌ Error opening Memory Control Center: {e}")
            print("🔗 Please visit: http://localhost:8501")

    def _process_memory_command(self, command: str):
        """Process memory commands starting with !"""
        try:
            from ui.memory_commands import get_command_processor

            print("🧠 Processing memory command...", end="", flush=True)

            # Get command processor
            command_processor = get_command_processor()

            # Process the command
            result = command_processor.process_command(command, user_id="chat_user")

            print("\r" + " " * 35 + "\r", end="")  # Clear processing message

            if result.success:
                print(f"✅ Memory Command Result ({result.execution_time_ms}ms):")
                print(result.message)

                # If there's data, show a summary
                if result.data and isinstance(result.data, list) and len(result.data) > 0:
                    print(f"\n📊 Found {len(result.data)} results")

            else:
                print(f"❌ Memory Command Failed:")
                print(result.message)
                print("💡 Type '!memhelp' for available commands")

        except ImportError:
            print("\r" + " " * 35 + "\r", end="")
            print("❌ Memory command system not available")
            print("💡 Make sure the memory system is properly initialized")
        except Exception as e:
            print("\r" + " " * 35 + "\r", end="")
            print(f"❌ Error processing memory command: {e}")

    def _is_memory_system_available(self) -> bool:
        """Check if the memory system is available."""
        try:
            from ui.memory_commands import get_command_processor
            return True
        except ImportError:
            return False

    def _display_status(self):
        """Display current transparency and semantic settings."""
        print("\n📊 Current Settings:")
        print(f"  🔍 Show Sources: {'✅ ON' if self.show_sources else '❌ OFF'}")
        print(f"  📋 Show Metadata: {'✅ ON' if self.show_metadata else '❌ OFF'}")
        print(f"  🏷️  Show Tags: {'✅ ON' if self.show_tags else '❌ OFF'}")
        print(f"  💾 Knowledge DB: {'✅ Available' if self.knowledge_db else '❌ Not Available'}")
        print(f"  🧠 Semantic Mode: {'✅ ON' if self.semantic_mode else '❌ OFF'}")
        print(f"  📊 Top-K Chunks: {self.semantic_top_k}")
        print(f"  🎯 Similarity Threshold: {self.semantic_threshold}")
        print(f"  🔧 Query Router: {'✅ Available' if self.query_router else '❌ Not Available'}")
        print(f"  📄 Multimodal Mode: {'✅ ON' if self.multimodal_mode else '❌ OFF'}")
        print(f"  🎨 Show Content Types: {'✅ ON' if self.show_content_types else '❌ OFF'}")
        print(f"  🔧 Multimodal Pipeline: {'✅ Available' if self.multimodal_pipeline else '❌ Not Available'}")
        print(f"  🛠️ Tool Mode: {'✅ ON' if self.tool_mode else '❌ OFF'}")
        print(f"  🤔 Show Reasoning: {'✅ ON' if self.show_reasoning else '❌ OFF'}")
        print(f"  🌐 Web Search: {'✅ ON' if self.enable_web_search else '❌ OFF'}")
        print(f"  🔧 SELF-DECIDE Framework: {'✅ Available' if self.self_decide else '❌ Not Available'}")
        print(f"  🛠️ Tool Selector: {'✅ Available' if self.tool_selector else '❌ Not Available'}")

        # Show vector store stats if available
        if self.query_router:
            try:
                stats = self.query_router.get_stats()
                vector_stats = stats.get('vector_store', {})
                print(f"  📚 Vector Store: {vector_stats.get('total_chunks', 0)} chunks")
            except Exception:
                pass

        # Show multimodal stats if available
        if self.multimodal_pipeline:
            try:
                multimodal_stats = self.multimodal_pipeline.get_processing_stats()
                print(f"  📄 Documents Processed: {multimodal_stats.get('documents_processed', 0)}")
                print(f"  🧩 Content Blocks: {multimodal_stats.get('total_content_blocks', 0)}")
            except Exception:
                pass

    def _display_source_info(self, source_metadata: List[dict]):
        """Display source document information."""
        if not source_metadata or not self.show_sources:
            return

        print("\n📚 Source Documents:")
        print("-" * 50)

        for source in source_metadata:
            print(f"[{source['source_id']}] Content: {source['content_preview']}")

            if self.show_metadata and source.get('metadata'):
                metadata = source['metadata']
                print(f"    📋 Metadata: {metadata}")

            if self.show_tags and source.get('metadata', {}).get('tags'):
                tags = source['metadata']['tags']
                print(f"    🏷️  Tags: {', '.join(tags[:10])}{'...' if len(tags) > 10 else ''}")

            print()
    
    def start_chat(self):
        """Start the interactive chat session."""
        self._display_welcome()
        
        try:
            while True:
                # Get user input
                try:
                    # Show memory control indicator
                    prompt = "You [🧠⚙️]: " if self._is_memory_system_available() else "You: "
                    user_input = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Chat session ended. Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process commands
                if self._process_command(user_input):
                    break

                # Check for memory commands
                if user_input.startswith('!'):
                    self._process_memory_command(user_input)
                    continue

                # Process query through tool-augmented reasoning or semantic router
                print("🔍 Thinking...", end="", flush=True)

                # Check if tool-augmented reasoning should be used
                if self.tool_mode and self.self_decide and self._should_use_tools(user_input):
                    # Use SELF-DECIDE framework with tool-augmented reasoning
                    response, context_chunks, query_metadata = self._process_with_tools(user_input)
                elif self.query_router and self.semantic_mode:
                    # Use semantic routing
                    query_result = self.query_router.route_query(
                        user_input=user_input,
                        semantic_mode=True,
                        top_k=self.semantic_top_k
                    )
                    prompt = query_result.formatted_prompt
                    context_chunks = query_result.context_chunks
                    query_metadata = query_result.metadata
                    response = self._generate_response(prompt)
                else:
                    # Fallback to original context retrieval
                    context, source_metadata = self._retrieve_context(user_input)
                    prompt = self._format_prompt(user_input, context)
                    context_chunks = []
                    query_metadata = {'mode': 'fallback'}
                    response = self._generate_response(prompt)

                print("\r" + " " * 15 + "\r", end="")  # Clear "Thinking..." message

                # Display response
                print(f"SAM: {response}")

                # Display semantic context information if available
                if context_chunks and (self.show_sources or self.show_metadata or self.show_tags):
                    self._display_semantic_context(context_chunks, query_metadata)
                elif not self.semantic_mode:
                    # Fallback source display
                    try:
                        context, source_metadata = self._retrieve_context(user_input)
                        self._display_source_info(source_metadata)
                    except:
                        pass

                print()
                
                # Log retrieval if logger is available
                if self.retrieval_logger:
                    try:
                        processing_time = query_metadata.get('processing_time_seconds', 0.0)
                        mode = query_metadata.get('mode', 'unknown')
                        self.retrieval_logger.log_retrieval(
                            query=user_input,
                            response=response,
                            context_chunks=context_chunks,
                            processing_time=processing_time,
                            mode=mode
                        )
                    except Exception as e:
                        logger.error(f"Error logging retrieval: {e}")

                # Store in conversation history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.error(f"Error in chat session: {e}")
            print(f"\n❌ An error occurred: {e}")
            print("Chat session ended.")

    def _display_semantic_context(self, context_chunks: List[dict], query_metadata: dict):
        """Display semantic search context information."""
        if not context_chunks:
            return

        print("\n🧠 Semantic Search Results:")
        print("-" * 60)
        print(f"Mode: {query_metadata.get('mode', 'unknown')}")
        print(f"Chunks Retrieved: {len(context_chunks)}")

        if 'processing_time_seconds' in query_metadata:
            print(f"Processing Time: {query_metadata['processing_time_seconds']:.3f}s")

        if self.show_sources or self.show_metadata or self.show_tags:
            print("\n📚 Source Chunks:")

            for i, chunk in enumerate(context_chunks[:3], 1):  # Show top 3
                metadata = chunk.get('metadata', {})
                similarity = chunk.get('similarity_score', 0)

                print(f"\n[{i}] Similarity: {similarity:.3f}")

                if self.show_sources:
                    source_file = metadata.get('source_file', 'Unknown')
                    print(f"    📁 Source: {source_file}")

                if self.show_tags and metadata.get('tags'):
                    tags = metadata['tags']
                    print(f"    🏷️  Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")

                if self.show_metadata:
                    enrichment_score = metadata.get('enrichment_score', 0)
                    timestamp = metadata.get('timestamp', 'Unknown')
                    print(f"    📊 Enrichment: {enrichment_score:.1f}")
                    print(f"    ⏰ Processed: {timestamp}")

                # Show content preview
                content_preview = chunk.get('text', '')[:150]
                print(f"    📝 Preview: {content_preview}...")

        print()

    def _process_multimodal_document(self, file_path: str) -> bool:
        """Process a multimodal document and add to knowledge base."""
        if not self.multimodal_pipeline:
            print("❌ Multimodal pipeline not available")
            return False

        try:
            from pathlib import Path

            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return False

            print(f"🔄 Processing multimodal document: {file_path}")

            # Process the document
            result = self.multimodal_pipeline.process_document(file_path)

            if result:
                print(f"✅ Document processed successfully!")
                print(f"  📄 Document ID: {result['document_id']}")
                print(f"  🧩 Content Blocks: {result['content_blocks']}")
                print(f"  📝 Summary Length: {result['summary_length']} chars")
                print(f"  🔑 Key Concepts: {result['key_concepts']}")
                print(f"  📊 Enrichment Score: {result['enrichment_score']:.2f}")
                print(f"  🎯 Priority: {result['priority_level'].upper()}")
                print(f"  🎨 Content Types: {', '.join(result['content_types'])}")
                print(f"  📁 Output: {result['output_directory']}")
                return False  # Don't process as regular chat
            else:
                print("❌ Failed to process document")
                return False

        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return False

    def _search_multimodal_content(self, query: str, content_type: Optional[str] = None) -> bool:
        """Search multimodal content in the knowledge base."""
        if not self.multimodal_pipeline:
            print("❌ Multimodal pipeline not available")
            return False

        try:
            print(f"🔍 Searching multimodal content: '{query}'")
            if content_type:
                print(f"  🎨 Content type filter: {content_type}")

            # Search multimodal content
            results = self.multimodal_pipeline.search_multimodal_content(
                query=query,
                top_k=5,
                content_type_filter=content_type
            )

            if results:
                print(f"✅ Found {len(results)} multimodal results:")
                print("-" * 60)

                for i, result in enumerate(results, 1):
                    similarity = result.get('similarity_score', 0)
                    content_type = result.get('content_type', 'unknown')
                    is_multimodal = result.get('is_multimodal', False)
                    enrichment_score = result.get('enrichment_score', 0)

                    print(f"\n[{i}] Similarity: {similarity:.3f} | Type: {content_type}")

                    if is_multimodal:
                        print(f"    🎨 Multimodal Content (Score: {enrichment_score:.2f})")

                        # Show key concepts if available
                        key_concepts = result.get('key_concepts', [])
                        if key_concepts:
                            print(f"    🔑 Key Concepts: {', '.join(key_concepts[:3])}{'...' if len(key_concepts) > 3 else ''}")

                        # Show multimodal data if available
                        multimodal_data = result.get('multimodal_data', {})
                        if multimodal_data:
                            content_types = multimodal_data.get('content_types_present', [])
                            if content_types:
                                print(f"    📄 Contains: {', '.join(content_types)}")

                    # Show content preview
                    content_preview = result.get('text', '')[:150]
                    print(f"    📝 Preview: {content_preview}...")

                print("\n" + "-" * 60)
            else:
                print("❌ No multimodal content found for this query")

            return False  # Don't process as regular chat

        except Exception as e:
            print(f"❌ Error searching multimodal content: {e}")
            return False

    def _should_use_tools(self, message: str) -> bool:
        """Determine if tools should be used for this query."""
        # Use tools for complex queries that might benefit from structured reasoning
        tool_indicators = [
            'calculate', 'compute', 'analyze', 'compare', 'find', 'search',
            'how to', 'what is', 'explain', 'solve', 'plot', 'graph',
            'table', 'list', 'organize', 'steps', 'process', 'create'
        ]

        message_lower = message.lower()
        return any(indicator in message_lower for indicator in tool_indicators) or len(message.split()) > 10

    def _process_with_tools(self, user_input: str) -> Tuple[str, List[dict], dict]:
        """Process query using SELF-DECIDE framework and tools."""
        try:
            logger.info("Using tool-augmented reasoning for query processing")

            # Execute SELF-DECIDE reasoning
            session = self.self_decide.reason(user_input)

            # Synthesize enhanced response
            if self.answer_synthesizer:
                synthesized = self.answer_synthesizer.synthesize_response(session)

                # Format response for chat display
                response = self.answer_synthesizer.format_response_for_chat(
                    synthesized,
                    show_sources=self.show_sources,
                    show_reasoning=self.show_reasoning
                )

                # Create context chunks from sources
                context_chunks = []
                for attribution in synthesized.source_attributions:
                    context_chunks.append({
                        'text': attribution.content_preview,
                        'similarity_score': attribution.confidence,
                        'metadata': {
                            'source_type': attribution.source_type,
                            'source_name': attribution.source_name,
                            **attribution.metadata
                        }
                    })

                # Create query metadata
                query_metadata = {
                    'mode': 'tool_augmented',
                    'session_id': session.session_id,
                    'tools_used': synthesized.tools_used,
                    'confidence': synthesized.confidence_score,
                    'processing_time_seconds': session.total_duration_ms / 1000.0
                }

                return response, context_chunks, query_metadata
            else:
                # Fallback: use session's final answer
                return session.final_answer, [], {'mode': 'tool_augmented_fallback'}

        except Exception as e:
            logger.error(f"Tool-augmented reasoning failed: {e}")
            # Fallback to standard processing
            if self.query_router and self.semantic_mode:
                query_result = self.query_router.route_query(
                    user_input=user_input,
                    semantic_mode=True,
                    top_k=self.semantic_top_k
                )
                response = self._generate_response(query_result.formatted_prompt)
                return response, query_result.context_chunks, query_result.metadata
            else:
                context, source_metadata = self._retrieve_context(user_input)
                prompt = self._format_prompt(user_input, context)
                response = self._generate_response(prompt)
                return response, [], {'mode': 'fallback_after_tool_error'}

    def _show_available_tools(self) -> bool:
        """Show available tools and their capabilities."""
        if not self.tool_selector:
            print("❌ Tool selector not available")
            return False

        try:
            tools_summary = self.tool_selector.get_tools_summary()

            print("\n🛠️ Available Tools:")
            print("=" * 60)

            for tool_name, description in tools_summary.items():
                tool_info = self.tool_selector.get_tool_info(tool_name)

                print(f"\n🔧 **{tool_name.replace('_', ' ').title()}**")
                print(f"   📝 {description}")

                if tool_info:
                    print(f"   ⚡ Complexity: {tool_info.complexity_level}")
                    print(f"   ⏱️ Speed: {tool_info.execution_time}")
                    print(f"   🎯 Use cases: {', '.join(tool_info.use_cases[:3])}{'...' if len(tool_info.use_cases) > 3 else ''}")

            print("\n" + "=" * 60)
            print("💡 Tools are automatically selected based on your query requirements.")
            print("   Use 'tools on/off' to enable/disable tool-augmented reasoning.")

            return False

        except Exception as e:
            print(f"❌ Error displaying tools: {e}")
            return False


def launch_chat_interface(model: Any, vector_index: Any, system_prompt: str):
    """
    Launch the chat interface.
    
    Args:
        model: The loaded language model
        vector_index: The vector store for knowledge retrieval
        system_prompt: The system prompt for the model
    """
    chat = ChatInterface(model, vector_index, system_prompt)
    chat.start_chat()
