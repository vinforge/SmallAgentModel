"""
SAM Web UI - Flask-based web interface for multimodal agent interaction
"""

import os

from sam.core.sam_model_client import create_legacy_ollama_client
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import uuid

# Configure logging first
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import SAM components
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Debug: Print the paths being used
logger.info(f"Current directory: {current_dir}")
logger.info(f"Parent directory: {parent_dir}")
logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

try:
    from utils.vector_manager import VectorManager
    from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
    from utils.embedding_utils import get_embedding_manager
    from utils.web_retrieval_suggester import WebRetrievalSuggester
    # Import security middleware
    from web_ui.security_middleware import (
        security_middleware, require_unlock, optional_security,
        get_secure_memory_store, create_security_routes, inject_security_context
    )
    # Import vetting interface
    from web_ui.vetting_interface import vetting_bp
    logger.info("✅ Successfully imported SAM components")
except ImportError as e:
    logger.warning(f"Import error: {e}")
    # Fallback imports for development
    VectorManager = None
    get_multimodal_pipeline = None
    get_embedding_manager = None
    WebRetrievalSuggester = None
    security_middleware = None
    require_unlock = lambda f: f
    optional_security = lambda f: f
    get_secure_memory_store = None

# Simple PDF viewer route (serves original file and accepts page param for front-end to anchor)
@app.route('/viewer')
@optional_security
def viewer():
    try:
        doc_id = request.args.get('doc')
        page = request.args.get('page')  # optional
        if not doc_id:
            return jsonify({'error': 'Missing doc parameter'}), 400
        # Load v2 metadata to locate original file
        meta_path = Path('uploads') / doc_id / 'metadata.json'
        if not meta_path.exists():
            return jsonify({'error': f'metadata not found for {doc_id}'}), 404
        meta = json.loads(meta_path.read_text())
        file_path = meta.get('file_path') or ''
        if not file_path or not os.path.exists(file_path):
            # Fallback to uploads path if file was moved
            candidate = Path('uploads') / doc_id / meta.get('filename', '')
            if candidate.exists():
                file_path = str(candidate)
            else:
                return jsonify({'error': 'Original file not found'}), 404
        # For now, serve the raw file; a front-end can use the page param to anchor.
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        resp = send_from_directory(directory, filename)
        # Hint header for page param that a client-side PDF viewer could use
        if page:
            resp.headers['X-Viewer-Page'] = str(page)
        return resp
    except Exception as e:
        logger.error(f"Viewer error: {e}")
        return jsonify({'error': str(e)}), 500

    create_security_routes = lambda app: None

@app.route('/view')
@optional_security
def view_document():
    try:
        doc_id = request.args.get('doc')
        page = request.args.get('page', type=int, default=1)
        if not doc_id:
            return jsonify({'error': 'Missing doc parameter'}), 400
        # Reuse the raw file via /viewer; the template uses PDF.js to render and handle paging
        return render_template('pdf_viewer.html', doc_id=doc_id, page=page)
    except Exception as e:
        logger.error(f"View error: {e}")
        return jsonify({'error': str(e)}), 500

    inject_security_context = lambda: {}
    vetting_bp = None

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'sam_multimodal_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'txt', 'md', 'pdf', 'docx', 'html', 'htm',
    'py', 'js', 'java', 'cpp', 'c'
}

# Global components
sam_model = None
vector_manager = None
multimodal_pipeline = None
embedding_manager = None
web_retrieval_suggester = None

# Tool-augmented reasoning components
self_decide_framework = None
tool_selector = None
tool_executor = None
answer_synthesizer = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_sam():
    """Initialize SAM components."""
    global sam_model, vector_manager, multimodal_pipeline, embedding_manager, web_retrieval_suggester
    global self_decide_framework, tool_selector, tool_executor, answer_synthesizer

    try:
        logger.info("Initializing SAM components...")

        # Initialize real SAM model with Ollama - NO MOCK MODEL

        # Initialize real SAM model with Ollama
        try:
            # Try to use Ollama directly via requests
            import requests

            class OllamaModel:
                def __init__(self):
                    self.base_url = "http://localhost:11434"
                    self.model_name = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
                    self.model_type = "REAL_OLLAMA_MODEL"  # For debugging
                    self.learned_knowledge = []  # Store learned facts for context injection
                    logger.info(f"🤖 Initialized REAL Ollama model with learning capability: {self.model_name}")

                def inject_learned_knowledge(self, knowledge_summary: str, key_concepts: list):
                    """Inject learned knowledge into the model's context for future queries."""
                    from datetime import datetime
                    knowledge_entry = {
                        'summary': knowledge_summary,
                        'concepts': key_concepts,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'document_learning'
                    }
                    self.learned_knowledge.append(knowledge_entry)

                    # Keep only the most recent 10 knowledge entries to avoid context overflow
                    if len(self.learned_knowledge) > 10:
                        self.learned_knowledge = self.learned_knowledge[-10:]

                    logger.info(f"🧠 WEB UI: Injected new knowledge into model context: {len(key_concepts)} concepts")

                def _build_enhanced_context(self, prompt: str) -> str:
                    """Build enhanced context with learned knowledge for better responses."""
                    if not self.learned_knowledge:
                        return prompt

                    # Create knowledge context
                    knowledge_context = "LEARNED KNOWLEDGE FROM UPLOADED DOCUMENTS:\n"
                    for i, knowledge in enumerate(self.learned_knowledge[-5:], 1):  # Use last 5 entries
                        knowledge_context += f"\n{i}. {knowledge['summary'][:200]}...\n"
                        knowledge_context += f"   Key concepts: {', '.join(knowledge['concepts'][:5])}\n"

                    # Inject knowledge before the actual prompt
                    enhanced_prompt = f"{knowledge_context}\n\nBased on the above learned knowledge from uploaded documents and your training, respond to:\n\n{prompt}"
                    return enhanced_prompt

                def generate(self, prompt, temperature=0.7, max_tokens=500, use_learned_knowledge=True):
                    logger.info(f"🤖 REAL Ollama model generating response for prompt: {prompt[:100]}...")
                    try:
                        # Enhance prompt with learned knowledge if requested
                        if use_learned_knowledge and self.learned_knowledge:
                            enhanced_prompt = self._build_enhanced_context(prompt)
                            logger.info(f"🧠 Enhanced prompt with {len(self.learned_knowledge)} learned knowledge entries")
                        else:
                            enhanced_prompt = prompt

                        response = create_legacy_ollama_client().generate(prompt)
                        if response.status_code == 200:
                            result = response.json().get("response", "No response generated")
                            logger.info(f"✅ REAL Ollama model generated {len(result)} character response")
                            return result
                        else:
                            raise Exception(f"Ollama API error: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Ollama generation error: {e}")
                        return f"I apologize, but I'm having trouble generating a response right now: {str(e)}"

            # Test Ollama connection
            test_response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
            if test_response.status_code == 200:
                sam_model = OllamaModel()
                logger.info("✅ Real SAM model initialized with Ollama")
            else:
                raise Exception("Ollama not responding")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama model: {e}")
            logger.error("CRITICAL: Ollama model is required for SAM to function properly")
            raise Exception(f"Cannot start SAM without Ollama model: {e}")

        # Initialize vector manager (optional)
        if VectorManager:
            vector_manager = VectorManager()
            logger.info("✅ Vector manager initialized")
        else:
            logger.warning("⚠️ Vector manager not available")

        # Initialize multimodal pipeline (optional)
        if get_multimodal_pipeline:
            multimodal_pipeline = get_multimodal_pipeline()
            logger.info("✅ Multimodal pipeline initialized")
        else:
            logger.warning("⚠️ Multimodal pipeline not available")

        # Initialize embedding manager (optional)
        if get_embedding_manager:
            embedding_manager = get_embedding_manager()
            logger.info("✅ Embedding manager initialized")
        else:
            logger.warning("⚠️ Embedding manager not available")

        # Initialize web retrieval suggester
        if WebRetrievalSuggester:
            web_retrieval_suggester = WebRetrievalSuggester()
            logger.info("✅ Web retrieval suggester initialized")
        else:
            logger.warning("⚠️ Web retrieval suggester not available")

        # Initialize tool-augmented reasoning components
        try:
            from reasoning.self_decide_framework import get_self_decide_framework
            from reasoning.tool_selector import get_tool_selector
            from reasoning.tool_executor import get_tool_executor
            from reasoning.answer_synthesizer import get_answer_synthesizer

            self_decide_framework = get_self_decide_framework()
            tool_selector = get_tool_selector(enable_web_search=False)
            tool_executor = get_tool_executor()
            answer_synthesizer = get_answer_synthesizer(model=sam_model)

            # Connect components
            self_decide_framework.tool_selector = tool_selector
            self_decide_framework.model = sam_model

            # CRITICAL FIX: Connect to memory vector store with adapter (Phase 3.2: Use ChromaDB)
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )

            # Create dynamic adapter to make memory store compatible with SELF-DECIDE framework
            class MemoryStoreAdapter:
                def __init__(self, memory_store):
                    self.memory_store = memory_store
                    self.current_query = None  # Store current query for context

                def search(self, query_embedding, top_k=5, score_threshold=0.1):
                    """Dynamic adapter method to make memory store compatible with SELF-DECIDE."""
                    try:
                        # Method 1: Try direct embedding search if available
                        if hasattr(self.memory_store, '_search_vector_index'):
                            try:
                                similar_chunks = self.memory_store._search_vector_index(query_embedding, top_k)
                                results = []

                                for chunk_id, similarity in similar_chunks:
                                    chunk = self.memory_store.memory_chunks.get(chunk_id)
                                    if chunk and similarity >= score_threshold:
                                        result = {
                                            'text': chunk.content,
                                            'similarity_score': similarity,
                                            'metadata': {
                                                'chunk_id': chunk.chunk_id,
                                                'source': chunk.source,
                                                'memory_type': chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type),
                                                'timestamp': chunk.timestamp,
                                                'importance_score': chunk.importance_score,
                                                **getattr(chunk, 'metadata', {})
                                            }
                                        }
                                        results.append(result)

                                if results:
                                    logger.info(f"Memory adapter: Found {len(results)} results via embedding search")
                                    return results
                            except Exception as e:
                                logger.warning(f"Embedding search failed: {e}")

                        # Method 2: Fallback to text-based search using current query context
                        if self.current_query:
                            memories = self.memory_store.search_memories(
                                self.current_query,
                                max_results=top_k,
                                min_similarity=score_threshold
                            )

                            results = []
                            for memory in memories:
                                result = {
                                    'text': memory.content,
                                    'similarity_score': getattr(memory, 'similarity_score', 0.8),
                                    'metadata': {
                                        'chunk_id': memory.chunk_id,
                                        'source': memory.source,
                                        'memory_type': memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                                        'timestamp': memory.timestamp,
                                        'importance_score': memory.importance_score,
                                        **getattr(memory, 'metadata', {})
                                    }
                                }
                                results.append(result)

                            logger.info(f"Memory adapter: Found {len(results)} results via text search for '{self.current_query}'")
                            return results

                        # Method 3: Last resort - search for any Blue Cloak content
                        memories = self.memory_store.search_memories("blue cloak", max_results=top_k)
                        results = []
                        for memory in memories:
                            result = {
                                'text': memory.content,
                                'similarity_score': 0.7,  # Default similarity
                                'metadata': {
                                    'chunk_id': memory.chunk_id,
                                    'source': memory.source,
                                    'memory_type': memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                                    'timestamp': memory.timestamp,
                                    'importance_score': memory.importance_score,
                                    **getattr(memory, 'metadata', {})
                                }
                            }
                            results.append(result)

                        logger.info(f"Memory adapter: Found {len(results)} results via fallback search")
                        return results

                    except Exception as e:
                        logger.error(f"Memory store adapter search failed: {e}")
                        return []

                def set_query_context(self, query):
                    """Set the current query context for better search."""
                    self.current_query = query

            # Use adapter instead of direct memory store
            memory_adapter = MemoryStoreAdapter(memory_store)
            self_decide_framework.vector_manager = memory_adapter

            logger.info(f"Connected SELF-DECIDE to memory store with {len(memory_store.get_all_memories())} memories")

            logger.info("✅ Tool-augmented reasoning initialized")

        except ImportError as e:
            logger.warning(f"Tool-augmented reasoning not available: {e}")

        return True

    except Exception as e:
        logger.error(f"Error initializing SAM: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')

@app.route('/vetting')
@optional_security
def vetting_dashboard():
    """Content vetting dashboard."""
    return render_template('vetting_dashboard.html')

@app.route('/dream-canvas')
@optional_security
def dream_canvas():
    """Dream Canvas - Cognitive synthesis visualization."""
    return render_template('dream_canvas.html')

@app.route('/api/trigger-web-search', methods=['POST'])
@optional_security
def trigger_web_search():
    """Phase 8.4: Trigger web search when user accepts escalation."""
    try:
        data = request.get_json()
        search_query = data.get('search_query', '').strip()
        original_query = data.get('original_query', '').strip()

        if not search_query:
            return jsonify({'error': 'Search query is required'}), 400

        logger.info(f"Triggering web search for: '{search_query}' (original: '{original_query}')")

        # Phase 8.5: Use the new intelligent web system with cocoindex
        try:
            from web_retrieval.intelligent_web_system import IntelligentWebSystem
            from config.config_manager import ConfigManager

            # Load configuration
            config_manager = ConfigManager()
            config = config_manager.get_config()

            # Initialize intelligent web system
            api_keys = {
                'serper': config.serper_api_key if config.serper_api_key else None,
                'newsapi': config.newsapi_api_key if config.newsapi_api_key else None
            }

            web_config = {
                'cocoindex_search_provider': config.cocoindex_search_provider,
                'cocoindex_num_pages': config.cocoindex_num_pages,
                'web_retrieval_provider': config.web_retrieval_provider
            }

            intelligent_web_system = IntelligentWebSystem(api_keys=api_keys, config=web_config)

            # Process the query using intelligent routing
            result = intelligent_web_system.process_query(search_query)

            if result['success']:
                logger.info(f"🌐 Intelligent web search completed successfully using {result['tool_used']}")
                logger.info(f"🔍 Web search result structure: {list(result.keys())}")
                logger.info(f"🔍 Web search data keys: {list(result.get('data', {}).keys()) if 'data' in result else 'NO_DATA'}")

                # Save to quarantine for vetting with enhanced debugging
                logger.info(f"🚨 ABOUT TO CALL save_intelligent_web_to_quarantine 🚨")
                logger.info(f"Query: '{search_query}'")
                logger.info(f"Result type: {type(result)}")

                try:
                    from secure_streamlit_app import save_intelligent_web_to_quarantine
                    logger.info(f"✅ Successfully imported save_intelligent_web_to_quarantine")

                    save_intelligent_web_to_quarantine(result, search_query)
                    logger.info(f"✅ save_intelligent_web_to_quarantine completed without exception")

                except Exception as quarantine_error:
                    logger.error(f"❌ QUARANTINE SAVE FAILED: {quarantine_error}")
                    logger.error(f"❌ Quarantine error type: {type(quarantine_error)}")
                    import traceback
                    logger.error(f"❌ Quarantine traceback: {traceback.format_exc()}")
                    # Don't re-raise, continue with vetting

                # DISABLED: Automatic vetting process (to allow users to see quarantined content)
                # The automatic vetting was immediately moving files to archive, preventing users
                # from seeing them in the "Quarantined Content Preview" section
                logger.info(f"🛡️ Skipping automatic vetting - content saved to quarantine for manual review")
                vetting_message = "Content saved to quarantine. Visit the Content Vetting page to review and approve."

                # Optional: Uncomment below to re-enable automatic vetting
                # try:
                #     logger.info(f"🛡️ About to trigger vetting process")
                #     from secure_streamlit_app import trigger_vetting_process
                #     vetting_result = trigger_vetting_process()
                #
                #     if vetting_result.get('status') == 'success':
                #         logger.info("✅ Automatic vetting completed successfully after web search")
                #         vetting_message = "Content automatically vetted and ready for use!"
                #     else:
                #         logger.warning(f"⚠️ Automatic vetting encountered issues: {vetting_result}")
                #         vetting_message = "Content saved to quarantine. Manual vetting may be required."
                # except Exception as e:
                #     logger.warning(f"Automatic vetting failed: {e}")
                #     vetting_message = "Content saved to quarantine. Manual vetting may be required."

                return jsonify({
                    'status': 'search_completed',
                    'message': f'Intelligent web search completed using {result["tool_used"].replace("_", " ").title()}. {vetting_message}',
                    'search_query': search_query,
                    'original_query': original_query,
                    'tool_used': result['tool_used'],
                    'content_count': result.get('data', {}).get('total_chunks', 0) or result.get('data', {}).get('total_articles', 0) or 1,
                    'next_step': 'content_ready',
                    'vetting_status': vetting_result.get('status', 'unknown') if 'vetting_result' in locals() else 'failed'
                })
            else:
                logger.error(f"Intelligent web search failed: {result.get('error', 'Unknown error')}")
                return jsonify({
                    'status': 'search_failed',
                    'error': f'Intelligent web search failed: {result.get("error", "Unknown error")}',
                    'fallback_message': 'I apologize, but I encountered an issue while searching the web. Please try again or ask me to answer with my current knowledge.'
                }), 500

        except subprocess.TimeoutExpired:
            logger.error("Web content fetch timed out")
            return jsonify({
                'status': 'search_timeout',
                'error': 'Web search timed out',
                'fallback_message': 'The web search is taking longer than expected. Please try again or ask me to answer with my current knowledge.'
            }), 500

        except Exception as e:
            logger.error(f"Error triggering web search: {e}")
            return jsonify({
                'status': 'search_error',
                'error': str(e),
                'fallback_message': 'I encountered an error while searching the web. Please try again or ask me to answer with my current knowledge.'
            }), 500

    except Exception as e:
        logger.error(f"Error in trigger_web_search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def test_endpoint():
    """Test endpoint to verify basic functionality."""
    try:
        return jsonify({
            'status': 'ok',
            'message': 'SAM Web UI is running',
            'sam_model_available': sam_model is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
@optional_security
def chat():
    """Handle chat messages with optional security."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Empty message'}), 400

        # Initialize session if needed
        if 'conversation_id' not in session:
            session['conversation_id'] = str(uuid.uuid4())

        # Check for special commands
        if message.startswith('/'):
            return handle_command(message)

        # Generate response using SAM
        response_data = generate_sam_response(message)

        # Generate unique message ID for feedback tracking
        message_id = str(uuid.uuid4())

        # Handle different response types
        if isinstance(response_data, dict):
            # Phase 8.3: Handle web search escalation
            if response_data.get('type') == 'web_search_escalation':
                assessment = response_data['confidence_assessment']
                return jsonify({
                    'type': 'web_search_escalation',
                    'message': response_data['message'],
                    'confidence_score': assessment.confidence_score,
                    'confidence_level': assessment.confidence_level.value,
                    'reasons': assessment.reasons,
                    'suggested_search_query': response_data['suggested_search_query'],
                    'original_query': response_data['original_query'],
                    'message_id': message_id,
                    'timestamp': datetime.now().isoformat(),
                    'conversation_id': session['conversation_id']
                })

            # Handle normal dict responses with thought blocks
            return jsonify({
                'response': response_data['response'],
                'message_id': message_id,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': session['conversation_id'],
                'has_thoughts': response_data.get('has_thoughts', False),
                'thought_blocks': response_data.get('thought_blocks', [])
            })
        else:
            # Handle string responses
            return jsonify({
                'response': response_data,
                'message_id': message_id,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': session['conversation_id'],
                'has_thoughts': False,
                'thought_blocks': []
            })

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

def handle_command(command):
    """Handle special commands."""
    try:
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower()

        if cmd == 'status':
            stats = get_system_status()
            return jsonify({
                'response': f"📊 **System Status**\n\n{format_status(stats)}",
                'type': 'status'
            })

        elif cmd == 'search' and len(cmd_parts) > 1:
            query = ' '.join(cmd_parts[1:])
            results = search_multimodal_content(query)
            return jsonify({
                'response': format_search_results(results),
                'type': 'search'
            })

        elif cmd == 'summarize' and len(cmd_parts) > 1:
            topic = ' '.join(cmd_parts[1:])
            summary_result = generate_smart_summary(topic)
            return jsonify({
                'response': summary_result,
                'type': 'summary'
            })

        elif cmd == 'thoughts' and len(cmd_parts) > 1:
            action = cmd_parts[1].lower()
            thoughts_result = handle_thoughts_command(action)
            return jsonify({
                'response': thoughts_result,
                'type': 'thoughts'
            })

        elif cmd == 'help':
            return jsonify({
                'response': get_help_text(),
                'type': 'help'
            })

        else:
            return jsonify({
                'response': f"Unknown command: {cmd}. Type `/help` for available commands.",
                'type': 'error'
            })

    except Exception as e:
        logger.error(f"Error handling command: {e}")
        return jsonify({'error': str(e)}), 500

def generate_sam_response(message):
    """Generate response using SAM model with enhanced smart routing."""
    try:
        # PRIORITY 1: Check for mathematical queries first (SmartQueryRouter)
        if should_use_smart_router(message):
            return generate_smart_router_response(message)

        # PRIORITY 2: Check for document queries (only if not mathematical)
        if is_document_query(message):
            return handle_document_query(message)

        # PRIORITY 3: Check if tool-augmented reasoning should be used
        if should_use_tools(message) and self_decide_framework:
            return generate_tool_augmented_response(message)
        else:
            return generate_standard_response(message)

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

def should_use_smart_router(message):
    """Check if the query should use the SmartQueryRouter (mathematical, STEF, etc.)."""
    try:
        # Import SmartQueryRouter to check if query should be routed
        from services.search_router import SmartQueryRouter

        router = SmartQueryRouter()

        # Check for pure mathematical expressions first
        if router.is_pure_math_expression(message):
            logger.info(f"🧮 Pure math detected: '{message}' - routing to SmartQueryRouter")
            return True

        # Check for other mathematical signals
        math_signals = router.detect_math_signals(message)
        if math_signals and any(score > 0.7 for score in math_signals.values()):
            logger.info(f"🧮 Math signals detected: '{message}' - routing to SmartQueryRouter")
            return True

        # Check for STEF program matches
        if router._stef_enabled and router._program_registry:
            stef_program = router._check_stef_match(message, {})
            if stef_program:
                logger.info(f"🚀 STEF program match: '{message}' - routing to SmartQueryRouter")
                return True

        return False

    except Exception as e:
        logger.warning(f"Error checking smart router eligibility: {e}")
        return False

def generate_smart_router_response(message):
    """Generate response using the SmartQueryHandler and RichAnswer system."""
    try:
        logger.info(f"🎯 Processing via SmartQueryRouter: '{message}'")

        # Import and use SmartQueryHandler
        from services.smart_query_handler import SmartQueryHandler

        handler = SmartQueryHandler()
        response = handler.process_query(message)

        if response.success:
            logger.info(f"✅ SmartQueryRouter success: {response.route_type} in {response.execution_time_ms:.1f}ms")
            return response.content
        else:
            logger.warning(f"❌ SmartQueryRouter failed: {response.metadata}")
            # Fallback to standard response
            return generate_standard_response(message)

    except Exception as e:
        logger.error(f"Error in SmartQueryRouter response: {e}")
        # Fallback to standard response
        return generate_standard_response(message)

def should_use_tools(message):
    """Determine if tools should be used for this query."""
    tool_indicators = [
        'calculate', 'compute', 'analyze', 'compare', 'find', 'search',
        'how to', 'what is', 'explain', 'solve', 'plot', 'graph',
        'table', 'list', 'organize', 'steps', 'process', 'create'
    ]

    message_lower = message.lower()
    return any(indicator in message_lower for indicator in tool_indicators) or len(message.split()) > 10

def generate_tool_augmented_response(message):
    """Generate response using SELF-DECIDE framework with confidence assessment."""
    try:
        logger.info("Using tool-augmented reasoning with confidence assessment for web UI response")

        # Phase 8.1: Perform initial vector store search for confidence assessment
        search_results = []
        if self_decide_framework and self_decide_framework.vector_manager:
            try:
                from utils.embedding_utils import get_embedding_manager
                embedding_manager = get_embedding_manager()

                # Generate query embedding
                query_embedding = embedding_manager.embed_query(message)

                # Set query context for memory adapter if available
                if hasattr(self_decide_framework.vector_manager, 'set_query_context'):
                    self_decide_framework.vector_manager.set_query_context(message)

                # Search for relevant chunks
                search_results = self_decide_framework.vector_manager.search(
                    query_embedding, top_k=5, score_threshold=0.1
                )

                logger.info(f"Retrieved {len(search_results)} results for confidence assessment")

            except Exception as e:
                logger.warning(f"Vector search for confidence assessment failed: {e}")
                search_results = []

        # Phase 8.2: Assess confidence in retrieval quality
        try:
            from reasoning.confidence_assessor import get_confidence_assessor
            confidence_assessor = get_confidence_assessor()

            assessment = confidence_assessor.assess_retrieval_quality(search_results, message)

            logger.info(f"Confidence assessment: {assessment.status} ({assessment.confidence_score:.2f})")

            # Phase 8.3: Check if web search escalation should be offered
            if assessment.status == "NOT_CONFIDENT":
                # Return special response object for web search escalation
                return {
                    'type': 'web_search_escalation',
                    'confidence_assessment': assessment,
                    'message': assessment.explanation,
                    'suggested_search_query': assessment.suggested_search_query,
                    'original_query': message
                }

        except Exception as e:
            logger.warning(f"Confidence assessment failed: {e}")
            # Continue with normal processing if confidence assessment fails

        # Normal processing: Execute SELF-DECIDE reasoning
        session = self_decide_framework.reason(message)

        # Synthesize enhanced response
        if answer_synthesizer:
            synthesized = answer_synthesizer.synthesize_response(session)

            # Format response for web display
            response = answer_synthesizer.format_response_for_chat(
                synthesized,
                show_sources=True,
                show_reasoning=False  # Keep reasoning compact for web UI
            )

            return response
        else:
            # Fallback: use session's final answer
            return session.final_answer

    except Exception as e:
        logger.error(f"Tool-augmented reasoning failed: {e}")
        # Fallback to standard response
        return generate_standard_response(message)

def generate_smart_summary(topic):
    """Generate smart summary for a given topic."""
    try:
        from memory.smart_summarizer import get_smart_summarizer, SummaryRequest, SummaryType, SummaryFormat
        from memory.memory_vectorstore import get_memory_store

        summarizer = get_smart_summarizer()
        # Use ChromaDB for document retrieval
        from memory.memory_vectorstore import VectorStoreType
        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )

        # Create summary request
        request = SummaryRequest(
            topic_keyword=topic,
            summary_type=SummaryType.TOPIC_SUMMARY,
            output_format=SummaryFormat.MARKDOWN,
            max_length=500,
            include_sources=True
        )

        # Generate summary
        summary = summarizer.generate_summary(request, memory_store)

        # Format response
        response_parts = []
        response_parts.append(f"📊 **Smart Summary: {topic}**\n")
        response_parts.append(summary.summary_text)
        response_parts.append(f"\n**Summary Statistics:**")
        response_parts.append(f"- Sources: {summary.source_count}")
        response_parts.append(f"- Word Count: {summary.word_count}")
        response_parts.append(f"- Confidence: {summary.confidence_score:.2f}")
        response_parts.append(f"- Key Topics: {', '.join(summary.key_topics)}")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error generating smart summary: {e}")
        return f"Error generating summary for '{topic}': {str(e)}"

def handle_thoughts_command(action):
    """Handle /thoughts command for Sprint 16."""
    try:
        from utils.thought_processor import get_thought_processor

        thought_processor = get_thought_processor()

        if action in ['on', 'enable', 'show']:
            return "🧠 **Thought visibility enabled.** SAM's thinking process will be available via toggle buttons in responses."

        elif action in ['off', 'disable', 'hide']:
            return "🔒 **Thought visibility disabled.** SAM's thinking process will be hidden from responses."

        elif action in ['status', 'check']:
            config_status = thought_processor.show_thoughts
            status_text = "enabled" if config_status else "disabled"
            return f"🧠 **Thought visibility is currently {status_text}.**\n\nUse `/thoughts on` or `/thoughts off` to change this setting."

        else:
            return "❓ **Usage:** `/thoughts [on|off|status]`\n\n- `on` - Enable thought visibility\n- `off` - Disable thought visibility\n- `status` - Check current status"

    except Exception as e:
        logger.error(f"Error handling thoughts command: {e}")
        return f"❌ Error processing thoughts command: {str(e)}"

def is_document_query(message):
    """Check if the message is asking about a specific document or content from uploaded documents."""
    message_lower = message.lower()

    # CRITICAL: Exclude mathematical queries first
    # Check for mathematical patterns that should NOT be treated as document queries
    import re
    math_patterns = [
        r'\b\d+\s*[\+\-\*\/]\s*\d+\b',  # Basic arithmetic: 5+4, 100-50
        r'what\s+is\s+\d+[\+\-\*\/]\d+',  # "what is 5+4"
        r'calculate\s+\d+',  # "calculate 100"
        r'\d+\s*\%\s*of\s*\d+',  # "15% of 100"
    ]

    for pattern in math_patterns:
        if re.search(pattern, message_lower):
            logger.info(f"🧮 Mathematical pattern detected in '{message}' - NOT treating as document query")
            return False

    # ENHANCED: Check for Deep Analysis patterns first (high priority)
    deep_analysis_patterns = [
        r'🔍\s*deep\s*analysis',  # "🔍 Deep Analysis"
        r'deep\s*analysis.*\.pdf',  # "Deep Analysis: filename.pdf"
        r'analyze.*\.pdf',  # "analyze filename.pdf"
        r'comprehensive\s*analysis',  # "comprehensive analysis"
        r'detailed\s*analysis',  # "detailed analysis"
    ]

    for pattern in deep_analysis_patterns:
        if re.search(pattern, message_lower):
            logger.info(f"🔍 Deep Analysis pattern detected in '{message}' - treating as document query")
            return True

    # ENHANCED: Check for arXiv and academic paper patterns
    arxiv_patterns = [
        r'\b\d{4}\.\d{5}v?\d*\.?pdf?\b',  # arXiv patterns like "2305.18290v3.pdf"
        r'\b\d{4}\.\d{5}v?\d*\b',        # arXiv without extension
        r'arxiv:\s*\d{4}\.\d{5}',        # "arxiv:2305.18290"
    ]

    for pattern in arxiv_patterns:
        if re.search(pattern, message_lower):
            logger.info(f"📄 arXiv pattern detected in '{message}' - treating as document query")
            return True

    # ENHANCED: Document query indicators with better coverage
    document_indicators = [
        'summary of', 'summarize', 'what is in', 'content of',
        'document', '.pdf', '.docx', '.md', '.txt',
        'file', 'paper', 'report', 'uploaded', 'upload',
        'analyze', 'analysis', 'review', 'examine',
        'explain', 'describe', 'discuss', 'breakdown'
    ]

    # Person/entity query indicators that might be in uploaded documents
    person_indicators = [
        'who is', 'who was', 'tell me about', 'what about',
        'describe', 'information about', 'details about',
        'background on', 'profile of'
    ]

    # Check for document indicators first
    if any(indicator in message_lower for indicator in document_indicators):
        logger.info(f"📄 Document indicator detected in '{message}' - treating as document query")
        return True

    # Check for person/entity queries that might be answered from uploaded documents
    if any(indicator in message_lower for indicator in person_indicators):
        # Check if we have any uploaded documents that might contain this information
        try:
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )
            all_memories = memory_store.get_all_memories()

            # If we have document memories, treat person queries as document queries
            document_memories = [m for m in all_memories if 'document:' in str(getattr(m, 'source', ''))]
            if document_memories:
                logger.info(f"Person/entity query detected with {len(document_memories)} document memories available")
                return True
        except ImportError as e:
            logger.warning(f"Error importing memory store: {e}")
            # If we can't check for documents, assume it's not a document query
            return False
        except Exception as e:
            logger.warning(f"Error checking for document memories: {e}")
            return False

    return False

def handle_document_query(message):
    """Handle document-specific queries using the new RAG pipeline."""
    try:
        logger.info(f"Document query detected: {message}")

        # Extract document name from the query
        document_name = extract_document_name(message)

        # Get memory store for document search (Phase 3.2: Use ChromaDB)
        try:
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )
        except ImportError as e:
            logger.error(f"Failed to import memory store: {e}")
            return f"I apologize, but I'm having trouble accessing the memory system: {str(e)}"
        all_memories = memory_store.get_all_memories()

        if document_name:
            logger.info(f"Extracted document name: {document_name}")

            # Check for document-specific memories
            document_memories = [m for m in all_memories if document_name in str(getattr(m, 'source', ''))]

            if not document_memories:
                # Document not found, provide helpful response
                available_docs = set()
                for memory in all_memories:
                    source = getattr(memory, 'source', '')
                    if source and 'document:' in source:
                        # Extract filename from various source formats
                        import re

                        # Handle "document:web_ui/uploads/20250606_154557_filename.pdf:block_1"
                        match = re.search(r'uploads/\d{8}_\d{6}_([^:]+)', source)
                        if match:
                            available_docs.add(match.group(1))
                            continue

                        # Handle "document:filename.pdf" or "document:filename.pdf:block_1"
                        if source.startswith('document:'):
                            filename_part = source[9:]  # Remove "document:" prefix
                            filename = filename_part.split(':')[0]  # Remove ":block_X" suffix
                            if '.' in filename and not filename.startswith('web_ui/'):
                                available_docs.add(filename)
                            continue

                        # Handle direct filenames
                        filename = source.split('/')[-1].split(':')[0]
                        if '.' in filename:
                            available_docs.add(filename)

                available_list = sorted(list(available_docs))[:10]  # Show up to 10 documents

                response = f"""I don't have access to "{document_name}" in my memory. This document hasn't been uploaded yet.

**Available documents I can summarize:**
"""
                for doc in available_list:
                    response += f"• {doc}\n"

                if len(available_docs) > 10:
                    response += f"• ... and {len(available_docs) - 10} more documents\n"

                response += f"""
**To get a summary of "{document_name}":**
1. Upload the document using the upload button in the interface
2. Wait for processing to complete
3. Ask me again for a summary

**Or try asking about one of the available documents above!**"""

                return response

            # Use the new RAG pipeline for document summarization
            try:
                from rag_pipeline.new_rag_pipeline import NewRAGPipeline
                pipeline = NewRAGPipeline(memory_store)
                # Generate summary using the enhanced pipeline
                summary_result = pipeline.generate_summary(document_name, message)
            except ImportError as e:
                logger.warning(f"RAG pipeline not available: {e}")
                # Fallback to simple document search
                summary_result = f"Found {len(document_memories)} memories related to {document_name}. Please check the Memory Control Center for detailed content."

            logger.info(f"Generated document summary: {len(summary_result)} chars")
            return summary_result
        else:
            # No specific document name found - this might be a person/entity query
            logger.info("No specific document name found, checking for person/entity query in uploaded documents")

            # Check if we have any document memories
            document_memories = [m for m in all_memories if 'document:' in str(getattr(m, 'source', ''))]

            if document_memories:
                logger.info(f"Found {len(document_memories)} document memories, searching for relevant content")

                # Use enhanced document search for person/entity queries
                return generate_enhanced_document_response(message, document_memories)
            else:
                # No documents available
                return """I don't have any uploaded documents to search through. Please upload a document first, then I can help answer questions about people, entities, or content mentioned in those documents.

**To get started:**
1. Click the upload button to add a document
2. Wait for processing to complete
3. Ask me questions about the content"""

    except Exception as e:
        logger.error(f"Error handling document query: {e}")
        return f"I apologize, but I encountered an error while processing your document query: {str(e)}"

def extract_document_name(message):
    """Extract document name from the query."""
    import re

    # Look for common document patterns - order matters!
    patterns = [
        r'"([^"]+\.(?:pdf|docx|md|txt))"',  # "filename.pdf" - quoted filenames first
        r'\'([^\']+\.(?:pdf|docx|md|txt))\'',  # 'filename.pdf' - single quoted filenames
        r'([A-Za-z0-9._-]+\.(?:pdf|docx|md|txt))',  # filename.pdf - include dots and more chars
        r'"([^"]+)"',  # "any quoted text" - fallback for quoted text
        r'\'([^\']+)\''   # 'any quoted text' - fallback for single quoted text
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            extracted = match.group(1)
            # Additional validation - must contain at least one letter and a dot
            if '.' in extracted and any(c.isalpha() for c in extracted):
                return extracted

    return None

def generate_enhanced_document_response(message, document_memories):
    """Generate enhanced response for person/entity queries using uploaded documents with citations."""
    try:
        logger.info(f"Generating enhanced document response for: {message}")

        # Search for relevant content in document memories
        try:
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
            memory_store = get_memory_store(
                store_type=VectorStoreType.CHROMA,
                storage_directory="memory_store",
                embedding_dimension=384
            )
        except ImportError as e:
            logger.error(f"Failed to import memory store: {e}")
            return f"I apologize, but I'm having trouble accessing the memory system: {str(e)}"

        # Phase 3.2: Use enhanced search with hybrid ranking
        try:
            if hasattr(memory_store, 'enhanced_search_memories'):
                # If we have a last uploaded document in this session, prefer filtering to that doc
                where_filter = None
                try:
                    doc_id_hint = session.get('last_uploaded_doc_id')
                    if doc_id_hint:
                        where_filter = {'document_id': doc_id_hint}
                        logger.info(f"Applying session doc filter for enhanced search: {doc_id_hint}")
                except Exception:
                    where_filter = None

                relevant_memories = memory_store.enhanced_search_memories(
                    query=message,
                    max_results=10,
                    initial_candidates=20,
                    where_filter=where_filter
                )
                logger.info(f"Enhanced document search returned {len(relevant_memories)} ranked results")
            else:
                # Apply doc filter if we have a session-scoped last uploaded doc
                try:
                    doc_id_hint = session.get('last_uploaded_doc_id')
                except Exception:
                    doc_id_hint = None
                if doc_id_hint and hasattr(memory_store, 'enhanced_search_memories'):
                    relevant_memories = memory_store.enhanced_search_memories(
                        query=message,
                        max_results=10,
                        initial_candidates=20,
                        where_filter={'document_id': doc_id_hint}
                    )
                else:
                    relevant_memories = memory_store.search_memories(message, max_results=10, min_similarity=0.2)
                logger.info(f"Fallback document search returned {len(relevant_memories)} results")
        except Exception as e:
            logger.warning(f"Enhanced document search failed, using fallback: {e}")
            relevant_memories = memory_store.search_memories(message, max_results=10, min_similarity=0.2)

        # Filter to only document memories and get the most relevant ones
        # DEBUG: Log source formats to understand filtering issue
        for i, m in enumerate(relevant_memories[:3]):
            # Phase 3.2: Handle both RankedMemoryResult and MemorySearchResult
            if hasattr(m, 'content') and hasattr(m, 'final_score'):
                # RankedMemoryResult
                source = m.metadata.get('source_path', 'NO_SOURCE')
                content_preview = str(m.content)[:100]
                memory_type = m.metadata.get('memory_type', 'NO_TYPE')
            elif hasattr(m, 'chunk'):
                # MemorySearchResult
                chunk = m.chunk
                source = getattr(chunk, 'source', 'NO_SOURCE')
                content_preview = str(getattr(chunk, 'content', 'NO_CONTENT'))[:100]
                memory_type = getattr(chunk, 'memory_type', 'NO_TYPE')
            else:
                # Fallback
                source = getattr(m, 'source', 'NO_SOURCE')
                content_preview = str(getattr(m, 'content', 'NO_CONTENT'))[:100]
                memory_type = getattr(m, 'memory_type', 'NO_TYPE')
            logger.info(f"DEBUG Memory {i}: source='{source}', content='{content_preview}...', type='{memory_type}'")

        # More flexible filtering - check for document indicators with relevance check
        document_relevant = []
        for m in relevant_memories:
            # Phase 3.2: Handle both RankedMemoryResult and MemorySearchResult
            if hasattr(m, 'content') and hasattr(m, 'final_score'):
                # RankedMemoryResult
                source = str(m.metadata.get('source_path', ''))
                memory_type = str(m.metadata.get('memory_type', ''))
                content = str(m.content)
            elif hasattr(m, 'chunk'):
                # MemorySearchResult
                chunk = m.chunk
                source = str(getattr(chunk, 'source', ''))
                memory_type = str(getattr(chunk, 'memory_type', ''))
                content = str(getattr(chunk, 'content', ''))
            else:
                # Fallback
                source = str(getattr(m, 'source', ''))
                memory_type = str(getattr(m, 'memory_type', ''))
                content = str(getattr(m, 'content', ''))

            # Check for document type or document indicators in source
            is_document = (memory_type.lower() == 'document' or
                          any(indicator in source.lower() for indicator in ['document:', 'uploads/', '.pdf', '.txt', '.docx']))

            # Additional relevance check for person queries
            if is_document:
                # For "Who is X" queries, check if the content actually mentions the person
                query_lower = message.lower()
                if any(phrase in query_lower for phrase in ['who is', 'about', 'tell me about']):
                    # Extract the person/entity name from the query
                    import re
                    query_terms = [word.strip() for word in re.findall(r'\b[A-Z][a-z]+\b', message)]
                    if query_terms:
                        content_lower = content.lower()
                        # Check if any query terms appear in the content
                        content_relevant = any(term.lower() in content_lower for term in query_terms)
                        if content_relevant:
                            document_relevant.append(m)
                        else:
                            logger.info(f"Filtering out irrelevant document: {extract_clean_source_name(source)} - no mention of {query_terms}")
                    else:
                        document_relevant.append(m)  # Fallback if no terms extracted
                else:
                    # For other document queries, include all documents
                    document_relevant.append(m)

        logger.info(f"Found {len(document_relevant)} document memories out of {len(relevant_memories)} total memories")

        if not document_relevant:
            # Try a broader search with key terms from the query, but with better relevance filtering
            import re
            # Extract potential names or key terms
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', message)
            if words:
                broader_query = ' '.join(words)
                logger.info(f"Trying broader search with: {broader_query}")
                # Re-run broader search with session doc filter when available
                try:
                    doc_id_hint = session.get('last_uploaded_doc_id')
                except Exception:
                    doc_id_hint = None
                if doc_id_hint and hasattr(memory_store, 'enhanced_search_memories'):
                    relevant_memories = memory_store.enhanced_search_memories(
                        query=broader_query,
                        max_results=10,
                        initial_candidates=20,
                        where_filter={'document_id': doc_id_hint}
                    )
                else:
                    relevant_memories = memory_store.search_memories(broader_query, max_results=10, min_similarity=0.15)  # Increased threshold

                # Apply stricter relevance filtering for broader search
                document_relevant = []
                for m in relevant_memories:
                    # Handle MemorySearchResult objects
                    if hasattr(m, 'chunk'):
                        chunk = m.chunk
                        source = str(getattr(chunk, 'source', ''))
                        memory_type = str(getattr(chunk, 'memory_type', ''))
                        content = str(getattr(chunk, 'content', ''))
                    else:
                        source = str(getattr(m, 'source', ''))
                        memory_type = str(getattr(m, 'memory_type', ''))
                        content = str(getattr(m, 'content', ''))

                    # Enhanced relevance check: document type AND content relevance
                    is_document = (memory_type.lower() == 'document' or
                                 any(indicator in source.lower() for indicator in ['document:', 'uploads/', '.pdf', '.txt', '.docx']))

                    # Check if content is actually relevant to the query
                    content_relevant = False
                    if is_document:
                        # For person queries, check if the content mentions the person
                        query_lower = message.lower()
                        content_lower = content.lower()

                        # Extract names from query
                        if any(word in query_lower for word in ['who is', 'about', 'tell me about']):
                            # This is likely a person/entity query
                            query_terms = [word.strip() for word in re.findall(r'\b[A-Z][a-z]+\b', message)]
                            if query_terms:
                                # Check if any query terms appear in content
                                content_relevant = any(term.lower() in content_lower for term in query_terms)
                        else:
                            # For other queries, use basic keyword matching
                            content_relevant = True  # Allow other document queries through

                    if is_document and content_relevant:
                        document_relevant.append(m)

                logger.info(f"Broader search found {len(document_relevant)} relevant documents after filtering")

        if document_relevant:
            logger.info(f"Found {len(document_relevant)} relevant document memories")

            # Build context with enhanced citations and confidence tracking
            context_parts = []
            sources = set()
            source_details = []  # Track detailed source information
            confidence_scores = []

            for i, memory in enumerate(document_relevant[:3]):  # Use top 3 results to reduce context size
                # Phase 3.2: Handle both RankedMemoryResult and MemorySearchResult
                if hasattr(memory, 'content') and hasattr(memory, 'final_score'):
                    # RankedMemoryResult
                    content = memory.content
                    source = memory.metadata.get('source_path', 'unknown')
                    similarity = memory.final_score  # Use final hybrid score
                elif hasattr(memory, 'chunk'):
                    # MemorySearchResult
                    chunk = memory.chunk
                    content = getattr(chunk, 'content', str(chunk))
                    source = getattr(chunk, 'source', 'unknown')
                    similarity = getattr(memory, 'similarity_score', 0.0)
                else:
                    # Fallback
                    content = getattr(memory, 'content', str(memory))
                    source = getattr(memory, 'source', 'unknown')
                    similarity = getattr(memory, 'similarity_score', 0.0)

                # Extract metadata robustly
                if hasattr(memory, 'chunk'):
                    mem_metadata = getattr(memory.chunk, 'metadata', {}) or {}
                else:
                    mem_metadata = getattr(memory, 'metadata', {}) or {}

                # Extract clean source name
                clean_source = extract_clean_source_name(source)
                sources.add(clean_source)

                # Track confidence scores
                confidence_scores.append(similarity)

                # Store detailed source information for citations
                citation_note = None
                anchor_candidate = None
                try:
                    from sam.utils.citation_utils import try_format_citation_for_chunk
                    citation_note = try_format_citation_for_chunk(content, mem_metadata)
                    if citation_note:
                        # Try to prepare an anchor URL if the UI has a viewer
                        # Example formats: /viewer?doc={document_id}&page={n} or /view_document/{id}?page={n}
                        doc_id = mem_metadata.get('document_id')
                        if doc_id:
                            import re as _re
                            m = _re.search(r"p\.\s*(\d+)", citation_note)
                            if m:
                                page_num = m.group(1)
                                anchor_candidate = f"/view?doc={doc_id}&page={page_num}"
                except Exception:
                    citation_note = None

                sd = {
                    'name': clean_source,
                    'similarity': similarity,
                    'content_preview': content[:100] + "..." if len(content) > 100 else content,
                    'citation': citation_note
                }
                if anchor_candidate:
                    sd['anchor_candidate'] = anchor_candidate
                source_details.append(sd)

                # Add content with citation marker
                if citation_note:
                    context_parts.append(f"[Source: {clean_source} {citation_note}] {content}")
                else:
                    context_parts.append(f"[Source: {clean_source}] {content}")

            context_text = "\n\n".join(context_parts)

            # Create comprehensive prompt for detailed responses
            prompt = f"""You are SAM, an intelligent assistant. Answer the user's question based on the provided document context.

User question: {message}

Context from documents:
{context_text}

Instructions:
- If you need to think through the problem, put your reasoning inside <think></think> tags
- After your thinking, provide a comprehensive, detailed answer based on the provided context
- Include relevant details, background information, and specific facts from the documents
- Be thorough and informative while remaining professional
- Your final answer should not include reasoning steps or thinking process

Example format:
<think>
Let me analyze the context... [your reasoning here]
</think>

[Comprehensive answer with details here]

Answer:"""

            response = sam_model.generate(prompt, temperature=0.3, max_tokens=800, use_learned_knowledge=False)

            # Clean up response - remove think blocks and internal reasoning
            import re
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = response.strip()

            # Clean up response prefixes
            if response.startswith("Response:"):
                response = response[9:].strip()
            if response.startswith("Answer:"):
                response = response[7:].strip()

            # Remove only the most obvious internal reasoning patterns (less aggressive)
            reasoning_patterns = [
                r'Okay, let me think.*?(?=\n\n|\Z)',  # "Okay, let me think..."
                r'📚 Sources:\s*\d+\..*?(?=\n\n|\Z)',          # Duplicate source sections only
                r'I should focus.*?(?=\n\n|\Z)',               # "I should focus..."
            ]

            for pattern in reasoning_patterns:
                response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)

            response = response.strip()

            # If response is empty or too short after cleaning, provide a fallback
            if len(response.strip()) < 20:
                response = "Based on the uploaded documents, I found relevant information but need more context to provide a complete answer."

            # Process thoughts to separate reasoning from answer
            try:
                from utils.thought_processor import get_thought_processor
                thought_processor = get_thought_processor()
                processed = thought_processor.process_response(response)
            except ImportError as e:
                logger.warning(f"Thought processor not available: {e}")
                # Create a simple processed object
                class SimpleProcessed:
                    def __init__(self, content):
                        self.visible_content = content
                        self.has_thoughts = False
                        self.thought_blocks = []
                processed = SimpleProcessed(response)

            # Use the clean visible content as the response
            response = processed.visible_content

            # Store thought blocks for potential "Show Thoughts" feature
            if processed.has_thoughts:
                logger.info(f"Extracted {len(processed.thought_blocks)} thought blocks from response")

            # Phase 3.2: Try to inject enhanced citations with rich metadata
            try:
                from memory.citation_engine import get_citation_engine
                citation_engine = get_citation_engine()

                logger.info(f"Attempting citation generation with {len(document_relevant)} memories")

                # Generate citations from the memories used
                cited_response = citation_engine.inject_citations(response, document_relevant, message)

                logger.info(f"Enhanced citations generated: {len(cited_response.citations)} citations, "
                           f"transparency: {cited_response.transparency_score:.1%}")

                # Use the cited response with enhanced formatting
                response = cited_response.response_text

            except ImportError as e:
                logger.warning(f"Citation engine not available: {e}")
                # Continue with manual citation formatting below
            except Exception as e:
                logger.warning(f"Citation generation failed, using fallback: {e}")
                # Continue with manual citation formatting below

            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            # Add clean, formatted source citations
            if source_details:
                response += f"\n\n**📚 Sources:**"

                # Add detailed source citations with confidence indicators
                for i, source_detail in enumerate(source_details, 1):
                    similarity_pct = source_detail['similarity'] * 100

                    # Create confidence indicator dots
                    confidence_dots = "●" * min(5, int(similarity_pct / 20)) + "○" * (5 - min(5, int(similarity_pct / 20)))

                    response += f"\n{i}. 📚 **{source_detail['name']}** {confidence_dots} ({similarity_pct:.1f}%)"

                # Add overall confidence score with color coding
                confidence_pct = overall_confidence * 100
                if confidence_pct >= 70:
                    confidence_emoji = "🟢"
                elif confidence_pct >= 50:
                    confidence_emoji = "🟡"
                else:
                    confidence_emoji = "🔴"

                response += f"\n\n{confidence_emoji} **Confidence:** {confidence_pct:.1f}%"

            logger.info(f"Clean document response generated: {len(response)} chars")

            # Return response with thought blocks if available
            if processed.has_thoughts:
                return {
                    'response': response,
                    'has_thoughts': True,
                    'thought_blocks': [
                        {
                            'content': block.content,
                            'token_count': block.token_count,
                            'block_id': block.block_id
                        }
                        for block in processed.thought_blocks
                    ]
                }
            else:
                return response
        else:
            # No relevant content found
            available_docs = set()
            for memory in document_memories:
                # Handle MemorySearchResult objects
                if hasattr(memory, 'chunk'):
                    chunk = memory.chunk
                    source = getattr(chunk, 'source', '')
                else:
                    source = getattr(memory, 'source', '')
                clean_source = extract_clean_source_name(source)
                if clean_source:
                    available_docs.add(clean_source)

            response = f"""I couldn't find information about "{message}" in the uploaded documents.

**Available documents I can search:**
"""
            for doc in sorted(available_docs):
                response += f"• {doc}\n"

            response += f"""
**Suggestions:**
1. Try rephrasing your question with different terms
2. Check if the person/entity is mentioned by a different name
3. Upload additional documents that might contain this information"""

            return response

    except Exception as e:
        logger.error(f"Error generating enhanced document response: {e}")
        return f"I apologize, but I encountered an error while searching the uploaded documents: {str(e)}"

def extract_clean_source_name(source):
    """Extract a clean, readable source name from the full source path."""
    if not source:
        return "Unknown Document"

    import re

    # Handle "document:web_ui/uploads/20250606_154557_filename.pdf:block_1"
    match = re.search(r'uploads/\d{8}_\d{6}_([^:]+)', source)
    if match:
        return match.group(1)

    # Handle "document:filename.pdf" or "document:filename.pdf:block_1"
    if source.startswith('document:'):
        filename_part = source[9:]  # Remove "document:" prefix
        filename = filename_part.split(':')[0]  # Remove ":block_X" suffix
        if not filename.startswith('web_ui/'):
            return filename

    # Handle direct filenames
    filename = source.split('/')[-1].split(':')[0]
    if '.' in filename:
        return filename

    return "Document"

def generate_general_document_response(message):
    """Generate response for general document queries."""
    try:
        # Search through all documents
        from memory.memory_vectorstore import get_memory_store, VectorStoreType

        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )

        # Phase 3.2: Use enhanced search with hybrid ranking
        try:
            # Try enhanced search first (Phase 3 upgrade)
            if hasattr(memory_store, 'enhanced_search_memories'):
                memories = memory_store.enhanced_search_memories(
                    query=message,
                    max_results=5,
                    initial_candidates=20
                )
                logger.info(f"Enhanced search returned {len(memories)} ranked results")
            else:
                # Fallback to regular search, but prefer session doc filter if available
                try:
                    doc_id_hint = session.get('last_uploaded_doc_id')
                except Exception:
                    doc_id_hint = None
                if doc_id_hint and hasattr(memory_store, 'enhanced_search_memories'):
                    memories = memory_store.enhanced_search_memories(
                        query=message,
                        max_results=5,
                        initial_candidates=20,
                        where_filter={'document_id': doc_id_hint}
                    )
                else:
                    memories = memory_store.search_memories(message, max_results=5)
                logger.info(f"Fallback search returned {len(memories)} results")
        except Exception as e:
            logger.warning(f"Enhanced search failed, using fallback: {e}")
            memories = memory_store.search_memories(message, max_results=5)

        if memories:
            # Phase 3.2: Handle both RankedMemoryResult and MemorySearchResult
            context_parts = []
            for memory in memories[:3]:
                try:
                    # Check if this is a RankedMemoryResult (Phase 3 enhanced)
                    if hasattr(memory, 'content') and hasattr(memory, 'final_score'):
                        # Enhanced result with ranking information
                        content = memory.content
                        source_name = memory.metadata.get('source_name', 'Unknown')
                        confidence = memory.confidence_score
                        context_parts.append(f"[Source: {source_name}, Confidence: {confidence:.2f}] {content}")
                    elif hasattr(memory, 'chunk'):
                        # Legacy MemorySearchResult
                        content = memory.chunk.content
                        source = getattr(memory.chunk, 'source', 'Unknown')
                        context_parts.append(f"[Source: {source}] {content}")
                    else:
                        # Fallback string representation
                        context_parts.append(str(memory))
                except Exception as e:
                    logger.warning(f"Error processing memory result: {e}")
                    context_parts.append(str(memory))

            context = "\n\n".join(context_parts)

            prompt = f"""You are SAM, an intelligent assistant. The user is asking about documents. Based on the following context from uploaded documents, provide a comprehensive and detailed response.

User question: {message}

Context from documents:
{context}

Please provide a thorough answer with relevant details from the documents.

Response:"""

            response = sam_model.generate(prompt, temperature=0.7, max_tokens=800)

            # Clean up response
            if response.startswith("Response:"):
                response = response[9:].strip()

            return response
        else:
            return "I don't have any documents that match your query. Please upload a document first or check the document name."

    except Exception as e:
        logger.error(f"Error generating general document response: {e}")
        return "I apologize, but I couldn't find relevant document information for your query."

def generate_standard_response(message):
    """Generate standard response using SAM model with enhanced context routing."""
    try:
        # CRITICAL FIX: Check for document-specific queries first
        if is_document_query(message):
            return handle_document_query(message)

        # Use enhanced context router for better context assembly
        try:
            from utils.context_router import get_enhanced_context_router
            context_router = get_enhanced_context_router()
        except ImportError as e:
            logger.error(f"Failed to import context router: {e}")
            # Fallback to simple response without context
            return generate_simple_response(message)

        # Assemble high-quality context with ranking and citations
        context_assembly = context_router.assemble_context(message, max_context_length=2000)

        # Extract context information
        context_text = context_assembly.context_text
        routing_explanation = context_assembly.routing_explanation
        transparency_score = context_assembly.transparency_score

        # Check if we should suggest web retrieval (Phase 7.1)
        # CRITICAL FIX: Only suggest web retrieval for specific queries, not general ones
        if (web_retrieval_suggester and
            web_retrieval_suggester.should_suggest_web_retrieval(message, getattr(context_assembly, 'ranked_memories', []))):
            logger.info(f"Suggesting web retrieval for query: {message[:50]}...")
            return web_retrieval_suggester.format_retrieval_suggestion(message)

        # Prepare context for prompt
        context = f"\n\n{context_text}" if context_text and context_text.strip() else ""

        # Determine query type from context assembly
        query_type = "general"
        if context_text and "document" in context_text.lower():
            query_type = "document_specific"
        elif context_text and ("memory" in context_text.lower() or "conversation" in context_text.lower()):
            query_type = "memory_search"

        # Create enhanced system prompt and user message for chat API
        if query_type == "document_specific":
            system_prompt = f"You are SAM, an intelligent assistant with access to uploaded documents. Answer based on the provided ranked and cited context from their uploaded documents. The context has been intelligently ranked and includes source citations for transparency.\n\nWhen thinking through complex questions, you can use <think>...</think> tags to show your reasoning process. This helps users understand how you arrived at your answer.\n\n{context}"
            user_message = message
        elif query_type == "memory_search":
            system_prompt = f"You are SAM, an intelligent assistant with conversation memory. Use the provided ranked memory context to answer appropriately.\n\nWhen thinking through complex questions, you can use <think>...</think> tags to show your reasoning process. This helps users understand how you arrived at your answer.\n\n{context}"
            user_message = message
        else:
            # CRITICAL FIX: Ensure general queries are handled properly even without context
            base_prompt = "You are SAM, an intelligent multimodal assistant. Answer the user's question helpfully and accurately. You can handle a wide variety of questions including jokes, math problems, explanations, creative writing, and general knowledge questions."
            if context.strip():
                system_prompt = f"{base_prompt}\n\nWhen thinking through complex questions, you can use <think>...</think> tags to show your reasoning process. This helps users understand how you arrived at your answer.\n\nAdditional context from your knowledge base:{context}"
            else:
                system_prompt = f"{base_prompt}\n\nWhen thinking through complex questions, you can use <think>...</think> tags to show your reasoning process. This helps users understand how you arrived at your answer."
            user_message = message

        # Debug: Check which model is being used
        model_type = getattr(sam_model, 'model_type', 'UNKNOWN_MODEL')
        logger.info(f"🔍 Using model type: {model_type} for query: {message[:50]}...")

        # Generate response using chat API for better prompt handling
        if hasattr(sam_model, 'chat'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            response = sam_model.chat(messages, temperature=0.7, max_tokens=800)
        else:
            # Fallback to generate method with proper prompt structure
            prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            response = sam_model.generate(prompt, temperature=0.7, max_tokens=800)

        # Clean up response
        if response.startswith("Response:"):
            response = response[9:].strip()

        # Process thoughts for Sprint 16
        try:
            from utils.thought_processor import get_thought_processor
            thought_processor = get_thought_processor()

            # Debug: Log the raw response to see if it contains <think> tags
            logger.info(f"🧠 Raw response length: {len(response)} chars")
            if '<think>' in response.lower():
                logger.info("🧠 Response contains <think> tags - processing thoughts")
            else:
                logger.info("🧠 Response does not contain <think> tags")

            processed = thought_processor.process_response(response)

            # Debug: Log the processing results
            logger.info(f"🧠 Processed response - has_thoughts: {processed.has_thoughts}, visible_content length: {len(processed.visible_content)}")
            if processed.has_thoughts:
                logger.info(f"🧠 Found {len(processed.thought_blocks)} thought blocks")

        except ImportError as e:
            logger.warning(f"Thought processor not available: {e}")
            # Create a simple processed object
            class SimpleProcessed:
                def __init__(self, content):
                    self.visible_content = content
                    self.has_thoughts = False
                    self.thought_blocks = []
            processed = SimpleProcessed(response)

        # Use the clean visible content as the response
        clean_response = processed.visible_content

        # Add transparency information if high transparency score
        if transparency_score > 0.7:
            clean_response += f"\n\n*✅ High transparency: {transparency_score:.1%} source coverage*"
        elif transparency_score > 0.4:
            clean_response += f"\n\n*🔸 Moderate transparency: {transparency_score:.1%} source coverage*"

        # Add routing information for debugging (only in debug mode)
        if app.debug:
            clean_response += f"\n\n*Debug: {routing_explanation}*"

        # Return response with thought blocks if available
        if processed.has_thoughts:
            return {
                'response': clean_response,
                'has_thoughts': True,
                'thought_blocks': [
                    {
                        'content': block.content,
                        'token_count': block.token_count,
                        'block_id': block.block_id
                    }
                    for block in processed.thought_blocks
                ]
            }
        else:
            return clean_response

    except Exception as e:
        logger.error(f"Error generating standard response: {e}")
        return generate_simple_response(message)

def generate_simple_response(message):
    """Generate a simple response without context routing as a fallback."""
    try:
        logger.info(f"Using simple response fallback for: {message[:50]}...")

        # Create a basic prompt without context
        prompt = f"You are SAM, an intelligent assistant. Answer the user's question helpfully and accurately.\n\nWhen thinking through complex questions, you can use <think>...</think> tags to show your reasoning process. This helps users understand how you arrived at your answer.\n\nUser: {message}\n\nAssistant:"

        # Generate response using the SAM model
        if sam_model:
            response = sam_model.generate(prompt, temperature=0.7, max_tokens=500)

            # Clean up response
            if response.startswith("Response:"):
                response = response[9:].strip()

            return response
        else:
            return "I apologize, but the SAM model is not available right now. Please try again later."

    except Exception as e:
        logger.error(f"Error in simple response generation: {e}")
        return f"I apologize, but I'm having trouble generating a response: {str(e)}"

# Removed old process_response_thoughts function - now using structured format

@app.route('/api/upload', methods=['POST'])
@require_unlock
def upload_file():
    """Handle file uploads for processing with encryption."""
    try:
        logger.info("File upload request received")

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"

        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(exist_ok=True)

        file_path = upload_dir / unique_filename
        file.save(file_path)

        logger.info(f"File saved: {file_path} ({file_path.stat().st_size:,} bytes)")

        # Process the file
        result = process_uploaded_file(file_path)

        # Check if processing was successful
        if 'error' in result:
            logger.error(f"File processing failed: {result['error']}")
            return jsonify({
                'error': f"Failed to process document: {result['error']}",
                'filename': filename
            }), 500

        # ENHANCED: Add knowledge consolidation confirmation
        consolidation_status = confirm_knowledge_consolidation(result, filename)
        result['knowledge_consolidation'] = consolidation_status

        # CRITICAL: Inject learned knowledge into web UI model for true learning
        if consolidation_status['status'] == 'successful':
            try:
                # Extract knowledge from processing result
                if 'summary_length' in result or 'key_concepts' in result:
                    # Get summary from memory storage if available
                    summary = f"Document processed: {filename}"
                    if 'memory_storage' in result:
                        summary = f"Document: {filename} - Successfully processed with {result.get('content_blocks', 0)} content blocks"

                    # Handle key_concepts which might be an integer count
                    key_concepts_data = result.get('key_concepts', [])
                    if isinstance(key_concepts_data, int):
                        # If it's just a count, create placeholder concepts
                        key_concepts = [f"concept_{i+1}" for i in range(min(key_concepts_data, 5))]
                    elif isinstance(key_concepts_data, list):
                        key_concepts = key_concepts_data
                    else:
                        key_concepts = []

                    # Inject into web UI model
                    if hasattr(sam_model, 'inject_learned_knowledge'):
                        sam_model.inject_learned_knowledge(summary, key_concepts)
                        logger.info(f"🎓 WEB UI MODEL LEARNING: Injected knowledge from {filename}")
                    else:
                        logger.warning("Web UI model does not support knowledge injection")

            except Exception as e:
                logger.error(f"Failed to inject knowledge into web UI model: {e}")

        logger.info(f"File processing successful: {filename}")
        logger.info(f"Knowledge consolidation status: {consolidation_status['status']}")

        return jsonify({
            'message': 'File processed successfully',
            'filename': filename,
            'result': result
        })

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def process_uploaded_file(file_path):
    """Process uploaded file through multimodal pipeline."""
    try:
        logger.info(f"Starting file processing: {file_path}")

        if not multimodal_pipeline:
            logger.error("Multimodal pipeline not available")
            return {'error': 'Multimodal pipeline not available'}

        logger.info("Multimodal pipeline available, processing document...")

        # Process the document
        result = multimodal_pipeline.process_document(file_path)

        logger.info(f"Processing result: {result is not None}")

        if result:
            logger.info(f"Document processed successfully: {result.get('document_id', 'unknown')}")

            # Remember the most recent uploaded document in the session for routing/filtering
            try:
                session['last_uploaded_doc_id'] = result.get('document_id')
                session['last_uploaded_filename'] = Path(file_path).name
            except Exception:
                pass

            response_data = {
                'document_id': result['document_id'],
                'content_blocks': result['content_blocks'],
                'enrichment_score': result['enrichment_score'],
                'priority_level': result['priority_level'],
                'content_types': result['content_types'],
                'summary_length': result['summary_length'],
                'key_concepts': result['key_concepts']
            }

            # Add memory storage information if available
            if 'memory_storage' in result:
                memory_info = result['memory_storage']
                logger.info(f"Memory storage successful: {memory_info.get('total_chunks_stored', 0)} chunks")
                response_data['memory_storage'] = {
                    'chunks_stored': memory_info.get('total_chunks_stored', 0),
                    'summary_chunk_id': memory_info.get('summary_chunk_id'),
                    'ready_for_qa': True
                }
                response_data['qa_ready_message'] = f"✅ Document is now ready for Q&A! {memory_info.get('total_chunks_stored', 0)} memory chunks created."
            else:
                logger.warning("No memory storage information in result - document may not be available for Q&A")
                response_data['qa_ready_message'] = "⚠️ Document processed but may not be immediately available for Q&A."

            logger.info("File processing completed successfully")
            return response_data
        else:
            logger.error("Document processing returned None/empty result")
            return {'error': 'Failed to process document - no result returned'}

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {'error': str(e)}

def confirm_knowledge_consolidation(processing_result, filename):
    """
    Confirm that knowledge consolidation was executed and provide detailed status.

    This function verifies that SAM has learned from the uploaded file and provides
    transparency about what knowledge was consolidated.
    """
    try:
        logger.info(f"🧠 KNOWLEDGE CONSOLIDATION CONFIRMATION for {filename}")

        consolidation_status = {
            'status': 'unknown',
            'summary_created': False,
            'key_concepts_extracted': 0,
            'memory_chunks_stored': 0,
            'enrichment_score': 0.0,
            'learning_confirmation': '',
            'consolidation_details': {},
            'timestamp': datetime.now().isoformat()
        }

        # Check if processing result contains consolidation indicators
        if processing_result and isinstance(processing_result, dict):

            # Check for summary creation (indicates knowledge consolidation)
            if 'summary_length' in processing_result and processing_result['summary_length'] > 0:
                consolidation_status['summary_created'] = True
                consolidation_status['consolidation_details']['summary_length'] = processing_result['summary_length']
                logger.info(f"✅ Knowledge summary created: {processing_result['summary_length']} characters")

            # Check for key concepts extraction
            if 'key_concepts' in processing_result:
                consolidation_status['key_concepts_extracted'] = processing_result['key_concepts']
                consolidation_status['consolidation_details']['key_concepts_count'] = processing_result['key_concepts']
                logger.info(f"✅ Key concepts extracted: {processing_result['key_concepts']} concepts")

            # Check for memory storage (indicates learning)
            if 'memory_storage' in processing_result:
                memory_info = processing_result['memory_storage']
                consolidation_status['memory_chunks_stored'] = memory_info.get('chunks_stored', 0)
                consolidation_status['consolidation_details']['memory_chunks'] = memory_info.get('chunks_stored', 0)
                logger.info(f"✅ Memory chunks stored: {memory_info.get('chunks_stored', 0)} chunks")

            # Check enrichment score
            if 'enrichment_score' in processing_result:
                consolidation_status['enrichment_score'] = processing_result['enrichment_score']
                consolidation_status['consolidation_details']['enrichment_score'] = processing_result['enrichment_score']
                logger.info(f"✅ Enrichment score: {processing_result['enrichment_score']:.3f}")

            # Determine overall consolidation status
            if (consolidation_status['summary_created'] and
                consolidation_status['key_concepts_extracted'] > 0 and
                consolidation_status['memory_chunks_stored'] > 0):

                consolidation_status['status'] = 'successful'
                consolidation_status['learning_confirmation'] = (
                    f"🎓 SAM has successfully learned from '{filename}'! "
                    f"Knowledge consolidated: {consolidation_status['key_concepts_extracted']} key concepts, "
                    f"{consolidation_status['memory_chunks_stored']} memory chunks, "
                    f"enrichment score: {consolidation_status['enrichment_score']:.3f}"
                )
                logger.info(f"🎓 KNOWLEDGE CONSOLIDATION SUCCESSFUL for {filename}")

            elif consolidation_status['summary_created'] or consolidation_status['memory_chunks_stored'] > 0:
                consolidation_status['status'] = 'partial'
                consolidation_status['learning_confirmation'] = (
                    f"⚠️ SAM has partially learned from '{filename}'. "
                    f"Some knowledge was consolidated but the process may be incomplete."
                )
                logger.warning(f"⚠️ KNOWLEDGE CONSOLIDATION PARTIAL for {filename}")

            else:
                consolidation_status['status'] = 'failed'
                consolidation_status['learning_confirmation'] = (
                    f"❌ Knowledge consolidation failed for '{filename}'. "
                    f"SAM may not have learned from this document."
                )
                logger.error(f"❌ KNOWLEDGE CONSOLIDATION FAILED for {filename}")

        else:
            consolidation_status['status'] = 'no_data'
            consolidation_status['learning_confirmation'] = (
                f"❓ Unable to confirm knowledge consolidation for '{filename}' - no processing data available."
            )
            logger.warning(f"❓ NO CONSOLIDATION DATA for {filename}")

        # Log consolidation summary
        logger.info(f"📊 CONSOLIDATION SUMMARY: {consolidation_status['status'].upper()} - "
                   f"Summary: {consolidation_status['summary_created']}, "
                   f"Concepts: {consolidation_status['key_concepts_extracted']}, "
                   f"Memory: {consolidation_status['memory_chunks_stored']}, "
                   f"Score: {consolidation_status['enrichment_score']:.3f}")

        return consolidation_status

    except Exception as e:
        logger.error(f"Error confirming knowledge consolidation for {filename}: {e}")
        return {
            'status': 'error',
            'summary_created': False,
            'key_concepts_extracted': 0,
            'memory_chunks_stored': 0,
            'enrichment_score': 0.0,
            'learning_confirmation': f"❌ Error confirming knowledge consolidation: {str(e)}",
            'consolidation_details': {},
            'timestamp': datetime.now().isoformat()
        }

def search_multimodal_content(query):
    """Search multimodal content."""
    try:
        if not multimodal_pipeline:
            return []

        results = multimodal_pipeline.search_multimodal_content(query, top_k=5)
        return results

    except Exception as e:
        logger.error(f"Error searching content: {e}")
        return []

def get_system_status():
    """Get system status information."""
    try:
        status = {
            'model_status': 'Connected' if sam_model else 'Not available',
            'vector_store_status': 'Connected' if vector_manager else 'Not available',
            'multimodal_pipeline_status': 'Available' if multimodal_pipeline else 'Not available'
        }

        if vector_manager:
            vector_stats = vector_manager.get_stats()
            status['total_chunks'] = vector_stats.get('total_chunks', 0)

        if multimodal_pipeline:
            processing_stats = multimodal_pipeline.get_processing_stats()
            status.update(processing_stats)

        return status

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {'error': str(e)}

def format_status(stats):
    """Format status information for display."""
    lines = []
    lines.append(f"🤖 **Model**: {stats.get('model_status', 'Unknown')}")
    lines.append(f"🗄️ **Vector Store**: {stats.get('vector_store_status', 'Unknown')}")
    lines.append(f"📄 **Multimodal Pipeline**: {stats.get('multimodal_pipeline_status', 'Unknown')}")
    lines.append(f"📚 **Total Chunks**: {stats.get('total_chunks', 0)}")
    lines.append(f"📄 **Documents Processed**: {stats.get('documents_processed', 0)}")
    lines.append(f"🧩 **Content Blocks**: {stats.get('total_content_blocks', 0)}")
    return "\n".join(lines)

def format_search_results(results):
    """Format search results for display."""
    if not results:
        return "No results found."

    lines = [f"🔍 **Found {len(results)} results:**\n"]

    for i, result in enumerate(results, 1):
        similarity = result.get('similarity_score', 0)
        content_type = result.get('content_type', 'unknown')
        is_multimodal = result.get('is_multimodal', False)

        lines.append(f"**{i}.** {content_type.title()} Content (Score: {similarity:.3f})")
        if is_multimodal:
            lines.append("   🎨 Multimodal content")

        preview = result.get('text', '')[:150]
        lines.append(f"   📝 {preview}...")
        lines.append("")

    return "\n".join(lines)

def get_help_text():
    """Get help text."""
    return """🤖 **SAM Web UI Help**

**Chat Commands:**
- Just type your questions normally
- `/status` - Show system status
- `/search <query>` - Search multimodal content
- `/summarize <topic>` - Generate smart summary about a topic
- `/thoughts [on|off|status]` - Control thought visibility (Sprint 16)
- `/help` - Show this help

**File Upload:**
- Click the upload button to process documents
- Supported formats: PDF, DOCX, Markdown, HTML, code files
- Files are automatically processed and added to knowledge base

**New in Sprint 16: Thought Transparency**
- 🧠 **Hidden by Default**: SAM's thinking process is now hidden for cleaner responses
- 🔘 **Toggle Buttons**: Click "SAM's Thoughts" to reveal reasoning process
- ⚙️ **Configurable**: Use `/thoughts on/off` to control visibility
- ⌨️ **Keyboard Shortcut**: `Alt + T` to toggle most recent thought

**Enhanced Features (Sprint 15):**
- 🏆 **Memory Ranking**: Intelligent prioritization of relevant content
- 📝 **Citation Engine**: Transparent source attribution and quotes
- 📊 **Smart Summaries**: AI-generated topic summaries with source tracking
- 📈 **Transparency Scores**: Quality indicators for response reliability

**Core Features:**
- 🧠 Intelligent responses with ranked context
- 📄 Multimodal document processing
- 🔍 Semantic search across content
- 📊 Content enrichment scoring
- 🎨 Support for text, code, tables, images"""

@app.route('/api/model-status')
def model_status():
    """Get detailed model status information for the status bar."""
    try:
        import requests

        # Check if sam_model is available and get its details
        if not sam_model:
            return jsonify({
                'status': 'offline',
                'error': 'SAM model not initialized',
                'model_name': 'Unknown',
                'model_type': 'Unknown',
                'parameter_size': 'Unknown',
                'quantization': 'Unknown',
                'learned_knowledge_count': 0
            })

        # Get model type and learned knowledge count
        model_type = getattr(sam_model, 'model_type', 'Unknown')
        learned_count = len(getattr(sam_model, 'learned_knowledge', []))

        # Test Ollama connection and get model details
        try:
            # Check if Ollama is responding
            ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)

            if ollama_response.status_code == 200:
                models_data = ollama_response.json()

                # Find our specific model
                target_model = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
                model_info = None

                for model in models_data.get('models', []):
                    if model.get('name') == target_model:
                        model_info = model
                        break

                if model_info:
                    # Extract model details
                    details = model_info.get('details', {})
                    parameter_size = details.get('parameter_size', '8.19B')
                    quantization = details.get('quantization_level', 'Q4_K_M')

                    return jsonify({
                        'status': 'online',
                        'model_name': 'DeepSeek-R1-Qwen3-8B',
                        'model_type': model_type,
                        'parameter_size': parameter_size,
                        'quantization': quantization,
                        'learned_knowledge_count': learned_count,
                        'full_model_name': target_model,
                        'ollama_status': 'connected'
                    })
                else:
                    return jsonify({
                        'status': 'offline',
                        'error': 'Target model not found in Ollama',
                        'model_name': 'DeepSeek-R1-Qwen3-8B',
                        'model_type': model_type,
                        'parameter_size': 'Unknown',
                        'quantization': 'Unknown',
                        'learned_knowledge_count': learned_count
                    })
            else:
                return jsonify({
                    'status': 'offline',
                    'error': f'Ollama API error: {ollama_response.status_code}',
                    'model_name': 'DeepSeek-R1-Qwen3-8B',
                    'model_type': model_type,
                    'parameter_size': 'Unknown',
                    'quantization': 'Unknown',
                    'learned_knowledge_count': learned_count
                })

        except requests.exceptions.RequestException as e:
            return jsonify({
                'status': 'offline',
                'error': f'Ollama connection failed: {str(e)}',
                'model_name': 'DeepSeek-R1-Qwen3-8B',
                'model_type': model_type,
                'parameter_size': 'Unknown',
                'quantization': 'Unknown',
                'learned_knowledge_count': learned_count
            })

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({
            'status': 'offline',
            'error': str(e),
            'model_name': 'Unknown',
            'model_type': 'Unknown',
            'parameter_size': 'Unknown',
            'quantization': 'Unknown',
            'learned_knowledge_count': 0
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    try:
        from utils.health_monitor import get_health_monitor

        health_monitor = get_health_monitor()
        health_report = health_monitor.get_health_report()

        # Determine HTTP status code based on health
        status_code = 200
        if health_report['overall_status'] == 'critical':
            status_code = 503
        elif health_report['overall_status'] == 'warning':
            status_code = 200  # Still operational

        return jsonify(health_report), status_code

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'overall_status': 'critical',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/status')
def status_endpoint():
    """Detailed status endpoint with system information."""
    try:
        from config.config_manager import get_config_manager
        from config.agent_mode import get_mode_controller
        from memory.memory_vectorstore import get_memory_store, VectorStoreType
        from utils.health_monitor import get_health_monitor

        config_manager = get_config_manager()
        mode_controller = get_mode_controller()
        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )
        health_monitor = get_health_monitor()

        # Get configuration summary
        config_summary = config_manager.get_config_summary()

        # Get agent mode status
        mode_status = mode_controller.get_mode_status()

        # Get memory statistics
        memory_stats = memory_store.get_memory_stats()

        # Get health report
        health_report = health_monitor.get_health_report()

        status_info = {
            'timestamp': datetime.now().isoformat(),
            'version': config_summary.get('version', '1.0.0'),
            'agent_mode': mode_status.current_mode.value,
            'memory_backend': config_summary.get('memory_backend', 'simple'),
            'memory_stats': memory_stats,
            'health': health_report,
            'configuration': config_summary,
            'uptime_seconds': health_report.get('uptime_seconds', 0),
            'ports': {
                'chat': config_summary.get('ports', {}).get('chat', 5001),
                'memory_ui': config_summary.get('ports', {}).get('memory_ui', 8501)
            }
        }

        return jsonify(status_info)

    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Handle user feedback for Phase 6 personalization learning."""
    try:
        data = request.get_json()
        message_id = data.get('message_id')
        rating = data.get('rating', 0.5)
        feedback_type = data.get('feedback_type', 'rating')
        correction_text = data.get('correction_text', '')

        logger.info(f"Feedback received: {feedback_type} for message {message_id}, rating: {rating}")

        # Try to integrate with Phase 6 feedback system
        try:
            from personalization.phase6_integration import get_phase6_engine

            engine = get_phase6_engine()
            user_id = "default_user"  # In production, get from session/auth

            # Submit feedback to Phase 6 system
            success = engine.submit_feedback(
                memory_id=message_id,
                user_id=user_id,
                rating=rating,
                correction=correction_text if correction_text else None
            )

            # Generate learning insights
            learning_applied = success
            learning_insight = ""

            if success and correction_text:
                learning_insight = "I've noted your specific feedback and will adjust my communication style accordingly."
            elif success and rating > 0.7:
                learning_insight = "Thank you for the positive feedback! I'll remember what worked well."
            elif success and rating < 0.4:
                learning_insight = "I'll work on improving responses like this one."

            return jsonify({
                'success': True,
                'learning_applied': learning_applied,
                'learning_insight': learning_insight,
                'message': 'Feedback processed successfully'
            })

        except ImportError:
            logger.warning("Phase 6 system not available, storing feedback locally")
            # Fallback: store feedback locally
            feedback_data = {
                'message_id': message_id,
                'rating': rating,
                'feedback_type': feedback_type,
                'correction_text': correction_text,
                'timestamp': datetime.now().isoformat(),
                'user_id': 'default_user'
            }

            # Store in simple JSON file for now
            feedback_file = Path('web_ui/feedback_log.json')
            feedback_log = []

            if feedback_file.exists():
                try:
                    with open(feedback_file, 'r') as f:
                        feedback_log = json.load(f)
                except:
                    feedback_log = []

            feedback_log.append(feedback_data)

            # Keep only last 1000 feedback entries
            if len(feedback_log) > 1000:
                feedback_log = feedback_log[-1000:]

            with open(feedback_file, 'w') as f:
                json.dump(feedback_log, f, indent=2)

            return jsonify({
                'success': True,
                'learning_applied': False,
                'learning_insight': 'Feedback stored for future learning improvements.',
                'message': 'Feedback recorded successfully'
            })

    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process feedback'
        }), 500

@app.route('/api/learning-history')
def learning_history():
    """Get SAM's learning history from processed documents."""
    try:
        from memory.memory_vectorstore import get_memory_store, VectorStoreType

        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )
        all_memories = memory_store.get_all_memories()

        # Filter for document summaries (learning events)
        learning_events = []
        for memory in all_memories:
            metadata = getattr(memory, 'metadata', {})
            if metadata.get('document_type') == 'summary':
                learning_event = {
                    'timestamp': metadata.get('upload_timestamp', metadata.get('processing_timestamp', 'unknown')),
                    'filename': metadata.get('file_name', 'unknown'),
                    'source_file': metadata.get('source_file', 'unknown'),
                    'enrichment_score': metadata.get('enrichment_score', 0.0),
                    'priority_level': metadata.get('priority_level', 'unknown'),
                    'key_concepts': metadata.get('key_concepts', []),
                    'content_types': metadata.get('content_types', []),
                    'content_blocks_count': metadata.get('content_blocks_count', 0),
                    'file_size': metadata.get('file_size', 0),
                    'memory_id': getattr(memory, 'memory_id', 'unknown')
                }
                learning_events.append(learning_event)

        # Sort by timestamp (most recent first)
        learning_events.sort(key=lambda x: x['timestamp'], reverse=True)

        # Calculate learning statistics
        total_documents = len(learning_events)
        total_concepts = sum(len(event.get('key_concepts', [])) for event in learning_events)
        avg_enrichment = sum(event.get('enrichment_score', 0) for event in learning_events) / max(total_documents, 1)
        total_content_blocks = sum(event.get('content_blocks_count', 0) for event in learning_events)

        learning_summary = {
            'total_documents_learned': total_documents,
            'total_key_concepts': total_concepts,
            'average_enrichment_score': round(avg_enrichment, 3),
            'total_content_blocks_processed': total_content_blocks,
            'learning_events': learning_events[:20],  # Return last 20 events
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(learning_summary)

    except Exception as e:
        logger.error(f"Error getting learning history: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Phase 8C: Dream Canvas API Endpoints
@app.route('/api/synthesis/trigger', methods=['POST'])
@optional_security
def trigger_synthesis():
    """Trigger cognitive synthesis with optional visualization."""
    try:
        data = request.get_json() or {}
        visualize = data.get('visualize', False)

        logger.info(f"🧠 Triggering cognitive synthesis (visualize: {visualize})")

        # Import synthesis components
        try:
            from memory.synthesis import SynthesisEngine, SynthesisConfig
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
        except ImportError as e:
            return jsonify({
                'status': 'error',
                'error': f'Synthesis components not available: {e}'
            }), 500

        # Get memory store
        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )

        # Configure synthesis engine
        config = SynthesisConfig(
            enable_reingestion=True,
            enable_deduplication=True
        )

        synthesis_engine = SynthesisEngine(config=config)

        # Run synthesis
        result = synthesis_engine.run_synthesis(memory_store, visualize=visualize)

        # Prepare response
        response_data = {
            'status': 'success',
            'run_id': result.run_id,
            'timestamp': result.timestamp,
            'clusters_found': result.clusters_found,
            'insights_generated': result.insights_generated,
            'output_file': result.output_file,
            'memory_count': len(memory_store.get_all_memories())
        }

        # Add cluster summary for Dream Canvas
        if hasattr(result, 'concept_clusters') and result.concept_clusters:
            cluster_summary = []
            for i, cluster in enumerate(result.concept_clusters[:6]):  # Top 6 clusters
                cluster_info = {
                    'id': cluster.cluster_id,
                    'name': f"Cluster {i+1}",  # Simple naming for now
                    'memory_count': cluster.size,
                    'coherence': cluster.coherence_score,
                    'themes': cluster.dominant_themes[:3] if cluster.dominant_themes else [],
                    'connections': len(cluster.chunks),
                    'memories': [
                        {
                            'title': chunk.content[:50] + "..." if len(chunk.content) > 50 else chunk.content,
                            'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                            'type': chunk.memory_type.value,
                            'score': chunk.importance_score
                        }
                        for chunk in cluster.chunks[:5]  # Top 5 memories per cluster
                    ]
                }
                cluster_summary.append(cluster_info)

            response_data['cluster_summary'] = cluster_summary

        # Add visualization data if requested
        if visualize and result.visualization_data:
            response_data['visualization_data'] = result.visualization_data
            response_data['visualization_enabled'] = True
        else:
            response_data['visualization_enabled'] = False

        logger.info(f"✅ Synthesis completed: {result.insights_generated} insights generated")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error triggering synthesis: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/synthesis/history', methods=['GET'])
@optional_security
def get_synthesis_history():
    """Get synthesis run history."""
    try:
        from memory.synthesis import SynthesisEngine

        synthesis_engine = SynthesisEngine()
        history = synthesis_engine.get_synthesis_history()

        return jsonify({
            'status': 'success',
            'history': history
        })

    except Exception as e:
        logger.error(f"Error getting synthesis history: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/synthesis/visualization/<run_id>', methods=['GET'])
@optional_security
def get_visualization_data(run_id):
    """Get visualization data for a specific synthesis run."""
    try:
        # Load synthesis output file
        from pathlib import Path
        import json

        synthesis_dir = Path("synthesis_output")
        output_file = synthesis_dir / f"synthesis_run_log_{run_id}.json"

        if not output_file.exists():
            return jsonify({
                'status': 'error',
                'error': 'Synthesis run not found'
            }), 404

        with open(output_file, 'r') as f:
            synthesis_data = json.load(f)

        # Check if visualization data exists
        if 'visualization_data' in synthesis_data:
            return jsonify({
                'status': 'success',
                'visualization_data': synthesis_data['visualization_data']
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'No visualization data available for this run'
            }), 404

    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/sof/execute', methods=['POST'])
@optional_security
def execute_sof_plan():
    """Execute a SOF (SAM Orchestration Framework) plan."""
    try:
        data = request.get_json() or {}

        # Extract plan and input data
        plan = data.get('plan', [])
        input_query = data.get('input_query', '')
        explicit_knowledge_chunks = data.get('explicit_knowledge_chunks', [])

        if not plan:
            return jsonify({
                'status': 'error',
                'error': 'No execution plan provided'
            }), 400

        if not input_query:
            return jsonify({
                'status': 'error',
                'error': 'No input query provided'
            }), 400

        # Import SOF components
        from sam.orchestration import get_sof_integration

        # Get SOF integration
        sof = get_sof_integration()
        if not sof.initialize():
            return jsonify({
                'status': 'error',
                'error': 'Failed to initialize SOF'
            }), 500

        # Create UIF with input data
        from sam.orchestration import SAM_UIF
        uif = SAM_UIF(input_query=input_query)

        # Add explicit knowledge chunks if provided
        if explicit_knowledge_chunks:
            uif.intermediate_data['explicit_knowledge_chunks'] = explicit_knowledge_chunks

        # Execute the plan
        execution_report = sof._coordinator.execute_plan(plan, uif)

        # Return execution results
        return jsonify({
            'status': 'success',
            'execution_report': {
                'result': execution_report.result.value,
                'execution_time': execution_report.execution_time,
                'skills_executed': execution_report.skills_executed,
                'intermediate_data': dict(uif.intermediate_data),
                'final_response': getattr(uif, 'final_response', None),
                'confidence_score': getattr(uif, 'confidence_score', None)
            }
        })

    except Exception as e:
        logger.error(f"Error executing SOF plan: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/synthesis/suggest-eps', methods=['POST'])
@optional_security
def suggest_optimal_eps():
    """Suggest optimal DBSCAN eps parameter using k-distance graph analysis."""
    try:
        data = request.get_json() or {}
        min_samples = data.get('min_samples', 5)
        target_clusters = data.get('target_clusters', 10)

        logger.info(f"🤔 Analyzing optimal eps with min_samples={min_samples}, target_clusters={target_clusters}")

        # Import required components
        try:
            from memory.synthesis.eps_optimizer import EpsOptimizer
            from memory.memory_vectorstore import get_memory_store, VectorStoreType
        except ImportError as e:
            return jsonify({
                'status': 'error',
                'error': f'Eps optimizer not available: {e}'
            }), 500

        # Get memory store and embeddings
        memory_store = get_memory_store(
            store_type=VectorStoreType.CHROMA,
            storage_directory="memory_store",
            embedding_dimension=384
        )

        all_memories = memory_store.get_all_memories()
        if len(all_memories) < 10:
            return jsonify({
                'status': 'error',
                'error': 'Insufficient memories for eps analysis (minimum 10 required)'
            }), 400

        # Extract embeddings
        embeddings = []
        for memory in all_memories:
            if memory.embedding and len(memory.embedding) > 0:
                embeddings.append(memory.embedding)

        if len(embeddings) < 10:
            return jsonify({
                'status': 'error',
                'error': 'Insufficient valid embeddings for analysis'
            }), 400

        # Run eps optimization
        optimizer = EpsOptimizer()
        suggestions = optimizer.suggest_clustering_params(
            np.array(embeddings),
            target_clusters=target_clusters
        )

        logger.info(f"✅ Optimal eps analysis complete: eps={suggestions['eps']:.4f}")

        return jsonify({
            'status': 'success',
            'suggestions': suggestions,
            'analysis_summary': f"Analyzed {len(embeddings)} embeddings to find optimal clustering parameters"
        })

    except Exception as e:
        logger.error(f"Error suggesting optimal eps: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Initialize security routes and context
create_security_routes(app)
app.context_processor(inject_security_context)

# Register vetting interface blueprint
if vetting_bp:
    app.register_blueprint(vetting_bp)
    logger.info("✅ Vetting interface registered")
else:
    logger.warning("⚠️ Vetting interface not available")

if __name__ == '__main__':
    # Create upload directory
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

    # Initialize SAM components
    if initialize_sam():
        print("🚀 SAM Web UI starting with security integration...")
        print("🌐 Access the interface at: http://localhost:5001")
        print("🔒 Security features: Setup/unlock, encrypted storage, session management")
        print("💡 For full SAM suite with Memory Control Center, use: python start_sam.py")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to initialize SAM components")
