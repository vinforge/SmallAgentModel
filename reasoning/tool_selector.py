"""
Tool Selector Module for SAM
Intelligent tool selection using rule-based and similarity heuristics.

Sprint 5 Task 2: Tool Selector Module
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Available tool types."""
    PYTHON_INTERPRETER = "python_interpreter"
    TABLE_GENERATOR = "table_generator"
    MULTIMODAL_QUERY = "multimodal_query"
    WEB_SEARCH = "web_search"

@dataclass
class ToolCapability:
    """Describes what a tool can do."""
    tool_name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    use_cases: List[str]
    keywords: List[str]
    complexity_level: str  # 'simple', 'moderate', 'complex'
    execution_time: str    # 'fast', 'medium', 'slow'

@dataclass
class ToolSelection:
    """Result of tool selection process."""
    tool_name: str
    confidence: float
    rationale: str
    input_params: Dict[str, Any]
    metadata: Dict[str, Any]

class ToolSelector:
    """
    Intelligent tool selector using rule-based and similarity heuristics.
    """
    
    def __init__(self, enable_web_search: bool = False):
        """
        Initialize the tool selector.
        
        Args:
            enable_web_search: Whether to enable web search tool
        """
        self.enable_web_search = enable_web_search
        
        # Define tool capabilities
        self.tool_capabilities = {
            ToolType.PYTHON_INTERPRETER.value: ToolCapability(
                tool_name="python_interpreter",
                description="Execute Python code for calculations, data analysis, and computations",
                input_types=["code", "mathematical_expression", "data"],
                output_types=["numerical_result", "plot", "analysis", "computation"],
                use_cases=[
                    "mathematical calculations",
                    "data analysis",
                    "statistical computations",
                    "plotting and visualization",
                    "algorithm implementation",
                    "numerical simulations"
                ],
                keywords=[
                    "calculate", "compute", "math", "plot", "graph", "analyze",
                    "statistics", "data", "algorithm", "code", "python",
                    "numerical", "simulation", "equation", "formula"
                ],
                complexity_level="moderate",
                execution_time="medium"
            ),
            
            ToolType.TABLE_GENERATOR.value: ToolCapability(
                tool_name="table_generator",
                description="Generate structured tables and data presentations",
                input_types=["data", "list", "comparison_request"],
                output_types=["table", "structured_data", "comparison_chart"],
                use_cases=[
                    "data organization",
                    "comparison tables",
                    "structured presentations",
                    "data formatting",
                    "summary tables"
                ],
                keywords=[
                    "table", "compare", "list", "organize", "structure",
                    "format", "summary", "data", "rows", "columns",
                    "comparison", "chart", "organize"
                ],
                complexity_level="simple",
                execution_time="fast"
            ),
            
            ToolType.MULTIMODAL_QUERY.value: ToolCapability(
                tool_name="multimodal_query",
                description="Search and analyze multimodal content across different formats",
                input_types=["query", "search_terms", "content_type"],
                output_types=["search_results", "content_analysis", "multimodal_insights"],
                use_cases=[
                    "content search",
                    "document analysis",
                    "multimodal information retrieval",
                    "cross-format search",
                    "content discovery"
                ],
                keywords=[
                    "search", "find", "look", "document", "content",
                    "multimodal", "image", "text", "code", "table",
                    "analyze", "discover", "retrieve"
                ],
                complexity_level="moderate",
                execution_time="medium"
            )
        }
        
        # Add web search if enabled
        if self.enable_web_search:
            self.tool_capabilities[ToolType.WEB_SEARCH.value] = ToolCapability(
                tool_name="web_search",
                description="Search the web for current information and external knowledge",
                input_types=["query", "search_terms"],
                output_types=["web_results", "external_information", "current_data"],
                use_cases=[
                    "current events",
                    "external information",
                    "fact checking",
                    "recent developments",
                    "web-based research"
                ],
                keywords=[
                    "current", "recent", "latest", "news", "web", "online",
                    "external", "internet", "search", "lookup", "find"
                ],
                complexity_level="simple",
                execution_time="medium"
            )
        
        logger.info(f"Tool selector initialized with {len(self.tool_capabilities)} tools")
        logger.info(f"Web search enabled: {self.enable_web_search}")
    
    def select_tool(self, query: str, context: Optional[Dict[str, Any]] = None, 
                   preferred_tool: Optional[str] = None) -> Optional[ToolSelection]:
        """
        Select the most appropriate tool for a given query.
        
        Args:
            query: User query or task description
            context: Additional context information
            preferred_tool: Preferred tool name (if any)
            
        Returns:
            ToolSelection object or None if no suitable tool found
        """
        try:
            logger.debug(f"Selecting tool for query: {query[:100]}...")
            
            # If preferred tool is specified and available, use it
            if preferred_tool and preferred_tool in self.tool_capabilities:
                return self._create_tool_selection(
                    preferred_tool, 
                    query, 
                    context, 
                    confidence=0.9,
                    rationale=f"Using preferred tool: {preferred_tool}"
                )
            
            # Analyze query to determine best tool
            tool_scores = self._score_tools_for_query(query, context)
            
            if not tool_scores:
                logger.warning("No suitable tools found for query")
                return None
            
            # Select tool with highest score
            best_tool, best_score = max(tool_scores.items(), key=lambda x: x[1])
            
            if best_score < 0.3:  # Minimum confidence threshold
                logger.warning(f"Best tool score ({best_score:.2f}) below threshold")
                return None
            
            # Create tool selection
            rationale = self._generate_selection_rationale(best_tool, query, best_score)
            
            return self._create_tool_selection(
                best_tool,
                query,
                context,
                confidence=best_score,
                rationale=rationale
            )
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            return None
    
    def select_multiple_tools(self, query: str, context: Optional[Dict[str, Any]] = None,
                            max_tools: int = 3) -> List[ToolSelection]:
        """
        Select multiple tools for complex queries.
        
        Args:
            query: User query or task description
            context: Additional context information
            max_tools: Maximum number of tools to select
            
        Returns:
            List of ToolSelection objects
        """
        try:
            tool_scores = self._score_tools_for_query(query, context)
            
            # Sort tools by score and select top ones
            sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tools = []
            
            for tool_name, score in sorted_tools[:max_tools]:
                if score >= 0.3:  # Minimum threshold
                    rationale = self._generate_selection_rationale(tool_name, query, score)
                    
                    tool_selection = self._create_tool_selection(
                        tool_name,
                        query,
                        context,
                        confidence=score,
                        rationale=rationale
                    )
                    
                    selected_tools.append(tool_selection)
            
            logger.debug(f"Selected {len(selected_tools)} tools for complex query")
            return selected_tools
            
        except Exception as e:
            logger.error(f"Error in multiple tool selection: {e}")
            return []
    
    def _score_tools_for_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Score all available tools for a given query."""
        query_lower = query.lower()
        tool_scores = {}
        
        for tool_name, capability in self.tool_capabilities.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in capability.keywords if keyword in query_lower)
            if keyword_matches > 0:
                score += min(keyword_matches * 0.2, 0.8)  # Max 0.8 from keywords
            
            # Use case matching
            use_case_matches = sum(1 for use_case in capability.use_cases 
                                 if any(word in query_lower for word in use_case.split()))
            if use_case_matches > 0:
                score += min(use_case_matches * 0.15, 0.6)  # Max 0.6 from use cases
            
            # Context-based scoring
            if context:
                score += self._score_tool_for_context(tool_name, capability, context)
            
            # Query pattern matching
            score += self._score_tool_for_patterns(tool_name, capability, query)
            
            # Normalize score
            tool_scores[tool_name] = min(score, 1.0)
        
        return tool_scores
    
    def _score_tool_for_context(self, tool_name: str, capability: ToolCapability, 
                               context: Dict[str, Any]) -> float:
        """Score a tool based on context information."""
        score = 0.0
        
        # Check knowledge gaps
        knowledge_gaps = context.get('knowledge_gaps', [])
        for gap in knowledge_gaps:
            if isinstance(gap, dict):
                gap_type = gap.get('gap_type', '')
                suggested_tools = gap.get('suggested_tools', [])
                
                if tool_name in suggested_tools:
                    score += 0.3
                
                # Match gap type to tool capability
                if gap_type == 'computational' and tool_name == 'python_interpreter':
                    score += 0.2
                elif gap_type == 'factual' and tool_name in ['multimodal_query', 'web_search']:
                    score += 0.2
        
        # Check reasoning plan
        plan = context.get('plan')
        if plan and isinstance(plan, dict):
            required_tools = plan.get('required_tools', [])
            if tool_name in required_tools:
                score += 0.2
        
        return score
    
    def _score_tool_for_patterns(self, tool_name: str, capability: ToolCapability, query: str) -> float:
        """Score a tool based on query patterns."""
        score = 0.0
        query_lower = query.lower()
        
        # Mathematical/computational patterns
        math_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Basic math
            r'calculate|compute|solve',   # Calculation keywords
            r'equation|formula|function', # Mathematical terms
            r'plot|graph|chart|visualize' # Visualization
        ]
        
        if tool_name == 'python_interpreter':
            for pattern in math_patterns:
                if re.search(pattern, query_lower):
                    score += 0.2
        
        # Table/organization patterns
        table_patterns = [
            r'table|list|organize',
            r'compare|comparison|versus|vs',
            r'summary|overview|breakdown'
        ]
        
        if tool_name == 'table_generator':
            for pattern in table_patterns:
                if re.search(pattern, query_lower):
                    score += 0.2
        
        # Search patterns
        search_patterns = [
            r'find|search|look\s+for',
            r'what\s+is|who\s+is|where\s+is',
            r'information\s+about|details\s+about'
        ]
        
        if tool_name in ['multimodal_query', 'web_search']:
            for pattern in search_patterns:
                if re.search(pattern, query_lower):
                    score += 0.15
        
        # Current/recent information patterns (web search)
        if tool_name == 'web_search':
            current_patterns = [
                r'current|recent|latest|now',
                r'today|this\s+year|2024|2025',
                r'news|updates|developments'
            ]
            
            for pattern in current_patterns:
                if re.search(pattern, query_lower):
                    score += 0.3
        
        return score
    
    def _create_tool_selection(self, tool_name: str, query: str, context: Optional[Dict[str, Any]],
                              confidence: float, rationale: str) -> ToolSelection:
        """Create a ToolSelection object."""
        capability = self.tool_capabilities[tool_name]
        
        # Prepare input parameters
        input_params = {
            'query': query,
            'tool_type': tool_name
        }
        
        # Add context-specific parameters
        if context:
            input_params['context'] = context
        
        # Tool-specific parameter preparation
        if tool_name == 'python_interpreter':
            input_params['execution_mode'] = 'safe'
            input_params['timeout'] = 30
        elif tool_name == 'table_generator':
            input_params['format'] = 'markdown'
            input_params['max_rows'] = 20
        elif tool_name == 'multimodal_query':
            input_params['search_types'] = ['text', 'code', 'table', 'image']
            input_params['max_results'] = 5
        elif tool_name == 'web_search':
            input_params['max_results'] = 5
            input_params['safe_search'] = True
        
        # Metadata
        metadata = {
            'tool_capability': capability.description,
            'complexity_level': capability.complexity_level,
            'execution_time': capability.execution_time,
            'selection_timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)) if logger.handlers else 'unknown'
        }
        
        return ToolSelection(
            tool_name=tool_name,
            confidence=confidence,
            rationale=rationale,
            input_params=input_params,
            metadata=metadata
        )
    
    def _generate_selection_rationale(self, tool_name: str, query: str, score: float) -> str:
        """Generate human-readable rationale for tool selection."""
        capability = self.tool_capabilities[tool_name]
        
        rationale_parts = [
            f"Selected {tool_name} (confidence: {score:.2f})"
        ]
        
        # Add specific reasons based on tool type
        if tool_name == 'python_interpreter':
            if any(word in query.lower() for word in ['calculate', 'compute', 'math']):
                rationale_parts.append("Query requires mathematical computation")
            if any(word in query.lower() for word in ['plot', 'graph', 'visualize']):
                rationale_parts.append("Query involves data visualization")
        
        elif tool_name == 'table_generator':
            if any(word in query.lower() for word in ['table', 'list', 'organize']):
                rationale_parts.append("Query requires structured data presentation")
            if any(word in query.lower() for word in ['compare', 'comparison']):
                rationale_parts.append("Query involves comparison or analysis")
        
        elif tool_name == 'multimodal_query':
            if any(word in query.lower() for word in ['search', 'find', 'look']):
                rationale_parts.append("Query requires information search")
            rationale_parts.append("Can search across multiple content types")
        
        elif tool_name == 'web_search':
            if any(word in query.lower() for word in ['current', 'recent', 'latest']):
                rationale_parts.append("Query requires current/recent information")
            rationale_parts.append("External information may be needed")
        
        return "; ".join(rationale_parts)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tool_capabilities.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolCapability]:
        """Get information about a specific tool."""
        return self.tool_capabilities.get(tool_name)
    
    def get_tools_summary(self) -> Dict[str, str]:
        """Get summary of all available tools."""
        return {
            tool_name: capability.description
            for tool_name, capability in self.tool_capabilities.items()
        }

# Global tool selector instance
_tool_selector = None

def get_tool_selector(enable_web_search: bool = False) -> ToolSelector:
    """Get or create a global tool selector instance."""
    global _tool_selector
    
    if _tool_selector is None:
        _tool_selector = ToolSelector(enable_web_search=enable_web_search)
    
    return _tool_selector
