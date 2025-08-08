"""
SAM Tool Registry Module

Maps SAM's existing tools and capabilities to planning actions for the A* search algorithm.
Provides a structured way to access SAM's document analysis, memory operations, 
research tools, and other capabilities within the planning framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools available in SAM."""
    DOCUMENT_ANALYSIS = "document_analysis"
    MEMORY_OPERATIONS = "memory_operations"
    RESEARCH_TOOLS = "research_tools"
    CONVERSATION_TOOLS = "conversation_tools"
    SYNTHESIS_TOOLS = "synthesis_tools"
    REASONING_TOOLS = "reasoning_tools"


@dataclass
class SAMTool:
    """
    Represents a SAM tool that can be used as an action in planning.
    """
    name: str
    """Unique name for the tool"""
    
    category: ToolCategory
    """Category this tool belongs to"""
    
    description: str
    """Human-readable description of what the tool does"""
    
    parameters: List[str]
    """List of parameter names this tool accepts"""
    
    cost_estimate: int = 1
    """Estimated cost (in planning steps) to execute this tool"""
    
    prerequisites: List[str] = None
    """List of conditions that must be met before using this tool"""
    
    context_requirements: List[str] = None
    """List of context types required (documents, memory, etc.)"""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.prerequisites is None:
            self.prerequisites = []
        if self.context_requirements is None:
            self.context_requirements = []
    
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """
        Check if this tool can be executed given the current context.
        
        Args:
            context: Current planning context
            
        Returns:
            True if tool can be executed
        """
        # Check context requirements
        for req in self.context_requirements:
            if req not in context or not context[req]:
                return False
        
        # Check prerequisites (can be extended with custom logic)
        for prereq in self.prerequisites:
            if not self._check_prerequisite(prereq, context):
                return False
        
        return True
    
    def _check_prerequisite(self, prerequisite: str, context: Dict[str, Any]) -> bool:
        """Check if a specific prerequisite is met."""
        # Simple prerequisite checking - can be enhanced
        if prerequisite == "document_uploaded":
            return context.get('documents') is not None
        elif prerequisite == "memory_available":
            return context.get('memory') is not None
        elif prerequisite == "conversation_active":
            return context.get('conversation') is not None
        else:
            # Unknown prerequisite - assume it's met
            return True


class SAMToolRegistry:
    """
    Registry of all SAM tools available for planning.
    
    This class maintains a catalog of SAM's capabilities and provides
    methods to query and filter tools based on context and requirements.
    """
    
    def __init__(self):
        """Initialize the tool registry with SAM's capabilities."""
        self._tools: Dict[str, SAMTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {}
        self._initialize_sam_tools()
    
    def _initialize_sam_tools(self):
        """Initialize the registry with SAM's known tools."""
        
        # Document Analysis Tools
        self._register_tool(SAMTool(
            name="summarize_document",
            category=ToolCategory.DOCUMENT_ANALYSIS,
            description="Generate comprehensive summary of uploaded document",
            parameters=["filename", "summary_type"],
            cost_estimate=2,
            context_requirements=["documents"]
        ))
        
        self._register_tool(SAMTool(
            name="extract_key_questions",
            category=ToolCategory.DOCUMENT_ANALYSIS,
            description="Generate strategic questions about document content",
            parameters=["filename", "question_type"],
            cost_estimate=2,
            context_requirements=["documents"]
        ))
        
        self._register_tool(SAMTool(
            name="deep_analyze_document",
            category=ToolCategory.DOCUMENT_ANALYSIS,
            description="Perform comprehensive analysis with insights and recommendations",
            parameters=["filename", "analysis_depth"],
            cost_estimate=3,
            context_requirements=["documents"]
        ))
        
        self._register_tool(SAMTool(
            name="extract_document_structure",
            category=ToolCategory.DOCUMENT_ANALYSIS,
            description="Analyze document structure and organization",
            parameters=["filename"],
            cost_estimate=1,
            context_requirements=["documents"]
        ))
        
        # Memory Operations Tools
        self._register_tool(SAMTool(
            name="search_memory",
            category=ToolCategory.MEMORY_OPERATIONS,
            description="Search SAM's memory for relevant information",
            parameters=["query", "memory_type"],
            cost_estimate=1,
            context_requirements=["memory"]
        ))
        
        self._register_tool(SAMTool(
            name="consolidate_knowledge",
            category=ToolCategory.MEMORY_OPERATIONS,
            description="Consolidate information into SAM's knowledge base",
            parameters=["content", "importance_score"],
            cost_estimate=2,
            context_requirements=["memory"]
        ))
        
        self._register_tool(SAMTool(
            name="retrieve_similar_memories",
            category=ToolCategory.MEMORY_OPERATIONS,
            description="Find memories similar to current context",
            parameters=["context", "similarity_threshold"],
            cost_estimate=1,
            context_requirements=["memory"]
        ))
        
        # Research Tools
        self._register_tool(SAMTool(
            name="web_search",
            category=ToolCategory.RESEARCH_TOOLS,
            description="Search the web for additional information",
            parameters=["query", "num_results"],
            cost_estimate=2
        ))
        
        self._register_tool(SAMTool(
            name="arxiv_search",
            category=ToolCategory.RESEARCH_TOOLS,
            description="Search ArXiv for academic papers",
            parameters=["query", "max_papers"],
            cost_estimate=3
        ))
        
        self._register_tool(SAMTool(
            name="deep_research",
            category=ToolCategory.RESEARCH_TOOLS,
            description="Conduct comprehensive research on a topic",
            parameters=["topic", "research_depth"],
            cost_estimate=5
        ))
        
        # Conversation Tools
        self._register_tool(SAMTool(
            name="analyze_conversation_context",
            category=ToolCategory.CONVERSATION_TOOLS,
            description="Analyze current conversation for context and intent",
            parameters=["conversation_history"],
            cost_estimate=1,
            context_requirements=["conversation"]
        ))
        
        self._register_tool(SAMTool(
            name="generate_clarifying_questions",
            category=ToolCategory.CONVERSATION_TOOLS,
            description="Generate questions to clarify user intent",
            parameters=["context"],
            cost_estimate=1,
            context_requirements=["conversation"]
        ))
        
        # Synthesis Tools
        self._register_tool(SAMTool(
            name="synthesize_information",
            category=ToolCategory.SYNTHESIS_TOOLS,
            description="Combine information from multiple sources",
            parameters=["sources", "synthesis_type"],
            cost_estimate=3
        ))
        
        self._register_tool(SAMTool(
            name="create_structured_response",
            category=ToolCategory.SYNTHESIS_TOOLS,
            description="Create well-structured response from analysis",
            parameters=["content", "response_format"],
            cost_estimate=2
        ))
        
        # Reasoning Tools
        self._register_tool(SAMTool(
            name="apply_logical_reasoning",
            category=ToolCategory.REASONING_TOOLS,
            description="Apply logical reasoning to draw conclusions",
            parameters=["premises", "reasoning_type"],
            cost_estimate=2
        ))
        
        self._register_tool(SAMTool(
            name="validate_conclusions",
            category=ToolCategory.REASONING_TOOLS,
            description="Validate conclusions against available evidence",
            parameters=["conclusions", "evidence"],
            cost_estimate=2
        ))
        
        logger.info(f"Initialized SAM tool registry with {len(self._tools)} tools")
    
    def _register_tool(self, tool: SAMTool):
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)
    
    def get_tool(self, name: str) -> Optional[SAMTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[SAMTool]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names]
    
    def get_available_tools(self, context: Dict[str, Any]) -> List[SAMTool]:
        """
        Get all tools that can be executed given the current context.
        
        Args:
            context: Current planning context
            
        Returns:
            List of tools that can be executed
        """
        available = []
        for tool in self._tools.values():
            if tool.can_execute(context):
                available.append(tool)
        
        return available
    
    def get_tools_for_task_type(self, task_description: str) -> List[SAMTool]:
        """
        Get tools that are relevant for a specific type of task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            List of relevant tools
        """
        task_lower = task_description.lower()
        relevant_tools = []
        
        # Simple keyword-based matching - can be enhanced with ML
        if any(word in task_lower for word in ['document', 'paper', 'file', 'analyze']):
            relevant_tools.extend(self.get_tools_by_category(ToolCategory.DOCUMENT_ANALYSIS))
        
        if any(word in task_lower for word in ['research', 'search', 'find', 'investigate']):
            relevant_tools.extend(self.get_tools_by_category(ToolCategory.RESEARCH_TOOLS))
        
        if any(word in task_lower for word in ['remember', 'recall', 'memory', 'previous']):
            relevant_tools.extend(self.get_tools_by_category(ToolCategory.MEMORY_OPERATIONS))
        
        if any(word in task_lower for word in ['synthesize', 'combine', 'merge', 'integrate']):
            relevant_tools.extend(self.get_tools_by_category(ToolCategory.SYNTHESIS_TOOLS))
        
        if any(word in task_lower for word in ['reason', 'logic', 'conclude', 'infer']):
            relevant_tools.extend(self.get_tools_by_category(ToolCategory.REASONING_TOOLS))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in relevant_tools:
            if tool.name not in seen:
                seen.add(tool.name)
                unique_tools.append(tool)
        
        return unique_tools
    
    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())
    
    def get_categories(self) -> List[ToolCategory]:
        """Get list of all tool categories."""
        return list(self._categories.keys())
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary information about the registry."""
        return {
            'total_tools': len(self._tools),
            'categories': {cat.value: len(tools) for cat, tools in self._categories.items()},
            'tools_by_category': {
                cat.value: [self._tools[name].name for name in tools]
                for cat, tools in self._categories.items()
            }
        }


# Global registry instance
_sam_tool_registry = None

def get_sam_tool_registry() -> SAMToolRegistry:
    """Get the global SAM tool registry instance."""
    global _sam_tool_registry
    if _sam_tool_registry is None:
        _sam_tool_registry = SAMToolRegistry()
    return _sam_tool_registry
