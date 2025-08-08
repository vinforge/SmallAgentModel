"""
Dynamic Toolchain Selection for SAM
Intelligent tool discovery, planning, and sequencing based on query context.

Sprint 8 Task 1: Dynamic Toolchain Selection
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Categories of tools."""
    COMPUTATION = "computation"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    SEARCH = "search"
    COMMUNICATION = "communication"
    FILE_PROCESSING = "file_processing"
    WEB_INTERACTION = "web_interaction"
    CUSTOM = "custom"

class ToolComplexity(Enum):
    """Tool complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"

class InputType(Enum):
    """Tool input types."""
    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    URL = "url"
    JSON = "json"
    LIST = "list"
    BOOLEAN = "boolean"

class OutputType(Enum):
    """Tool output types."""
    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    JSON = "json"
    TABLE = "table"
    CHART = "chart"
    BOOLEAN = "boolean"

@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    complexity: ToolComplexity
    input_types: List[InputType]
    output_types: List[OutputType]
    tags: List[str]
    limitations: List[str]
    dependencies: List[str]
    execution_time_estimate: int  # seconds
    requires_approval: bool
    version: str
    created_at: str
    last_updated: str
    usage_count: int
    success_rate: float
    average_execution_time: float
    metadata: Dict[str, Any]

@dataclass
class ToolPlan:
    """A plan for tool execution."""
    plan_id: str
    goal: str
    steps: List[Dict[str, Any]]
    estimated_duration: int
    confidence: float
    fallback_options: List[str]
    created_at: str

@dataclass
class ToolStep:
    """A single step in a tool execution plan."""
    step_id: str
    tool_id: str
    tool_name: str
    input_data: Dict[str, Any]
    expected_output: str
    dependencies: List[str]
    fallback_tools: List[str]

class ToolRegistry:
    """
    Registry for managing tool metadata and capabilities.
    """
    
    def __init__(self, registry_file: str = "tool_registry.json"):
        """
        Initialize the tool registry.
        
        Args:
            registry_file: Path to tool registry storage file
        """
        self.registry_file = Path(registry_file)
        self.tools: Dict[str, ToolMetadata] = {}
        
        # Load existing registry
        self._load_registry()
        
        # Initialize with built-in tools if registry is empty
        if not self.tools:
            self._initialize_builtin_tools()
        
        logger.info(f"Tool registry initialized with {len(self.tools)} tools")
    
    def register_tool(self, tool_metadata: ToolMetadata) -> bool:
        """
        Register a new tool in the registry.
        
        Args:
            tool_metadata: ToolMetadata for the tool
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.tools[tool_metadata.tool_id] = tool_metadata
            self._save_registry()
            
            logger.info(f"Registered tool: {tool_metadata.name} ({tool_metadata.tool_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool {tool_metadata.tool_id}: {e}")
            return False
    
    def get_tool(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool metadata by ID."""
        return self.tools.get(tool_id)
    
    def search_tools(self, query: str = "", category: Optional[ToolCategory] = None,
                    input_type: Optional[InputType] = None,
                    output_type: Optional[OutputType] = None,
                    tags: Optional[List[str]] = None) -> List[ToolMetadata]:
        """
        Search for tools based on criteria.
        
        Args:
            query: Text query to match against name/description
            category: Tool category filter
            input_type: Required input type
            output_type: Required output type
            tags: Required tags (any match)
            
        Returns:
            List of matching tools
        """
        try:
            matching_tools = []
            query_lower = query.lower() if query else ""
            
            for tool in self.tools.values():
                # Text query matching
                if query_lower:
                    if not (query_lower in tool.name.lower() or 
                           query_lower in tool.description.lower() or
                           any(query_lower in tag.lower() for tag in tool.tags)):
                        continue
                
                # Category filter
                if category and tool.category != category:
                    continue
                
                # Input type filter
                if input_type and input_type not in tool.input_types:
                    continue
                
                # Output type filter
                if output_type and output_type not in tool.output_types:
                    continue
                
                # Tags filter
                if tags and not any(tag in tool.tags for tag in tags):
                    continue
                
                matching_tools.append(tool)
            
            # Sort by success rate and usage
            matching_tools.sort(
                key=lambda t: (t.success_rate, t.usage_count),
                reverse=True
            )
            
            return matching_tools
            
        except Exception as e:
            logger.error(f"Error searching tools: {e}")
            return []
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolMetadata]:
        """Get all tools in a specific category."""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def update_tool_stats(self, tool_id: str, success: bool, 
                         execution_time: float) -> bool:
        """
        Update tool usage statistics.
        
        Args:
            tool_id: Tool ID
            success: Whether execution was successful
            execution_time: Execution time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tool = self.tools.get(tool_id)
            if not tool:
                return False
            
            # Update usage count
            tool.usage_count += 1
            
            # Update success rate
            total_successes = tool.success_rate * (tool.usage_count - 1)
            if success:
                total_successes += 1
            tool.success_rate = total_successes / tool.usage_count
            
            # Update average execution time
            total_time = tool.average_execution_time * (tool.usage_count - 1)
            total_time += execution_time
            tool.average_execution_time = total_time / tool.usage_count
            
            # Update timestamp
            tool.last_updated = datetime.now().isoformat()
            
            self._save_registry()
            
            logger.debug(f"Updated stats for tool {tool_id}: success_rate={tool.success_rate:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating tool stats for {tool_id}: {e}")
            return False
    
    def _initialize_builtin_tools(self):
        """Initialize registry with built-in tools."""
        builtin_tools = [
            ToolMetadata(
                tool_id="python_interpreter",
                name="Python Interpreter",
                description="Execute Python code for calculations, data processing, and analysis",
                category=ToolCategory.COMPUTATION,
                complexity=ToolComplexity.MODERATE,
                input_types=[InputType.TEXT],
                output_types=[OutputType.TEXT, OutputType.JSON],
                tags=["python", "computation", "analysis", "programming"],
                limitations=["No file system access", "Limited execution time"],
                dependencies=[],
                execution_time_estimate=5,
                requires_approval=False,
                version="1.0.0",
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                usage_count=0,
                success_rate=0.9,
                average_execution_time=3.2,
                metadata={}
            ),
            ToolMetadata(
                tool_id="table_generator",
                name="Table Generator",
                description="Create structured tables from data for comparison and analysis",
                category=ToolCategory.VISUALIZATION,
                complexity=ToolComplexity.SIMPLE,
                input_types=[InputType.JSON, InputType.LIST],
                output_types=[OutputType.TABLE, OutputType.TEXT],
                tags=["table", "visualization", "data", "formatting"],
                limitations=["Limited to tabular data"],
                dependencies=[],
                execution_time_estimate=2,
                requires_approval=False,
                version="1.0.0",
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                usage_count=0,
                success_rate=0.95,
                average_execution_time=1.8,
                metadata={}
            ),
            ToolMetadata(
                tool_id="multimodal_query",
                name="Multimodal Query",
                description="Search through documents and knowledge base for relevant information",
                category=ToolCategory.SEARCH,
                complexity=ToolComplexity.MODERATE,
                input_types=[InputType.TEXT],
                output_types=[OutputType.JSON, OutputType.TEXT],
                tags=["search", "documents", "knowledge", "retrieval"],
                limitations=["Depends on indexed content"],
                dependencies=["document_index"],
                execution_time_estimate=3,
                requires_approval=False,
                version="1.0.0",
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                usage_count=0,
                success_rate=0.85,
                average_execution_time=2.5,
                metadata={}
            ),
            ToolMetadata(
                tool_id="web_search",
                name="Web Search",
                description="Search the internet for current information and data",
                category=ToolCategory.WEB_INTERACTION,
                complexity=ToolComplexity.SIMPLE,
                input_types=[InputType.TEXT],
                output_types=[OutputType.JSON, OutputType.TEXT],
                tags=["web", "search", "internet", "current"],
                limitations=["Rate limited", "Requires internet connection"],
                dependencies=["internet_access"],
                execution_time_estimate=4,
                requires_approval=False,
                version="1.0.0",
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                usage_count=0,
                success_rate=0.8,
                average_execution_time=3.8,
                metadata={}
            )
        ]
        
        for tool in builtin_tools:
            self.tools[tool.tool_id] = tool
        
        self._save_registry()
        logger.info(f"Initialized {len(builtin_tools)} built-in tools")
    
    def _load_registry(self):
        """Load tool registry from storage."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for tool_data in data.get('tools', []):
                    tool = ToolMetadata(
                        tool_id=tool_data['tool_id'],
                        name=tool_data['name'],
                        description=tool_data['description'],
                        category=ToolCategory(tool_data['category']),
                        complexity=ToolComplexity(tool_data['complexity']),
                        input_types=[InputType(t) for t in tool_data['input_types']],
                        output_types=[OutputType(t) for t in tool_data['output_types']],
                        tags=tool_data['tags'],
                        limitations=tool_data['limitations'],
                        dependencies=tool_data['dependencies'],
                        execution_time_estimate=tool_data['execution_time_estimate'],
                        requires_approval=tool_data['requires_approval'],
                        version=tool_data['version'],
                        created_at=tool_data['created_at'],
                        last_updated=tool_data['last_updated'],
                        usage_count=tool_data.get('usage_count', 0),
                        success_rate=tool_data.get('success_rate', 0.5),
                        average_execution_time=tool_data.get('average_execution_time', 0.0),
                        metadata=tool_data.get('metadata', {})
                    )
                    
                    self.tools[tool.tool_id] = tool
                
                logger.info(f"Loaded {len(self.tools)} tools from registry")
            
        except Exception as e:
            logger.error(f"Error loading tool registry: {e}")
    
    def _save_registry(self):
        """Save tool registry to storage."""
        try:
            tools_data = []
            
            for tool in self.tools.values():
                tool_dict = asdict(tool)
                # Convert enums to strings
                tool_dict['category'] = tool.category.value
                tool_dict['complexity'] = tool.complexity.value
                tool_dict['input_types'] = [t.value for t in tool.input_types]
                tool_dict['output_types'] = [t.value for t in tool.output_types]
                tools_data.append(tool_dict)
            
            data = {
                'tools': tools_data,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.tools)} tools to registry")
            
        except Exception as e:
            logger.error(f"Error saving tool registry: {e}")

class ToolPlanner:
    """
    Plans tool execution sequences for complex goals.
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the tool planner.
        
        Args:
            tool_registry: ToolRegistry instance
        """
        self.tool_registry = tool_registry
        
        logger.info("Tool planner initialized")
    
    def create_execution_plan(self, goal: str, context: Dict[str, Any] = None) -> ToolPlan:
        """
        Create an execution plan for a goal.
        
        Args:
            goal: The goal to achieve
            context: Additional context for planning
            
        Returns:
            ToolPlan with execution steps
        """
        try:
            import uuid
            plan_id = f"plan_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Creating execution plan for: {goal[:50]}...")
            
            # Analyze goal to determine required tools
            required_capabilities = self._analyze_goal_requirements(goal, context)
            
            # Find suitable tools
            candidate_tools = self._find_candidate_tools(required_capabilities)
            
            # Create execution steps
            steps = self._create_execution_steps(goal, candidate_tools, context)
            
            # Estimate duration
            estimated_duration = sum(
                self.tool_registry.get_tool(step['tool_id']).execution_time_estimate
                for step in steps
                if self.tool_registry.get_tool(step['tool_id'])
            )
            
            # Calculate confidence
            confidence = self._calculate_plan_confidence(steps)
            
            # Identify fallback options
            fallback_options = self._identify_fallback_options(required_capabilities)
            
            plan = ToolPlan(
                plan_id=plan_id,
                goal=goal,
                steps=steps,
                estimated_duration=estimated_duration,
                confidence=confidence,
                fallback_options=fallback_options,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"Created execution plan: {len(steps)} steps, "
                       f"duration: {estimated_duration}s, confidence: {confidence:.2f}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            return self._create_fallback_plan(goal)
    
    def _analyze_goal_requirements(self, goal: str, context: Dict[str, Any] = None) -> List[str]:
        """Analyze goal to determine required capabilities."""
        requirements = []
        goal_lower = goal.lower()
        
        # Computation requirements
        if any(word in goal_lower for word in ['calculate', 'compute', 'math', 'formula']):
            requirements.append('computation')
        
        # Data analysis requirements
        if any(word in goal_lower for word in ['analyze', 'data', 'statistics', 'trends']):
            requirements.append('data_analysis')
        
        # Visualization requirements
        if any(word in goal_lower for word in ['chart', 'graph', 'plot', 'visualize', 'table']):
            requirements.append('visualization')
        
        # Search requirements
        if any(word in goal_lower for word in ['search', 'find', 'lookup', 'research']):
            requirements.append('search')
        
        # File processing requirements
        if any(word in goal_lower for word in ['file', 'document', 'csv', 'json', 'read']):
            requirements.append('file_processing')
        
        # Web interaction requirements
        if any(word in goal_lower for word in ['web', 'internet', 'online', 'current', 'latest']):
            requirements.append('web_interaction')
        
        return requirements
    
    def _find_candidate_tools(self, requirements: List[str]) -> List[ToolMetadata]:
        """Find tools that match the requirements."""
        candidate_tools = []
        
        for requirement in requirements:
            # Map requirements to categories
            category_map = {
                'computation': ToolCategory.COMPUTATION,
                'data_analysis': ToolCategory.DATA_ANALYSIS,
                'visualization': ToolCategory.VISUALIZATION,
                'search': ToolCategory.SEARCH,
                'file_processing': ToolCategory.FILE_PROCESSING,
                'web_interaction': ToolCategory.WEB_INTERACTION
            }
            
            if requirement in category_map:
                category_tools = self.tool_registry.get_tools_by_category(category_map[requirement])
                candidate_tools.extend(category_tools)
        
        # Remove duplicates and sort by success rate
        unique_tools = list({tool.tool_id: tool for tool in candidate_tools}.values())
        unique_tools.sort(key=lambda t: t.success_rate, reverse=True)
        
        return unique_tools
    
    def _create_execution_steps(self, goal: str, tools: List[ToolMetadata], 
                               context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Create execution steps from tools."""
        steps = []
        
        for i, tool in enumerate(tools[:5]):  # Limit to top 5 tools
            step = {
                'step_id': f"step_{i+1}",
                'tool_id': tool.tool_id,
                'tool_name': tool.name,
                'input_data': self._generate_input_data(tool, goal, context),
                'expected_output': self._generate_expected_output(tool, goal),
                'dependencies': [],
                'fallback_tools': [t.tool_id for t in tools[i+1:i+3]]  # Next 2 tools as fallbacks
            }
            steps.append(step)
        
        return steps
    
    def _generate_input_data(self, tool: ToolMetadata, goal: str, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate input data for a tool based on goal."""
        input_data = {}
        
        # Basic input based on tool's primary input type
        if InputType.TEXT in tool.input_types:
            input_data['query'] = goal
        
        if InputType.JSON in tool.input_types and context:
            input_data['context'] = context
        
        # Tool-specific input generation
        if tool.tool_id == 'python_interpreter':
            input_data['code'] = f"# Code to help with: {goal}\nprint('Processing...')"
        elif tool.tool_id == 'multimodal_query':
            input_data['query'] = goal
            input_data['top_k'] = 5
        elif tool.tool_id == 'web_search':
            input_data['query'] = goal
            input_data['num_results'] = 5
        
        return input_data
    
    def _generate_expected_output(self, tool: ToolMetadata, goal: str) -> str:
        """Generate expected output description for a tool."""
        if tool.category == ToolCategory.COMPUTATION:
            return f"Computational result or analysis related to: {goal}"
        elif tool.category == ToolCategory.VISUALIZATION:
            return f"Visual representation or table for: {goal}"
        elif tool.category == ToolCategory.SEARCH:
            return f"Relevant information found for: {goal}"
        else:
            return f"Output from {tool.name} for: {goal}"
    
    def _calculate_plan_confidence(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the execution plan."""
        if not steps:
            return 0.0
        
        total_confidence = 0.0
        
        for step in steps:
            tool = self.tool_registry.get_tool(step['tool_id'])
            if tool:
                total_confidence += tool.success_rate
        
        return total_confidence / len(steps)
    
    def _identify_fallback_options(self, requirements: List[str]) -> List[str]:
        """Identify fallback options for requirements."""
        fallbacks = []
        
        for requirement in requirements:
            if requirement == 'computation':
                fallbacks.append("Manual calculation with step-by-step explanation")
            elif requirement == 'search':
                fallbacks.append("Use available knowledge base or ask user for information")
            elif requirement == 'visualization':
                fallbacks.append("Provide textual description or ASCII representation")
        
        return fallbacks
    
    def _create_fallback_plan(self, goal: str) -> ToolPlan:
        """Create a fallback plan when planning fails."""
        import uuid
        
        return ToolPlan(
            plan_id=f"fallback_{uuid.uuid4().hex[:8]}",
            goal=goal,
            steps=[{
                'step_id': 'fallback_step',
                'tool_id': 'manual_reasoning',
                'tool_name': 'Manual Reasoning',
                'input_data': {'goal': goal},
                'expected_output': 'Reasoning-based response',
                'dependencies': [],
                'fallback_tools': []
            }],
            estimated_duration=30,
            confidence=0.3,
            fallback_options=["Ask user for clarification", "Break down goal into simpler parts"],
            created_at=datetime.now().isoformat()
        )

# Global instances
_tool_registry = None
_tool_planner = None

def get_tool_registry(registry_file: str = "tool_registry.json") -> ToolRegistry:
    """Get or create a global tool registry instance."""
    global _tool_registry
    
    if _tool_registry is None:
        _tool_registry = ToolRegistry(registry_file=registry_file)
    
    return _tool_registry

def get_tool_planner(tool_registry: ToolRegistry = None) -> ToolPlanner:
    """Get or create a global tool planner instance."""
    global _tool_planner
    
    if _tool_planner is None:
        if tool_registry is None:
            tool_registry = get_tool_registry()
        _tool_planner = ToolPlanner(tool_registry)
    
    return _tool_planner
