"""
Custom Tool Creation Interface for SAM
Allows users to define new tools via prompt-based templates with testing and versioning.

Sprint 8 Task 4: Custom Tool Creation Interface
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ToolTemplateType(Enum):
    """Types of tool templates."""
    COMPUTATION = "computation"
    DATA_PROCESSING = "data_processing"
    TEXT_ANALYSIS = "text_analysis"
    API_INTEGRATION = "api_integration"
    CUSTOM_LOGIC = "custom_logic"

class ToolStatus(Enum):
    """Status of custom tools."""
    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ToolTemplate:
    """Template for creating custom tools."""
    template_id: str
    name: str
    description: str
    template_type: ToolTemplateType
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    logic_template: str
    example_inputs: List[Dict[str, Any]]
    example_outputs: List[Any]
    validation_rules: List[str]
    created_at: str
    metadata: Dict[str, Any]

@dataclass
class CustomTool:
    """A custom tool created by a user."""
    tool_id: str
    name: str
    description: str
    creator_id: str
    template_id: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    logic_code: str
    status: ToolStatus
    version: str
    test_results: List[Dict[str, Any]]
    usage_count: int
    success_rate: float
    created_at: str
    last_updated: str
    metadata: Dict[str, Any]

@dataclass
class ToolVersion:
    """Version of a custom tool."""
    version_id: str
    tool_id: str
    version_number: str
    changes: str
    logic_code: str
    test_results: List[Dict[str, Any]]
    created_at: str
    created_by: str
    is_active: bool

class CustomToolCreator:
    """
    Interface for creating and managing custom tools.
    """
    
    def __init__(self, tools_directory: str = "custom_tools"):
        """
        Initialize the custom tool creator.
        
        Args:
            tools_directory: Directory for storing custom tools
        """
        self.tools_dir = Path(tools_directory)
        self.tools_dir.mkdir(exist_ok=True)
        
        # Storage
        self.templates: Dict[str, ToolTemplate] = {}
        self.custom_tools: Dict[str, CustomTool] = {}
        self.tool_versions: Dict[str, List[ToolVersion]] = {}
        
        # Initialize built-in templates
        self._initialize_templates()
        
        # Load existing tools
        self._load_custom_tools()
        
        logger.info(f"Custom tool creator initialized with {len(self.templates)} templates and {len(self.custom_tools)} custom tools")
    
    def get_available_templates(self) -> List[ToolTemplate]:
        """Get all available tool templates."""
        return list(self.templates.values())
    
    def create_tool_from_template(self, template_id: str, tool_name: str,
                                 tool_description: str, creator_id: str,
                                 custom_logic: str, input_customization: Dict[str, Any] = None,
                                 output_customization: Dict[str, Any] = None) -> str:
        """
        Create a custom tool from a template.
        
        Args:
            template_id: Template to use
            tool_name: Name for the new tool
            tool_description: Description of the tool
            creator_id: User creating the tool
            custom_logic: Custom logic code
            input_customization: Custom input schema modifications
            output_customization: Custom output schema modifications
            
        Returns:
            Tool ID of created tool
        """
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            
            tool_id = f"custom_{uuid.uuid4().hex[:12]}"
            
            # Customize input/output schemas
            input_schema = template.input_schema.copy()
            if input_customization:
                input_schema.update(input_customization)
            
            output_schema = template.output_schema.copy()
            if output_customization:
                output_schema.update(output_customization)
            
            # Create custom tool
            custom_tool = CustomTool(
                tool_id=tool_id,
                name=tool_name,
                description=tool_description,
                creator_id=creator_id,
                template_id=template_id,
                input_schema=input_schema,
                output_schema=output_schema,
                logic_code=custom_logic,
                status=ToolStatus.DRAFT,
                version="1.0.0",
                test_results=[],
                usage_count=0,
                success_rate=0.0,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                metadata={}
            )
            
            self.custom_tools[tool_id] = custom_tool
            
            # Create initial version
            self._create_tool_version(tool_id, "1.0.0", "Initial version", custom_logic, creator_id)
            
            # Save to file
            self._save_custom_tool(custom_tool)
            
            logger.info(f"Created custom tool: {tool_name} ({tool_id})")
            return tool_id
            
        except Exception as e:
            logger.error(f"Error creating custom tool: {e}")
            raise
    
    def test_custom_tool(self, tool_id: str, test_inputs: List[Dict[str, Any]],
                        expected_outputs: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Test a custom tool with provided inputs.
        
        Args:
            tool_id: Tool ID to test
            test_inputs: List of test input data
            expected_outputs: Optional expected outputs for validation
            
        Returns:
            Test results
        """
        try:
            tool = self.custom_tools.get(tool_id)
            if not tool:
                raise ValueError(f"Custom tool not found: {tool_id}")
            
            test_results = []
            
            for i, test_input in enumerate(test_inputs):
                try:
                    # Simulate tool execution
                    result = self._simulate_tool_execution(tool, test_input)
                    
                    test_case = {
                        'test_case': i + 1,
                        'input': test_input,
                        'output': result,
                        'success': True,
                        'error': None,
                        'execution_time_ms': 100  # Simulated
                    }
                    
                    # Validate against expected output if provided
                    if expected_outputs and i < len(expected_outputs):
                        expected = expected_outputs[i]
                        test_case['expected'] = expected
                        test_case['matches_expected'] = self._compare_outputs(result, expected)
                    
                    test_results.append(test_case)
                    
                except Exception as e:
                    test_case = {
                        'test_case': i + 1,
                        'input': test_input,
                        'output': None,
                        'success': False,
                        'error': str(e),
                        'execution_time_ms': 0
                    }
                    test_results.append(test_case)
            
            # Update tool test results
            tool.test_results = test_results
            tool.last_updated = datetime.now().isoformat()
            
            # Calculate success rate
            successful_tests = sum(1 for result in test_results if result['success'])
            tool.success_rate = successful_tests / len(test_results) if test_results else 0.0
            
            # Update status based on test results
            if tool.success_rate >= 0.8:
                tool.status = ToolStatus.TESTING
            else:
                tool.status = ToolStatus.FAILED
            
            self._save_custom_tool(tool)
            
            summary = {
                'tool_id': tool_id,
                'total_tests': len(test_results),
                'successful_tests': successful_tests,
                'success_rate': tool.success_rate,
                'status': tool.status.value,
                'test_results': test_results
            }
            
            logger.info(f"Tested custom tool {tool_id}: {successful_tests}/{len(test_results)} tests passed")
            return summary
            
        except Exception as e:
            logger.error(f"Error testing custom tool {tool_id}: {e}")
            raise
    
    def activate_tool(self, tool_id: str) -> bool:
        """
        Activate a custom tool for use.
        
        Args:
            tool_id: Tool ID to activate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tool = self.custom_tools.get(tool_id)
            if not tool:
                logger.error(f"Custom tool not found: {tool_id}")
                return False
            
            # Check if tool has passed testing
            if tool.status != ToolStatus.TESTING or tool.success_rate < 0.8:
                logger.error(f"Tool {tool_id} has not passed testing requirements")
                return False
            
            tool.status = ToolStatus.ACTIVE
            tool.last_updated = datetime.now().isoformat()
            
            self._save_custom_tool(tool)
            
            logger.info(f"Activated custom tool: {tool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error activating tool {tool_id}: {e}")
            return False
    
    def update_tool(self, tool_id: str, new_logic: str, version_notes: str,
                   updater_id: str) -> str:
        """
        Update a custom tool with new logic.
        
        Args:
            tool_id: Tool ID to update
            new_logic: New logic code
            version_notes: Notes about the changes
            updater_id: User making the update
            
        Returns:
            New version number
        """
        try:
            tool = self.custom_tools.get(tool_id)
            if not tool:
                raise ValueError(f"Custom tool not found: {tool_id}")
            
            # Generate new version number
            current_version = tool.version
            version_parts = current_version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            new_version = '.'.join(version_parts)
            
            # Create new version
            self._create_tool_version(tool_id, new_version, version_notes, new_logic, updater_id)
            
            # Update tool
            tool.logic_code = new_logic
            tool.version = new_version
            tool.status = ToolStatus.DRAFT  # Reset to draft for testing
            tool.test_results = []
            tool.last_updated = datetime.now().isoformat()
            
            self._save_custom_tool(tool)
            
            logger.info(f"Updated custom tool {tool_id} to version {new_version}")
            return new_version
            
        except Exception as e:
            logger.error(f"Error updating tool {tool_id}: {e}")
            raise
    
    def get_tool_versions(self, tool_id: str) -> List[ToolVersion]:
        """Get all versions of a custom tool."""
        return self.tool_versions.get(tool_id, [])
    
    def rollback_tool(self, tool_id: str, target_version: str) -> bool:
        """
        Rollback a tool to a previous version.
        
        Args:
            tool_id: Tool ID to rollback
            target_version: Version to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tool = self.custom_tools.get(tool_id)
            if not tool:
                logger.error(f"Custom tool not found: {tool_id}")
                return False
            
            # Find target version
            versions = self.tool_versions.get(tool_id, [])
            target_version_obj = None
            
            for version in versions:
                if version.version_number == target_version:
                    target_version_obj = version
                    break
            
            if not target_version_obj:
                logger.error(f"Version {target_version} not found for tool {tool_id}")
                return False
            
            # Rollback tool
            tool.logic_code = target_version_obj.logic_code
            tool.version = target_version
            tool.status = ToolStatus.DRAFT  # Reset to draft
            tool.test_results = target_version_obj.test_results.copy()
            tool.last_updated = datetime.now().isoformat()
            
            self._save_custom_tool(tool)
            
            logger.info(f"Rolled back tool {tool_id} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back tool {tool_id}: {e}")
            return False
    
    def get_custom_tool(self, tool_id: str) -> Optional[CustomTool]:
        """Get a custom tool by ID."""
        return self.custom_tools.get(tool_id)
    
    def list_custom_tools(self, creator_id: Optional[str] = None,
                         status: Optional[ToolStatus] = None) -> List[CustomTool]:
        """
        List custom tools with optional filtering.
        
        Args:
            creator_id: Filter by creator
            status: Filter by status
            
        Returns:
            List of custom tools
        """
        tools = list(self.custom_tools.values())
        
        if creator_id:
            tools = [tool for tool in tools if tool.creator_id == creator_id]
        
        if status:
            tools = [tool for tool in tools if tool.status == status]
        
        # Sort by creation date (newest first)
        tools.sort(key=lambda t: t.created_at, reverse=True)
        
        return tools
    
    def _initialize_templates(self):
        """Initialize built-in tool templates."""
        templates = [
            ToolTemplate(
                template_id="computation_template",
                name="Computation Tool Template",
                description="Template for creating computational tools",
                template_type=ToolTemplateType.COMPUTATION,
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
                        "variables": {"type": "object", "description": "Variables to use in computation"}
                    },
                    "required": ["expression"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "number", "description": "Computation result"},
                        "steps": {"type": "array", "description": "Computation steps"}
                    }
                },
                logic_template="""
def execute(input_data):
    expression = input_data.get('expression', '')
    variables = input_data.get('variables', {})
    
    # Add your computation logic here
    # Example: result = eval(expression, variables)
    
    return {
        'result': 0,  # Replace with actual computation
        'steps': ['Step 1', 'Step 2']  # Replace with actual steps
    }
""",
                example_inputs=[
                    {"expression": "2 + 2"},
                    {"expression": "x * y", "variables": {"x": 5, "y": 3}}
                ],
                example_outputs=[
                    {"result": 4, "steps": ["2 + 2 = 4"]},
                    {"result": 15, "steps": ["5 * 3 = 15"]}
                ],
                validation_rules=[
                    "Expression must be a valid mathematical expression",
                    "Variables must be numeric values"
                ],
                created_at=datetime.now().isoformat(),
                metadata={}
            ),
            ToolTemplate(
                template_id="text_analysis_template",
                name="Text Analysis Tool Template",
                description="Template for creating text analysis tools",
                template_type=ToolTemplateType.TEXT_ANALYSIS,
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"},
                        "analysis_type": {"type": "string", "description": "Type of analysis to perform"}
                    },
                    "required": ["text"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "analysis_result": {"type": "object", "description": "Analysis results"},
                        "metrics": {"type": "object", "description": "Text metrics"}
                    }
                },
                logic_template="""
def execute(input_data):
    text = input_data.get('text', '')
    analysis_type = input_data.get('analysis_type', 'basic')
    
    # Add your text analysis logic here
    # Example: word_count = len(text.split())
    
    return {
        'analysis_result': {},  # Replace with actual analysis
        'metrics': {'word_count': 0}  # Replace with actual metrics
    }
""",
                example_inputs=[
                    {"text": "Hello world", "analysis_type": "basic"}
                ],
                example_outputs=[
                    {"analysis_result": {"sentiment": "neutral"}, "metrics": {"word_count": 2}}
                ],
                validation_rules=[
                    "Text must not be empty",
                    "Analysis type must be supported"
                ],
                created_at=datetime.now().isoformat(),
                metadata={}
            )
        ]
        
        for template in templates:
            self.templates[template.template_id] = template
        
        logger.info(f"Initialized {len(templates)} tool templates")
    
    def _simulate_tool_execution(self, tool: CustomTool, input_data: Dict[str, Any]) -> Any:
        """Simulate execution of a custom tool."""
        # This is a simplified simulation
        # In a real implementation, this would safely execute the tool's logic
        
        template = self.templates.get(tool.template_id)
        if not template:
            raise ValueError(f"Template not found: {tool.template_id}")
        
        # Return a simulated result based on template type
        if template.template_type == ToolTemplateType.COMPUTATION:
            return {"result": 42, "steps": ["Simulated computation"]}
        elif template.template_type == ToolTemplateType.TEXT_ANALYSIS:
            text = input_data.get('text', '')
            return {
                "analysis_result": {"sentiment": "neutral"},
                "metrics": {"word_count": len(text.split())}
            }
        else:
            return {"simulated": True, "input_processed": input_data}
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual output with expected output."""
        # Simple comparison - could be more sophisticated
        return actual == expected
    
    def _create_tool_version(self, tool_id: str, version_number: str,
                           changes: str, logic_code: str, creator_id: str):
        """Create a new version of a tool."""
        version = ToolVersion(
            version_id=f"ver_{uuid.uuid4().hex[:12]}",
            tool_id=tool_id,
            version_number=version_number,
            changes=changes,
            logic_code=logic_code,
            test_results=[],
            created_at=datetime.now().isoformat(),
            created_by=creator_id,
            is_active=True
        )
        
        if tool_id not in self.tool_versions:
            self.tool_versions[tool_id] = []
        
        # Deactivate previous versions
        for prev_version in self.tool_versions[tool_id]:
            prev_version.is_active = False
        
        self.tool_versions[tool_id].append(version)
    
    def _save_custom_tool(self, tool: CustomTool):
        """Save a custom tool to file."""
        try:
            tool_file = self.tools_dir / f"{tool.tool_id}.json"
            
            tool_dict = asdict(tool)
            tool_dict['status'] = tool.status.value
            
            with open(tool_file, 'w', encoding='utf-8') as f:
                json.dump(tool_dict, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved custom tool: {tool.tool_id}")
            
        except Exception as e:
            logger.error(f"Error saving custom tool {tool.tool_id}: {e}")
    
    def _load_custom_tools(self):
        """Load custom tools from files."""
        try:
            for tool_file in self.tools_dir.glob("*.json"):
                try:
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        tool_data = json.load(f)
                    
                    tool = CustomTool(
                        tool_id=tool_data['tool_id'],
                        name=tool_data['name'],
                        description=tool_data['description'],
                        creator_id=tool_data['creator_id'],
                        template_id=tool_data['template_id'],
                        input_schema=tool_data['input_schema'],
                        output_schema=tool_data['output_schema'],
                        logic_code=tool_data['logic_code'],
                        status=ToolStatus(tool_data['status']),
                        version=tool_data['version'],
                        test_results=tool_data['test_results'],
                        usage_count=tool_data.get('usage_count', 0),
                        success_rate=tool_data.get('success_rate', 0.0),
                        created_at=tool_data['created_at'],
                        last_updated=tool_data['last_updated'],
                        metadata=tool_data.get('metadata', {})
                    )
                    
                    self.custom_tools[tool.tool_id] = tool
                    
                except Exception as e:
                    logger.warning(f"Error loading tool file {tool_file}: {e}")
            
            logger.info(f"Loaded {len(self.custom_tools)} custom tools")
            
        except Exception as e:
            logger.error(f"Error loading custom tools: {e}")

# Global custom tool creator instance
_custom_tool_creator = None

def get_custom_tool_creator(tools_directory: str = "custom_tools") -> CustomToolCreator:
    """Get or create a global custom tool creator instance."""
    global _custom_tool_creator
    
    if _custom_tool_creator is None:
        _custom_tool_creator = CustomToolCreator(tools_directory=tools_directory)
    
    return _custom_tool_creator
