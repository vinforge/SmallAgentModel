"""
Capsule Chaining & Strategy Replay for SAM
Supports chaining and re-running saved capsules to solve bigger problems.

Sprint 7 Task 4: Capsule Chaining & Strategy Replay
"""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ChainStatus(Enum):
    """Status of a capsule chain."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class CapsuleExecutionStatus(Enum):
    """Status of individual capsule execution in a chain."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class CapsuleExecution:
    """Execution of a single capsule in a chain."""
    execution_id: str
    capsule_id: str
    capsule_name: str
    status: CapsuleExecutionStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    dependencies: List[str]  # execution_ids this depends on
    started_at: Optional[str]
    completed_at: Optional[str]
    execution_time_ms: Optional[int]
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class CapsuleChain:
    """A chain of capsules to be executed in sequence or parallel."""
    chain_id: str
    name: str
    description: str
    status: ChainStatus
    executions: List[CapsuleExecution]
    created_by: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    total_execution_time_ms: Optional[int]
    success_rate: float
    context: Dict[str, Any]  # Shared context across executions
    metadata: Dict[str, Any]

class CapsuleChainManager:
    """
    Manages capsule chains and strategy replay functionality.
    """
    
    def __init__(self, chains_file: str = "capsule_chains.json", 
                 capsule_manager=None):
        """
        Initialize the capsule chain manager.
        
        Args:
            chains_file: Path to chains storage file
            capsule_manager: Knowledge capsule manager instance
        """
        self.chains_file = Path(chains_file)
        self.capsule_manager = capsule_manager
        self.chains: Dict[str, CapsuleChain] = {}
        
        # Load existing chains
        self._load_chains()
        
        logger.info(f"Capsule chain manager initialized with {len(self.chains)} chains")
    
    def create_chain(self, name: str, description: str, 
                    capsule_sequence: List[Dict[str, Any]], 
                    user_id: str, context: Dict[str, Any] = None) -> str:
        """
        Create a new capsule chain.
        
        Args:
            name: Chain name
            description: Chain description
            capsule_sequence: List of capsule configurations
            user_id: User creating the chain
            context: Initial context
            
        Returns:
            Chain ID
        """
        try:
            chain_id = f"chain_{uuid.uuid4().hex[:12]}"
            
            # Create executions from capsule sequence
            executions = []
            
            for i, capsule_config in enumerate(capsule_sequence):
                execution_id = f"{chain_id}_exec_{i+1}"
                
                # Extract capsule information
                capsule_id = capsule_config.get('capsule_id')
                capsule_name = capsule_config.get('capsule_name', f'Capsule {i+1}')
                input_data = capsule_config.get('input_data', {})
                dependencies = capsule_config.get('dependencies', [])
                
                # Convert dependency indices to execution IDs
                dependency_ids = []
                for dep in dependencies:
                    if isinstance(dep, int) and 0 <= dep < i:
                        dependency_ids.append(f"{chain_id}_exec_{dep+1}")
                    elif isinstance(dep, str):
                        dependency_ids.append(dep)
                
                execution = CapsuleExecution(
                    execution_id=execution_id,
                    capsule_id=capsule_id,
                    capsule_name=capsule_name,
                    status=CapsuleExecutionStatus.PENDING,
                    input_data=input_data,
                    output_data=None,
                    dependencies=dependency_ids,
                    started_at=None,
                    completed_at=None,
                    execution_time_ms=None,
                    error_message=None,
                    metadata=capsule_config.get('metadata', {})
                )
                
                executions.append(execution)
            
            # Create chain
            chain = CapsuleChain(
                chain_id=chain_id,
                name=name,
                description=description,
                status=ChainStatus.CREATED,
                executions=executions,
                created_by=user_id,
                created_at=datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                total_execution_time_ms=None,
                success_rate=0.0,
                context=context or {},
                metadata={}
            )
            
            # Store chain
            self.chains[chain_id] = chain
            self._save_chains()
            
            logger.info(f"Created capsule chain: {name} ({chain_id}) with {len(executions)} capsules")
            return chain_id
            
        except Exception as e:
            logger.error(f"Error creating capsule chain: {e}")
            raise
    
    def execute_chain(self, chain_id: str, 
                     execution_context: Dict[str, Any] = None) -> bool:
        """
        Execute a capsule chain.
        
        Args:
            chain_id: ID of the chain to execute
            execution_context: Additional context for execution
            
        Returns:
            True if execution started successfully, False otherwise
        """
        try:
            chain = self.chains.get(chain_id)
            if not chain:
                logger.error(f"Chain not found: {chain_id}")
                return False
            
            if chain.status == ChainStatus.RUNNING:
                logger.warning(f"Chain already running: {chain_id}")
                return False
            
            # Update chain status
            chain.status = ChainStatus.RUNNING
            chain.started_at = datetime.now().isoformat()
            
            # Merge execution context
            if execution_context:
                chain.context.update(execution_context)
            
            logger.info(f"Starting chain execution: {chain.name} ({chain_id})")
            
            # Execute capsules in dependency order
            execution_results = self._execute_capsules_in_order(chain)
            
            # Update chain status based on results
            successful_executions = sum(1 for result in execution_results if result)
            chain.success_rate = successful_executions / len(execution_results) if execution_results else 0
            
            if chain.success_rate == 1.0:
                chain.status = ChainStatus.COMPLETED
            else:
                chain.status = ChainStatus.FAILED
            
            chain.completed_at = datetime.now().isoformat()
            
            # Calculate total execution time
            if chain.started_at:
                start_time = datetime.fromisoformat(chain.started_at)
                end_time = datetime.now()
                chain.total_execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self._save_chains()
            
            logger.info(f"Chain execution completed: {chain.name} "
                       f"(success rate: {chain.success_rate:.1%})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing chain {chain_id}: {e}")
            
            # Update chain status to failed
            if chain_id in self.chains:
                self.chains[chain_id].status = ChainStatus.FAILED
                self._save_chains()
            
            return False
    
    def replay_chain(self, chain_id: str, 
                    new_input_data: Dict[str, Any] = None) -> Optional[str]:
        """
        Replay a chain with new input data.
        
        Args:
            chain_id: ID of the chain to replay
            new_input_data: New input data for the replay
            
        Returns:
            New chain ID if successful, None otherwise
        """
        try:
            original_chain = self.chains.get(chain_id)
            if not original_chain:
                logger.error(f"Chain not found for replay: {chain_id}")
                return None
            
            # Create new chain based on original
            replay_chain_id = f"replay_{uuid.uuid4().hex[:12]}"
            
            # Copy executions with new input data
            new_executions = []
            for i, original_exec in enumerate(original_chain.executions):
                new_execution_id = f"{replay_chain_id}_exec_{i+1}"
                
                # Use new input data if provided, otherwise use original
                input_data = new_input_data.get(original_exec.execution_id, original_exec.input_data) if new_input_data else original_exec.input_data
                
                # Update dependency IDs
                new_dependencies = []
                for dep in original_exec.dependencies:
                    if dep.startswith(chain_id):
                        # Update to new chain's execution IDs
                        dep_index = dep.split('_exec_')[-1]
                        new_dependencies.append(f"{replay_chain_id}_exec_{dep_index}")
                    else:
                        new_dependencies.append(dep)
                
                new_execution = CapsuleExecution(
                    execution_id=new_execution_id,
                    capsule_id=original_exec.capsule_id,
                    capsule_name=original_exec.capsule_name,
                    status=CapsuleExecutionStatus.PENDING,
                    input_data=input_data,
                    output_data=None,
                    dependencies=new_dependencies,
                    started_at=None,
                    completed_at=None,
                    execution_time_ms=None,
                    error_message=None,
                    metadata=original_exec.metadata.copy()
                )
                
                new_executions.append(new_execution)
            
            # Create replay chain
            replay_chain = CapsuleChain(
                chain_id=replay_chain_id,
                name=f"{original_chain.name} (Replay)",
                description=f"Replay of {original_chain.name}: {original_chain.description}",
                status=ChainStatus.CREATED,
                executions=new_executions,
                created_by=original_chain.created_by,
                created_at=datetime.now().isoformat(),
                started_at=None,
                completed_at=None,
                total_execution_time_ms=None,
                success_rate=0.0,
                context=original_chain.context.copy(),
                metadata={
                    'original_chain_id': chain_id,
                    'replay': True
                }
            )
            
            # Store replay chain
            self.chains[replay_chain_id] = replay_chain
            self._save_chains()
            
            logger.info(f"Created replay chain: {replay_chain.name} ({replay_chain_id})")
            return replay_chain_id
            
        except Exception as e:
            logger.error(f"Error replaying chain {chain_id}: {e}")
            return None
    
    def get_chain_status(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a chain."""
        try:
            chain = self.chains.get(chain_id)
            if not chain:
                return None
            
            # Count execution statuses
            status_counts = {}
            for execution in chain.executions:
                status = execution.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get current execution
            current_execution = None
            for execution in chain.executions:
                if execution.status == CapsuleExecutionStatus.RUNNING:
                    current_execution = {
                        'execution_id': execution.execution_id,
                        'capsule_name': execution.capsule_name,
                        'started_at': execution.started_at
                    }
                    break
            
            return {
                'chain_id': chain.chain_id,
                'name': chain.name,
                'status': chain.status.value,
                'progress': {
                    'completed': status_counts.get('completed', 0),
                    'total': len(chain.executions),
                    'percentage': (status_counts.get('completed', 0) / len(chain.executions)) * 100 if chain.executions else 0
                },
                'success_rate': chain.success_rate,
                'execution_counts': status_counts,
                'current_execution': current_execution,
                'created_at': chain.created_at,
                'started_at': chain.started_at,
                'completed_at': chain.completed_at,
                'total_execution_time_ms': chain.total_execution_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error getting chain status for {chain_id}: {e}")
            return None
    
    def list_chains(self, user_id: Optional[str] = None, 
                   status_filter: Optional[ChainStatus] = None) -> List[Dict[str, Any]]:
        """List chains with optional filtering."""
        try:
            chain_list = []
            
            for chain in self.chains.values():
                # Apply filters
                if user_id and chain.created_by != user_id:
                    continue
                
                if status_filter and chain.status != status_filter:
                    continue
                
                # Get chain summary
                status_info = self.get_chain_status(chain.chain_id)
                if status_info:
                    chain_list.append(status_info)
            
            # Sort by creation date (newest first)
            chain_list.sort(key=lambda x: x['created_at'], reverse=True)
            
            return chain_list
            
        except Exception as e:
            logger.error(f"Error listing chains: {e}")
            return []
    
    def _execute_capsules_in_order(self, chain: CapsuleChain) -> List[bool]:
        """Execute capsules in dependency order."""
        execution_results = []
        completed_executions = set()
        
        # Keep executing until all are done or we can't proceed
        max_iterations = len(chain.executions) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(completed_executions) < len(chain.executions) and iteration < max_iterations:
            iteration += 1
            progress_made = False
            
            for execution in chain.executions:
                if execution.execution_id in completed_executions:
                    continue
                
                # Check if dependencies are satisfied
                if self._check_execution_dependencies(execution, completed_executions):
                    # Execute this capsule
                    success = self._execute_single_capsule(execution, chain.context)
                    execution_results.append(success)
                    completed_executions.add(execution.execution_id)
                    progress_made = True
                    
                    # Update chain context with execution output
                    if success and execution.output_data:
                        chain.context[f"output_{execution.execution_id}"] = execution.output_data
            
            if not progress_made:
                # No progress made, check for circular dependencies or missing capsules
                logger.warning(f"No progress in chain execution: {chain.chain_id}")
                break
        
        # Mark remaining executions as failed
        for execution in chain.executions:
            if execution.execution_id not in completed_executions:
                execution.status = CapsuleExecutionStatus.FAILED
                execution.error_message = "Dependency not satisfied or circular dependency"
                execution_results.append(False)
        
        return execution_results
    
    def _execute_single_capsule(self, execution: CapsuleExecution, 
                               chain_context: Dict[str, Any]) -> bool:
        """Execute a single capsule."""
        try:
            execution.status = CapsuleExecutionStatus.RUNNING
            execution.started_at = datetime.now().isoformat()
            
            logger.info(f"Executing capsule: {execution.capsule_name} ({execution.execution_id})")
            
            # Load capsule if capsule manager is available
            if self.capsule_manager and execution.capsule_id:
                capsule = self.capsule_manager.load_capsule(execution.capsule_id)
                
                if capsule:
                    # Simulate capsule execution (would need actual execution logic)
                    execution_result = self._simulate_capsule_execution(
                        capsule, execution.input_data, chain_context
                    )
                    
                    execution.output_data = execution_result
                    execution.status = CapsuleExecutionStatus.COMPLETED
                    
                    logger.info(f"Capsule executed successfully: {execution.capsule_name}")
                else:
                    execution.status = CapsuleExecutionStatus.FAILED
                    execution.error_message = f"Capsule not found: {execution.capsule_id}"
                    logger.error(f"Capsule not found: {execution.capsule_id}")
            else:
                # Fallback: simulate execution without actual capsule
                execution.output_data = {
                    'simulated': True,
                    'input_processed': execution.input_data,
                    'execution_id': execution.execution_id
                }
                execution.status = CapsuleExecutionStatus.COMPLETED
                logger.info(f"Simulated capsule execution: {execution.capsule_name}")
            
            # Calculate execution time
            if execution.started_at:
                start_time = datetime.fromisoformat(execution.started_at)
                end_time = datetime.now()
                execution.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            execution.completed_at = datetime.now().isoformat()
            
            return execution.status == CapsuleExecutionStatus.COMPLETED
            
        except Exception as e:
            execution.status = CapsuleExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now().isoformat()
            
            logger.error(f"Error executing capsule {execution.capsule_name}: {e}")
            return False
    
    def _simulate_capsule_execution(self, capsule, input_data: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate capsule execution (placeholder for actual execution logic)."""
        # This would be replaced with actual capsule execution logic
        return {
            'capsule_name': capsule.name,
            'methodology': capsule.content.methodology,
            'tools_used': capsule.content.tools_used,
            'key_insights': capsule.content.key_insights,
            'input_data': input_data,
            'context_used': list(context.keys()),
            'simulated_result': f"Executed {capsule.name} with {len(input_data)} input parameters"
        }
    
    def _check_execution_dependencies(self, execution: CapsuleExecution, 
                                    completed_executions: set) -> bool:
        """Check if execution dependencies are satisfied."""
        for dep_id in execution.dependencies:
            if dep_id not in completed_executions:
                return False
        return True
    
    def _load_chains(self):
        """Load chains from storage."""
        try:
            if self.chains_file.exists():
                with open(self.chains_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct chains
                for chain_data in data.get('chains', []):
                    # Reconstruct executions
                    executions = []
                    for exec_data in chain_data['executions']:
                        execution = CapsuleExecution(
                            execution_id=exec_data['execution_id'],
                            capsule_id=exec_data['capsule_id'],
                            capsule_name=exec_data['capsule_name'],
                            status=CapsuleExecutionStatus(exec_data['status']),
                            input_data=exec_data['input_data'],
                            output_data=exec_data.get('output_data'),
                            dependencies=exec_data['dependencies'],
                            started_at=exec_data.get('started_at'),
                            completed_at=exec_data.get('completed_at'),
                            execution_time_ms=exec_data.get('execution_time_ms'),
                            error_message=exec_data.get('error_message'),
                            metadata=exec_data.get('metadata', {})
                        )
                        executions.append(execution)
                    
                    # Reconstruct chain
                    chain = CapsuleChain(
                        chain_id=chain_data['chain_id'],
                        name=chain_data['name'],
                        description=chain_data['description'],
                        status=ChainStatus(chain_data['status']),
                        executions=executions,
                        created_by=chain_data['created_by'],
                        created_at=chain_data['created_at'],
                        started_at=chain_data.get('started_at'),
                        completed_at=chain_data.get('completed_at'),
                        total_execution_time_ms=chain_data.get('total_execution_time_ms'),
                        success_rate=chain_data.get('success_rate', 0.0),
                        context=chain_data.get('context', {}),
                        metadata=chain_data.get('metadata', {})
                    )
                    
                    self.chains[chain.chain_id] = chain
                
                logger.info(f"Loaded {len(self.chains)} chains from storage")
            
        except Exception as e:
            logger.error(f"Error loading chains: {e}")
    
    def _save_chains(self):
        """Save chains to storage."""
        try:
            # Convert chains to serializable format
            chains_data = []
            for chain in self.chains.values():
                chain_dict = asdict(chain)
                # Convert enums to strings
                chain_dict['status'] = chain.status.value
                for exec_dict in chain_dict['executions']:
                    exec_dict['status'] = CapsuleExecutionStatus(exec_dict['status']).value
                chains_data.append(chain_dict)
            
            data = {
                'chains': chains_data,
                'last_updated': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            self.chains_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.chains_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.chains)} chains to storage")
            
        except Exception as e:
            logger.error(f"Error saving chains: {e}")

# Global chain manager instance
_chain_manager = None

def get_chain_manager(chains_file: str = "capsule_chains.json", 
                     capsule_manager=None) -> CapsuleChainManager:
    """Get or create a global chain manager instance."""
    global _chain_manager
    
    if _chain_manager is None:
        _chain_manager = CapsuleChainManager(
            chains_file=chains_file,
            capsule_manager=capsule_manager
        )
    
    return _chain_manager
