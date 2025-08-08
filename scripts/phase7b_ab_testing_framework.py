"""
Phase 7B A/B Testing Framework for SLP Validation
================================================

Enhanced A/B testing framework specifically designed to validate the
Scalable Latent Program (SLP) Cognitive Automation Engine.

This framework implements the three test arms specified in task3.md:
- Arm A (Baseline): Original SAM with no TPV or SLP
- Arm B (TPV Only): Phase 2 system with Active Reasoning Control only
- Arm C (SLP Active): Full Phase 7 system with both TPV and SLP enabled

Key Features:
- Stateful testing for pattern learning validation
- Groups of similar queries for pattern reuse testing
- Enhanced metrics collection for SLP performance
- Sequential execution to enable program capture and reuse
"""

import time
import json
import logging
import requests
import sys
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory to path for sam module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class TestArm(Enum):
    """Test arms for Phase 7B SLP validation."""
    A_BASELINE = "A_BASELINE"           # No TPV, No SLP
    B_TPV_ONLY = "B_TPV_ONLY"          # TPV enabled, SLP disabled
    C_SLP_ACTIVE = "C_SLP_ACTIVE"      # TPV + SLP enabled

class PromptGroup(Enum):
    """Groups of similar prompts for pattern reuse testing."""
    CYBERSECURITY_ANALYSIS = "cybersecurity_analysis"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    TECHNICAL_EXPLANATION = "technical_explanation"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class TestPrompt:
    """Enhanced test prompt with group information."""
    id: str
    text: str
    group: PromptGroup
    sequence_order: int  # Order within the group (1, 2, 3...)
    expected_pattern_reuse: bool  # Whether this should trigger pattern reuse
    description: str

@dataclass
class SLPTestResult:
    """Enhanced test result with SLP-specific metrics."""
    prompt_id: str
    test_arm: TestArm
    end_to_end_latency_ms: float
    total_tokens_generated: int
    response_text: str
    timestamp: float
    error: Optional[str] = None
    
    # TPV-specific metrics
    tpv_enabled: bool = False
    tpv_steps: int = 0
    tpv_halt_reason: Optional[str] = None
    control_decision: Optional[str] = None
    
    # SLP-specific metrics (new for Phase 7B)
    slp_enabled: bool = False
    program_used_id: Optional[str] = None
    program_was_captured: bool = False
    signature_match_confidence: float = 0.0
    program_execution_time_ms: float = 0.0
    slp_hit_rate: float = 0.0
    total_programs_available: int = 0

class Phase7BBenchmarkDataset:
    """Enhanced benchmark dataset for SLP validation."""
    
    def __init__(self):
        self.prompts: List[TestPrompt] = []
        self._create_slp_benchmark_prompts()
    
    def _create_slp_benchmark_prompts(self):
        """Create benchmark prompts designed for SLP pattern reuse testing."""
        
        # Group 1: Cybersecurity Analysis (3 similar prompts)
        cybersec_prompts = [
            TestPrompt(
                id="cyber_01",
                text="Analyze the cybersecurity risks in the quarterly security report. Focus on network vulnerabilities and access control issues.",
                group=PromptGroup.CYBERSECURITY_ANALYSIS,
                sequence_order=1,
                expected_pattern_reuse=False,  # First in group - should capture pattern
                description="First cybersecurity analysis - pattern capture expected"
            ),
            TestPrompt(
                id="cyber_02", 
                text="Analyze the cybersecurity risks in the monthly security assessment. Focus on network vulnerabilities and access control issues.",
                group=PromptGroup.CYBERSECURITY_ANALYSIS,
                sequence_order=2,
                expected_pattern_reuse=True,  # Should reuse pattern from cyber_01
                description="Second cybersecurity analysis - pattern reuse expected"
            ),
            TestPrompt(
                id="cyber_03",
                text="Analyze the cybersecurity risks in the annual security review. Focus on network vulnerabilities and access control issues.",
                group=PromptGroup.CYBERSECURITY_ANALYSIS,
                sequence_order=3,
                expected_pattern_reuse=True,  # Should reuse pattern from cyber_01
                description="Third cybersecurity analysis - pattern reuse expected"
            )
        ]
        
        # Group 2: Document Summarization (3 similar prompts)
        doc_summary_prompts = [
            TestPrompt(
                id="summary_01",
                text="Summarize the key findings and recommendations from the technical documentation. Include main conclusions and action items.",
                group=PromptGroup.DOCUMENT_SUMMARIZATION,
                sequence_order=1,
                expected_pattern_reuse=False,  # First in group
                description="First document summary - pattern capture expected"
            ),
            TestPrompt(
                id="summary_02",
                text="Summarize the key findings and recommendations from the project report. Include main conclusions and action items.",
                group=PromptGroup.DOCUMENT_SUMMARIZATION,
                sequence_order=2,
                expected_pattern_reuse=True,  # Should reuse pattern
                description="Second document summary - pattern reuse expected"
            ),
            TestPrompt(
                id="summary_03",
                text="Summarize the key findings and recommendations from the research paper. Include main conclusions and action items.",
                group=PromptGroup.DOCUMENT_SUMMARIZATION,
                sequence_order=3,
                expected_pattern_reuse=True,  # Should reuse pattern
                description="Third document summary - pattern reuse expected"
            )
        ]
        
        # Group 3: Technical Explanation (3 similar prompts)
        tech_explain_prompts = [
            TestPrompt(
                id="tech_01",
                text="Explain the technical architecture and implementation details of the distributed system. Focus on scalability and performance considerations.",
                group=PromptGroup.TECHNICAL_EXPLANATION,
                sequence_order=1,
                expected_pattern_reuse=False,  # First in group
                description="First technical explanation - pattern capture expected"
            ),
            TestPrompt(
                id="tech_02",
                text="Explain the technical architecture and implementation details of the microservices platform. Focus on scalability and performance considerations.",
                group=PromptGroup.TECHNICAL_EXPLANATION,
                sequence_order=2,
                expected_pattern_reuse=True,  # Should reuse pattern
                description="Second technical explanation - pattern reuse expected"
            ),
            TestPrompt(
                id="tech_03",
                text="Explain the technical architecture and implementation details of the cloud infrastructure. Focus on scalability and performance considerations.",
                group=PromptGroup.TECHNICAL_EXPLANATION,
                sequence_order=3,
                expected_pattern_reuse=True,  # Should reuse pattern
                description="Third technical explanation - pattern reuse expected"
            )
        ]
        
        # Group 4: Risk Assessment (3 similar prompts)
        risk_assess_prompts = [
            TestPrompt(
                id="risk_01",
                text="Conduct a comprehensive risk assessment of the proposed changes. Evaluate potential impacts, likelihood, and mitigation strategies.",
                group=PromptGroup.RISK_ASSESSMENT,
                sequence_order=1,
                expected_pattern_reuse=False,  # First in group
                description="First risk assessment - pattern capture expected"
            ),
            TestPrompt(
                id="risk_02",
                text="Conduct a comprehensive risk assessment of the system upgrade. Evaluate potential impacts, likelihood, and mitigation strategies.",
                group=PromptGroup.RISK_ASSESSMENT,
                sequence_order=2,
                expected_pattern_reuse=True,  # Should reuse pattern
                description="Second risk assessment - pattern reuse expected"
            ),
            TestPrompt(
                id="risk_03",
                text="Conduct a comprehensive risk assessment of the deployment plan. Evaluate potential impacts, likelihood, and mitigation strategies.",
                group=PromptGroup.RISK_ASSESSMENT,
                sequence_order=3,
                expected_pattern_reuse=True,  # Should reuse pattern
                description="Third risk assessment - pattern reuse expected"
            )
        ]
        
        # Combine all prompts in sequence order for stateful testing
        self.prompts = cybersec_prompts + doc_summary_prompts + tech_explain_prompts + risk_assess_prompts
    
    def get_prompts_by_group(self, group: PromptGroup) -> List[TestPrompt]:
        """Get all prompts in a specific group."""
        return [p for p in self.prompts if p.group == group]
    
    def get_sequential_prompts(self) -> List[TestPrompt]:
        """Get prompts in sequential order for stateful testing."""
        # Sort by group and sequence order to ensure proper pattern learning
        return sorted(self.prompts, key=lambda p: (p.group.value, p.sequence_order))

class Phase7BTestFramework:
    """Enhanced A/B testing framework for Phase 7B SLP validation."""
    
    def __init__(self):
        self.dataset = Phase7BBenchmarkDataset()
        self.results: List[SLPTestResult] = []
        
        # Test arm configurations
        self.test_arms = {
            TestArm.A_BASELINE: {
                'description': 'Baseline - No TPV, No SLP',
                'tpv_enabled': False,
                'slp_enabled': False
            },
            TestArm.B_TPV_ONLY: {
                'description': 'TPV Only - Active Reasoning Control enabled, SLP disabled',
                'tpv_enabled': True,
                'slp_enabled': False
            },
            TestArm.C_SLP_ACTIVE: {
                'description': 'SLP Active - Both TPV and SLP enabled',
                'tpv_enabled': True,
                'slp_enabled': True
            }
        }
    
    def setup_test_environment(self) -> bool:
        """Setup and validate the Phase 7B test environment."""
        logger.info("🔧 Setting up Phase 7B SLP test environment...")
        
        try:
            # Test Arm A (Baseline) - No TPV, No SLP
            logger.info("  Testing Arm A (Baseline)...")
            # Standard Ollama without any enhancements
            
            # Test Arm B (TPV Only) - Phase 2 system
            logger.info("  Testing Arm B (TPV Only)...")
            from sam.cognition.tpv import sam_tpv_integration
            if sam_tpv_integration:
                logger.info("    ✅ TPV integration available")
            else:
                logger.warning("    ⚠️ TPV integration not available")
            
            # Test Arm C (SLP Active) - Full Phase 7 system
            logger.info("  Testing Arm C (SLP Active)...")
            from sam.cognition.slp import get_slp_integration
            slp_integration = get_slp_integration()
            if slp_integration:
                logger.info("    ✅ SLP integration available")
                # Clear any existing programs for clean testing (if method exists)
                try:
                    if hasattr(slp_integration.program_manager.store, 'clear_all_programs'):
                        slp_integration.program_manager.store.clear_all_programs()
                        logger.info("    🧹 Cleared existing programs for clean testing")
                    else:
                        logger.info("    ℹ️ Program store cleanup not available (continuing with existing programs)")
                except Exception as e:
                    logger.warning(f"    ⚠️ Could not clear programs: {e}")
            else:
                logger.warning("    ⚠️ SLP integration not available")
            
            logger.info("✅ Phase 7B test environment validated")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup Phase 7B test environment: {e}")
            return False

    def execute_stateful_testing(self) -> List[SLPTestResult]:
        """
        Execute stateful testing as specified in task3.md.

        Processes prompts sequentially to enable pattern learning and reuse.
        Each group's first prompt should capture a pattern, subsequent prompts
        should reuse that pattern in Arm C (SLP Active).
        """
        logger.info("🧪 Starting Phase 7B Stateful Testing")
        logger.info("📋 Testing Strategy: Sequential execution for pattern learning validation")

        all_results = []
        sequential_prompts = self.dataset.get_sequential_prompts()

        # Execute each prompt across all three arms in sequence
        for i, prompt in enumerate(sequential_prompts, 1):
            logger.info(f"\n🔍 Prompt {i}/{len(sequential_prompts)}: {prompt.id}")
            logger.info(f"   Group: {prompt.group.value}, Sequence: {prompt.sequence_order}")
            logger.info(f"   Expected Pattern Reuse: {prompt.expected_pattern_reuse}")

            # Execute across all test arms
            for arm in TestArm:
                logger.info(f"  🎯 Testing {arm.value}...")

                try:
                    result = self._execute_single_test(prompt, arm)
                    all_results.append(result)

                    # Log key metrics
                    if arm == TestArm.C_SLP_ACTIVE:
                        if result.program_used_id:
                            logger.info(f"    ✅ Program used: {result.program_used_id[:8]}... (confidence: {result.signature_match_confidence:.2f})")
                        elif result.program_was_captured:
                            logger.info(f"    📚 New program captured")
                        else:
                            logger.info(f"    🔍 Standard processing (no pattern match)")

                    logger.info(f"    ⏱️ Latency: {result.end_to_end_latency_ms:.0f}ms, Tokens: {result.total_tokens_generated}")

                except Exception as e:
                    logger.error(f"    ❌ Test failed: {e}")
                    # Create error result
                    error_result = SLPTestResult(
                        prompt_id=prompt.id,
                        test_arm=arm,
                        end_to_end_latency_ms=0.0,
                        total_tokens_generated=0,
                        response_text="",
                        timestamp=time.time(),
                        error=str(e)
                    )
                    all_results.append(error_result)

        self.results = all_results
        logger.info(f"🎉 Phase 7B Stateful Testing Complete! {len(all_results)} total tests executed")
        return all_results

    def _execute_single_test(self, prompt: TestPrompt, arm: TestArm) -> SLPTestResult:
        """Execute a single test for a specific prompt and arm."""
        start_time = time.time()

        try:
            if arm == TestArm.A_BASELINE:
                return self._execute_baseline_arm(prompt, start_time)
            elif arm == TestArm.B_TPV_ONLY:
                return self._execute_tpv_only_arm(prompt, start_time)
            elif arm == TestArm.C_SLP_ACTIVE:
                return self._execute_slp_active_arm(prompt, start_time)
            else:
                raise ValueError(f"Unknown test arm: {arm}")

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Test execution failed for {prompt.id} on {arm.value}: {e}")

            return SLPTestResult(
                prompt_id=prompt.id,
                test_arm=arm,
                end_to_end_latency_ms=execution_time,
                total_tokens_generated=0,
                response_text="",
                timestamp=time.time(),
                error=str(e)
            )

    def _execute_baseline_arm(self, prompt: TestPrompt, start_time: float) -> SLPTestResult:
        """Execute test for Arm A (Baseline - No TPV, No SLP)."""
        try:
            # Direct Ollama call without any enhancements
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                    "prompt": prompt.text,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=60
            )

            execution_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get('response', '').strip()
                token_count = len(response_text.split()) if response_text else 0

                return SLPTestResult(
                    prompt_id=prompt.id,
                    test_arm=TestArm.A_BASELINE,
                    end_to_end_latency_ms=execution_time,
                    total_tokens_generated=token_count,
                    response_text=response_text,
                    timestamp=time.time(),
                    tpv_enabled=False,
                    slp_enabled=False
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            raise Exception(f"Baseline execution failed: {e}")

    def _execute_tpv_only_arm(self, prompt: TestPrompt, start_time: float) -> SLPTestResult:
        """Execute test for Arm B (TPV Only - Phase 2 system)."""
        try:
            from sam.cognition.tpv import sam_tpv_integration, UserProfile

            if not sam_tpv_integration or not sam_tpv_integration.is_initialized:
                raise Exception("TPV integration not available or not initialized")

            # Execute with TPV enabled but SLP disabled
            tpv_response = sam_tpv_integration.generate_response_with_tpv(
                prompt=prompt.text,
                user_profile=UserProfile.GENERAL,
                initial_confidence=0.7,
                context={'test_mode': True},
                ollama_params={
                    "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
            )

            execution_time = (time.time() - start_time) * 1000

            return SLPTestResult(
                prompt_id=prompt.id,
                test_arm=TestArm.B_TPV_ONLY,
                end_to_end_latency_ms=execution_time,
                total_tokens_generated=tpv_response.token_count,
                response_text=tpv_response.content,
                timestamp=time.time(),
                tpv_enabled=True,
                tpv_steps=len(tpv_response.reasoning_steps),
                tpv_halt_reason=tpv_response.halt_reason,
                control_decision=tpv_response.control_decision,
                slp_enabled=False
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            raise Exception(f"TPV-only execution failed: {e}")

    def _execute_slp_active_arm(self, prompt: TestPrompt, start_time: float) -> SLPTestResult:
        """Execute test for Arm C (SLP Active - Full Phase 7 system)."""
        try:
            from sam.cognition.slp import get_slp_integration
            from sam.cognition.tpv import sam_tpv_integration

            slp_integration = get_slp_integration(sam_tpv_integration)

            if not slp_integration:
                raise Exception("SLP integration not available")

            # Ensure SLP is enabled for this test
            slp_integration.enable_slp()

            # Prepare context for SLP
            context = {
                'memory_results': [],
                'sources': [],
                'test_mode': True,
                'prompt_group': prompt.group.value,
                'sequence_order': prompt.sequence_order
            }

            # Define fallback generator that uses TPV
            def tpv_fallback_generator(query, ctx):
                if sam_tpv_integration and sam_tpv_integration.is_initialized:
                    from sam.cognition.tpv import UserProfile
                    tpv_response = sam_tpv_integration.generate_response_with_tpv(
                        prompt=query,
                        user_profile=UserProfile.GENERAL,
                        initial_confidence=0.7,
                        context=ctx,
                        ollama_params={
                            "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "max_tokens": 500
                            }
                        }
                    )
                    return tpv_response.content
                else:
                    # Fallback to basic Ollama
                    import requests
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M",
                            "prompt": query,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "max_tokens": 500
                            }
                        },
                        timeout=60
                    )
                    if response.status_code == 200:
                        return response.json().get('response', '').strip()
                    else:
                        return f"Error: {response.status_code}"

            # Execute with SLP (which includes TPV integration)
            slp_result = slp_integration.generate_response_with_slp(
                query=prompt.text,
                context=context,
                user_profile='test_user',
                fallback_generator=tpv_fallback_generator
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract SLP metadata
            slp_metadata = slp_result.get('slp_metadata', {})

            # Get current SLP statistics
            slp_stats = slp_integration.get_slp_statistics()
            integration_stats = slp_stats.get('integration_stats', {})

            return SLPTestResult(
                prompt_id=prompt.id,
                test_arm=TestArm.C_SLP_ACTIVE,
                end_to_end_latency_ms=execution_time,
                total_tokens_generated=slp_metadata.get('token_count', len(slp_result.get('response', '').split())),
                response_text=slp_result.get('response', ''),
                timestamp=time.time(),
                tpv_enabled=True,
                slp_enabled=True,
                program_used_id=slp_metadata.get('program_id'),
                program_was_captured=slp_metadata.get('captured_program', False),
                signature_match_confidence=slp_metadata.get('signature_match_confidence', 0.0),
                program_execution_time_ms=slp_metadata.get('execution_time_ms', 0.0),
                slp_hit_rate=integration_stats.get('hit_rate_percent', 0.0),
                total_programs_available=len(slp_integration.program_manager.store.get_all_programs())
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            raise Exception(f"SLP Active execution failed: {e}")

    def save_results(self, output_dir: Path) -> Path:
        """Save test results to JSON file."""
        timestamp = int(time.time())
        results_file = output_dir / f"phase7b_slp_validation_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            result_dict['test_arm'] = result.test_arm.value
            serializable_results.append(result_dict)

        # Create comprehensive results structure
        results_data = {
            'metadata': {
                'test_type': 'Phase 7B SLP Validation',
                'timestamp': timestamp,
                'total_tests': len(self.results),
                'test_arms': [arm.value for arm in TestArm],
                'prompt_groups': [group.value for group in PromptGroup],
                'total_prompts': len(self.dataset.prompts)
            },
            'test_configuration': {
                'arms': self.test_arms,
                'dataset_info': {
                    'total_prompts': len(self.dataset.prompts),
                    'groups': {group.value: len(self.dataset.get_prompts_by_group(group)) for group in PromptGroup}
                }
            },
            'results': serializable_results
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"📁 Results saved to: {results_file}")
        return results_file

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the test results."""
        if not self.results:
            return {}

        summary = {
            'total_tests': len(self.results),
            'by_arm': {},
            'by_group': {},
            'slp_performance': {}
        }

        # Statistics by test arm
        for arm in TestArm:
            arm_results = [r for r in self.results if r.test_arm == arm]
            if arm_results:
                avg_latency = sum(r.end_to_end_latency_ms for r in arm_results) / len(arm_results)
                avg_tokens = sum(r.total_tokens_generated for r in arm_results) / len(arm_results)
                error_count = sum(1 for r in arm_results if r.error)

                summary['by_arm'][arm.value] = {
                    'test_count': len(arm_results),
                    'avg_latency_ms': avg_latency,
                    'avg_tokens': avg_tokens,
                    'error_count': error_count,
                    'success_rate': (len(arm_results) - error_count) / len(arm_results) * 100
                }

        # SLP-specific performance metrics
        slp_results = [r for r in self.results if r.test_arm == TestArm.C_SLP_ACTIVE and not r.error]
        if slp_results:
            program_uses = sum(1 for r in slp_results if r.program_used_id)
            program_captures = sum(1 for r in slp_results if r.program_was_captured)

            summary['slp_performance'] = {
                'total_slp_tests': len(slp_results),
                'program_uses': program_uses,
                'program_captures': program_captures,
                'hit_rate': (program_uses / len(slp_results)) * 100 if slp_results else 0,
                'capture_rate': (program_captures / len(slp_results)) * 100 if slp_results else 0
            }

        return summary
