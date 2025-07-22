#!/usr/bin/env python3
"""
Deep-Dive Diagnosis for Table-to-Code Expert Tool
=================================================

Performs detailed root cause analysis on failing test cases with extensive logging
to identify exact failure points and implement targeted fixes.

This implements Task 1 of Phase 2B: Reliability & Hardening.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import traceback
from datetime import datetime

# Add SAM modules to path
sys.path.append(str(Path(__file__).parent.parent))

from sam.orchestration.skills.table_to_code_expert import TableToCodeExpert, AnalysisRequest
from sam.orchestration.uif import SAM_UIF, UIFStatus
from sam.cognition.table_processing.sam_integration import get_table_aware_retrieval
from memory.memory_vectorstore import get_memory_store, MemoryType

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deep_diagnosis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepDiagnosisAnalyzer:
    """Deep diagnosis analyzer for Table-to-Code Expert failures."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.expert = TableToCodeExpert()
        self.memory_store = get_memory_store()
        self.diagnosis_results = {}
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
    def run_deep_diagnosis(self) -> Dict[str, Any]:
        """Run comprehensive diagnosis on failing test cases."""
        logger.info("🔍 Starting Deep-Dive Diagnosis for Table-to-Code Expert")
        
        # Set up test data with known characteristics
        self.setup_controlled_test_data()
        
        # Diagnose the two failing test scenarios
        self.diagnose_code_generation_failures()
        self.diagnose_end_to_end_workflow_failures()
        
        # Generate comprehensive diagnosis report
        self.generate_diagnosis_report()
        
        return self.diagnosis_results
    
    def setup_controlled_test_data(self):
        """Set up controlled test data with known characteristics."""
        logger.info("🔧 Setting up controlled test data")
        
        # Test data with various edge cases
        test_sales_data = [
            {'Product': 'Software Licenses', 'Q1_Sales': 2500000, 'Q2_Sales': 2800000, 'Q3_Sales': 3100000, 'Growth': 24.0},
            {'Product': 'Hardware Sales', 'Q1_Sales': 1800000, 'Q2_Sales': 2000000, 'Q3_Sales': 2200000, 'Growth': 22.2},
            {'Product': 'Professional Services', 'Q1_Sales': 1200000, 'Q2_Sales': 1400000, 'Q3_Sales': 1600000, 'Growth': 33.3},
            {'Product': 'Support Contracts', 'Q1_Sales': 800000, 'Q2_Sales': 900000, 'Q3_Sales': 1000000, 'Growth': 25.0},
            {'Product': 'Cloud Services', 'Q1_Sales': 0, 'Q2_Sales': 500000, 'Q3_Sales': 750000, 'Growth': float('inf')}  # Edge case: division by zero
        ]
        
        # Store with detailed metadata
        table_id = "diagnosis_sales_data"
        stored_chunks = []
        
        for row_idx, row_data in enumerate(test_sales_data):
            for col_idx, (header, value) in enumerate(row_data.items()):
                chunk_metadata = {
                    'content': str(value),
                    'is_table_part': True,
                    'table_id': table_id,
                    'table_title': 'Sales Performance Analysis - Diagnosis Data',
                    'cell_role': 'HEADER' if row_idx == 0 and col_idx == 0 else 'DATA',
                    'cell_coordinates': (row_idx, col_idx),
                    'cell_data_type': 'text' if col_idx == 0 else 'currency' if 'Sales' in header else 'percentage',
                    'confidence_score': 0.95
                }
                
                chunk_id = self.memory_store.add_memory(
                    content=str(value),
                    memory_type=MemoryType.DOCUMENT,
                    source="diagnosis_sales_data.csv",
                    tags=['table', 'sales', 'diagnosis', 'test'],
                    importance_score=0.9,
                    metadata=chunk_metadata
                )
                stored_chunks.append(chunk_id)
        
        logger.info(f"✅ Stored {len(stored_chunks)} test chunks for diagnosis")
        return table_id
    
    def diagnose_code_generation_failures(self):
        """Diagnose Test 4: Code Generation Types failures."""
        logger.info("🔬 DIAGNOSIS 1: Code Generation Failures")
        
        diagnosis = {
            'test_name': 'Code Generation Types',
            'failure_points': [],
            'root_causes': [],
            'data_issues': [],
            'code_issues': [],
            'execution_issues': []
        }
        
        # Initialize expert
        if not self.expert.table_retrieval:
            self.expert._initialize_table_retrieval(SAM_UIF(input_query="diagnosis"))
        
        # Test different analysis types with detailed logging
        test_cases = [
            {
                'name': 'Visualization Generation',
                'request': AnalysisRequest(
                    intent='visualize',
                    table_query='sales chart',
                    specific_columns=['Product', 'Q3_Sales'],
                    operation='summary',
                    visualization_type='bar',
                    filters={}
                )
            },
            {
                'name': 'Calculation Generation',
                'request': AnalysisRequest(
                    intent='calculate',
                    table_query='total sales',
                    specific_columns=['Q3_Sales'],
                    operation='sum',
                    visualization_type=None,
                    filters={}
                )
            },
            {
                'name': 'Analysis Generation',
                'request': AnalysisRequest(
                    intent='analyze',
                    table_query='comprehensive analysis',
                    specific_columns=[],
                    operation='summary',
                    visualization_type=None,
                    filters={}
                )
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"🧪 Testing: {test_case['name']}")
            
            try:
                # Step 1: Log the analysis request
                logger.debug(f"Analysis Request: {test_case['request'].__dict__}")
                
                # Step 2: Test table retrieval
                relevant_tables = self.expert._find_relevant_tables(test_case['request'])
                logger.debug(f"Relevant tables found: {relevant_tables}")
                
                if not relevant_tables:
                    diagnosis['failure_points'].append(f"{test_case['name']}: No relevant tables found")
                    continue
                
                # Step 3: Test table data retrieval
                table_data = self.expert.table_retrieval.get_table_data_for_analysis(relevant_tables[0])
                logger.debug(f"Table data retrieved: {table_data is not None}")
                
                if table_data:
                    logger.debug(f"Table structure: {table_data.get('headers', [])}")
                    logger.debug(f"Data rows: {len(table_data.get('data', []))}")
                    logger.debug(f"Sample data: {table_data.get('data', [])[:2]}")
                else:
                    diagnosis['data_issues'].append(f"{test_case['name']}: Failed to retrieve table data")
                    continue
                
                # Step 4: Test code generation
                code_result = self.expert._generate_analysis_code(test_case['request'], relevant_tables)
                logger.debug(f"Code generation success: {code_result.success}")
                
                if code_result.success:
                    logger.debug(f"Generated code length: {len(code_result.code)} characters")
                    logger.debug(f"Code preview: {code_result.code[:200]}...")
                    
                    # Step 5: Test code execution
                    if len(code_result.code) > 0:
                        execution_result = self.expert._execute_code_safely(code_result.code)
                        logger.debug(f"Execution result: {execution_result[:200] if execution_result else 'None'}...")
                        
                        if "Error:" in execution_result:
                            diagnosis['execution_issues'].append({
                                'test': test_case['name'],
                                'error': execution_result,
                                'code': code_result.code
                            })
                else:
                    diagnosis['code_issues'].append({
                        'test': test_case['name'],
                        'error': code_result.error_message,
                        'explanation': code_result.explanation
                    })
                
            except Exception as e:
                logger.error(f"Exception in {test_case['name']}: {e}")
                diagnosis['failure_points'].append({
                    'test': test_case['name'],
                    'exception': str(e),
                    'traceback': traceback.format_exc()
                })
        
        # Analyze patterns in failures
        if diagnosis['execution_issues']:
            diagnosis['root_causes'].append("Code execution failures detected - likely DataFrame variable scoping or library import issues")
        
        if diagnosis['data_issues']:
            diagnosis['root_causes'].append("Table data retrieval failures - possible memory integration issues")
        
        if diagnosis['code_issues']:
            diagnosis['root_causes'].append("Code generation logic failures - template or prompt issues")
        
        self.diagnosis_results['code_generation_diagnosis'] = diagnosis
        logger.info(f"🔬 Code Generation Diagnosis Complete: {len(diagnosis['failure_points'])} failure points identified")
    
    def diagnose_end_to_end_workflow_failures(self):
        """Diagnose Test 6: End-to-End Workflow failures."""
        logger.info("🔬 DIAGNOSIS 2: End-to-End Workflow Failures")
        
        diagnosis = {
            'test_name': 'End-to-End Workflow',
            'failure_points': [],
            'root_causes': [],
            'uif_issues': [],
            'integration_issues': [],
            'execution_flow': []
        }
        
        try:
            # Step 1: Create UIF and log its state
            uif = SAM_UIF(input_query="Show me a comprehensive analysis of the sales data")
            uif.intermediate_data["execute_code"] = False
            
            logger.debug(f"UIF created - Input: {uif.input_query}")
            logger.debug(f"UIF status: {uif.status}")
            logger.debug(f"UIF data keys: {list(uif.intermediate_data.keys())}")
            
            diagnosis['execution_flow'].append("UIF created successfully")
            
            # Step 2: Test skill execution step by step
            logger.debug("Testing skill execution...")
            
            # Initialize table retrieval
            if not self.expert.table_retrieval:
                self.expert._initialize_table_retrieval(uif)
                diagnosis['execution_flow'].append("Table retrieval initialized")
            
            # Step 3: Test user request parsing
            user_request = uif.input_query.strip()
            analysis_request = self.expert._parse_user_request(user_request)
            logger.debug(f"Parsed request: {analysis_request.__dict__}")
            diagnosis['execution_flow'].append(f"Request parsed - Intent: {analysis_request.intent}")
            
            # Step 4: Test table finding
            relevant_tables = self.expert._find_relevant_tables(analysis_request)
            logger.debug(f"Relevant tables: {relevant_tables}")
            
            if not relevant_tables:
                diagnosis['failure_points'].append("No relevant tables found in end-to-end test")
                diagnosis['root_causes'].append("Table search functionality not finding test data")
            else:
                diagnosis['execution_flow'].append(f"Found {len(relevant_tables)} relevant tables")
            
            # Step 5: Test full execution
            result_uif = self.expert.execute(uif)
            logger.debug(f"Execution result status: {result_uif.status}")
            logger.debug(f"Result UIF data keys: {list(result_uif.intermediate_data.keys())}")
            
            if result_uif.status == UIFStatus.SUCCESS:
                diagnosis['execution_flow'].append("Execution completed successfully")
                
                # Check expected outputs
                if "generated_code" not in result_uif.intermediate_data:
                    diagnosis['uif_issues'].append("Missing 'generated_code' in result UIF")
                
                if "analysis_result" not in result_uif.intermediate_data:
                    diagnosis['uif_issues'].append("Missing 'analysis_result' in result UIF")
                    
            else:
                diagnosis['failure_points'].append(f"Execution failed with status: {result_uif.status}")
                if hasattr(result_uif, 'error_details'):
                    diagnosis['failure_points'].append(f"Error details: {result_uif.error_details}")
        
        except Exception as e:
            logger.error(f"Exception in end-to-end diagnosis: {e}")
            diagnosis['failure_points'].append({
                'exception': str(e),
                'traceback': traceback.format_exc()
            })
        
        # Analyze patterns
        if diagnosis['uif_issues']:
            diagnosis['root_causes'].append("UIF data structure issues - missing expected output keys")
        
        if diagnosis['integration_issues']:
            diagnosis['root_causes'].append("Integration issues between components")
        
        self.diagnosis_results['end_to_end_diagnosis'] = diagnosis
        logger.info(f"🔬 End-to-End Diagnosis Complete: {len(diagnosis['failure_points'])} failure points identified")
    
    def generate_diagnosis_report(self):
        """Generate comprehensive diagnosis report."""
        logger.info("📊 Generating Comprehensive Diagnosis Report")
        
        report = {
            'diagnosis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_failure_points': 0,
                'critical_issues': [],
                'recommended_fixes': []
            },
            'detailed_findings': self.diagnosis_results
        }
        
        # Count total failure points
        for diagnosis in self.diagnosis_results.values():
            report['summary']['total_failure_points'] += len(diagnosis.get('failure_points', []))
        
        # Identify critical issues
        code_gen_issues = self.diagnosis_results.get('code_generation_diagnosis', {})
        if code_gen_issues.get('execution_issues'):
            report['summary']['critical_issues'].append("Code execution failures - DataFrame scoping issues")
            report['summary']['recommended_fixes'].append("Enhance code templates with proper variable scoping")
        
        if code_gen_issues.get('data_issues'):
            report['summary']['critical_issues'].append("Table data retrieval failures")
            report['summary']['recommended_fixes'].append("Fix table reconstruction and data conversion logic")
        
        e2e_issues = self.diagnosis_results.get('end_to_end_diagnosis', {})
        if e2e_issues.get('uif_issues'):
            report['summary']['critical_issues'].append("UIF integration issues")
            report['summary']['recommended_fixes'].append("Fix UIF data structure handling")
        
        # Save detailed report
        report_file = Path("logs/deep_diagnosis_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Diagnosis report saved to: {report_file}")
        
        # Print summary
        print(f"\n🔍 DEEP DIAGNOSIS SUMMARY")
        print(f"=" * 50)
        print(f"Total Failure Points: {report['summary']['total_failure_points']}")
        print(f"Critical Issues: {len(report['summary']['critical_issues'])}")
        
        for issue in report['summary']['critical_issues']:
            print(f"  ❌ {issue}")
        
        print(f"\nRecommended Fixes:")
        for fix in report['summary']['recommended_fixes']:
            print(f"  🔧 {fix}")
        
        self.diagnosis_results['report'] = report


def main():
    """Main diagnosis function."""
    analyzer = DeepDiagnosisAnalyzer()
    results = analyzer.run_deep_diagnosis()
    
    print(f"\n✅ Deep diagnosis complete. Check logs/deep_diagnosis_report.json for details.")
    return 0


if __name__ == "__main__":
    exit(main())
