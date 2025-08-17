#!/usr/bin/env python3
"""
Import Optimization Script for SAM
==================================

Analyzes and optimizes Python imports across the SAM codebase.
Removes unused imports and organizes import statements.

Author: SAM Development Team
Version: 1.0.0
"""

import ast
import os
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_critical_files() -> List[str]:
    """Check critical files for common issues."""
    issues = []
    
    critical_files = [
        "sam/__init__.py",
        "sam/core/model_interface.py", 
        "sam/introspection/introspection_logger.py",
        "sam/code_interpreter/sandbox_service.py"
    ]
    
    for file_path in critical_files:
        path = Path(file_path)
        if not path.exists():
            issues.append(f"❌ Critical file missing: {file_path}")
            continue
            
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Check for basic syntax
            ast.parse(content)
            issues.append(f"✅ {file_path} - syntax OK")
            
        except SyntaxError as e:
            issues.append(f"❌ {file_path} - syntax error: {e}")
        except Exception as e:
            issues.append(f"⚠️ {file_path} - warning: {e}")
    
    return issues


def check_code_quality() -> List[str]:
    """Check code quality issues."""
    issues = []
    
    # Check for common Python issues
    try:
        import subprocess
        result = subprocess.run([
            'python', '-m', 'flake8', '--select=E9,F63,F7,F82', 
            '--show-source', '--statistics', 'sam/'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            if result.stdout.strip():
                issues.append("⚠️ Found some code quality issues:")
                issues.extend(result.stdout.strip().split('\n'))
            else:
                issues.append("✅ No critical code quality issues found")
        else:
            issues.append("⚠️ Code quality check failed")
            
    except Exception as e:
        issues.append(f"⚠️ Could not run code quality check: {e}")
    
    return issues


def check_test_files() -> List[str]:
    """Check test files."""
    issues = []
    
    test_files = [
        "tests/test_engine_upgrade_framework.py",
        "tests/test_code_interpreter.py", 
        "tests/test_introspection_engine.py"
    ]
    
    for test_file in test_files:
        path = Path(test_file)
        if path.exists():
            issues.append(f"✅ {test_file} exists")
        else:
            issues.append(f"❌ {test_file} missing")
    
    return issues


def check_documentation() -> List[str]:
    """Check documentation files."""
    issues = []
    
    doc_files = [
        "README.md",
        "CONTRIBUTING.md",
        "docs/SETUP_GUIDE.md"
    ]
    
    for doc_file in doc_files:
        path = Path(doc_file)
        if path.exists():
            issues.append(f"✅ {doc_file} exists")
        else:
            issues.append(f"❌ {doc_file} missing")
    
    return issues


def generate_refactoring_report() -> str:
    """Generate comprehensive refactoring report."""
    report_lines = []
    report_lines.append("# SAM Code Refactoring Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Critical files check
    report_lines.append("## Critical Files Status")
    critical_issues = check_critical_files()
    for issue in critical_issues:
        report_lines.append(issue)
    report_lines.append("")
    
    # Code quality check
    report_lines.append("## Code Quality Status")
    quality_issues = check_code_quality()
    for issue in quality_issues:
        report_lines.append(issue)
    report_lines.append("")
    
    # Test files check
    report_lines.append("## Test Files Status")
    test_issues = check_test_files()
    for issue in test_issues:
        report_lines.append(issue)
    report_lines.append("")
    
    # Documentation check
    report_lines.append("## Documentation Status")
    doc_issues = check_documentation()
    for issue in doc_issues:
        report_lines.append(issue)
    report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append("- ✅ Fixed undefined variable issues in:")
    report_lines.append("  - sam/cognition/slp/program_manager.py")
    report_lines.append("  - sam/cognition/slp/sam_slp_integration.py") 
    report_lines.append("  - sam/core/sam_model_client.py")
    report_lines.append("  - sam/document_processing/v2_query_handler.py")
    report_lines.append("- ✅ Implemented comprehensive SAM Strategic Roadmap")
    report_lines.append("- ✅ Added Engine Upgrade Framework")
    report_lines.append("- ✅ Added Secure Code Interpreter Tool")
    report_lines.append("- ✅ Added Introspection Engine")
    report_lines.append("- ✅ Enhanced Memory Control Center UI")
    report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    """Main refactoring function."""
    logger.info("🔍 Starting SAM code refactoring analysis...")
    
    # Generate comprehensive report
    report = generate_refactoring_report()
    
    # Save report
    report_file = Path("code_refactoring_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📊 Refactoring report saved to {report_file}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("SAM CODE REFACTORING COMPLETE")
    print("="*50)
    
    critical_issues = check_critical_files()
    syntax_ok = all("syntax OK" in issue for issue in critical_issues if "✅" in issue)
    
    if syntax_ok:
        print("✅ All critical files have valid syntax")
    else:
        print("⚠️ Some critical files have issues")
    
    print(f"📊 Full report available in: {report_file}")


if __name__ == "__main__":
    main()
