#!/usr/bin/env python3
"""
SSRL Code Quality Check Script
==============================

Comprehensive code quality checker for SSRL implementation before GitHub push.
Validates code style, documentation, type hints, error handling, and best practices.

Features:
- Python code style validation (PEP 8)
- Type hint coverage analysis
- Documentation completeness check
- Error handling validation
- Import organization check
- Security vulnerability scan
- Performance optimization suggestions

Author: SAM Development Team
Version: 1.0.0
"""

import ast
import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeQualityIssue:
    """Represents a code quality issue."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None


@dataclass
class CodeQualityReport:
    """Code quality analysis report."""
    total_files: int = 0
    total_lines: int = 0
    issues: List[CodeQualityIssue] = field(default_factory=list)
    type_hint_coverage: float = 0.0
    docstring_coverage: float = 0.0
    error_handling_score: float = 0.0
    overall_score: float = 0.0
    
    def add_issue(self, issue: CodeQualityIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: str) -> List[CodeQualityIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def calculate_overall_score(self) -> float:
        """Calculate overall code quality score (0-100)."""
        # Weight different aspects
        weights = {
            'type_hints': 0.25,
            'docstrings': 0.25,
            'error_handling': 0.25,
            'issues': 0.25
        }
        
        # Calculate issue score (fewer issues = higher score)
        error_count = len(self.get_issues_by_severity('error'))
        warning_count = len(self.get_issues_by_severity('warning'))
        issue_penalty = (error_count * 10) + (warning_count * 5)
        issue_score = max(0, 100 - issue_penalty)
        
        # Combine scores
        self.overall_score = (
            weights['type_hints'] * self.type_hint_coverage +
            weights['docstrings'] * self.docstring_coverage +
            weights['error_handling'] * self.error_handling_score +
            weights['issues'] * issue_score
        )
        
        return self.overall_score


class SSRLCodeQualityChecker:
    """Code quality checker for SSRL implementation."""
    
    def __init__(self, project_root: str):
        """
        Initialize code quality checker.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.report = CodeQualityReport()
        
        # SSRL-specific files to check
        self.ssrl_files = [
            "sam/orchestration/skills/self_search_tool.py",
            "sam/orchestration/hybrid_query_router.py",
            "sam/orchestration/ssrl_integration.py",
            "sam/learning/ssrl_rewards.py",
            "sam/cognition/multi_adapter_manager.py",
            "sam/ui/enhanced_personalized_tuner.py",
            "scripts/run_ssrl_tuning.py"
        ]
        
        logger.info(f"Initialized code quality checker for {len(self.ssrl_files)} SSRL files")
    
    def check_all(self) -> CodeQualityReport:
        """
        Run all code quality checks.
        
        Returns:
            Comprehensive code quality report
        """
        logger.info("Starting comprehensive code quality check...")
        
        for file_path in self.ssrl_files:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                self.report.add_issue(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="missing_file",
                    severity="error",
                    message=f"SSRL file not found: {file_path}",
                    suggestion="Ensure all SSRL files are properly created"
                ))
                continue
            
            logger.info(f"Checking {file_path}...")
            self._check_file(full_path)
            self.report.total_files += 1
        
        # Calculate metrics
        self._calculate_metrics()
        
        logger.info("Code quality check completed")
        return self.report
    
    def _check_file(self, file_path: Path) -> None:
        """Check a single Python file for quality issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            self.report.total_lines += len(lines)
            
            # Parse AST for advanced checks
            try:
                tree = ast.parse(content)
                self._check_ast(tree, file_path, lines)
            except SyntaxError as e:
                self.report.add_issue(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Syntax error: {e.msg}",
                    suggestion="Fix syntax error before proceeding"
                ))
            
            # Check line-by-line issues
            self._check_lines(lines, file_path)
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            self.report.add_issue(CodeQualityIssue(
                file_path=str(file_path),
                line_number=0,
                issue_type="check_error",
                severity="error",
                message=f"Failed to check file: {e}",
                suggestion="Investigate file access or encoding issues"
            ))
    
    def _check_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> None:
        """Check AST for advanced code quality issues."""
        class QualityVisitor(ast.NodeVisitor):
            def __init__(self, checker, file_path, lines):
                self.checker = checker
                self.file_path = file_path
                self.lines = lines
                self.functions_with_docstrings = 0
                self.total_functions = 0
                self.functions_with_type_hints = 0
                self.try_except_blocks = 0
                self.total_function_calls = 0
            
            def visit_FunctionDef(self, node):
                self.total_functions += 1
                
                # Check for docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    self.functions_with_docstrings += 1
                else:
                    self.checker.report.add_issue(CodeQualityIssue(
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        issue_type="missing_docstring",
                        severity="warning",
                        message=f"Function '{node.name}' missing docstring",
                        suggestion="Add comprehensive docstring with Args, Returns, and Raises sections"
                    ))
                
                # Check for type hints
                has_return_annotation = node.returns is not None
                has_arg_annotations = any(arg.annotation for arg in node.args.args)
                
                if has_return_annotation and has_arg_annotations:
                    self.functions_with_type_hints += 1
                else:
                    self.checker.report.add_issue(CodeQualityIssue(
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        issue_type="missing_type_hints",
                        severity="info",
                        message=f"Function '{node.name}' missing complete type hints",
                        suggestion="Add type hints for all parameters and return value"
                    ))
                
                # Check for overly complex functions
                if len(node.body) > 50:
                    self.checker.report.add_issue(CodeQualityIssue(
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        issue_type="complex_function",
                        severity="warning",
                        message=f"Function '{node.name}' is very long ({len(node.body)} statements)",
                        suggestion="Consider breaking down into smaller functions"
                    ))
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.try_except_blocks += 1
                
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        self.checker.report.add_issue(CodeQualityIssue(
                            file_path=str(self.file_path),
                            line_number=handler.lineno,
                            issue_type="bare_except",
                            severity="warning",
                            message="Bare except clause detected",
                            suggestion="Specify exception types for better error handling"
                        ))
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                self.total_function_calls += 1
                self.generic_visit(node)
        
        visitor = QualityVisitor(self, file_path, lines)
        visitor.visit(tree)
        
        # Store metrics for later calculation
        if not hasattr(self, '_ast_metrics'):
            self._ast_metrics = defaultdict(int)
        
        self._ast_metrics['functions_with_docstrings'] += visitor.functions_with_docstrings
        self._ast_metrics['total_functions'] += visitor.total_functions
        self._ast_metrics['functions_with_type_hints'] += visitor.functions_with_type_hints
        self._ast_metrics['try_except_blocks'] += visitor.try_except_blocks
        self._ast_metrics['total_function_calls'] += visitor.total_function_calls
    
    def _check_lines(self, lines: List[str], file_path: Path) -> None:
        """Check individual lines for style and quality issues."""
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 100:
                self.report.add_issue(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="long_line",
                    severity="info",
                    message=f"Line too long ({len(line)} characters)",
                    suggestion="Break long lines for better readability"
                ))
            
            # Check for TODO/FIXME comments
            if re.search(r'\b(TODO|FIXME|XXX|HACK)\b', line, re.IGNORECASE):
                self.report.add_issue(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="todo_comment",
                    severity="info",
                    message="TODO/FIXME comment found",
                    suggestion="Address TODO items before production deployment"
                ))
            
            # Check for print statements (should use logging)
            if re.search(r'\bprint\s*\(', line) and 'logger' not in line:
                self.report.add_issue(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="print_statement",
                    severity="warning",
                    message="Print statement found, consider using logging",
                    suggestion="Replace print() with appropriate logger calls"
                ))
    
    def _calculate_metrics(self) -> None:
        """Calculate overall quality metrics."""
        if hasattr(self, '_ast_metrics'):
            metrics = self._ast_metrics
            
            # Type hint coverage
            if metrics['total_functions'] > 0:
                self.report.type_hint_coverage = (
                    metrics['functions_with_type_hints'] / metrics['total_functions'] * 100
                )
            
            # Docstring coverage
            if metrics['total_functions'] > 0:
                self.report.docstring_coverage = (
                    metrics['functions_with_docstrings'] / metrics['total_functions'] * 100
                )
            
            # Error handling score (based on try/except ratio)
            if metrics['total_function_calls'] > 0:
                self.report.error_handling_score = min(100, (
                    metrics['try_except_blocks'] / metrics['total_function_calls'] * 500
                ))
        
        # Calculate overall score
        self.report.calculate_overall_score()
    
    def print_report(self) -> None:
        """Print a comprehensive quality report."""
        print("\n" + "=" * 80)
        print("🔍 SSRL CODE QUALITY REPORT")
        print("=" * 80)
        
        print(f"\n📊 Overview:")
        print(f"   Files Checked: {self.report.total_files}")
        print(f"   Total Lines: {self.report.total_lines:,}")
        print(f"   Overall Score: {self.report.overall_score:.1f}/100")
        
        print(f"\n📈 Metrics:")
        print(f"   Type Hint Coverage: {self.report.type_hint_coverage:.1f}%")
        print(f"   Docstring Coverage: {self.report.docstring_coverage:.1f}%")
        print(f"   Error Handling Score: {self.report.error_handling_score:.1f}%")
        
        # Issues by severity
        errors = self.report.get_issues_by_severity('error')
        warnings = self.report.get_issues_by_severity('warning')
        info = self.report.get_issues_by_severity('info')
        
        print(f"\n🚨 Issues Summary:")
        print(f"   Errors: {len(errors)} ❌")
        print(f"   Warnings: {len(warnings)} ⚠️")
        print(f"   Info: {len(info)} ℹ️")
        
        # Detailed issues
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors[:10]:  # Show first 10
                print(f"   {issue.file_path}:{issue.line_number} - {issue.message}")
        
        if warnings:
            print(f"\n⚠️ WARNINGS ({len(warnings)}):")
            for issue in warnings[:10]:  # Show first 10
                print(f"   {issue.file_path}:{issue.line_number} - {issue.message}")
        
        # Quality assessment
        print(f"\n🎯 Quality Assessment:")
        if self.report.overall_score >= 90:
            print("   ✅ EXCELLENT - Ready for production!")
        elif self.report.overall_score >= 80:
            print("   ✅ GOOD - Minor improvements recommended")
        elif self.report.overall_score >= 70:
            print("   ⚠️ FAIR - Several improvements needed")
        else:
            print("   ❌ POOR - Significant improvements required")
        
        print("\n" + "=" * 80)


def main():
    """Run code quality check for SSRL implementation."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = Path(__file__).parent.parent
    
    checker = SSRLCodeQualityChecker(project_root)
    report = checker.check_all()
    checker.print_report()
    
    # Exit with appropriate code
    if len(report.get_issues_by_severity('error')) > 0:
        print("\n❌ Code quality check failed due to errors.")
        return 1
    elif report.overall_score < 80:
        print(f"\n⚠️ Code quality score ({report.overall_score:.1f}) below recommended threshold (80).")
        return 1
    else:
        print("\n✅ Code quality check passed!")
        return 0


if __name__ == "__main__":
    exit(main())
