"""
ContentVettingSkill - Security Analysis and Content Validation
==============================================================

Provides comprehensive security analysis for external content including
credibility assessment, bias detection, and content purity validation.
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from ..uif import SAM_UIF
from .base import BaseSkillModule, SkillExecutionError

logger = logging.getLogger(__name__)


class ContentVettingSkill(BaseSkillModule):
    """
    Security analysis skill for content validation and vetting.
    
    Performs four-dimension security assessment:
    1. Credibility & Bias Analysis
    2. Persuasive/Manipulative Language Detection
    3. Speculation vs Fact Classification
    4. Content Purity & Injection Detection
    """
    
    skill_name = "ContentVettingSkill"
    skill_version = "1.0.0"
    skill_description = "Performs comprehensive security analysis and content validation"
    skill_category = "security"
    
    # Dependency declarations
    required_inputs = ["external_content"]
    optional_inputs = ["vetting_context", "security_level"]
    output_keys = ["vetting_report", "security_score", "approved_content"]
    
    # Skill characteristics
    requires_external_access = False
    requires_vetting = False  # This skill does the vetting
    can_run_parallel = True
    estimated_execution_time = 2.0
    max_execution_time = 10.0
    
    def __init__(self):
        super().__init__()
        self._bias_indicators = self._load_bias_indicators()
        self._manipulation_patterns = self._load_manipulation_patterns()
        self._fact_indicators = self._load_fact_indicators()
        self._injection_patterns = self._load_injection_patterns()
    
    def execute(self, uif: SAM_UIF) -> SAM_UIF:
        """
        Execute comprehensive content vetting and security analysis.

        Args:
            uif: Universal Interface Format with external content

        Returns:
            Updated UIF with vetting results
        """
        # Initialize tracing
        trace_id = uif.intermediate_data.get('trace_id')
        start_time = time.time()

        if trace_id:
            self._log_trace_event(
                trace_id=trace_id,
                event_type="start",
                severity="info",
                message="Starting comprehensive content vetting and security analysis",
                payload={
                    "tool": self.skill_name,
                    "requires_vetting": self.requires_vetting,
                    "skill_category": self.skill_category
                }
            )

        try:
            # Extract content for vetting
            external_content = self._extract_external_content(uif)
            if not external_content:
                if trace_id:
                    self._log_trace_event(
                        trace_id=trace_id,
                        event_type="error",
                        severity="error",
                        message="No external content found for vetting",
                        payload={"uif_keys": list(uif.intermediate_data.keys())}
                    )
                raise SkillExecutionError("No external content found for vetting")

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_in",
                    severity="info",
                    message=f"Extracted {len(external_content)} content items for vetting",
                    payload={
                        "content_count": len(external_content),
                        "content_types": [item.get("type", "unknown") for item in external_content],
                        "content_sources": [item.get("source", "unknown") for item in external_content]
                    }
                )

            self.logger.info(f"Vetting {len(external_content)} content items")

            # Get vetting parameters
            vetting_context = uif.intermediate_data.get("vetting_context", {})
            security_level = uif.intermediate_data.get("security_level", "standard")

            # Perform comprehensive vetting
            vetting_start_time = time.time()
            vetting_report = self._perform_comprehensive_vetting(
                external_content, vetting_context, security_level
            )
            vetting_duration = (time.time() - vetting_start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="tool_call",
                    severity="info",
                    message=f"Content vetting analysis completed: {len(vetting_report['items'])} items analyzed",
                    duration_ms=vetting_duration,
                    payload={
                        "items_analyzed": len(vetting_report["items"]),
                        "security_level": security_level,
                        "risk_summary": vetting_report["risk_summary"],
                        "aggregate_metrics": vetting_report["aggregate_metrics"],
                        "vetting_duration_ms": vetting_duration
                    }
                )

            # Calculate overall security score
            security_score = self._calculate_security_score(vetting_report)

            # Determine approved content
            approved_content = self._filter_approved_content(external_content, vetting_report)

            # Store results in UIF
            uif.intermediate_data["vetting_report"] = vetting_report
            uif.intermediate_data["security_score"] = security_score
            uif.intermediate_data["approved_content"] = approved_content

            # Update security context
            uif.security_context.update({
                "vetting_completed": True,
                "security_score": security_score,
                "content_approved": len(approved_content),
                "content_rejected": len(external_content) - len(approved_content)
            })

            # Set skill outputs
            uif.set_skill_output(self.skill_name, {
                "items_vetted": len(external_content),
                "items_approved": len(approved_content),
                "security_score": security_score,
                "high_risk_items": len([item for item in vetting_report["items"] if item["risk_level"] == "high"])
            })

            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="data_out",
                    severity="info",
                    message=f"Content vetting completed: {len(approved_content)}/{len(external_content)} items approved",
                    duration_ms=total_duration,
                    payload={
                        "items_vetted": len(external_content),
                        "items_approved": len(approved_content),
                        "items_rejected": len(external_content) - len(approved_content),
                        "security_score": security_score,
                        "high_risk_items": len([item for item in vetting_report["items"] if item["risk_level"] == "high"]),
                        "execution_time_ms": total_duration,
                        "vetting_time_ms": vetting_duration,
                        "overhead_ms": total_duration - vetting_duration
                    }
                )

            self.logger.info(f"Vetting completed: {len(approved_content)}/{len(external_content)} items approved")

            return uif

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000

            if trace_id:
                self._log_trace_event(
                    trace_id=trace_id,
                    event_type="error",
                    severity="error",
                    message=f"Content vetting failed: {str(e)}",
                    duration_ms=total_duration,
                    payload={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": total_duration
                    }
                )

            self.logger.exception("Error during content vetting")
            raise SkillExecutionError(f"Content vetting failed: {str(e)}")
    
    def _extract_external_content(self, uif: SAM_UIF) -> List[Dict[str, Any]]:
        """
        Extract external content from UIF for vetting.
        
        Returns:
            List of content items to vet
        """
        content_items = []
        
        # Check intermediate data for external content
        if "external_content" in uif.intermediate_data:
            external_content = uif.intermediate_data["external_content"]
            if isinstance(external_content, list):
                content_items.extend(external_content)
            elif isinstance(external_content, dict):
                content_items.append(external_content)
        
        # Check for web search results
        if "web_search_results" in uif.intermediate_data:
            web_results = uif.intermediate_data["web_search_results"]
            for result in web_results:
                content_items.append({
                    "type": "web_search_result",
                    "source": result.get("url", "unknown"),
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),
                    "metadata": result
                })
        
        # Check for extracted web content
        if "extracted_content" in uif.intermediate_data:
            extracted_content = uif.intermediate_data["extracted_content"]
            for content in extracted_content:
                content_items.append({
                    "type": "extracted_web_content",
                    "source": content.get("url", "unknown"),
                    "title": content.get("title", ""),
                    "content": content.get("content", ""),
                    "metadata": content
                })
        
        return content_items
    
    def _perform_comprehensive_vetting(self, content_items: List[Dict[str, Any]], 
                                     context: Dict[str, Any], security_level: str) -> Dict[str, Any]:
        """
        Perform comprehensive four-dimension security analysis.
        
        Returns:
            Comprehensive vetting report
        """
        vetting_report = {
            "timestamp": datetime.now().isoformat(),
            "security_level": security_level,
            "total_items": len(content_items),
            "items": [],
            "aggregate_metrics": {
                "credibility_score": 0.0,
                "bias_score": 0.0,
                "fact_score": 0.0,
                "purity_score": 0.0
            },
            "risk_summary": {
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0
            }
        }
        
        total_credibility = 0.0
        total_bias = 0.0
        total_fact = 0.0
        total_purity = 0.0
        
        for item in content_items:
            # Perform four-dimension analysis
            item_analysis = self._analyze_content_item(item, security_level)
            vetting_report["items"].append(item_analysis)
            
            # Accumulate scores
            total_credibility += item_analysis["credibility_score"]
            total_bias += item_analysis["bias_score"]
            total_fact += item_analysis["fact_score"]
            total_purity += item_analysis["purity_score"]
            
            # Count risk levels
            risk_level = item_analysis["risk_level"]
            vetting_report["risk_summary"][f"{risk_level}_risk"] += 1
        
        # Calculate aggregate metrics
        if content_items:
            vetting_report["aggregate_metrics"] = {
                "credibility_score": total_credibility / len(content_items),
                "bias_score": total_bias / len(content_items),
                "fact_score": total_fact / len(content_items),
                "purity_score": total_purity / len(content_items)
            }
        
        return vetting_report
    
    def _analyze_content_item(self, item: Dict[str, Any], security_level: str) -> Dict[str, Any]:
        """
        Analyze a single content item across four security dimensions.
        
        Returns:
            Detailed analysis report for the item
        """
        content = item.get("content", "")
        title = item.get("title", "")
        source = item.get("source", "unknown")
        
        # Dimension 1: Credibility & Bias Analysis
        credibility_analysis = self._analyze_credibility_bias(content, title, source)
        
        # Dimension 2: Persuasive/Manipulative Language Detection
        manipulation_analysis = self._analyze_manipulation(content, title)
        
        # Dimension 3: Speculation vs Fact Classification
        fact_analysis = self._analyze_fact_speculation(content, title)
        
        # Dimension 4: Content Purity & Injection Detection
        purity_analysis = self._analyze_content_purity(content, title)
        
        # Calculate overall risk level
        risk_level = self._calculate_risk_level(
            credibility_analysis, manipulation_analysis, fact_analysis, purity_analysis
        )
        
        return {
            "source": source,
            "title": title,
            "content_length": len(content),
            "credibility_score": credibility_analysis["score"],
            "credibility_issues": credibility_analysis["issues"],
            "bias_score": manipulation_analysis["score"],
            "manipulation_indicators": manipulation_analysis["indicators"],
            "fact_score": fact_analysis["score"],
            "speculation_indicators": fact_analysis["speculation_indicators"],
            "purity_score": purity_analysis["score"],
            "injection_indicators": purity_analysis["injection_indicators"],
            "risk_level": risk_level,
            "approved": risk_level != "high"
        }
    
    def _analyze_credibility_bias(self, content: str, title: str, source: str) -> Dict[str, Any]:
        """Analyze credibility and bias indicators."""
        score = 0.8  # Start with neutral score
        issues = []
        
        text = f"{title} {content}".lower()
        
        # Check for bias indicators
        for indicator in self._bias_indicators:
            if indicator in text:
                score -= 0.1
                issues.append(f"Bias indicator: {indicator}")
        
        # Check source credibility
        if self._is_questionable_source(source):
            score -= 0.2
            issues.append("Questionable source domain")
        
        # Check for emotional language
        emotional_words = ["shocking", "unbelievable", "amazing", "terrible", "outrageous"]
        emotional_count = sum(1 for word in emotional_words if word in text)
        if emotional_count > 2:
            score -= 0.1
            issues.append("High emotional language content")
        
        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues
        }
    
    def _analyze_manipulation(self, content: str, title: str) -> Dict[str, Any]:
        """Analyze persuasive and manipulative language patterns."""
        score = 0.8  # Start with neutral score
        indicators = []
        
        text = f"{title} {content}".lower()
        
        # Check for manipulation patterns
        for pattern in self._manipulation_patterns:
            if re.search(pattern, text):
                score -= 0.15
                indicators.append(f"Manipulation pattern: {pattern}")
        
        # Check for urgency indicators
        urgency_words = ["urgent", "immediate", "now", "hurry", "limited time"]
        urgency_count = sum(1 for word in urgency_words if word in text)
        if urgency_count > 1:
            score -= 0.1
            indicators.append("Urgency manipulation detected")
        
        return {
            "score": max(0.0, min(1.0, score)),
            "indicators": indicators
        }
    
    def _analyze_fact_speculation(self, content: str, title: str) -> Dict[str, Any]:
        """Analyze fact vs speculation content."""
        score = 0.7  # Start slightly lower for speculation
        speculation_indicators = []
        
        text = f"{title} {content}".lower()
        
        # Check for fact indicators (positive)
        fact_count = 0
        for indicator in self._fact_indicators:
            if indicator in text:
                fact_count += 1
        
        if fact_count > 0:
            score += min(0.3, fact_count * 0.1)
        
        # Check for speculation indicators (negative)
        speculation_words = ["might", "could", "possibly", "allegedly", "rumored", "unconfirmed"]
        speculation_count = sum(1 for word in speculation_words if word in text)
        if speculation_count > 2:
            score -= 0.2
            speculation_indicators.append("High speculation language")
        
        return {
            "score": max(0.0, min(1.0, score)),
            "speculation_indicators": speculation_indicators
        }
    
    def _analyze_content_purity(self, content: str, title: str) -> Dict[str, Any]:
        """Analyze content for injection attempts and purity."""
        score = 0.9  # Start high for purity
        injection_indicators = []
        
        text = f"{title} {content}"
        
        # Check for injection patterns
        for pattern in self._injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3
                injection_indicators.append(f"Injection pattern: {pattern}")
        
        # Check for suspicious code patterns
        code_patterns = ["<script", "javascript:", "eval(", "exec(", "system("]
        for pattern in code_patterns:
            if pattern in text.lower():
                score -= 0.4
                injection_indicators.append(f"Suspicious code: {pattern}")
        
        return {
            "score": max(0.0, min(1.0, score)),
            "injection_indicators": injection_indicators
        }
    
    def _calculate_risk_level(self, credibility: Dict, manipulation: Dict, 
                            fact: Dict, purity: Dict) -> str:
        """Calculate overall risk level based on all dimensions."""
        avg_score = (credibility["score"] + manipulation["score"] + 
                    fact["score"] + purity["score"]) / 4
        
        if avg_score < 0.4:
            return "high"
        elif avg_score < 0.7:
            return "medium"
        else:
            return "low"
    
    def _calculate_security_score(self, vetting_report: Dict[str, Any]) -> float:
        """Calculate overall security score from vetting report."""
        metrics = vetting_report["aggregate_metrics"]
        
        # Weighted average of all dimensions
        weights = {
            "credibility_score": 0.3,
            "bias_score": 0.25,
            "fact_score": 0.25,
            "purity_score": 0.2
        }
        
        weighted_score = sum(metrics[key] * weights[key] for key in weights)
        return round(weighted_score, 3)
    
    def _filter_approved_content(self, content_items: List[Dict[str, Any]], 
                               vetting_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter content items to only include approved ones."""
        approved_content = []
        
        for i, item in enumerate(content_items):
            if i < len(vetting_report["items"]):
                analysis = vetting_report["items"][i]
                if analysis["approved"]:
                    approved_content.append({
                        **item,
                        "vetting_score": analysis["credibility_score"],
                        "approved_at": datetime.now().isoformat()
                    })
        
        return approved_content
    
    def _load_bias_indicators(self) -> List[str]:
        """Load bias indicator patterns."""
        return [
            "always", "never", "all", "none", "everyone", "no one",
            "obviously", "clearly", "undoubtedly", "certainly"
        ]
    
    def _load_manipulation_patterns(self) -> List[str]:
        """Load manipulation detection patterns."""
        return [
            r"you must\s+\w+",
            r"don't miss\s+\w+",
            r"act now\s+\w+",
            r"limited offer\s+\w+",
            r"exclusive\s+\w+"
        ]
    
    def _load_fact_indicators(self) -> List[str]:
        """Load fact indicator patterns."""
        return [
            "according to", "research shows", "study found", "data indicates",
            "statistics show", "evidence suggests", "published in"
        ]
    
    def _load_injection_patterns(self) -> List[str]:
        """Load injection detection patterns."""
        return [
            r"<script.*?>",
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"system\s*\(",
            r"__import__",
            r"subprocess"
        ]
    
    def _is_questionable_source(self, source: str) -> bool:
        """Check if source is from a questionable domain."""
        questionable_domains = [
            "example.com", "test.com", "fake.com", "spam.com"
        ]
        
        return any(domain in source.lower() for domain in questionable_domains)

    def _log_trace_event(self, trace_id: str, event_type: str, severity: str,
                        message: str, duration_ms: Optional[float] = None,
                        payload: Optional[Dict[str, Any]] = None) -> None:
        """Log a trace event for the content vetting tool."""
        try:
            from sam.cognition.trace_logger import log_event
            log_event(
                trace_id=trace_id,
                source_module=self.skill_name,
                event_type=event_type,
                severity=severity,
                message=message,
                duration_ms=duration_ms,
                payload=payload or {},
                metadata={
                    "tool_version": self.skill_version,
                    "tool_category": self.skill_category,
                    "requires_external_access": self.requires_external_access,
                    "requires_vetting": self.requires_vetting
                }
            )
        except ImportError:
            # Tracing not available, continue without logging
            pass
        except Exception as e:
            # Don't let tracing errors break the tool
            self.logger.debug(f"Trace logging failed: {e}")
