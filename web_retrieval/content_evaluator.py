"""
Content Evaluator for Phase 7.2: Automated Vetting Engine

This module provides the main ContentEvaluator class that orchestrates
the entire automated vetting process for web content retrieved through
SAM's web fetching system.

Features:
- Comprehensive content sanitization and security analysis
- Dimensional analysis using SAM's ConceptualDimensionProber
- Source reputation assessment
- Risk scoring and recommendation generation
- Integration with SAM's existing intelligence systems
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
import time

from .content_sanitizer import ContentSanitizer
from .exceptions import (
    WebRetrievalError,
    ValidationError,
    ContentExtractionError
)


class ContentEvaluator:
    """
    Automated content vetting engine using SAM's intelligence.
    
    This class orchestrates the entire vetting process:
    1. Content sanitization and cleaning
    2. Security-focused dimensional analysis
    3. Source reputation assessment
    4. Risk assessment and scoring
    5. Recommendation generation
    """
    
    def __init__(self, 
                 vetting_profile: str = "web_vetting",
                 safety_threshold: float = 0.7,
                 analysis_mode: str = "balanced"):
        """
        Initialize the content evaluator.
        
        Args:
            vetting_profile: Dimension profile for security analysis
            safety_threshold: Threshold for PASS/FAIL recommendations
            analysis_mode: Analysis weight mode (security_focus, quality_focus, balanced)
        """
        self.vetting_profile = vetting_profile
        self.safety_threshold = safety_threshold
        self.analysis_mode = analysis_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sanitizer = ContentSanitizer()
        self.dimension_prober = None
        self.profile_config = None
        self.reputation_data = None
        
        # Load configurations
        self._load_configurations()
        self._initialize_dimension_prober()
        
        self.logger.info(f"ContentEvaluator initialized with profile: {vetting_profile}, "
                        f"threshold: {safety_threshold}, mode: {analysis_mode}")
    
    def evaluate_quarantined_file(self, filepath: str) -> Dict[str, Any]:
        """
        Main evaluation method for quarantined content files.
        
        Args:
            filepath: Path to quarantined JSON file
            
        Returns:
            Enhanced data with comprehensive vetting results
        """
        start_time = time.time()
        filepath = Path(filepath)
        
        self.logger.info(f"Starting evaluation of quarantined file: {filepath.name}")
        
        try:
            # Step 1: Load and validate quarantined data
            raw_data = self._load_quarantined_data(filepath)
            
            # Step 2: Sanitize content for analysis
            sanitization_result = self.sanitizer.sanitize_content(raw_data['content'])
            
            # Step 3: Analyze source reputation
            source_reputation = self._analyze_source_reputation(raw_data['url'])
            
            # Step 4: Run dimensional analysis
            dimension_scores = self._analyze_dimensions(
                sanitization_result['clean_text'],
                raw_data['url']
            )
            
            # Step 5: Calculate comprehensive risk assessment
            risk_assessment = self._calculate_risk_assessment(
                dimension_scores, sanitization_result, source_reputation
            )
            
            # Step 6: Generate final recommendation
            recommendation = self._generate_recommendation(
                risk_assessment, dimension_scores, source_reputation
            )
            
            # Step 7: Compile comprehensive results
            processing_time = time.time() - start_time
            
            vetting_results = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'recommendation': recommendation['decision'],
                'confidence': recommendation['confidence'],
                'reason': recommendation['reason'],
                'overall_score': risk_assessment['overall_score'],
                'scores': dimension_scores,
                'source_reputation': source_reputation,
                'risk_assessment': risk_assessment,
                'sanitization': {
                    'removed_elements': sanitization_result['removed_elements'],
                    'suspicious_patterns': sanitization_result['suspicious_patterns'],
                    'purity_score': sanitization_result['purity_score'],
                    'encoding_issues': sanitization_result['encoding_issues'],
                    'content_stats': sanitization_result['content_stats']
                },
                'metadata': {
                    'profile_used': self.vetting_profile,
                    'analysis_mode': self.analysis_mode,
                    'evaluator_version': '1.0',
                    'safety_threshold': self.safety_threshold
                }
            }
            
            # Step 8: Enrich original data
            enriched_data = raw_data.copy()
            enriched_data['vetting_results'] = vetting_results
            
            self.logger.info(f"Evaluation complete: {recommendation['decision']} "
                           f"(score: {risk_assessment['overall_score']:.3f}, "
                           f"confidence: {recommendation['confidence']:.1%}, "
                           f"time: {processing_time:.2f}s)")
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Error evaluating {filepath.name}: {e}")
            
            # Return error result
            error_data = raw_data if 'raw_data' in locals() else {'url': 'unknown', 'content': '', 'timestamp': ''}
            error_data['vetting_results'] = {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'FAIL',
                'confidence': 0.0,
                'reason': f"Evaluation error: {str(e)}",
                'overall_score': 0.0,
                'error': str(e)
            }
            
            return error_data
    
    def _load_configurations(self):
        """Load vetting profile and reputation data."""
        try:
            # Load vetting profile
            profile_path = Path(__file__).parent.parent / "multimodal_processing" / "dimension_profiles" / f"{self.vetting_profile}.json"
            
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    self.profile_config = json.load(f)
                self.logger.info(f"Loaded vetting profile: {self.vetting_profile}")
            else:
                self.logger.warning(f"Vetting profile not found: {profile_path}")
                self.profile_config = self._get_default_profile()
            
            # Load reputation data
            reputation_path = Path(__file__).parent / "source_reputation.json"
            
            if reputation_path.exists():
                with open(reputation_path, 'r') as f:
                    self.reputation_data = json.load(f)
                self.logger.info("Loaded source reputation database")
            else:
                self.logger.warning(f"Reputation database not found: {reputation_path}")
                self.reputation_data = self._get_default_reputation_data()
                
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
            self.profile_config = self._get_default_profile()
            self.reputation_data = self._get_default_reputation_data()
    
    def _initialize_dimension_prober(self):
        """Initialize the ConceptualDimensionProber for analysis."""
        try:
            from multimodal_processing.conceptual_dimension_prober import ConceptualDimensionProber
            
            self.dimension_prober = ConceptualDimensionProber(
                profile_name=self.vetting_profile
            )
            self.logger.info("ConceptualDimensionProber initialized successfully")
            
        except ImportError as e:
            self.logger.warning(f"ConceptualDimensionProber not available: {e}")
            self.dimension_prober = None
        except Exception as e:
            self.logger.error(f"Error initializing dimension prober: {e}")
            self.dimension_prober = None
    
    def _load_quarantined_data(self, filepath: Path) -> Dict[str, Any]:
        """Load and validate quarantined data file."""
        if not filepath.exists():
            raise ValidationError(f"Quarantined file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ['url', 'content', 'timestamp']
            for field in required_fields:
                if field not in data:
                    raise ValidationError(f"Missing required field: {field}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in quarantined file: {e}")
        except Exception as e:
            raise ValidationError(f"Error loading quarantined file: {e}")
    
    def _analyze_source_reputation(self, url: str) -> Dict[str, Any]:
        """Analyze source reputation based on domain and URL characteristics."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Get base score from TLD
            tld = self._extract_tld(domain)
            base_score = self._get_tld_score(tld)
            
            # Check trusted domains
            trusted_bonus = self._check_trusted_domains(domain)
            
            # Check suspicious domains
            suspicious_penalty = self._check_suspicious_domains(domain)
            
            # Apply reputation factors
            factor_adjustments = self._calculate_reputation_factors(parsed_url)
            
            # Calculate final score
            final_score = base_score + trusted_bonus + suspicious_penalty + factor_adjustments
            final_score = max(0.0, min(1.0, final_score))
            
            # Determine risk category
            risk_category = self._get_risk_category(final_score)
            
            return {
                'domain': domain,
                'tld': tld,
                'base_score': base_score,
                'trusted_bonus': trusted_bonus,
                'suspicious_penalty': suspicious_penalty,
                'factor_adjustments': factor_adjustments,
                'final_score': final_score,
                'risk_category': risk_category,
                'https_used': parsed_url.scheme == 'https'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing source reputation for {url}: {e}")
            return {
                'domain': 'unknown',
                'tld': 'unknown',
                'final_score': 0.5,
                'risk_category': 'medium_risk',
                'error': str(e)
            }
    
    def _analyze_dimensions(self, clean_text: str, url: str) -> Dict[str, float]:
        """Analyze content using dimensional analysis."""
        if not self.dimension_prober or not clean_text:
            # Fallback to basic analysis
            return self._basic_dimension_analysis(clean_text, url)
        
        try:
            # Use ConceptualDimensionProber for sophisticated analysis
            analysis_result = self.dimension_prober.analyze_text(clean_text)
            
            # Extract dimension scores
            dimension_scores = {}
            for dimension_name in self.profile_config.get('dimensions', {}):
                score = analysis_result.get(dimension_name, 0.5)
                dimension_scores[dimension_name] = float(score)
            
            self.logger.info(f"Dimensional analysis complete: {len(dimension_scores)} dimensions")
            return dimension_scores
            
        except Exception as e:
            self.logger.warning(f"Dimensional analysis failed, using fallback: {e}")
            return self._basic_dimension_analysis(clean_text, url)
    
    def _basic_dimension_analysis(self, text: str, url: str) -> Dict[str, float]:
        """Basic fallback dimensional analysis using pattern matching."""
        scores = {}
        
        if not text:
            return {dim: 0.0 for dim in ['credibility', 'persuasion', 'speculation', 'purity', 'source_reputation']}
        
        text_lower = text.lower()
        
        # Basic credibility analysis
        credibility_indicators = ['study', 'research', 'data', 'evidence', 'according to']
        credibility_score = min(1.0, sum(1 for indicator in credibility_indicators if indicator in text_lower) * 0.2)
        scores['credibility'] = credibility_score
        
        # Basic persuasion analysis
        persuasion_indicators = ['amazing', 'incredible', 'urgent', 'act now', 'limited time']
        persuasion_score = min(1.0, sum(1 for indicator in persuasion_indicators if indicator in text_lower) * 0.3)
        scores['persuasion'] = persuasion_score
        
        # Basic speculation analysis
        speculation_indicators = ['might', 'could', 'may', 'probably', 'i think', 'believe']
        speculation_score = min(1.0, sum(1 for indicator in speculation_indicators if indicator in text_lower) * 0.25)
        scores['speculation'] = speculation_score
        
        # Purity score from sanitizer
        scores['purity'] = 0.8  # Default high purity for basic analysis
        
        # Source reputation (will be calculated separately)
        scores['source_reputation'] = 0.6  # Default neutral
        
        return scores

    def _calculate_risk_assessment(self, dimension_scores: Dict[str, float],
                                 sanitization_result: Dict[str, Any],
                                 source_reputation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment."""

        # Get analysis weights based on mode
        weights = self.profile_config.get('analysis_weights', {}).get(
            self.analysis_mode,
            self.profile_config.get('analysis_weights', {}).get('balanced', {})
        )

        # Calculate weighted overall score
        overall_score = 0.0
        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0.2)  # Default weight
            overall_score += score * weight

        # Apply source reputation
        source_weight = weights.get('source_reputation', 0.1)
        overall_score += source_reputation.get('final_score', 0.5) * source_weight

        # Apply purity score from sanitization
        purity_weight = weights.get('purity', 0.15)
        overall_score += sanitization_result.get('purity_score', 0.8) * purity_weight

        # Normalize score
        overall_score = max(0.0, min(1.0, overall_score))

        # Identify risk factors
        risk_factors = self._identify_risk_factors(dimension_scores, sanitization_result, source_reputation)

        # Calculate confidence based on consistency of scores
        confidence = self._calculate_confidence(dimension_scores, overall_score)

        return {
            'overall_score': overall_score,
            'risk_factors': risk_factors,
            'confidence': confidence,
            'analysis_mode': self.analysis_mode,
            'weights_used': weights
        }

    def _generate_recommendation(self, risk_assessment: Dict[str, Any],
                               dimension_scores: Dict[str, float],
                               source_reputation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation based on all analysis results."""

        overall_score = risk_assessment['overall_score']
        risk_factors = risk_assessment['risk_factors']

        # Get thresholds from profile
        thresholds = self.profile_config.get('scoring_thresholds', {})
        pass_threshold = thresholds.get('pass_threshold', 0.70)
        warning_threshold = thresholds.get('warning_threshold', 0.50)
        fail_threshold = thresholds.get('fail_threshold', 0.30)

        # Determine decision
        decision = 'FAIL'
        reason_parts = []

        # Check for critical security issues
        purity_score = dimension_scores.get('purity', 1.0)
        if purity_score < 0.4:
            decision = 'FAIL'
            reason_parts.append(f"Critical security concerns (purity: {purity_score:.2f})")

        # Check for multiple high-risk factors
        elif len([rf for rf in risk_factors if rf.get('severity') in ['critical', 'high']]) >= 2:
            decision = 'FAIL'
            reason_parts.append("Multiple high-risk factors detected")

        # Apply score-based thresholds
        elif overall_score >= pass_threshold:
            decision = 'PASS'
            reason_parts.append(f"High overall quality score ({overall_score:.2f})")

        elif overall_score >= warning_threshold:
            decision = 'REVIEW'
            reason_parts.append(f"Moderate quality score ({overall_score:.2f}) requires review")

        else:
            decision = 'FAIL'
            reason_parts.append(f"Low overall quality score ({overall_score:.2f})")

        # Add specific risk factor details
        for risk_factor in risk_factors:
            if risk_factor.get('severity') in ['critical', 'high']:
                reason_parts.append(f"{risk_factor['description']} ({risk_factor['dimension']}: {risk_factor['score']:.2f})")

        # Compile final reason
        reason = '; '.join(reason_parts) if reason_parts else "Standard analysis complete"

        return {
            'decision': decision,
            'confidence': risk_assessment['confidence'],
            'reason': reason,
            'score_breakdown': {
                'overall': overall_score,
                'vs_pass_threshold': overall_score - pass_threshold,
                'vs_warning_threshold': overall_score - warning_threshold
            }
        }

    def _identify_risk_factors(self, dimension_scores: Dict[str, float],
                             sanitization_result: Dict[str, Any],
                             source_reputation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific risk factors based on analysis results."""

        risk_factors = []
        risk_config = self.profile_config.get('risk_factors', {})

        # Check dimension-based risk factors
        for dimension, score in dimension_scores.items():
            risk_key = f"high_{dimension}" if dimension == 'persuasion' or dimension == 'speculation' else f"low_{dimension}"

            if risk_key in risk_config:
                threshold = risk_config[risk_key]['threshold']

                # For persuasion and speculation, high scores are risky
                if dimension in ['persuasion', 'speculation'] and score >= threshold:
                    risk_factors.append({
                        'dimension': dimension,
                        'score': score,
                        'threshold': threshold,
                        'severity': risk_config[risk_key]['severity'],
                        'description': risk_config[risk_key]['description']
                    })

                # For credibility and purity, low scores are risky
                elif dimension in ['credibility', 'purity'] and score <= threshold:
                    risk_factors.append({
                        'dimension': dimension,
                        'score': score,
                        'threshold': threshold,
                        'severity': risk_config[risk_key]['severity'],
                        'description': risk_config[risk_key]['description']
                    })

        # Check source reputation risk
        source_score = source_reputation.get('final_score', 0.5)
        if 'poor_source' in risk_config and source_score <= risk_config['poor_source']['threshold']:
            risk_factors.append({
                'dimension': 'source_reputation',
                'score': source_score,
                'threshold': risk_config['poor_source']['threshold'],
                'severity': risk_config['poor_source']['severity'],
                'description': risk_config['poor_source']['description']
            })

        # Check sanitization-based risks
        suspicious_patterns = sanitization_result.get('suspicious_patterns', [])
        if len(suspicious_patterns) > 0:
            risk_factors.append({
                'dimension': 'content_security',
                'score': 1.0 - (len(suspicious_patterns) * 0.2),
                'threshold': 0.8,
                'severity': 'high',
                'description': f"Suspicious patterns detected: {len(suspicious_patterns)} issues"
            })

        return risk_factors

    def _calculate_confidence(self, dimension_scores: Dict[str, float], overall_score: float) -> float:
        """Calculate confidence in the analysis based on score consistency."""

        if not dimension_scores:
            return 0.0

        scores = list(dimension_scores.values())

        # Calculate variance in scores
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

        # Lower variance = higher confidence
        consistency_factor = max(0.0, 1.0 - (variance * 2))

        # Higher absolute scores (very high or very low) = higher confidence
        extremity_factor = abs(overall_score - 0.5) * 2

        # Combine factors
        confidence = (consistency_factor * 0.6) + (extremity_factor * 0.4)

        return max(0.0, min(1.0, confidence))

    # Helper methods for source reputation analysis
    def _extract_tld(self, domain: str) -> str:
        """Extract top-level domain from domain name."""
        parts = domain.split('.')
        if len(parts) >= 2:
            return '.' + parts[-1]
        return '.unknown'

    def _get_tld_score(self, tld: str) -> float:
        """Get base score for top-level domain."""
        domain_scores = self.reputation_data.get('domain_scores', {})

        for category, scores in domain_scores.items():
            if tld in scores:
                return scores[tld]

        return 0.5  # Default neutral score

    def _check_trusted_domains(self, domain: str) -> float:
        """Check if domain is in trusted domains list."""
        trusted_domains = self.reputation_data.get('trusted_domains', {})

        for category, domains in trusted_domains.items():
            if domain in domains:
                return 0.20  # Trusted domain bonus

        return 0.0

    def _check_suspicious_domains(self, domain: str) -> float:
        """Check if domain is in suspicious domains list."""
        suspicious_domains = self.reputation_data.get('suspicious_domains', [])

        if domain in suspicious_domains:
            return -0.50  # Suspicious domain penalty

        return 0.0

    def _calculate_reputation_factors(self, parsed_url) -> float:
        """Calculate reputation adjustments based on URL factors."""
        factors = self.reputation_data.get('reputation_factors', {})
        adjustment = 0.0

        # HTTPS bonus
        if parsed_url.scheme == 'https':
            adjustment += factors.get('https_bonus', 0.05)

        # Subdomain penalty
        domain_parts = parsed_url.netloc.split('.')
        if len(domain_parts) > 2:
            adjustment += factors.get('subdomain_penalty', -0.10)

        # Short domain penalty
        if len(parsed_url.netloc) < 8:
            adjustment += factors.get('short_domain_penalty', -0.15)

        # Numeric domain penalty
        if any(char.isdigit() for char in parsed_url.netloc):
            adjustment += factors.get('numeric_domain_penalty', -0.20)

        return adjustment

    def _get_risk_category(self, score: float) -> str:
        """Get risk category based on score."""
        risk_categories = self.reputation_data.get('risk_categories', {})

        for category, config in risk_categories.items():
            score_range = config.get('score_range', [0.0, 1.0])
            if score_range[0] <= score <= score_range[1]:
                return category

        return 'medium_risk'

    def _get_default_profile(self) -> Dict[str, Any]:
        """Get default vetting profile if file not found."""
        return {
            'dimensions': {
                'credibility': {'weight': 0.30},
                'persuasion': {'weight': 0.25},
                'speculation': {'weight': 0.20},
                'purity': {'weight': 0.15},
                'source_reputation': {'weight': 0.10}
            },
            'scoring_thresholds': {
                'pass_threshold': 0.70,
                'warning_threshold': 0.50,
                'fail_threshold': 0.30
            },
            'risk_factors': {}
        }

    def _get_default_reputation_data(self) -> Dict[str, Any]:
        """Get default reputation data if file not found."""
        return {
            'domain_scores': {
                '.gov': 0.95, '.edu': 0.90, '.org': 0.75, '.com': 0.60
            },
            'trusted_domains': {},
            'suspicious_domains': [],
            'reputation_factors': {
                'https_bonus': 0.05,
                'subdomain_penalty': -0.10
            },
            'risk_categories': {}
        }

    def get_evaluator_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics and configuration."""
        return {
            'vetting_profile': self.vetting_profile,
            'safety_threshold': self.safety_threshold,
            'analysis_mode': self.analysis_mode,
            'dimension_prober_available': self.dimension_prober is not None,
            'profile_loaded': self.profile_config is not None,
            'reputation_data_loaded': self.reputation_data is not None,
            'version': '1.0'
        }
