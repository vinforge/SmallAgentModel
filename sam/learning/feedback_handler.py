"""
MEMOIR Feedback Handler

Automatic learning system that processes user feedback and corrections
to continuously improve SAM's knowledge through MEMOIR edits.

Author: SAM Development Team
Version: 1.0.0
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..orchestration.uif import SAM_UIF
from ..orchestration.skills.internal.memoir_edit import MEMOIR_EditSkill
from ..orchestration.memoir_sof_integration import get_memoir_sof_integration

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback."""
    FACTUAL_CORRECTION = "factual_correction"
    PREFERENCE_LEARNING = "preference_learning"
    BEHAVIOR_ADJUSTMENT = "behavior_adjustment"
    KNOWLEDGE_UPDATE = "knowledge_update"
    CLARIFICATION = "clarification"

@dataclass
class FeedbackEvent:
    """Represents a user feedback event."""
    feedback_id: str
    feedback_type: FeedbackType
    original_query: str
    sam_response: str
    user_feedback: str
    corrected_response: Optional[str]
    confidence_score: float
    timestamp: datetime
    context: Dict[str, Any]
    processed: bool = False

class MEMOIRFeedbackHandler:
    """
    Handles user feedback and automatically creates MEMOIR edits for learning.
    
    This system enables SAM to learn from user corrections and feedback,
    automatically creating MEMOIR edits to improve future responses.
    
    Key Features:
    - Automatic feedback classification
    - Intelligent correction extraction
    - MEMOIR edit generation
    - Learning validation and verification
    - Feedback analytics and monitoring
    """
    
    def __init__(self, memoir_integration: Optional[Any] = None):
        """
        Initialize the feedback handler.
        
        Args:
            memoir_integration: MEMOIR SOF integration instance
        """
        self.logger = logging.getLogger(f"{__name__}.MEMOIRFeedbackHandler")
        self.memoir_integration = memoir_integration or get_memoir_sof_integration()
        
        # Feedback processing configuration
        self.config = {
            'auto_process_feedback': True,
            'confidence_threshold': 0.7,
            'max_edit_attempts': 3,
            'enable_validation': True,
            'store_feedback_history': True
        }
        
        # Feedback storage
        self.feedback_history: List[FeedbackEvent] = []
        self.processing_stats = {
            'total_feedback': 0,
            'successful_edits': 0,
            'failed_edits': 0,
            'feedback_types': {ft.value: 0 for ft in FeedbackType}
        }
        
        # Pattern matching for feedback classification
        self.feedback_patterns = {
            FeedbackType.FACTUAL_CORRECTION: [
                r"actually,?\s*(.*)",
                r"that'?s not right,?\s*(.*)",
                r"incorrect,?\s*(.*)",
                r"wrong,?\s*(.*)",
                r"the correct answer is\s*(.*)",
                r"it should be\s*(.*)",
                r"fix:?\s*(.*)",
                r"correction:?\s*(.*)"
            ],
            FeedbackType.PREFERENCE_LEARNING: [
                r"i prefer\s*(.*)",
                r"i like\s*(.*)",
                r"i usually\s*(.*)",
                r"my preference is\s*(.*)",
                r"remember that i\s*(.*)",
                r"note that i\s*(.*)",
                r"for me,?\s*(.*)"
            ],
            FeedbackType.KNOWLEDGE_UPDATE: [
                r"update:?\s*(.*)",
                r"new information:?\s*(.*)",
                r"learn this:?\s*(.*)",
                r"remember:?\s*(.*)",
                r"save this:?\s*(.*)"
            ]
        }
        
        self.logger.info("MEMOIR Feedback Handler initialized")
    
    def process_feedback(
        self,
        original_query: str,
        sam_response: str,
        user_feedback: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback and create MEMOIR edits if appropriate.
        
        Args:
            original_query: The original user query
            sam_response: SAM's response that received feedback
            user_feedback: User's feedback/correction
            context: Additional context information
            
        Returns:
            Processing results including edit information
        """
        try:
            # Create feedback event
            feedback_event = self._create_feedback_event(
                original_query, sam_response, user_feedback, context or {}
            )
            
            self.logger.info(f"Processing feedback: {feedback_event.feedback_id}")
            
            # Store feedback
            if self.config['store_feedback_history']:
                self.feedback_history.append(feedback_event)
            
            # Update statistics
            self.processing_stats['total_feedback'] += 1
            self.processing_stats['feedback_types'][feedback_event.feedback_type.value] += 1
            
            # Process feedback if auto-processing is enabled
            if self.config['auto_process_feedback']:
                return self._auto_process_feedback(feedback_event)
            else:
                return {
                    'feedback_id': feedback_event.feedback_id,
                    'feedback_type': feedback_event.feedback_type.value,
                    'auto_processed': False,
                    'message': 'Feedback stored for manual processing'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to process feedback: {e}")
            return {
                'success': False,
                'error': str(e),
                'feedback_id': None
            }
    
    def _create_feedback_event(
        self,
        original_query: str,
        sam_response: str,
        user_feedback: str,
        context: Dict[str, Any]
    ) -> FeedbackEvent:
        """Create a feedback event from user input."""
        
        # Generate unique feedback ID
        feedback_id = f"feedback_{int(datetime.now().timestamp())}_{len(self.feedback_history)}"
        
        # Classify feedback type
        feedback_type = self._classify_feedback(user_feedback)
        
        # Extract corrected response if applicable
        corrected_response = self._extract_correction(user_feedback, feedback_type)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(user_feedback, feedback_type)
        
        return FeedbackEvent(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            original_query=original_query,
            sam_response=sam_response,
            user_feedback=user_feedback,
            corrected_response=corrected_response,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            context=context
        )
    
    def _classify_feedback(self, user_feedback: str) -> FeedbackType:
        """Classify the type of user feedback."""
        user_feedback_lower = user_feedback.lower().strip()
        
        # Check patterns for each feedback type
        for feedback_type, patterns in self.feedback_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_feedback_lower):
                    return feedback_type
        
        # Default classification based on keywords
        if any(word in user_feedback_lower for word in ['wrong', 'incorrect', 'actually', 'correction']):
            return FeedbackType.FACTUAL_CORRECTION
        elif any(word in user_feedback_lower for word in ['prefer', 'like', 'usually']):
            return FeedbackType.PREFERENCE_LEARNING
        elif any(word in user_feedback_lower for word in ['remember', 'learn', 'save']):
            return FeedbackType.KNOWLEDGE_UPDATE
        else:
            return FeedbackType.CLARIFICATION
    
    def _extract_correction(self, user_feedback: str, feedback_type: FeedbackType) -> Optional[str]:
        """Extract the corrected information from user feedback."""
        if feedback_type not in [FeedbackType.FACTUAL_CORRECTION, FeedbackType.KNOWLEDGE_UPDATE]:
            return None
        
        user_feedback_lower = user_feedback.lower().strip()
        
        # Try to extract correction using patterns
        patterns = self.feedback_patterns.get(feedback_type, [])
        for pattern in patterns:
            match = re.search(pattern, user_feedback_lower)
            if match:
                correction = match.group(1).strip()
                if correction:
                    return correction
        
        # Fallback: use the entire feedback as correction
        return user_feedback.strip()
    
    def _calculate_confidence(self, user_feedback: str, feedback_type: FeedbackType) -> float:
        """Calculate confidence score for the feedback."""
        base_confidence = 0.7
        
        # Adjust based on feedback type
        type_adjustments = {
            FeedbackType.FACTUAL_CORRECTION: 0.1,
            FeedbackType.PREFERENCE_LEARNING: 0.0,
            FeedbackType.KNOWLEDGE_UPDATE: 0.1,
            FeedbackType.BEHAVIOR_ADJUSTMENT: -0.1,
            FeedbackType.CLARIFICATION: -0.2
        }
        
        confidence = base_confidence + type_adjustments.get(feedback_type, 0.0)
        
        # Adjust based on feedback characteristics
        user_feedback_lower = user_feedback.lower()
        
        # Strong indicators increase confidence
        if any(word in user_feedback_lower for word in ['definitely', 'absolutely', 'certain', 'sure']):
            confidence += 0.1
        
        # Weak indicators decrease confidence
        if any(word in user_feedback_lower for word in ['maybe', 'perhaps', 'might', 'possibly']):
            confidence -= 0.1
        
        # Length and detail increase confidence
        if len(user_feedback.split()) > 10:
            confidence += 0.05
        
        return max(0.1, min(1.0, confidence))
    
    def _auto_process_feedback(self, feedback_event: FeedbackEvent) -> Dict[str, Any]:
        """Automatically process feedback and create MEMOIR edits."""
        
        # Check confidence threshold
        if feedback_event.confidence_score < self.config['confidence_threshold']:
            return {
                'feedback_id': feedback_event.feedback_id,
                'auto_processed': False,
                'reason': f'Confidence {feedback_event.confidence_score:.2f} below threshold {self.config["confidence_threshold"]}',
                'success': False
            }
        
        # Process based on feedback type
        if feedback_event.feedback_type in [
            FeedbackType.FACTUAL_CORRECTION,
            FeedbackType.KNOWLEDGE_UPDATE,
            FeedbackType.PREFERENCE_LEARNING
        ]:
            return self._create_memoir_edit(feedback_event)
        else:
            return {
                'feedback_id': feedback_event.feedback_id,
                'auto_processed': False,
                'reason': f'Feedback type {feedback_event.feedback_type.value} not suitable for MEMOIR edit',
                'success': True
            }
    
    def _create_memoir_edit(self, feedback_event: FeedbackEvent) -> Dict[str, Any]:
        """Create a MEMOIR edit from feedback event."""
        try:
            # Get MEMOIR edit skill
            memoir_skills = self.memoir_integration.get_memoir_skills()
            if 'MEMOIR_EditSkill' not in memoir_skills:
                return {
                    'feedback_id': feedback_event.feedback_id,
                    'success': False,
                    'error': 'MEMOIR_EditSkill not available'
                }
            
            edit_skill = self.memoir_integration.memoir_skills['MEMOIR_EditSkill']
            
            # Prepare edit data
            edit_prompt = feedback_event.original_query
            correct_answer = feedback_event.corrected_response or feedback_event.user_feedback
            
            # Create UIF for the edit
            edit_uif = SAM_UIF(
                input_query=f"Feedback correction: {feedback_event.feedback_id}",
                intermediate_data={
                    'edit_prompt': edit_prompt,
                    'correct_answer': correct_answer,
                    'edit_context': f"User feedback correction - {feedback_event.feedback_type.value}",
                    'confidence_score': feedback_event.confidence_score,
                    'edit_metadata': {
                        'source': 'user_feedback',
                        'feedback_id': feedback_event.feedback_id,
                        'feedback_type': feedback_event.feedback_type.value,
                        'original_response': feedback_event.sam_response,
                        'user_feedback': feedback_event.user_feedback,
                        'timestamp': feedback_event.timestamp.isoformat(),
                        'context': feedback_event.context
                    }
                }
            )
            
            # Execute the edit
            result_uif = edit_skill.execute(edit_uif)
            
            # Process results
            if result_uif.intermediate_data.get('edit_success', False):
                edit_id = result_uif.intermediate_data['edit_id']
                feedback_event.processed = True
                self.processing_stats['successful_edits'] += 1
                
                self.logger.info(f"✅ Created MEMOIR edit {edit_id} from feedback {feedback_event.feedback_id}")
                
                return {
                    'feedback_id': feedback_event.feedback_id,
                    'success': True,
                    'edit_id': edit_id,
                    'auto_processed': True,
                    'confidence_score': feedback_event.confidence_score,
                    'feedback_type': feedback_event.feedback_type.value
                }
            else:
                self.processing_stats['failed_edits'] += 1
                error_msg = result_uif.intermediate_data.get('error', 'Unknown error')
                
                return {
                    'feedback_id': feedback_event.feedback_id,
                    'success': False,
                    'error': f'MEMOIR edit failed: {error_msg}',
                    'auto_processed': True
                }
                
        except Exception as e:
            self.processing_stats['failed_edits'] += 1
            self.logger.error(f"Failed to create MEMOIR edit from feedback: {e}")

            return {
                'feedback_id': feedback_event.feedback_id,
                'success': False,
                'error': str(e),
                'auto_processed': True
            }

    # SELF-REFLECT Integration (Phase 5C)

    def process_autonomous_correction(self, uif: SAM_UIF) -> Dict[str, Any]:
        """
        Process autonomous corrections from SELF-REFLECT system.

        This method is called by the CoordinatorEngine when the AutonomousFactualCorrectionSkill
        completes with was_revised: True, automatically feeding corrections into MEMOIR.

        Args:
            uif: Universal Interface Format containing SELF-REFLECT results

        Returns:
            Processing result with MEMOIR edit information
        """
        try:
            # Extract SELF-REFLECT results
            was_revised = uif.intermediate_data.get("was_revised", False)
            revision_notes = uif.intermediate_data.get("revision_notes", "")
            final_response = uif.intermediate_data.get("final_response", "")
            original_query = uif.intermediate_data.get("original_query", "")
            initial_response = uif.intermediate_data.get("response_text", "")

            if not was_revised or not revision_notes:
                return {
                    'success': False,
                    'reason': 'No autonomous corrections to process',
                    'was_revised': was_revised
                }

            self.logger.info("Processing autonomous correction from SELF-REFLECT")

            # Parse revision notes for high-confidence factual corrections
            corrections = self._parse_revision_notes(revision_notes)

            if not corrections:
                return {
                    'success': False,
                    'reason': 'No parseable corrections found in revision notes',
                    'revision_notes': revision_notes
                }

            # Process each correction
            memoir_results = []
            for correction in corrections:
                if correction['confidence'] >= 0.8:  # High confidence threshold
                    memoir_result = self._create_memoir_edit_from_correction(
                        correction, original_query, initial_response, final_response
                    )
                    memoir_results.append(memoir_result)

            # Update statistics
            successful_edits = sum(1 for result in memoir_results if result.get('success', False))
            self.processing_stats['successful_edits'] += successful_edits
            self.processing_stats['failed_edits'] += len(memoir_results) - successful_edits

            self.logger.info(f"Processed {len(memoir_results)} autonomous corrections, {successful_edits} successful")

            return {
                'success': True,
                'corrections_processed': len(corrections),
                'memoir_edits_created': successful_edits,
                'memoir_results': memoir_results,
                'revision_notes': revision_notes
            }

        except Exception as e:
            self.logger.error(f"Failed to process autonomous correction: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _parse_revision_notes(self, revision_notes: str) -> List[Dict[str, Any]]:
        """
        Parse revision notes to extract structured corrections.

        Args:
            revision_notes: Raw revision notes from critique step

        Returns:
            List of structured correction objects
        """
        corrections = []

        try:
            # Pattern 1: Bullet point corrections
            bullet_pattern = r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)'
            bullet_matches = re.findall(bullet_pattern, revision_notes, re.DOTALL)

            for match in bullet_matches:
                correction_text = match.strip()
                if len(correction_text) > 10:  # Filter out very short items
                    corrections.append({
                        'text': correction_text,
                        'type': 'bullet_correction',
                        'confidence': self._assess_correction_confidence(correction_text)
                    })

            # Pattern 2: "Correction:" format
            correction_pattern = r'Correction:\s*(.+?)(?=\nCorrection:|\n\n|$)'
            correction_matches = re.findall(correction_pattern, revision_notes, re.DOTALL | re.IGNORECASE)

            for match in correction_matches:
                correction_text = match.strip()
                corrections.append({
                    'text': correction_text,
                    'type': 'explicit_correction',
                    'confidence': self._assess_correction_confidence(correction_text)
                })

            # Pattern 3: "X should be Y" format
            should_be_pattern = r'(.+?)\s+should be\s+(.+?)(?=\.|,|\n|$)'
            should_be_matches = re.findall(should_be_pattern, revision_notes, re.IGNORECASE)

            for incorrect, correct in should_be_matches:
                corrections.append({
                    'text': f"{incorrect.strip()} should be {correct.strip()}",
                    'type': 'should_be_correction',
                    'incorrect': incorrect.strip(),
                    'correct': correct.strip(),
                    'confidence': 0.9  # High confidence for explicit corrections
                })

            return corrections

        except Exception as e:
            self.logger.error(f"Error parsing revision notes: {e}")
            return []

    def _assess_correction_confidence(self, correction_text: str) -> float:
        """
        Assess confidence level of a correction based on language patterns.

        Args:
            correction_text: Text of the correction

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # High confidence indicators
        high_confidence_words = ['incorrect', 'wrong', 'error', 'should be', 'actually', 'fact']
        if any(word in correction_text.lower() for word in high_confidence_words):
            confidence += 0.3

        # Specific factual indicators
        factual_indicators = ['date', 'year', 'number', 'name', 'location', 'capital', 'president']
        if any(indicator in correction_text.lower() for indicator in factual_indicators):
            confidence += 0.2

        # Uncertainty indicators decrease confidence
        uncertainty_words = ['might', 'possibly', 'perhaps', 'unclear', 'uncertain']
        if any(word in correction_text.lower() for word in uncertainty_words):
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    def _create_memoir_edit_from_correction(
        self,
        correction: Dict[str, Any],
        original_query: str,
        initial_response: str,
        final_response: str
    ) -> Dict[str, Any]:
        """
        Create a MEMOIR edit from a parsed correction.

        Args:
            correction: Structured correction object
            original_query: Original user query
            initial_response: Initial response before correction
            final_response: Final corrected response

        Returns:
            MEMOIR edit result
        """
        try:
            # Get MEMOIR edit skill
            memoir_skills = self.memoir_integration.get_memoir_skills()
            if 'MEMOIR_EditSkill' not in memoir_skills:
                return {
                    'success': False,
                    'error': 'MEMOIR_EditSkill not available'
                }

            edit_skill = self.memoir_integration.memoir_skills['MEMOIR_EditSkill']

            # Create correction ID
            correction_id = f"auto_correct_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(correction['text']) % 10000}"

            # Prepare edit data
            edit_prompt = original_query
            correct_answer = final_response

            # Create UIF for the edit
            edit_uif = SAM_UIF(
                input_query=f"Autonomous correction: {correction_id}",
                intermediate_data={
                    'edit_prompt': edit_prompt,
                    'correct_answer': correct_answer,
                    'edit_context': f"Autonomous SELF-REFLECT correction - {correction['type']}",
                    'confidence_score': correction['confidence'],
                    'edit_metadata': {
                        'source': 'autonomous_self_reflect',
                        'correction_id': correction_id,
                        'correction_type': correction['type'],
                        'correction_text': correction['text'],
                        'original_response': initial_response,
                        'corrected_response': final_response,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': correction['confidence']
                    }
                }
            )

            # Execute the edit
            result_uif = edit_skill.execute(edit_uif)

            # Process results
            if result_uif.intermediate_data.get('edit_success', False):
                edit_id = result_uif.intermediate_data['edit_id']

                self.logger.info(f"✅ Created MEMOIR edit {edit_id} from autonomous correction")

                return {
                    'success': True,
                    'edit_id': edit_id,
                    'correction_id': correction_id,
                    'correction_type': correction['type'],
                    'confidence': correction['confidence'],
                    'correction_text': correction['text']
                }
            else:
                error_msg = result_uif.intermediate_data.get('error', 'Unknown error')
                return {
                    'success': False,
                    'error': f'MEMOIR edit failed: {error_msg}',
                    'correction_id': correction_id
                }

        except Exception as e:
            self.logger.error(f"Failed to create MEMOIR edit from autonomous correction: {e}")
            return {
                'success': False,
                'error': str(e),
                'correction_text': correction.get('text', 'Unknown')
            }
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback processing statistics."""
        success_rate = 0.0
        if self.processing_stats['total_feedback'] > 0:
            success_rate = self.processing_stats['successful_edits'] / self.processing_stats['total_feedback']
        
        return {
            'total_feedback_events': self.processing_stats['total_feedback'],
            'successful_edits': self.processing_stats['successful_edits'],
            'failed_edits': self.processing_stats['failed_edits'],
            'success_rate': success_rate,
            'feedback_by_type': self.processing_stats['feedback_types'],
            'recent_feedback_count': len([f for f in self.feedback_history if 
                                        (datetime.now() - f.timestamp).days < 7]),
            'processed_feedback_count': len([f for f in self.feedback_history if f.processed]),
            'configuration': self.config
        }
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent feedback events."""
        recent_feedback = sorted(
            self.feedback_history, 
            key=lambda x: x.timestamp, 
            reverse=True
        )[:limit]
        
        return [
            {
                'feedback_id': f.feedback_id,
                'feedback_type': f.feedback_type.value,
                'original_query': f.original_query[:100] + '...' if len(f.original_query) > 100 else f.original_query,
                'user_feedback': f.user_feedback[:100] + '...' if len(f.user_feedback) > 100 else f.user_feedback,
                'confidence_score': f.confidence_score,
                'processed': f.processed,
                'timestamp': f.timestamp.isoformat()
            }
            for f in recent_feedback
        ]
    
    def configure_feedback_handler(self, **config) -> bool:
        """
        Configure the feedback handler.
        
        Args:
            **config: Configuration parameters
            
        Returns:
            True if configuration successful
        """
        try:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.info(f"✅ Configured {key} = {value}")
                else:
                    self.logger.warning(f"⚠️  Unknown configuration key: {key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure feedback handler: {e}")
            return False


# Global feedback handler instance
_feedback_handler = None

def get_feedback_handler() -> MEMOIRFeedbackHandler:
    """Get the global feedback handler instance."""
    global _feedback_handler
    
    if _feedback_handler is None:
        _feedback_handler = MEMOIRFeedbackHandler()
    
    return _feedback_handler
