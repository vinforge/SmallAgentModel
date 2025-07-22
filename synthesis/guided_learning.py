"""
User-Guided Learning Feedback for SAM
Allows users to guide SAM's learning through interactive feedback and corrections.

Sprint 7 Task 5: User-Guided Learning Feedback
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback."""
    CORRECTION = "correction"
    IMPROVEMENT = "improvement"
    PREFERENCE = "preference"
    VALIDATION = "validation"
    GUIDANCE = "guidance"

class LearningPriority(Enum):
    """Priority levels for learning feedback."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class FeedbackEntry:
    """A single piece of user feedback."""
    feedback_id: str
    user_id: str
    session_id: str
    feedback_type: FeedbackType
    priority: LearningPriority
    original_query: str
    original_response: str
    user_feedback: str
    suggested_improvement: Optional[str]
    context: Dict[str, Any]
    timestamp: str
    processed: bool
    learning_applied: bool
    metadata: Dict[str, Any]

@dataclass
class LearningRule:
    """A learning rule derived from user feedback."""
    rule_id: str
    rule_type: str
    condition: str
    action: str
    confidence: float
    source_feedback_ids: List[str]
    created_at: str
    last_applied: Optional[str]
    application_count: int
    effectiveness_score: float
    metadata: Dict[str, Any]

class UserGuidedLearningManager:
    """
    Manages user-guided learning through interactive feedback and corrections.
    """

    def __init__(self, feedback_file: str = "user_feedback.json",
                 rules_file: str = "learning_rules.json"):
        """
        Initialize the user-guided learning manager.

        Args:
            feedback_file: Path to feedback storage file
            rules_file: Path to learning rules storage file
        """
        self.feedback_file = feedback_file
        self.rules_file = rules_file

        # Storage
        self.feedback_entries: Dict[str, FeedbackEntry] = {}
        self.learning_rules: Dict[str, LearningRule] = {}

        # Configuration
        self.config = {
            'auto_rule_generation': True,
            'min_feedback_for_rule': 2,
            'rule_confidence_threshold': 0.6,
            'max_rules': 1000
        }

        # Load existing data
        self._load_feedback()
        self._load_rules()

        logger.info(f"User-guided learning manager initialized: "
                   f"{len(self.feedback_entries)} feedback entries, "
                   f"{len(self.learning_rules)} rules")

    def collect_feedback(self, user_id: str, session_id: str,
                        original_query: str, original_response: str,
                        feedback_text: str, feedback_type: FeedbackType = FeedbackType.IMPROVEMENT,
                        priority: LearningPriority = LearningPriority.MEDIUM,
                        suggested_improvement: Optional[str] = None,
                        context: Dict[str, Any] = None) -> str:
        """
        Collect user feedback on a response.

        Args:
            user_id: User providing feedback
            session_id: Session ID
            original_query: Original query
            original_response: Original response
            feedback_text: User's feedback
            feedback_type: Type of feedback
            priority: Priority level
            suggested_improvement: User's suggested improvement
            context: Additional context

        Returns:
            Feedback ID
        """
        try:
            import uuid
            feedback_id = f"feedback_{uuid.uuid4().hex[:12]}"

            feedback_entry = FeedbackEntry(
                feedback_id=feedback_id,
                user_id=user_id,
                session_id=session_id,
                feedback_type=feedback_type,
                priority=priority,
                original_query=original_query,
                original_response=original_response,
                user_feedback=feedback_text,
                suggested_improvement=suggested_improvement,
                context=context or {},
                timestamp=datetime.now().isoformat(),
                processed=False,
                learning_applied=False,
                metadata={}
            )

            self.feedback_entries[feedback_id] = feedback_entry
            self._save_feedback()

            # Process feedback immediately if auto-rule generation is enabled
            if self.config['auto_rule_generation']:
                self._process_feedback(feedback_entry)

            logger.info(f"Collected feedback: {feedback_type.value} from user {user_id}")
            return feedback_id

        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            raise

    def provide_correction(self, user_id: str, session_id: str,
                          original_query: str, original_response: str,
                          corrected_response: str, explanation: str = "") -> str:
        """
        Provide a correction to SAM's response.

        Args:
            user_id: User providing correction
            session_id: Session ID
            original_query: Original query
            original_response: Original (incorrect) response
            corrected_response: Corrected response
            explanation: Explanation of the correction

        Returns:
            Feedback ID
        """
        feedback_text = f"Correction: {explanation}" if explanation else "Response corrected"

        return self.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            original_query=original_query,
            original_response=original_response,
            feedback_text=feedback_text,
            feedback_type=FeedbackType.CORRECTION,
            priority=LearningPriority.HIGH,
            suggested_improvement=corrected_response,
            context={'correction_type': 'full_response'}
        )

    def provide_preference(self, user_id: str, session_id: str,
                          query: str, response_a: str, response_b: str,
                          preferred_response: str, reason: str = "") -> str:
        """
        Provide preference between two responses.

        Args:
            user_id: User providing preference
            session_id: Session ID
            query: Original query
            response_a: First response option
            response_b: Second response option
            preferred_response: Which response was preferred ('a' or 'b')
            reason: Reason for preference

        Returns:
            Feedback ID
        """
        feedback_text = f"Preferred response {preferred_response}: {reason}"

        context = {
            'preference_type': 'response_comparison',
            'response_a': response_a,
            'response_b': response_b,
            'preferred': preferred_response,
            'reason': reason
        }

        return self.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            original_query=query,
            original_response=response_a if preferred_response == 'b' else response_b,
            feedback_text=feedback_text,
            feedback_type=FeedbackType.PREFERENCE,
            priority=LearningPriority.MEDIUM,
            suggested_improvement=response_a if preferred_response == 'a' else response_b,
            context=context
        )

    def validate_response(self, user_id: str, session_id: str,
                         query: str, response: str, is_correct: bool,
                         confidence: float = 1.0, notes: str = "") -> str:
        """
        Validate whether a response is correct.

        Args:
            user_id: User providing validation
            session_id: Session ID
            query: Original query
            response: Response to validate
            is_correct: Whether the response is correct
            confidence: Confidence in the validation (0.0-1.0)
            notes: Additional notes

        Returns:
            Feedback ID
        """
        feedback_text = f"Response {'validated as correct' if is_correct else 'marked as incorrect'}"
        if notes:
            feedback_text += f": {notes}"

        context = {
            'validation_type': 'correctness',
            'is_correct': is_correct,
            'confidence': confidence,
            'notes': notes
        }

        priority = LearningPriority.HIGH if not is_correct else LearningPriority.MEDIUM

        return self.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            original_query=query,
            original_response=response,
            feedback_text=feedback_text,
            feedback_type=FeedbackType.VALIDATION,
            priority=priority,
            context=context
        )

    def provide_guidance(self, user_id: str, session_id: str,
                        topic: str, guidance_text: str,
                        guidance_type: str = "general") -> str:
        """
        Provide general guidance for future responses.

        Args:
            user_id: User providing guidance
            session_id: Session ID
            topic: Topic the guidance applies to
            guidance_text: The guidance text
            guidance_type: Type of guidance

        Returns:
            Feedback ID
        """
        context = {
            'guidance_type': guidance_type,
            'topic': topic,
            'applies_to': 'future_responses'
        }

        return self.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            original_query=f"Guidance for topic: {topic}",
            original_response="",
            feedback_text=guidance_text,
            feedback_type=FeedbackType.GUIDANCE,
            priority=LearningPriority.MEDIUM,
            context=context
        )

    def get_applicable_rules(self, query: str, context: Dict[str, Any] = None) -> List[LearningRule]:
        """
        Get learning rules applicable to a query.

        Args:
            query: Query to find applicable rules for
            context: Additional context

        Returns:
            List of applicable learning rules
        """
        try:
            applicable_rules = []
            query_lower = query.lower()

            for rule in self.learning_rules.values():
                if rule.confidence < self.config['rule_confidence_threshold']:
                    continue

                # Simple condition matching (would be more sophisticated in practice)
                condition_lower = rule.condition.lower()

                # Check if rule condition matches query
                if self._rule_matches_query(rule, query, context):
                    applicable_rules.append(rule)

            # Sort by effectiveness and confidence
            applicable_rules.sort(
                key=lambda r: (r.effectiveness_score, r.confidence),
                reverse=True
            )

            return applicable_rules[:5]  # Return top 5 rules

        except Exception as e:
            logger.error(f"Error getting applicable rules: {e}")
            return []

    def apply_learning_rule(self, rule: LearningRule, query: str,
                           context: Dict[str, Any] = None) -> Optional[str]:
        """
        Apply a learning rule to modify response generation.

        Args:
            rule: Learning rule to apply
            query: Current query
            context: Current context

        Returns:
            Guidance text if applicable, None otherwise
        """
        try:
            # Update rule usage statistics
            rule.last_applied = datetime.now().isoformat()
            rule.application_count += 1

            # Generate guidance based on rule action
            guidance = self._generate_guidance_from_rule(rule, query, context)

            self._save_rules()

            logger.debug(f"Applied learning rule: {rule.rule_id}")
            return guidance

        except Exception as e:
            logger.error(f"Error applying learning rule {rule.rule_id}: {e}")
            return None

    def get_feedback_summary(self, user_id: Optional[str] = None,
                           days_back: int = 30) -> Dict[str, Any]:
        """
        Get summary of feedback received.

        Args:
            user_id: Filter by user ID
            days_back: Number of days to look back

        Returns:
            Feedback summary
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Filter feedback entries
            relevant_feedback = []
            for feedback in self.feedback_entries.values():
                feedback_date = datetime.fromisoformat(feedback.timestamp)

                if feedback_date >= cutoff_date:
                    if user_id is None or feedback.user_id == user_id:
                        relevant_feedback.append(feedback)

            # Calculate statistics
            total_feedback = len(relevant_feedback)
            feedback_by_type = {}
            feedback_by_priority = {}
            processed_count = 0
            applied_count = 0

            for feedback in relevant_feedback:
                # Count by type
                feedback_type = feedback.feedback_type.value
                feedback_by_type[feedback_type] = feedback_by_type.get(feedback_type, 0) + 1

                # Count by priority
                priority = feedback.priority.value
                feedback_by_priority[priority] = feedback_by_priority.get(priority, 0) + 1

                # Count processed and applied
                if feedback.processed:
                    processed_count += 1
                if feedback.learning_applied:
                    applied_count += 1

            return {
                'total_feedback': total_feedback,
                'feedback_by_type': feedback_by_type,
                'feedback_by_priority': feedback_by_priority,
                'processed_count': processed_count,
                'applied_count': applied_count,
                'processing_rate': processed_count / total_feedback if total_feedback > 0 else 0,
                'application_rate': applied_count / total_feedback if total_feedback > 0 else 0,
                'rules_generated': len(self.learning_rules),
                'period_days': days_back
            }

        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            return {}

    def _process_feedback(self, feedback: FeedbackEntry):
        """Process feedback to potentially generate learning rules."""
        try:
            # Mark as processed
            feedback.processed = True

            # Generate learning rule based on feedback type
            if feedback.feedback_type == FeedbackType.CORRECTION:
                self._generate_correction_rule(feedback)
            elif feedback.feedback_type == FeedbackType.PREFERENCE:
                self._generate_preference_rule(feedback)
            elif feedback.feedback_type == FeedbackType.VALIDATION:
                self._generate_validation_rule(feedback)
            elif feedback.feedback_type == FeedbackType.GUIDANCE:
                self._generate_guidance_rule(feedback)

            self._save_feedback()

        except Exception as e:
            logger.error(f"Error processing feedback {feedback.feedback_id}: {e}")

    def _generate_correction_rule(self, feedback: FeedbackEntry):
        """Generate a learning rule from correction feedback."""
        try:
            import uuid
            rule_id = f"rule_{uuid.uuid4().hex[:12]}"

            # Extract patterns from the correction
            query_pattern = self._extract_query_pattern(feedback.original_query)

            rule = LearningRule(
                rule_id=rule_id,
                rule_type="correction",
                condition=f"query_pattern:{query_pattern}",
                action=f"avoid_response_pattern:{self._extract_response_pattern(feedback.original_response)}",
                confidence=0.8,
                source_feedback_ids=[feedback.feedback_id],
                created_at=datetime.now().isoformat(),
                last_applied=None,
                application_count=0,
                effectiveness_score=0.5,
                metadata={
                    'original_query': feedback.original_query,
                    'corrected_response': feedback.suggested_improvement,
                    'user_feedback': feedback.user_feedback
                }
            )

            self.learning_rules[rule_id] = rule
            feedback.learning_applied = True

            self._save_rules()

            logger.info(f"Generated correction rule: {rule_id}")

        except Exception as e:
            logger.error(f"Error generating correction rule: {e}")

    def _generate_preference_rule(self, feedback: FeedbackEntry):
        """Generate a learning rule from preference feedback."""
        try:
            import uuid
            rule_id = f"rule_{uuid.uuid4().hex[:12]}"

            preferred_response = feedback.suggested_improvement
            reason = feedback.context.get('reason', '')

            rule = LearningRule(
                rule_id=rule_id,
                rule_type="preference",
                condition=f"query_type:{self._classify_query_type(feedback.original_query)}",
                action=f"prefer_style:{self._extract_style_preference(preferred_response, reason)}",
                confidence=0.6,
                source_feedback_ids=[feedback.feedback_id],
                created_at=datetime.now().isoformat(),
                last_applied=None,
                application_count=0,
                effectiveness_score=0.5,
                metadata={
                    'preferred_response': preferred_response,
                    'reason': reason
                }
            )

            self.learning_rules[rule_id] = rule
            feedback.learning_applied = True

            self._save_rules()

            logger.info(f"Generated preference rule: {rule_id}")

        except Exception as e:
            logger.error(f"Error generating preference rule: {e}")

    def _generate_validation_rule(self, feedback: FeedbackEntry):
        """Generate a learning rule from validation feedback."""
        if not feedback.context.get('is_correct', True):
            # Only generate rules for incorrect responses
            self._generate_correction_rule(feedback)

    def _generate_guidance_rule(self, feedback: FeedbackEntry):
        """Generate a learning rule from guidance feedback."""
        try:
            import uuid
            rule_id = f"rule_{uuid.uuid4().hex[:12]}"

            topic = feedback.context.get('topic', 'general')
            guidance_type = feedback.context.get('guidance_type', 'general')

            rule = LearningRule(
                rule_id=rule_id,
                rule_type="guidance",
                condition=f"topic:{topic}",
                action=f"apply_guidance:{feedback.user_feedback}",
                confidence=0.7,
                source_feedback_ids=[feedback.feedback_id],
                created_at=datetime.now().isoformat(),
                last_applied=None,
                application_count=0,
                effectiveness_score=0.5,
                metadata={
                    'topic': topic,
                    'guidance_type': guidance_type,
                    'guidance_text': feedback.user_feedback
                }
            )

            self.learning_rules[rule_id] = rule
            feedback.learning_applied = True

            self._save_rules()

            logger.info(f"Generated guidance rule: {rule_id}")

        except Exception as e:
            logger.error(f"Error generating guidance rule: {e}")

    def _rule_matches_query(self, rule: LearningRule, query: str,
                           context: Dict[str, Any] = None) -> bool:
        """Check if a rule's condition matches the current query."""
        condition = rule.condition.lower()
        query_lower = query.lower()

        # Simple pattern matching (would be more sophisticated in practice)
        if condition.startswith('query_pattern:'):
            pattern = condition.split(':', 1)[1]
            return pattern in query_lower
        elif condition.startswith('query_type:'):
            query_type = condition.split(':', 1)[1]
            return self._classify_query_type(query) == query_type
        elif condition.startswith('topic:'):
            topic = condition.split(':', 1)[1]
            return topic in query_lower

        return False

    def _generate_guidance_from_rule(self, rule: LearningRule, query: str,
                                   context: Dict[str, Any] = None) -> str:
        """Generate guidance text from a learning rule."""
        action = rule.action

        if action.startswith('avoid_response_pattern:'):
            return f"Avoid patterns similar to: {action.split(':', 1)[1]}"
        elif action.startswith('prefer_style:'):
            style = action.split(':', 1)[1]
            return f"Use {style} style for this type of query"
        elif action.startswith('apply_guidance:'):
            guidance = action.split(':', 1)[1]
            return guidance

        return f"Apply learning rule: {rule.rule_type}"

    def _extract_query_pattern(self, query: str) -> str:
        """Extract a pattern from a query for rule matching."""
        # Simplified pattern extraction
        words = query.lower().split()
        if len(words) > 3:
            return ' '.join(words[:3])  # First 3 words
        return query.lower()

    def _extract_response_pattern(self, response: str) -> str:
        """Extract a pattern from a response to avoid."""
        # Simplified pattern extraction
        sentences = response.split('.')
        if sentences:
            return sentences[0][:50]  # First 50 chars of first sentence
        return response[:50]

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return 'definitional'
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'because', 'reason']):
            return 'causal'
        elif any(word in query_lower for word in ['compare', 'difference']):
            return 'comparative'
        else:
            return 'general'

    def _extract_style_preference(self, preferred_response: str, reason: str) -> str:
        """Extract style preference from preferred response and reason."""
        # Simplified style extraction
        if 'concise' in reason.lower() or len(preferred_response) < 200:
            return 'concise'
        elif 'detailed' in reason.lower() or len(preferred_response) > 500:
            return 'detailed'
        elif 'technical' in reason.lower():
            return 'technical'
        elif 'simple' in reason.lower():
            return 'simple'
        else:
            return 'balanced'

    def _load_feedback(self):
        """Load feedback from storage."""
        try:
            from pathlib import Path

            feedback_path = Path(self.feedback_file)
            if feedback_path.exists():
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for feedback_data in data.get('feedback', []):
                    feedback = FeedbackEntry(
                        feedback_id=feedback_data['feedback_id'],
                        user_id=feedback_data['user_id'],
                        session_id=feedback_data['session_id'],
                        feedback_type=FeedbackType(feedback_data['feedback_type']),
                        priority=LearningPriority(feedback_data['priority']),
                        original_query=feedback_data['original_query'],
                        original_response=feedback_data['original_response'],
                        user_feedback=feedback_data['user_feedback'],
                        suggested_improvement=feedback_data.get('suggested_improvement'),
                        context=feedback_data.get('context', {}),
                        timestamp=feedback_data['timestamp'],
                        processed=feedback_data.get('processed', False),
                        learning_applied=feedback_data.get('learning_applied', False),
                        metadata=feedback_data.get('metadata', {})
                    )

                    self.feedback_entries[feedback.feedback_id] = feedback

                logger.info(f"Loaded {len(self.feedback_entries)} feedback entries")

        except Exception as e:
            logger.error(f"Error loading feedback: {e}")

    def _save_feedback(self):
        """Save feedback to storage."""
        try:
            from pathlib import Path

            # Convert to serializable format
            feedback_data = []
            for feedback in self.feedback_entries.values():
                feedback_dict = {
                    'feedback_id': feedback.feedback_id,
                    'user_id': feedback.user_id,
                    'session_id': feedback.session_id,
                    'feedback_type': feedback.feedback_type.value,
                    'priority': feedback.priority.value,
                    'original_query': feedback.original_query,
                    'original_response': feedback.original_response,
                    'user_feedback': feedback.user_feedback,
                    'suggested_improvement': feedback.suggested_improvement,
                    'context': feedback.context,
                    'timestamp': feedback.timestamp,
                    'processed': feedback.processed,
                    'learning_applied': feedback.learning_applied,
                    'metadata': feedback.metadata
                }
                feedback_data.append(feedback_dict)

            data = {
                'feedback': feedback_data,
                'last_updated': datetime.now().isoformat()
            }

            # Ensure directory exists
            feedback_path = Path(self.feedback_file)
            feedback_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.feedback_entries)} feedback entries")

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")

    def _load_rules(self):
        """Load learning rules from storage."""
        try:
            from pathlib import Path

            rules_path = Path(self.rules_file)
            if rules_path.exists():
                with open(rules_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for rule_data in data.get('rules', []):
                    rule = LearningRule(
                        rule_id=rule_data['rule_id'],
                        rule_type=rule_data['rule_type'],
                        condition=rule_data['condition'],
                        action=rule_data['action'],
                        confidence=rule_data['confidence'],
                        source_feedback_ids=rule_data['source_feedback_ids'],
                        created_at=rule_data['created_at'],
                        last_applied=rule_data.get('last_applied'),
                        application_count=rule_data.get('application_count', 0),
                        effectiveness_score=rule_data.get('effectiveness_score', 0.5),
                        metadata=rule_data.get('metadata', {})
                    )

                    self.learning_rules[rule.rule_id] = rule

                logger.info(f"Loaded {len(self.learning_rules)} learning rules")

        except Exception as e:
            logger.error(f"Error loading learning rules: {e}")

    def _save_rules(self):
        """Save learning rules to storage."""
        try:
            from pathlib import Path

            # Convert to serializable format
            rules_data = []
            for rule in self.learning_rules.values():
                rule_dict = {
                    'rule_id': rule.rule_id,
                    'rule_type': rule.rule_type,
                    'condition': rule.condition,
                    'action': rule.action,
                    'confidence': rule.confidence,
                    'source_feedback_ids': rule.source_feedback_ids,
                    'created_at': rule.created_at,
                    'last_applied': rule.last_applied,
                    'application_count': rule.application_count,
                    'effectiveness_score': rule.effectiveness_score,
                    'metadata': rule.metadata
                }
                rules_data.append(rule_dict)

            data = {
                'rules': rules_data,
                'last_updated': datetime.now().isoformat()
            }

            # Ensure directory exists
            rules_path = Path(self.rules_file)
            rules_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(rules_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.learning_rules)} learning rules")

        except Exception as e:
            logger.error(f"Error saving learning rules: {e}")

# Global guided learning manager instance
_guided_learning_manager = None

def get_guided_learning_manager(feedback_file: str = "user_feedback.json",
                               rules_file: str = "learning_rules.json") -> UserGuidedLearningManager:
    """Get or create a global guided learning manager instance."""
    global _guided_learning_manager

    if _guided_learning_manager is None:
        _guided_learning_manager = UserGuidedLearningManager(
            feedback_file=feedback_file,
            rules_file=rules_file
        )

    return _guided_learning_manager