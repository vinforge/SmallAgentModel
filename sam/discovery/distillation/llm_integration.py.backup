"""
LLM Integration for Cognitive Distillation Engine
================================================

Handles LLM calls for principle discovery using the Symbolic Analyst prompt.

Author: SAM Development Team
Version: 1.0.0
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Represents an LLM response for principle discovery."""
    discovered_principle: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    raw_response: str

class LLMIntegration:
    """Handles LLM integration for principle discovery."""
    
    def __init__(self):
        """Initialize LLM integration."""
        self.model_name = "gpt-4"  # Default model
        self.max_tokens = 1000
        self.temperature = 0.3  # Lower temperature for more consistent analysis
        
        logger.info("LLM integration initialized")
    
    async def discover_principle_async(self, interaction_data: List[Dict]) -> Optional[LLMResponse]:
        """
        Discover a cognitive principle using LLM analysis (async version).
        
        Args:
            interaction_data: List of successful interaction data
            
        Returns:
            LLMResponse with discovered principle or None if failed
        """
        try:
            # Create the Symbolic Analyst prompt
            prompt = self._create_symbolic_analyst_prompt(interaction_data)
            
            # Make LLM call
            response = await self._call_llm_async(prompt)
            
            if not response:
                logger.error("LLM call failed")
                return None
            
            # Parse the response
            parsed_response = self._parse_llm_response(response)
            
            if not parsed_response:
                logger.error("Failed to parse LLM response")
                return None
            
            logger.info(f"Successfully discovered principle: {parsed_response.discovered_principle[:50]}...")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Principle discovery failed: {e}")
            return None
    
    def discover_principle(self, interaction_data: List[Dict]) -> Optional[LLMResponse]:
        """
        Discover a cognitive principle using LLM analysis (sync version).
        
        Args:
            interaction_data: List of successful interaction data
            
        Returns:
            LLMResponse with discovered principle or None if failed
        """
        try:
            # Run async version in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.discover_principle_async(interaction_data))
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Sync principle discovery failed: {e}")
            return None
    
    def _create_symbolic_analyst_prompt(self, interaction_data: List[Dict]) -> str:
        """Create the specialized Symbolic Analyst prompt for principle discovery."""
        
        # Enhanced prompt with better structure and examples
        prompt = """You are an expert cognitive scientist and AI strategist. Your task is to discover the underlying, simple, human-readable 'Principle of Reasoning' that explains the success of the following interactions.

ANALYSIS INSTRUCTIONS:
1. Analyze the relationship between queries, context, and successful responses
2. Look for patterns that make these responses particularly effective
3. Identify the abstract reasoning rule that governs success
4. Avoid simply describing the process - find the deeper principle
5. Express the principle as a clear, actionable guideline

SUCCESSFUL INTERACTION DATA:
{}

ANALYSIS FRAMEWORK:
- What makes these responses successful?
- What patterns appear across multiple interactions?
- What reasoning approach is consistently applied?
- How does context influence the response strategy?
- What can be generalized as a reusable principle?

RESPONSE FORMAT:
Provide your analysis in the following JSON format:
{{
    "discovered_principle": "A clear, actionable principle statement",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why this principle explains the success",
    "supporting_evidence": ["Evidence point 1", "Evidence point 2", "Evidence point 3"],
    "domain_applicability": ["domain1", "domain2"],
    "limitations": "Any limitations or caveats for this principle"
}}

EXAMPLE GOOD PRINCIPLES:
- "For financial queries, prioritize source recency and cite specific data points"
- "For technical problems, provide concrete examples before abstract explanations"
- "For research questions, synthesize multiple authoritative sources with explicit attribution"

Now analyze the provided interactions and discover the underlying principle:""".format(
            json.dumps(interaction_data, indent=2)
        )
        
        return prompt
    
    async def _call_llm_async(self, prompt: str) -> Optional[str]:
        """Make async LLM call."""
        try:
            # Try to use SAM's existing LLM infrastructure
            response = await self._call_sam_llm(prompt)
            if response:
                return response
            
            # Fallback to direct API calls
            response = await self._call_openai_api(prompt)
            if response:
                return response
            
            # Final fallback to local models
            response = await self._call_local_llm(prompt)
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    async def _call_sam_llm(self, prompt: str) -> Optional[str]:
        """Call SAM's existing LLM infrastructure."""
        try:
            # Import SAM's LLM components
            import sys
            from pathlib import Path
            
            # Try to import SAM's reasoning or core LLM modules
            try:
                from reasoning.tool_executor import ToolExecutor
                
                # Create a tool executor instance
                executor = ToolExecutor()
                
                # Make LLM call through SAM's infrastructure
                response = await executor.execute_llm_call(
                    prompt=prompt,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                if response and response.get('content'):
                    logger.info("Successfully used SAM's LLM infrastructure")
                    return response['content']
                    
            except ImportError:
                logger.warning("SAM's LLM infrastructure not available")
                return None
                
        except Exception as e:
            logger.warning(f"SAM LLM call failed: {e}")
            return None
    
    async def _call_openai_api(self, prompt: str) -> Optional[str]:
        """Call OpenAI API directly."""
        try:
            import openai
            import os
            
            # Check for API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found")
                return None
            
            openai.api_key = api_key
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cognitive scientist analyzing AI reasoning patterns."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if response.choices and response.choices[0].message:
                logger.info("Successfully used OpenAI API")
                return response.choices[0].message.content
                
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            return None
    
    async def _call_local_llm(self, prompt: str) -> Optional[str]:
        """Call local LLM (Ollama or similar)."""
        try:
            import aiohttp
            
            # Try Ollama API
            ollama_url = "http://localhost:11434/api/generate"
            
            payload = {
                "model": "llama2",  # Default local model
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(ollama_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('response'):
                            logger.info("Successfully used local LLM (Ollama)")
                            return result['response']
                            
        except Exception as e:
            logger.warning(f"Local LLM call failed: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[LLMResponse]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Validate required fields
                if 'discovered_principle' not in data:
                    logger.error("No discovered_principle in LLM response")
                    return None
                
                return LLMResponse(
                    discovered_principle=data['discovered_principle'],
                    confidence=data.get('confidence', 0.5),
                    reasoning=data.get('reasoning', ''),
                    metadata={
                        'supporting_evidence': data.get('supporting_evidence', []),
                        'domain_applicability': data.get('domain_applicability', []),
                        'limitations': data.get('limitations', ''),
                        'model_used': self.model_name
                    },
                    raw_response=response
                )
            else:
                # Fallback: extract principle from text
                principle = self._extract_principle_from_text(response)
                if principle:
                    return LLMResponse(
                        discovered_principle=principle,
                        confidence=0.4,  # Lower confidence for text extraction
                        reasoning="Extracted from unstructured response",
                        metadata={'extraction_method': 'text_parsing'},
                        raw_response=response
                    )
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            # Try text extraction fallback
            principle = self._extract_principle_from_text(response)
            if principle:
                return LLMResponse(
                    discovered_principle=principle,
                    confidence=0.3,
                    reasoning="Extracted from malformed JSON response",
                    metadata={'parse_error': str(e)},
                    raw_response=response
                )
        
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
    
    def _extract_principle_from_text(self, text: str) -> Optional[str]:
        """Extract principle from unstructured text response."""
        try:
            # Look for common principle patterns
            patterns = [
                r'principle[:\s]+([^.!?]+[.!?])',
                r'discovered[:\s]+([^.!?]+[.!?])',
                r'rule[:\s]+([^.!?]+[.!?])',
                r'guideline[:\s]+([^.!?]+[.!?])',
                r'for\s+\w+\s+queries?[,\s]+([^.!?]+[.!?])'
            ]
            
            import re
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    principle = match.group(1).strip()
                    if len(principle) > 10:  # Reasonable length
                        return principle
            
            # Fallback: look for sentences that sound like principles
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 20 and 
                    any(word in sentence.lower() for word in ['for', 'when', 'should', 'prioritize', 'emphasize'])):
                    return sentence + '.'
            
            return None
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return None

# Global LLM integration instance
llm_integration = LLMIntegration()
