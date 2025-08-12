"""
Retrieval and injection utilities for procedural memory.
- retrieve_top1: returns top procedure for a task, if any
- inject_into_context: stores procedure into SAMContextManager for planner access
"""
from __future__ import annotations

from typing import Optional, Dict, Any

from .store import ProceduralMemoryStore
from sam.agent_zero.planning.sam_context_manager import SAMContextManager


def retrieve_top1(task_description: str, min_similarity: float = 0.25) -> Optional[Dict[str, Any]]:
    store = ProceduralMemoryStore()
    result = store.retrieve_top1(task_description, min_similarity=min_similarity)
    if not result:
        return None
    return {
        'procedure': result.procedure_json,
        'similarity': result.similarity,
        'name': result.name,
        'procedure_id': result.procedure_id,
    }


def inject_into_context(context_manager: SAMContextManager, retrieval_payload: Dict[str, Any]) -> None:
    context_manager.set_procedural_guidance(
        procedure=retrieval_payload['procedure'],
        similarity=retrieval_payload['similarity'],
    )

