#!/usr/bin/env python3
"""
Citation utilities for grounding chat sources to document pages/anchors.
Best-effort mapping using v2 storage artifacts (uploads/{document_id}/metadata.json and chunk_metadata.json).
Falls back gracefully if artifacts are missing.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    # Collapse whitespace and lowercase for fuzzy matching
    return re.sub(r"\s+", " ", s or "").strip().lower()


@lru_cache(maxsize=64)
def _load_v2_artifacts(document_id: str) -> Tuple[str, List[Dict]]:
    """Load text_content and chunk metadata for a v2 document.
    Returns (text_content, chunk_meta_list). Empty values on failure.
    """
    try:
        base = Path("uploads") / document_id
        meta_path = base / "metadata.json"
        chunk_meta_path = base / "chunk_metadata.json"
        if not (meta_path.exists() and chunk_meta_path.exists()):
            return "", []
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = meta.get("text_content", "")
        cmeta = json.loads(chunk_meta_path.read_text(encoding="utf-8")).get("chunks", [])
        return text or "", cmeta or []
    except Exception as e:
        logger.debug(f"Failed to load v2 artifacts for {document_id}: {e}")
        return "", []


def _find_span_index(full_text: str, snippet: str, window: int = 300) -> Optional[int]:
    """Roughly locate snippet inside full_text; returns starting char index or None."""
    try:
        sn = _normalize(snippet)[:window]
        if not sn:
            return None
        ft = _normalize(full_text)
        return ft.find(sn)
    except Exception:
        return None


def try_format_citation_for_chunk(chunk_content: str, chunk_metadata: Dict) -> Optional[str]:
    """Return a compact citation like 'p. 12' if we can resolve it; else None.
    Uses v2 storage artifacts based on document_id.
    """
    try:
        doc_id = (chunk_metadata or {}).get("document_id")
        if not doc_id:
            return None
        text, chunks = _load_v2_artifacts(doc_id)
        if not text or not chunks:
            return None
        # Find approx start index in full text
        start_idx = _find_span_index(text, chunk_content[:500])
        if start_idx is None:
            return None
        # Find a chunk_meta whose doc span overlaps
        target = None
        for cm in chunks:
            s = cm.get("doc_char_start")
            e = cm.get("doc_char_end")
            if isinstance(s, int) and isinstance(e, int) and s <= start_idx < e:
                target = cm
                break
        if not target:
            # Fallback: pick nearest by distance
            best = None
            best_dist = 1e12
            for cm in chunks:
                s = cm.get("doc_char_start")
                e = cm.get("doc_char_end")
                if isinstance(s, int) and isinstance(e, int):
                    dist = 0 if s <= start_idx <= e else min(abs(start_idx - s), abs(start_idx - e))
                    if dist < best_dist:
                        best_dist = dist
                        best = cm
            target = best
        if not target:
            return None
        ps = target.get("page_spans") or []
        if not ps:
            return None
        page_number = ps[0].get("page_number")
        if not isinstance(page_number, int):
            return None
        return f"p. {page_number}"
    except Exception as e:
        logger.debug(f"Citation resolution failed: {e}")
        return None

