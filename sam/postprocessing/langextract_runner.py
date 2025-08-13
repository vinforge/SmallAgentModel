#!/usr/bin/env python3
"""
LangExtract Post-Processing Runner (Placeholder)

Safely integrates LangExtract if installed; otherwise logs a skip.
No direct dependency is introduced; controlled via config in v2_upload_handler.load_pipeline_config().
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_langextract_for_document(
    file_path: str,
    document_id: str,
    model_id: str = "gemini-2.5-flash",
    extraction_passes: int = 2,
    max_workers: int = 8,
    max_char_buffer: int = 1000,
    output_dir: Optional[str] = None,
) -> bool:
    """Run LangExtract on a single document path if available.

    Writes outputs to uploads/{document_id}/ by default:
      - extractions.jsonl
      - visualization.html
    """
    try:
        # Import lazily; if missing, skip gracefully
        import langextract as lx  # type: ignore
    except Exception as e:
        logger.warning(f"LangExtract not available: {e}. Skipping extraction for {document_id}")
        return False

    try:
        # Resolve output directory
        out_dir = (
            Path(output_dir) if output_dir else Path("uploads") / document_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load the text: for now, let LangExtract read file_path directly if it's a URL or text.
        # If it's a PDF, lx can take raw text or content; here we pass the path and let lx handle URLs.
        # For binary PDFs, a prior text extraction would be ideal; this is a minimal placeholder.
        # Users can enhance by providing text_content from v2 storage.
        logger.info(
            f"Running LangExtract: model={model_id}, passes={extraction_passes}, workers={max_workers}"
        )

        result = lx.extract(
            text_or_documents=str(file_path),
            prompt_description="Extract key entities and facts relevant to QA. Use exact text spans.",
            examples=[],
            model_id=model_id,
            extraction_passes=extraction_passes,
            max_workers=max_workers,
            max_char_buffer=max_char_buffer,
        )

        # Save JSONL
        jsonl_path = out_dir / "extractions.jsonl"
        lx.io.save_annotated_documents([result], output_name=str(jsonl_path.name), output_dir=str(out_dir))

        # Generate HTML visualization
        html_content = lx.visualize(str(jsonl_path))
        (out_dir / "visualization.html").write_text(html_content, encoding="utf-8")

        logger.info(f"LangExtract outputs written to {out_dir}")
        return True

    except Exception as e:
        logger.error(f"LangExtract extraction failed for {document_id}: {e}")
        return False

