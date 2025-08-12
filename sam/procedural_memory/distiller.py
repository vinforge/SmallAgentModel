"""
Reference distiller script for Memρ-style procedural memory Build operation.
- Accepts a trajectory (task text + tool call logs)
- Produces a distilled_procedure_json instance per schema
This is a minimal reference; integrate with your LLM of choice.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .validator import validate_procedure

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    observation: Any

@dataclass
class Trajectory:
    task_text: str
    steps: List[ToolCall]
    outcome: str  # "success" | "failure"


def load_tool_catalog() -> List[Dict[str, Any]]:
    """Return a minimal catalog of tool names and signatures.
    Replace or integrate with tools/tool_registry.py if desired.
    """
    return [
        {"name": "WebSearchTool", "sig": "(query: string) -> results[]"},
        {"name": "HeadlessBrowserTool", "sig": "(url: string) -> html"},
        {"name": "HttpGetTool", "sig": "(url: string, timeout_ms?: number) -> {status_code, body}"},
        {"name": "RSSReaderTool", "sig": "(url: string) -> items[]"},
        {"name": "CocoIndexTool", "sig": "(seed_url: string, depth?: number) -> index_id"},
        {"name": "CocoQueryTool", "sig": "(index_id: string, query: string) -> passages[]"},
        {"name": "FileWriteTool", "sig": "(path: string, content: string) -> ok"},
        {"name": "ArticleExtractorTool", "sig": "(html: string) -> text"},
        {"name": "SummarizeTool", "sig": "(text: string, bullets?: number) -> summary"},
    ]


def build_distillation_prompt(traj: Trajectory) -> str:
    catalog = load_tool_catalog()
    return (
        "You are a skill distiller for SAM. Convert an execution trajectory into a reusable, concise procedure. "
        "Use the distilled_procedure_json schema. Minimize steps while preserving reliability. Generalize with slots. "
        "Replace secrets with ${slot} placeholders. Use existing tool names exactly. Output JSON only.\n\n"
        + "Tool catalog:\n"
        + json.dumps(catalog, indent=2)
        + "\n\nTrajectory:\n"
        + json.dumps({
            "task_text": traj.task_text,
            "steps": [{"name": s.name, "args": s.args, "observation": str(s.observation)[:500]} for s in traj.steps],
            "outcome": traj.outcome,
        }, indent=2)
        + "\n\nInstructions: Keep 4–12 steps, add minimal pre/postconditions and success_criteria, set provenance.origin based on outcome."
    )


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call SAM's existing LLM integration; fallback to local if unavailable.
    Returns raw text string from model.
    """
    try:
        # Prefer SAM's distillation LLM integration
        from sam.discovery.distillation.llm_integration import LLMIntegration
        llm = LLMIntegration()
        llm.model_name = model
        # Use sync discover call path as a generic call via its internal methods
        # We re-use its private _call_* methods to send a single message prompt.
        import asyncio
        async def _run():
            resp = await llm._call_sam_llm(prompt)  # type: ignore[attr-defined]
            if not resp:
                resp = await llm._call_openai_api(prompt)  # type: ignore[attr-defined]
            if not resp:
                resp = await llm._call_local_llm(prompt)  # type: ignore[attr-defined]
            return resp
        return asyncio.get_event_loop().run_until_complete(_run()) or ""
    except Exception:
        # Last resort: return empty to force caller to provide mock
        return ""


def distill(traj: Trajectory, llm_response: Optional[str] = None) -> Dict[str, Any]:
    """Distill a trajectory into a procedure JSON object.
    If llm_response is provided, use it; otherwise call_llm().
    """
    prompt = build_distillation_prompt(traj)
    if llm_response is None:
        llm_response = call_llm(prompt)

    # Extract JSON block from llm_response
    start = llm_response.find("{")
    end = llm_response.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain JSON.")
    obj = json.loads(llm_response[start:end])

    report = validate_procedure(obj)
    if not report.is_valid:
        raise ValueError("Procedure failed validation: " + "; ".join(report.errors))

    return obj


def save_procedure(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distill a trajectory into a procedure JSON")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory.json")
    parser.add_argument("--out", type=str, required=True, help="Path to output procedure.json")
    parser.add_argument("--mock", type=str, help="Optional path to mock LLM response JSON")
    args = parser.parse_args()

    traj_data = json.loads(Path(args.trajectory).read_text(encoding="utf-8"))
    traj = Trajectory(
        task_text=traj_data["task_text"],
        steps=[ToolCall(**s) for s in traj_data["steps"]],
        outcome=traj_data.get("outcome", "success"),
    )

    mock_resp = None
    if args.mock:
        mock_resp = Path(args.mock).read_text(encoding="utf-8")

    proc = distill(traj, llm_response=mock_resp)
    save_procedure(proc, Path(args.out))
    print(f"Saved procedure to {args.out}")

