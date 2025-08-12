"""
Validator for distilled_procedure_json using JSON Schema.
Integrates with SAM's Python stack (jsonschema required).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel
from jsonschema import Draft202012Validator  # jsonschema is required

SCHEMA_PATH = Path(__file__).parent / "schema" / "distilled_procedure.schema.json"


class ValidationReport(BaseModel):
    is_valid: bool
    errors: list[str] = []


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_procedure(data: Dict[str, Any]) -> ValidationReport:
    """
    Validate a distilled procedure object against the JSON Schema.
    """
    schema = load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        formatted = [f"{'/'.join([str(p) for p in e.path])}: {e.message}" for e in errors]
        return ValidationReport(is_valid=False, errors=formatted)
    return ValidationReport(is_valid=True, errors=[])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sam.procedural_memory.validator <path_to_procedure.json>")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    obj = json.loads(path.read_text(encoding="utf-8"))
    report = validate_procedure(obj)
    print(json.dumps(report.model_dump(), indent=2))

