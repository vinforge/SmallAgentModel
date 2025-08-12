# SAM Procedural Memory (Memρ-style)

This module contains:
- JSON Schema for `distilled_procedure_json`
- TypeScript type definition (for tooling and editors)
- Python validator
- Reference distiller script (Build operation) expecting a trajectory
- Seed procedures for 10 common tasks (Phase 1 testing)

Directories:
- schema/: JSON Schema file
- types/: TypeScript `.d.ts`
- seeds/: JSON procedures that conform to the schema

Usage:
- Validate a procedure JSON:
  ```bash
  python -m sam.procedural_memory.validator sam/procedural_memory/seeds/fetch_and_summarize.json
  ```
- Distill from a trajectory with a mocked LLM output:
  ```bash
  python -m sam.procedural_memory.distiller --trajectory /path/to/trajectory.json --out /tmp/out.json --mock /path/to/mock_llm.json
  ```

Integration notes:
- Hook retrieval before planning; inject top-1 procedure as soft guidance.
- Enqueue distillation after task completion; store validated procedures.

