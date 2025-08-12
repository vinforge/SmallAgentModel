"""
CLI for procedural memory operations:
- import-seeds: upsert all seeds from manifest into the encrypted store
Usage:
  python -m sam.procedural_memory.cli import-seeds [--manifest sam/procedural_memory/seeds/tasks_manifest.json]
Env for key management:
  SAM_PM_KEY_HEX, SAM_PM_KEY_B64, SAM_PM_KEY_PATH (one of these to set 32-byte key)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import ProceduralMemoryStore


def import_seeds(manifest_path: str) -> None:
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding='utf-8'))
    store = ProceduralMemoryStore()
    base = mp.parent
    count = 0
    for item in manifest['tasks']:
        fp = base / item['file']
        obj = json.loads(fp.read_text(encoding='utf-8'))
        store.upsert(obj)
        count += 1
    print(f"Imported {count} procedures from {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Procedural Memory CLI")
    sub = parser.add_subparsers(dest='cmd')

    p_import = sub.add_parser('import-seeds', help='Import seed procedures from manifest')
    p_import.add_argument('--manifest', type=str, default='sam/procedural_memory/seeds/tasks_manifest.json')

    args = parser.parse_args()
    if args.cmd == 'import-seeds':
        import_seeds(args.manifest)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

