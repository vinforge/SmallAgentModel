"""
ProceduralMemoryStore: stores/retrieves distilled procedures with embeddings.
Vectors via EncryptedChromaStore; metadata and JSON encrypted-at-rest via CryptoManager.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .validator import validate_procedure
from utils.embedding_utils import get_embedding_manager
from security.encrypted_chroma_store import EncryptedChromaStore
from security.crypto_utils import CryptoManager

DB_PATH = Path("memory_store/procedural_memory.db")
VECTOR_DIR = "memory_store/encrypted_vectors"
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS procedures (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  intent_signature TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  scope TEXT,
  provenance_origin TEXT,
  author_model TEXT,
  semver TEXT NOT NULL,
  success_count INTEGER DEFAULT 0,
  failure_count INTEGER DEFAULT 0,
  last_used_at TEXT,
  deprecation_status TEXT DEFAULT 'active',
  deprecation_reason TEXT,
  tags TEXT,
  domain TEXT,
  json_encrypted TEXT NOT NULL
);
"""

@dataclass
class RetrievalResult:
    procedure_id: str
    name: str
    similarity: float
    procedure_json: Dict[str, Any]

class ProceduralMemoryStore:
    def __init__(self, db_path: Path = DB_PATH, vector_dir: str = VECTOR_DIR, embedding_model: str = "all-MiniLM-L6-v2", crypto: Optional[CryptoManager] = None):
        self.db_path = db_path
        self.embedder = get_embedding_manager(embedding_model)
        self.crypto = crypto or CryptoManager()
        # Persistent key management: prefer env/file-based key if available
        import os, base64
        if not self.crypto.is_initialized():
            key_hex = os.environ.get('SAM_PM_KEY_HEX')
            key_b64 = os.environ.get('SAM_PM_KEY_B64')
            key_path = os.environ.get('SAM_PM_KEY_PATH')
            key_bytes: Optional[bytes] = None
            try:
                if key_hex:
                    key_bytes = bytes.fromhex(key_hex.strip())
                elif key_b64:
                    key_bytes = base64.b64decode(key_b64.strip())
                elif key_path and os.path.exists(key_path):
                    with open(key_path, 'rb') as f:
                        data = f.read().strip()
                        try:
                            key_bytes = bytes.fromhex(data.decode('utf-8'))
                        except Exception:
                            key_bytes = base64.b64decode(data)
            except Exception:
                key_bytes = None
            if key_bytes and len(key_bytes) == 32:
                self.crypto.set_session_key(key_bytes)
            else:
                key = os.urandom(32)
                self.crypto.set_session_key(key)
        # Initialize encrypted vector store
        self.vector = EncryptedChromaStore(collection_name="sam_secure_proc_vectors", crypto_manager=self.crypto, storage_path=vector_dir)
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()

    def upsert(self, proc: Dict[str, Any], proc_id: Optional[str] = None) -> str:
        report = validate_procedure(proc)
        if not report.is_valid:
            raise ValueError("Invalid procedure: " + "; ".join(report.errors))
        import uuid, datetime
        pid = proc_id or uuid.uuid4().hex
        intent = proc.get("intent", {})
        sig = intent.get("signature_text", "")
        if not sig:
            raise ValueError("intent.signature_text required")
        vec = self.embedder.embed_query(sig)
        # Encrypted vector store: embedding list expected
        self.vector.add_memory_chunk(
            chunk_text=sig,
            metadata={"name": proc.get("name"), "tags": intent.get("tags", []), "domain": intent.get("domain", None)},
            embedding=list(map(float, vec.tolist() if hasattr(vec, 'tolist') else vec)),
            chunk_id=pid,
        )
        # Encrypt full JSON for at-rest storage (SQLite)
        encrypted_json = self.crypto.encrypt_data(json.dumps(proc, ensure_ascii=False))
        now = datetime.datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO procedures (id, name, intent_signature, embedding_model, created_at, updated_at, scope,
                  provenance_origin, author_model, semver, success_count, failure_count, last_used_at, deprecation_status,
                  deprecation_reason, tags, domain, json_encrypted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, 'active', NULL, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  name=excluded.name,
                  intent_signature=excluded.intent_signature,
                  embedding_model=excluded.embedding_model,
                  updated_at=excluded.updated_at,
                  scope=excluded.scope,
                  provenance_origin=excluded.provenance_origin,
                  author_model=excluded.author_model,
                  semver=excluded.semver,
                  tags=excluded.tags,
                  domain=excluded.domain,
                  json_encrypted=excluded.json_encrypted
                """,
                (
                    pid,
                    proc.get("name"),
                    sig,
                    intent.get("embedding_model", "all-MiniLM-L6-v2"),
                    now,
                    now,
                    proc.get("scope"),
                    (proc.get("provenance") or {}).get("origin"),
                    (proc.get("provenance") or {}).get("author_model"),
                    proc.get("semver"),
                    json.dumps(intent.get("tags")),
                    intent.get("domain"),
                    encrypted_json,
                ),
            )
            conn.commit()
        return pid

    def retrieve_top1(self, task_text: str, min_similarity: float = 0.25) -> Optional[RetrievalResult]:
        q = self.embedder.embed_query(task_text)
        results = self.vector.query_memories(query_embedding=list(map(float, q.tolist() if hasattr(q, 'tolist') else q)), n_results=5)
        if not results:
            return None
        # Convert 'distance' to similarity proxy (1 - distance) if distance is provided; else default
        # Chroma returns smaller distances for closer matches; we invert to similarity.
        results = sorted(results, key=lambda r: r.get('distance', 1.0))
        top = results[0]
        pid = top.get('id')
        # Apply threshold if distance present
        if top.get('distance') is not None and (1.0 - float(top['distance'])) < min_similarity:
            return None
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT json_encrypted, name FROM procedures WHERE id=?", (pid,))
            row = cur.fetchone()
            if not row:
                return None
            decrypted = self.crypto.decrypt_data(row[0])
            proc_json = json.loads(decrypted)
            similarity = 1.0 - float(top.get('distance', 1.0))
            return RetrievalResult(
                procedure_id=pid,
                name=row[1],
                similarity=similarity,
                procedure_json=proc_json,
            )

