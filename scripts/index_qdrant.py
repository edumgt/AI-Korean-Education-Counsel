#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""documents.jsonl을 Qdrant에 적재한다 (전체 + 도메인별 컬렉션)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_collection_name(domain_name: str) -> str:
    s = (domain_name or "").strip().replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in s)
    return f"career_{(s or 'unknown').lower()}"


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def point_id(doc_id: str, idx: int) -> int:
    h = hashlib.sha1(f"{doc_id}:{idx}".encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )


def upsert_points(client: QdrantClient, collection: str, vectors: np.ndarray, payloads: List[Dict], ids: List[int], batch: int = 256) -> None:
    for i in range(0, len(ids), batch):
        client.upsert(
            collection_name=collection,
            points=rest.Batch(
                ids=ids[i : i + batch],
                vectors=vectors[i : i + batch].tolist(),
                payloads=payloads[i : i + batch],
            ),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.docs).expanduser().resolve())
    model = SentenceTransformer(DEFAULT_MODEL)
    vector_size = model.get_sentence_embedding_dimension()
    client = QdrantClient(url=QDRANT_URL)

    all_texts, all_payloads, all_ids = [], [], []
    per_domain: Dict[str, List[int]] = {}

    for r in rows:
        doc_id = str(r.get("doc_id") or "unknown")
        domain = str(r.get("domain_name") or "unknown")
        for idx, ch in enumerate(chunk_text(r.get("text", ""), args.max_chars, args.overlap)):
            pid = point_id(doc_id, idx)
            payload = {
                "doc_id": doc_id,
                "chunk_idx": idx,
                "domain_name": domain,
                "source_spec": r.get("source_spec"),
                "creation_year": r.get("creation_year"),
                "text": ch,
            }
            all_texts.append("passage: " + ch)
            all_payloads.append(payload)
            all_ids.append(pid)
            per_domain.setdefault(domain, []).append(pid)

    if not all_ids:
        print("[WARN] no chunks found")
        return

    vectors = np.asarray(model.encode(all_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True), dtype=np.float32)
    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}

    all_collection = "career_all"
    ensure_collection(client, all_collection, vector_size)
    upsert_points(client, all_collection, vectors, all_payloads, all_ids)

    for domain, ids in per_domain.items():
        col = normalize_collection_name(domain)
        ensure_collection(client, col, vector_size)
        payloads = [all_payloads[id_to_idx[pid]] for pid in ids]
        vecs = vectors[[id_to_idx[pid] for pid in ids], :]
        upsert_points(client, col, vecs, payloads, ids)

    print(f"[OK] indexed {len(all_ids)} chunks to {QDRANT_URL}")
    print("[OK] collections: career_all + per-domain")


if __name__ == "__main__":
    main()
