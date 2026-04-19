"""Split legal corpus (e.g. all_cases.txt) into retrievable chunks with stable IDs."""

from __future__ import annotations

import json
import re
from typing import Any

CASE_START = re.compile(r"^\[case\s+(\d+)\]\s*", re.IGNORECASE | re.MULTILINE)


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if end < len(text):
            nl = chunk.rfind("\n\n")
            if nl > max_chars // 4:
                end = start + nl
                chunk = text[start:end]
        piece = chunk.strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def chunk_legal_corpus(
    text: str,
    *,
    max_chunk_chars: int = 2500,
    source_file: str = "",
) -> list[dict[str, Any]]:
    """
    Split on [case N] line starts; sub-chunk long cases by character windows.
    Each chunk: {"id", "text", "case_id", "source_file"}.
    """
    text = text.strip()
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    matches = list(CASE_START.finditer(text))
    chunks: list[dict[str, Any]] = []

    if not matches:
        for j, piece in enumerate(_split_oversized(text, max_chunk_chars)):
            chunks.append(
                {
                    "id": f"doc_{j}",
                    "text": piece,
                    "case_id": None,
                    "source_file": source_file,
                }
            )
        return chunks

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        for j, piece in enumerate(_split_oversized(preamble, max_chunk_chars)):
            chunks.append(
                {
                    "id": f"preamble_{j}",
                    "text": piece,
                    "case_id": None,
                    "source_file": source_file,
                }
            )

    for i, m in enumerate(matches):
        case_num = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        for j, piece in enumerate(_split_oversized(body, max_chunk_chars)):
            cid = f"case_{case_num}" if j == 0 else f"case_{case_num}_part{j}"
            chunks.append(
                {
                    "id": cid,
                    "text": piece,
                    "case_id": str(case_num),
                    "source_file": source_file,
                }
            )

    return chunks


def load_corpus_file(path: str, max_chunk_chars: int = 2500) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return chunk_legal_corpus(raw, max_chunk_chars=max_chunk_chars, source_file=path)


def save_chunks_jsonl(chunks: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
