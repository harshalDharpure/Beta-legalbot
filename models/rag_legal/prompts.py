"""Citation-grounded prompts aligned with `User: ...\\nAssistant:` generation format."""

from __future__ import annotations

from typing import Any


def build_rag_user_block(
    user_query: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    max_chars_per_chunk: int = 1500,
) -> str:
    """
    Build the **user** side content: instructions + passages + original question.
    The outer wrapper adds `User:` / `Assistant:` in the generation helper.
    """
    lines = [
        "You are helping with Indian legal information for lay users.",
        "Rules:",
        "- Base your answer ONLY on the numbered passages below when stating legal facts.",
        "- After important factual claims taken from a passage, cite its bracket id exactly, e.g. [case_3] or [case_3_part1].",
        "- If passages do not contain enough information, say so briefly and give only general, non-specific guidance.",
        "- Keep a supportive tone; this is not formal legal advice.",
        "",
        "PASSAGES:",
    ]
    for c in retrieved_chunks:
        cid = c["id"]
        body = (c.get("text") or "")[:max_chars_per_chunk].strip()
        lines.append(f"[{cid}]")
        lines.append(body)
        lines.append("")
    lines.append("QUESTION:")
    lines.append(user_query.strip())
    return "\n".join(lines)
