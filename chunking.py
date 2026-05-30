"""Structure-aware document chunking.

Supports: markdown headings, paragraphs, and fixed-size fallback.
Each chunk carries metadata about its source structure.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_by_structure(text: str, doc_name: str, max_chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split document using structural cues (headings, paragraphs), falling back to fixed-size."""
    lines = text.split("\n")
    sections = _split_by_headings(lines)

    doc_type = _detect_doc_type(text)
    chunks = []
    for heading, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        paragraphs = _split_by_paragraphs(section_text)

        for para in paragraphs:
            if len(para) <= max_chunk_size:
                chunks.append(Chunk(
                    text=para,
                    metadata={
                        "doc_name": doc_name,
                        "heading": heading,
                        "doc_type": doc_type,
                        "chunk_type": "paragraph",
                    }
                ))
            else:
                # Fixed-size fallback with overlap for long paragraphs
                sub_chunks = _fixed_size_split(para, max_chunk_size, overlap)
                for i, sc in enumerate(sub_chunks):
                    chunks.append(Chunk(
                        text=sc,
                        metadata={
                            "doc_name": doc_name,
                            "heading": heading,
                            "doc_type": doc_type,
                            "chunk_type": "fixed_split",
                            "sub_index": i,
                        }
                    ))
    return chunks


def _split_by_headings(lines: list[str]) -> list[tuple[str, str]]:
    """Split lines into (heading, content) sections."""
    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    return sections if sections else [("Introduction", "\n".join(lines))]


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text on double newlines into paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _fixed_size_split(text: str, max_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with overlap, breaking on word boundaries."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start
        current_len = 0
        while end < len(words) and current_len + len(words[end]) + 1 <= max_size:
            current_len += len(words[end]) + 1
            end += 1

        if end == start:
            end = start + 1

        chunks.append(" ".join(words[start:end]))

        # Calculate overlap in words
        overlap_words = 0
        overlap_len = 0
        for i in range(end - 1, start - 1, -1):
            if overlap_len + len(words[i]) + 1 > overlap:
                break
            overlap_len += len(words[i]) + 1
            overlap_words += 1

        start = end - overlap_words

    return chunks


def _detect_doc_type(text: str) -> str:
    """Simple heuristic to detect document type."""
    lower = text[:500].lower()
    if any(kw in lower for kw in ["api", "endpoint", "request", "response", "http"]):
        return "api_doc"
    if any(kw in lower for kw in ["tutorial", "step", "learn", "guide"]):
        return "tutorial"
    if any(kw in lower for kw in ["faq", "question", "answer"]):
        return "faq"
    if any(kw in lower for kw in ["changelog", "release", "version"]):
        return "changelog"
    return "general"
