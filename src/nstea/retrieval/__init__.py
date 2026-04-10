"""Document loader — chunks clinical guideline text files for RAG indexing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    """A single chunk of a clinical guideline document."""

    chunk_id: str
    source: str          # file name / guideline title
    text: str
    metadata: dict       # page, section, etc.


def load_and_chunk(
    path: Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Load a text file and split into overlapping chunks.

    Args:
        path: Path to .txt or .md guideline file.
        chunk_size: Target number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of DocumentChunk objects.
    """
    text = path.read_text(encoding="utf-8")
    source = path.stem
    chunks: list[DocumentChunk] = []

    # Split on paragraph boundaries first, then assemble into ~chunk_size blocks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current_block: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current_block:
            chunk_text = "\n\n".join(current_block)
            chunk_id = _make_id(source, chunk_text)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                source=source,
                text=chunk_text,
                metadata={"file": path.name, "char_offset": 0},
            ))
            # Keep tail for overlap
            overlap_text = chunk_text[-overlap:] if overlap else ""
            current_block = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        current_block.append(para)
        current_len += len(para)

    # Final chunk
    if current_block:
        chunk_text = "\n\n".join(current_block)
        chunk_id = _make_id(source, chunk_text)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            source=source,
            text=chunk_text,
            metadata={"file": path.name, "char_offset": 0},
        ))

    return chunks


def load_directory(
    directory: Path,
    extensions: tuple[str, ...] = (".txt", ".md"),
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Load all guideline files from a directory."""
    all_chunks: list[DocumentChunk] = []
    for ext in extensions:
        for file_path in sorted(directory.glob(f"*{ext}")):
            all_chunks.extend(load_and_chunk(file_path, chunk_size, overlap))
    return all_chunks


def _make_id(source: str, text: str) -> str:
    h = hashlib.sha256(f"{source}:{text[:100]}".encode()).hexdigest()[:12]
    return f"{source}_{h}"
