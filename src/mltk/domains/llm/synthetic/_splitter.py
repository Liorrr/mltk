"""Zero-dependency word-count text splitter.

Splits raw text into overlapping chunks sized by word count.
Respects paragraph boundaries where possible so chunks start
and end at natural break points.

Why word-count (not token-count)?  Token counting requires a
tokenizer library (tiktoken, sentencepiece, etc.).  Word-count
is a good-enough proxy that keeps the module dependency-free
and deterministic across environments.
"""

from __future__ import annotations


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_words: int = 30,
) -> list[str]:
    """Split *text* into overlapping word-count-based chunks.

    The splitter first divides text on paragraph boundaries
    (double newlines), then greedily groups paragraphs into
    chunks that stay within *chunk_size* words.  If a single
    paragraph exceeds *chunk_size* it is split at sentence
    boundaries, falling back to raw word boundaries.

    Args:
        text: The input document text.
        chunk_size: Maximum number of words per chunk.
            Default 512.
        chunk_overlap: Number of trailing words from the
            previous chunk to prepend to the next one.
            Default 50.
        min_chunk_words: Minimum word count for a chunk to
            be included in the output.  Chunks below this
            threshold are merged into the previous chunk or
            discarded.  Default 30.

    Returns:
        List of text chunks.  May be empty if *text* has
        fewer than *min_chunk_words* words.

    Example::

        from mltk.domains.llm.synthetic._splitter import (
            split_text,
        )

        chunks = split_text(article, chunk_size=256)
        for chunk in chunks:
            print(len(chunk.split()), "words")
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    raw_chunks = _group_paragraphs(
        paragraphs, chunk_size,
    )

    chunks = _apply_overlap(raw_chunks, chunk_overlap)

    return [
        c for c in chunks
        if len(c.split()) >= min_chunk_words
    ]


def _split_paragraphs(text: str) -> list[str]:
    """Split on double-newlines, keeping non-empty blocks."""
    blocks = text.split("\n\n")
    return [b.strip() for b in blocks if b.strip()]


def _group_paragraphs(
    paragraphs: list[str],
    chunk_size: int,
) -> list[str]:
    """Greedily group paragraphs into word-count chunks.

    If a single paragraph exceeds *chunk_size*, it is split
    at sentence boundaries first, then at word boundaries.
    """
    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for para in paragraphs:
        para_words = para.split()
        para_len = len(para_words)

        if para_len > chunk_size:
            # Flush current accumulator first
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_count = 0
            # Split the oversized paragraph
            chunks.extend(
                _split_long_paragraph(para, chunk_size)
            )
            continue

        if current_count + para_len > chunk_size:
            # Flush current chunk
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = list(para_words)
            current_count = para_len
        else:
            current_words.extend(para_words)
            current_count += para_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _split_long_paragraph(
    paragraph: str,
    chunk_size: int,
) -> list[str]:
    """Break a paragraph that exceeds *chunk_size* words.

    Tries sentence boundaries (period + space) first.
    Falls back to raw word splitting.
    """
    sentences = _split_sentences(paragraph)
    if len(sentences) > 1:
        return _group_paragraphs(sentences, chunk_size)

    # Single long sentence -- split on word boundaries
    words = paragraph.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitting on '. ' boundary.

    Good enough for chunking -- not meant to be a
    general-purpose sentence tokenizer.
    """
    parts: list[str] = []
    current = []
    for word in text.split():
        current.append(word)
        if word.endswith((".","!","?")):
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return [p for p in parts if p.strip()]


def _apply_overlap(
    chunks: list[str],
    overlap: int,
) -> list[str]:
    """Prepend *overlap* trailing words from chunk N to chunk N+1."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        overlap_words = prev_words[-overlap:]
        overlap_text = " ".join(overlap_words)
        result.append(
            overlap_text + " " + chunks[i]
        )
    return result
