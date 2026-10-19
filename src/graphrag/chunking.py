"""Document chunking for the vector side of hybrid retrieval.

Sentence-aware chunks: passages are split into sentences and grouped up to a
character budget, so a chunk never cuts a sentence in half. Each chunk keeps
its document id and index — that pair is the citation unit for chunk-only RAG.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Document
from .extraction import split_sentences


@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # f"{doc_id}#{index}"
    doc_id: str
    index: int
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.chunk_id}]"


def chunk_document(doc: Document, max_chars: int = 320) -> list[Chunk]:
    sentences = split_sentences(doc.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > max_chars:
            chunks.append(
                Chunk(chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id,
                      index=len(chunks), text=" ".join(current))
            )
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(
            Chunk(chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id,
                  index=len(chunks), text=" ".join(current))
        )
    return chunks


def chunk_corpus(documents: list[Document], max_chars: int = 320) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in documents:
        out.extend(chunk_document(doc, max_chars=max_chars))
    return out
