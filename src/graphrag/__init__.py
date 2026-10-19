"""GraphRAG: knowledge-graph-powered retrieval-augmented generation.

Pipeline: synthetic corpus -> pattern extraction -> entity resolution ->
NetworkX knowledge graph + TF-IDF chunk index -> query planner -> hybrid
retrieval (chunks + multi-hop graph paths) -> cited, extractive answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .answering import Answer, compose_answer
from .chunking import Chunk, chunk_corpus
from .corpus import DOCUMENTS, Document
from .extraction import ExtractionResult, extract_corpus
from .graph_store import KnowledgeGraph
from .planner import QueryPlan, plan_query
from .retrieval import ContextPack, HybridRetriever

__all__ = [
    "Answer",
    "Assistant",
    "Chunk",
    "ContextPack",
    "Document",
    "ExtractionResult",
    "HybridRetriever",
    "KnowledgeGraph",
    "QueryPlan",
    "build_assistant",
]


@dataclass
class Assistant:
    """End-to-end GraphRAG assistant over a fixed corpus."""

    kg: KnowledgeGraph
    retriever: HybridRetriever
    extraction: ExtractionResult
    chunks: list[Chunk]

    def plan(self, question: str) -> QueryPlan:
        return plan_query(question, self.extraction.resolver, self.kg.entities())

    def retrieve(self, question: str) -> ContextPack:
        return self.retriever.retrieve(self.plan(question))

    def ask(self, question: str, mode: str = "hybrid") -> Answer:
        """Answer a question.

        ``mode``: "hybrid" (planner route + graph-first composition),
        "graph" (graph paths only), or "chunks" (chunk-only RAG baseline).
        """
        plan = self.plan(question)
        if mode == "chunks":
            plan.route = "chunks"
        pack = self.retriever.retrieve(plan)
        return compose_answer(pack, self.kg, mode=mode)


def build_assistant(documents: list[Document] | None = None) -> Assistant:
    docs = documents if documents is not None else DOCUMENTS
    extraction = extract_corpus(docs)
    kg = KnowledgeGraph.from_extraction(extraction)
    chunks = chunk_corpus(docs)
    retriever = HybridRetriever(kg, chunks)
    return Assistant(kg=kg, retriever=retriever, extraction=extraction, chunks=chunks)
