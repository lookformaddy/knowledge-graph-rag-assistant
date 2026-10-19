"""Answer composition over a ContextPack.

The answerer is *extractive and deterministic*: it commits to a single answer
entity and shows its evidence. That is a feature for an eval harness — every
answer can be scored exactly, and every claim traces to a citation. In
production, the same ContextPack is handed to an LLM for fluent generation;
the pack's paths and citations are what keep that generation grounded.

Strategies:
* **graph/hybrid** — if the pack contains a path from a question seed to an
  entity of the target type, the path endpoint is the answer, and the path
  itself is the headline citation.
* **chunks** (the chunk-only RAG baseline) — within the top-ranked chunk,
  pick the sentence with the highest question overlap that contains an
  entity of the target type, and commit to that entity. If the best chunk
  never connects a target entity to the question, the baseline abstains —
  which is precisely how chunk-only RAG fails multi-hop questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import tokenize
from .extraction import split_sentences
from .graph_store import KnowledgeGraph
from .planner import QueryPlan
from .retrieval import ContextPack

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "whom", "does", "do", "did", "for", "of", "to", "in", "on", "by",
    "that", "whose", "with", "and", "or", "it", "its", "this",
}


@dataclass
class Answer:
    question: str
    answer: str | None                 # committed answer entity, or None (abstain)
    strategy: str                      # "graph-path" | "chunk-sentence" | "none"
    explanation: str
    citations: list[str] = field(default_factory=list)
    route: str = ""


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS}


def _entities_of_type(kg: KnowledgeGraph, target_type: str | None) -> list[str]:
    if target_type is None:
        return kg.entities()
    return [n for n, d in kg.graph.nodes(data=True) if d.get("type") == target_type]


def answer_with_graph(pack: ContextPack) -> Answer | None:
    if not pack.paths:
        return None
    path = pack.paths[0]
    answer_entity = path.nodes[-1]
    evidence = "; ".join(f'"{s.sentence}" ({s.doc_id})' for s in path.steps)
    return Answer(
        question=pack.question,
        answer=answer_entity,
        strategy="graph-path",
        explanation=(
            f"{answer_entity}. Reached in {path.hops} hop(s): {path.render()}. "
            f"Evidence: {evidence}."
        ),
        citations=pack.citations(),
        route=pack.plan.route,
    )


def answer_with_chunks(pack: ContextPack, kg: KnowledgeGraph) -> Answer | None:
    if not pack.chunks:
        return None
    plan: QueryPlan = pack.plan
    targets = _entities_of_type(kg, plan.target_type)
    if plan.target_type and not targets:
        return None
    top_chunk = pack.chunks[0].chunk
    q_tokens = _content_tokens(plan.question)
    best_sentence: str | None = None
    best_entity: str | None = None
    best_score = -1.0
    for sentence in split_sentences(top_chunk.text):
        overlap = len(q_tokens & _content_tokens(sentence))
        mentioned = [t for t in targets if t.lower() in sentence.lower()]
        if plan.target_type and not mentioned:
            continue
        # Prefer sentences that both overlap the question and name a target.
        score = overlap + (2.0 if mentioned else 0.0)
        if score > best_score:
            best_score, best_sentence = score, sentence
            best_entity = mentioned[0] if mentioned else None
    if best_sentence is None:
        return None
    return Answer(
        question=plan.question,
        answer=best_entity,
        strategy="chunk-sentence",
        explanation=(
            f'Best passage {top_chunk.citation}: "{best_sentence}"'
            + (f" Answer entity: {best_entity}." if best_entity else " No answer entity found.")
        ),
        citations=[top_chunk.citation],
        route=plan.route,
    )


def compose_answer(pack: ContextPack, kg: KnowledgeGraph, mode: str) -> Answer:
    """Compose an answer under an explicit mode: chunks | graph | hybrid."""
    if mode in ("graph", "hybrid"):
        graph_answer = answer_with_graph(pack)
        if graph_answer is not None:
            return graph_answer
        if mode == "graph":
            return Answer(
                question=pack.question, answer=None, strategy="none",
                explanation="No graph path from the question's entities to a "
                            "target of the requested type was found.",
                citations=pack.citations(), route=pack.plan.route,
            )
    chunk_answer = answer_with_chunks(pack, kg)
    if chunk_answer is not None:
        return chunk_answer
    return Answer(
        question=pack.question, answer=None, strategy="none",
        explanation="Retrieved context did not contain a grounded answer.",
        citations=pack.citations(), route=pack.plan.route,
    )
