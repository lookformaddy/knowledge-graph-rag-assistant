"""FastAPI service exposing the GraphRAG assistant.

Run:  uvicorn graphrag.api:app --reload   (with ``src`` on PYTHONPATH,
      or after ``pip install -e .``)

Endpoints:
* ``GET  /health``              — liveness + graph size
* ``POST /ask``                 — answer a question (mode: hybrid|graph|chunks)
* ``GET  /graph/stats``         — entity/relation statistics
* ``GET  /graph/entity/{name}`` — facts touching one entity
* ``GET  /documents``           — corpus listing
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import build_assistant

app = FastAPI(
    title="Knowledge Graph-Powered RAG Assistant",
    version="0.1.0",
    description="Hybrid retrieval: TF-IDF chunk search fused with multi-hop "
                "knowledge-graph traversal, with path-level citations.",
)

_assistant = build_assistant()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=[
        "Which supplier supplies parts for the product whose returns are "
        "handled by the team led by Priya Nair?"
    ])
    mode: Literal["hybrid", "graph", "chunks"] = "hybrid"


class AskResponse(BaseModel):
    question: str
    answer: str | None
    strategy: str
    explanation: str
    citations: list[str]
    route: str
    plan: dict


@app.get("/health")
def health() -> dict:
    stats = _assistant.kg.stats()
    return {"status": "ok", "entities": stats["entities"], "edges": stats["edges"]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    plan = _assistant.plan(req.question)
    answer = _assistant.ask(req.question, mode=req.mode)
    return AskResponse(
        question=req.question,
        answer=answer.answer,
        strategy=answer.strategy,
        explanation=answer.explanation,
        citations=answer.citations,
        route=answer.route,
        plan={
            "seed_entities": plan.seed_entities,
            "target_type": plan.target_type,
            "relation_hints": plan.relation_hints,
            "rationale": plan.rationale,
        },
    )


@app.get("/graph/stats")
def graph_stats() -> dict:
    return _assistant.kg.stats()


@app.get("/graph/entity/{name}")
def graph_entity(name: str) -> dict:
    canonical = _assistant.extraction.resolver.resolve(name)
    if canonical is None or canonical not in _assistant.kg:
        raise HTTPException(status_code=404, detail=f"Unknown entity: {name}")
    facts = _assistant.kg.entity_facts(canonical)
    return {
        "entity": canonical,
        "type": _assistant.kg.graph.nodes[canonical].get("type"),
        "facts": [
            {"subject": t.subject, "relation": t.relation, "object": t.object,
             "doc_id": t.doc_id, "sentence": t.sentence}
            for t in facts
        ],
    }


@app.get("/documents")
def documents() -> dict:
    from .corpus import DOCUMENTS

    return {"documents": [{"doc_id": d.doc_id, "title": d.title} for d in DOCUMENTS]}
