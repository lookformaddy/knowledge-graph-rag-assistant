"""Hybrid retrieval: TF-IDF chunk search fused with graph traversal.

The fused **context pack** is the unit an answerer (or, in production, an LLM)
consumes. Crucially, graph facts arrive with the *path* that produced them, so
a citation can show the reasoning chain, not just a document id:

    [path] Priya Nair -[leads]-> Support Operations Team -[operates]-> ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .embeddings import TfidfVectorStore
from .extraction import Triple
from .graph_store import KnowledgeGraph, PathResult
from .planner import QueryPlan


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class ContextPack:
    question: str
    plan: QueryPlan
    chunks: list[ScoredChunk] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    paths: list[PathResult] = field(default_factory=list)

    def citations(self) -> list[str]:
        cites = [sc.chunk.citation for sc in self.chunks]
        for path in self.paths:
            cites.append(f"[path] {path.render()}")
        for t in self.triples:
            cites.append(f"[{t.doc_id}] {t.as_path_step()}")
        # Stable de-duplication.
        seen: set[str] = set()
        return [c for c in cites if not (c in seen or seen.add(c))]

    def render_context(self) -> str:
        lines: list[str] = []
        if self.paths:
            lines.append("Graph paths:")
            lines += [f"  - {p.render()}" for p in self.paths]
        if self.triples:
            lines.append("Graph facts:")
            lines += [f"  - {t.as_path_step()} (source: {t.sentence})" for t in self.triples]
        if self.chunks:
            lines.append("Passages:")
            lines += [f"  - {sc.chunk.citation} {sc.chunk.text}" for sc in self.chunks]
        return "\n".join(lines)


class HybridRetriever:
    def __init__(self, kg: KnowledgeGraph, chunks: list[Chunk]) -> None:
        self.kg = kg
        self.chunks = {c.chunk_id: c for c in chunks}
        self.store = TfidfVectorStore()
        for c in chunks:
            self.store.add(c.chunk_id, c.text)

    # -- chunk side ---------------------------------------------------------
    def search_chunks(self, question: str, top_k: int = 4) -> list[ScoredChunk]:
        hits = self.store.search(question, top_k=top_k)
        return [ScoredChunk(chunk=self.chunks[cid], score=s) for cid, s in hits]

    # -- graph side ---------------------------------------------------------
    def graph_paths_to_type(self, plan: QueryPlan, max_hops: int = 4) -> list[PathResult]:
        """Shortest paths from any seed entity to an entity of the target
        type, ranked by (hops, whether the path uses a hinted relation,
        answer name) so results are deterministic."""
        if not plan.seed_entities or not plan.target_type:
            return []
        targets = [
            name for name, data in self.kg.graph.nodes(data=True)
            if data.get("type") == plan.target_type and name not in plan.seed_entities
        ]
        def hint_coverage(p: PathResult) -> int:
            return len({s.relation for s in p.steps} & set(plan.relation_hints))

        results: list[PathResult] = []
        for seed in plan.seed_entities:
            for target in targets:
                path = self.kg.find_path(seed, target, max_hops=max_hops)
                if path is not None:
                    results.append(path)
        # Prefer paths that exercise more of the relations the question asks
        # about (the full reasoning chain), then shorter paths, then name.
        results.sort(key=lambda p: (-hint_coverage(p), p.hops, p.nodes[-1]))
        # Keep the best path per distinct answer entity.
        best: dict[str, PathResult] = {}
        for p in results:
            best.setdefault(p.nodes[-1], p)
        return sorted(best.values(),
                      key=lambda p: (-hint_coverage(p), p.hops, p.nodes[-1]))

    def neighbourhood_triples(self, plan: QueryPlan, max_hops: int = 2) -> list[Triple]:
        out: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for seed in plan.seed_entities:
            for t in self.kg.triples_around(seed, max_hops=max_hops):
                key = (t.subject, t.relation, t.object)
                if key not in seen:
                    seen.add(key)
                    out.append(t)
        return out

    # -- fusion -------------------------------------------------------------
    def retrieve(self, plan: QueryPlan, top_k: int = 4) -> ContextPack:
        pack = ContextPack(question=plan.question, plan=plan)
        if plan.route in ("chunks", "hybrid"):
            pack.chunks = self.search_chunks(plan.question, top_k=top_k)
        if plan.route in ("graph", "hybrid"):
            pack.paths = self.graph_paths_to_type(plan)
            pack.triples = self.neighbourhood_triples(plan)
            if plan.route == "graph":
                # Graph-first answers still carry the best passage as backup.
                pack.chunks = self.search_chunks(plan.question, top_k=2)
        return pack
