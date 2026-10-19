# Architecture

## Overview

The system answers enterprise questions by reasoning over a knowledge graph
built from the same documents a conventional RAG system would chunk. Both
retrieval paths share one corpus; the graph path adds typed entities,
relations, and multi-hop traversal on top.

```
            ┌────────────────────── offline build ──────────────────────┐
Documents ──► Pattern extraction ──► Entity resolution ──► Knowledge graph (NetworkX)
    │              (triples)            (aliases,               │
    │                                    dedup, types)          │
    └──► Sentence-aware chunking ──► TF-IDF vector store        │
                                                                │
            ┌────────────────────── online query ───────────────▼───────┐
Question ──► Query planner ──► Hybrid retriever ──► Context pack ──► Answer composer
              (link, type,      (chunks + paths     (passages,        (extractive,
               route)            fused)              triples, paths)   cited)
```

## Components

### Corpus (`corpus.py`)
Eight synthetic documents about a fictional retailer. Facts needed for
multi-hop questions are deliberately spread across documents so no single
chunk contains a full reasoning chain — that is the gap being measured.

### Extraction (`extraction.py`)
A deterministic pattern grammar maps enterprise phrasing to typed triples:
`leads`, `owns`, `depends_on`, `operates`, `handles_returns_for`,
`supplies_part_for` (with the supplied part kept as a `via` qualifier),
`part_of`, `reports_to`, `stores_for`, `uses`. Two passes: pass 1 builds the
entity gazetteer; pass 2 re-resolves all mentions against it. Extraction is
fully offline and reproducible — the same input always yields the same graph,
which is what makes the eval a regression test rather than a vibe check.

### Entity resolution (`extraction.EntityResolver`)
Three deterministic rules, applied in order, each recorded in an audit list:
exact canonical match → alias table ("Atlas" → "Atlas Warehouse") → unique
token containment. In production this layer is where a blocking + embedding
similarity resolver would live; the interface (mention in, canonical out,
decision logged) stays the same.

### Graph store (`graph_store.py`)
A NetworkX `MultiDiGraph`. Every edge carries `relation`, the supporting
`sentence`, and the source `doc_id` — provenance is stored at build time so
retrieval never has to reconstruct it. Node types come first from name shape,
then are *refined by relation participation* (whatever supplies parts is a
supplier; whatever is supplied is a product), which is stronger evidence.

Traversal offers two primitives: bounded undirected BFS neighbourhoods (for
context expansion) and shortest-path search that may walk edges backwards —
each rendered step marks reversed hops (`<-[relation]-`) so citations never
misrepresent direction.

### Vector store (`embeddings.py`)
TF-IDF + cosine over sentence-aware chunks. Chosen deliberately: zero model
downloads, deterministic, hermetic CI. The interface (`add`/`search`) is the
seam where a dense embedding model drops in for production.

### Query planner (`planner.py`)
A transparent heuristic router:
1. **Entity linking** — canonical names and aliases matched against the
   question (longest first).
2. **Target typing** — question words select the answer entity type
   ("who supplies…" → supplier, not person; "which team…" → team).
3. **Multi-hop detection** — relative clauses ("whose", "that owns",
   "led by"), nested clauses, or multiple linked entities route to the
   graph; simple factoids route to hybrid (chunks answer, graph verifies);
   entity-free questions fall back to chunks.

### Hybrid retrieval (`retrieval.py`)
Produces a `ContextPack`: scored chunks + neighbourhood triples + ranked
seed→target paths. Paths are ranked by *relation-hint coverage* (how many of
the relations the question asks about the path actually exercises), then by
hop count — this prefers the full reasoning chain over a shortcut that
happens to reach the same answer. Citations include the path itself:
`[path] Priya Nair -[leads]-> ... <-[supplies_part_for]- Helix Components`.

### Answering (`answering.py`)
Extractive and deterministic. Graph mode commits to the best path's endpoint
and quotes each hop's source sentence. Chunk mode (the baseline) commits only
if the top chunk contains a sentence that both overlaps the question and
names a target-type entity — otherwise it abstains. Abstention is scored as a
miss, which is exactly how ungrounded chunk RAG behaves on multi-hop
questions in production: it either guesses or stays silent.

### API (`api.py`)
FastAPI over the same `Assistant` object the tests exercise; `/ask` returns
the answer, the strategy that produced it, the plan, and all citations, so
clients can render the reasoning chain, not just the result.

## Failure modes by design

| Situation | Chunk-only | GraphRAG |
|---|---|---|
| Factoid in one passage | answers | answers |
| Answer needs 2–4 hops across documents | abstains / guesses | answers with path |
| Entity mentioned by alias ("Atlas") | token mismatch risk | resolver canonicalises |
| Question about an unlinked topic | answers from text | falls back to chunks |

## Scaling notes (roadmap pointers)

* Swap NetworkX for a property-graph store (Neo4j/Neptune) behind
  `KnowledgeGraph`; traversal API is already graph-shaped.
* Swap TF-IDF for dense embeddings behind `TfidfVectorStore.search`.
* Swap pattern extraction for LLM-based extraction with the pattern grammar
  as a validation layer; keep the resolver and provenance model.
* Add incremental graph updates and per-edge confidence once extraction is
  probabilistic.
