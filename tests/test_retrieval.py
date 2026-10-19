"""Hybrid retrieval: chunk search, graph paths, and context-pack fusion."""

from graphrag.planner import plan_query

MULTIHOP_Q = (
    "Which supplier supplies parts for the product whose returns are "
    "handled by the team led by Priya Nair?"
)


def _pack(assistant, question):
    plan = plan_query(question, assistant.extraction.resolver, assistant.kg.entities())
    return assistant.retriever.retrieve(plan)


def test_chunk_search_ranks_relevant_passage_first(assistant):
    hits = assistant.retriever.search_chunks("Who leads the Data Platform Team?")
    assert hits
    assert hits[0].chunk.doc_id == "org-chart"
    assert hits[0].score > 0


def test_graph_paths_found_for_multihop(assistant):
    pack = _pack(assistant, MULTIHOP_Q)
    assert pack.paths, "expected at least one seed->supplier path"
    best = pack.paths[0]
    assert best.nodes[-1] == "Helix Components"
    assert best.hops == 4  # leader -> team -> portal -> product -> supplier


def test_context_pack_citations_include_path(assistant):
    pack = _pack(assistant, MULTIHOP_Q)
    citations = pack.citations()
    assert any(c.startswith("[path]") for c in citations)
    path_cite = next(c for c in citations if c.startswith("[path]"))
    assert "Priya Nair" in path_cite and "Helix Components" in path_cite


def test_context_pack_render_mentions_all_layers(assistant):
    pack = _pack(assistant, MULTIHOP_Q)
    rendered = pack.render_context()
    assert "Graph paths:" in rendered
    assert "Graph facts:" in rendered
    assert "Passages:" in rendered


def test_chunk_route_pack_has_no_graph_content(assistant):
    pack = _pack(assistant, "How often do reconciliation jobs run?")
    assert pack.chunks
    assert pack.paths == []
