"""End-to-end answering: the chunk-only vs GraphRAG contrast."""

MULTIHOP_Q = (
    "Which supplier supplies parts for the product whose returns are "
    "handled by the team led by Priya Nair?"
)


def test_factoid_answered_by_both_modes(assistant):
    q = "Who leads the Data Platform Team?"
    assert assistant.ask(q, mode="chunks").answer == "Maria Chen"
    assert assistant.ask(q, mode="hybrid").answer == "Maria Chen"


def test_multihop_chunk_baseline_abstains(assistant):
    ans = assistant.ask(MULTIHOP_Q, mode="chunks")
    assert ans.answer is None  # no single chunk connects the chain


def test_multihop_hybrid_answers_with_path(assistant):
    ans = assistant.ask(MULTIHOP_Q, mode="hybrid")
    assert ans.answer == "Helix Components"
    assert ans.strategy == "graph-path"
    assert any(c.startswith("[path]") for c in ans.citations)
    assert "Priya Nair" in ans.explanation


def test_multihop_graph_mode(assistant):
    ans = assistant.ask(MULTIHOP_Q, mode="graph")
    assert ans.answer == "Helix Components"


def test_second_multihop_chain(assistant):
    q = "Who leads the team that owns the Atlas Warehouse?"
    assert assistant.ask(q, mode="hybrid").answer == "Maria Chen"
    assert assistant.ask(q, mode="chunks").answer is None


def test_answer_citations_are_traceable(assistant):
    ans = assistant.ask("Which team owns the Atlas Warehouse?", mode="hybrid")
    assert ans.answer == "Data Platform Team"
    assert ans.citations  # every answer ships evidence
