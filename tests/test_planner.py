"""Query planner: entity linking, target typing, and routing."""

from graphrag.planner import plan_query

MULTIHOP_Q = (
    "Which supplier supplies parts for the product whose returns are "
    "handled by the team led by Priya Nair?"
)


def _plan(assistant, question):
    return plan_query(question, assistant.extraction.resolver, assistant.kg.entities())


def test_multihop_question_routes_to_graph(assistant):
    plan = _plan(assistant, MULTIHOP_Q)
    assert plan.route == "graph"
    assert "Priya Nair" in plan.seed_entities
    assert plan.target_type == "supplier"
    assert "handles_returns_for" in plan.relation_hints
    assert "supplies_part_for" in plan.relation_hints


def test_factoid_routes_to_hybrid(assistant):
    plan = _plan(assistant, "Who leads the Data Platform Team?")
    assert plan.route == "hybrid"
    assert plan.seed_entities == ["Data Platform Team"]
    assert plan.target_type == "person"


def test_who_supplies_targets_supplier_not_person(assistant):
    plan = _plan(assistant, "Who supplies packaging for the Trail Backpack 40L?")
    assert plan.target_type == "supplier"


def test_entity_linking_uses_aliases(assistant):
    plan = _plan(assistant, "What does Atlas store for the Checkout Service?")
    assert "Atlas Warehouse" in plan.seed_entities


def test_no_entity_question_routes_to_chunks(assistant):
    plan = _plan(assistant, "How often do reconciliation jobs run?")
    assert plan.route == "chunks"
    assert plan.seed_entities == []


def test_team_target_type(assistant):
    plan = _plan(assistant, "Which team owns the Atlas Warehouse?")
    assert plan.target_type == "team"
