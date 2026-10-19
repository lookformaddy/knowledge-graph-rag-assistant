"""FastAPI service contract tests (in-process via TestClient)."""

import pytest
from fastapi.testclient import TestClient

from graphrag.api import app

client = TestClient(app)

MULTIHOP_Q = (
    "Which supplier supplies parts for the product whose returns are "
    "handled by the team led by Priya Nair?"
)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["entities"] >= 20


def test_ask_hybrid_multihop():
    resp = client.post("/ask", json={"question": MULTIHOP_Q, "mode": "hybrid"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Helix Components"
    assert body["strategy"] == "graph-path"
    assert any(c.startswith("[path]") for c in body["citations"])
    assert body["plan"]["target_type"] == "supplier"


def test_ask_chunks_mode_contrast():
    resp = client.post("/ask", json={"question": MULTIHOP_Q, "mode": "chunks"})
    assert resp.status_code == 200
    assert resp.json()["answer"] is None


def test_ask_factoid_default_mode():
    resp = client.post("/ask", json={"question": "Who leads the Data Platform Team?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Maria Chen"


def test_ask_validation_rejects_short_question():
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422


def test_graph_stats():
    resp = client.get("/graph/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["relations"]["leads"] == 4
    assert body["entity_types"]["supplier"] == 3


def test_graph_entity_lookup():
    resp = client.get("/graph/entity/Atlas Warehouse")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity"] == "Atlas Warehouse"
    assert body["facts"]


def test_graph_entity_alias_lookup():
    resp = client.get("/graph/entity/Atlas")
    assert resp.status_code == 200
    assert resp.json()["entity"] == "Atlas Warehouse"


def test_graph_entity_unknown_404():
    resp = client.get("/graph/entity/Nope Industries")
    assert resp.status_code == 404


def test_documents_listing():
    resp = client.get("/documents")
    assert resp.status_code == 200
    ids = {d["doc_id"] for d in resp.json()["documents"]}
    assert "org-chart" in ids and "supplier-catalog" in ids


@pytest.mark.parametrize("mode", ["hybrid", "graph", "chunks"])
def test_all_modes_respond(mode):
    resp = client.post("/ask", json={"question": "Who leads the Data Platform Team?",
                                     "mode": mode})
    assert resp.status_code == 200
