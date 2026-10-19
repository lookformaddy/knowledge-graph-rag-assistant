"""Eval regression: hybrid GraphRAG must dominate the chunk-only baseline,
especially on multi-hop questions. Loads eval/run_eval.py as a module."""

import importlib.util
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("run_eval", EVAL_DIR / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_hybrid_beats_chunks_on_multihop():
    module = _load_eval_module()
    questions = json.loads((EVAL_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    results = module.run(questions)
    acc = results["accuracy"]
    assert acc["multihop"]["hybrid"] == 1.0
    assert acc["multihop"]["chunks"] == 0.0
    assert acc["factoid"]["hybrid"] == 1.0
    assert acc["factoid"]["chunks"] == 1.0
    assert acc["overall"]["hybrid"] > acc["overall"]["chunks"]


def test_eval_questions_are_well_formed():
    questions = json.loads((EVAL_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    for q in questions:
        assert q["kind"] in ("factoid", "multihop")
        assert q["expected_answers"]
        if q["kind"] == "multihop":
            assert q.get("gold_chain"), "multi-hop questions must declare their gold chain"
