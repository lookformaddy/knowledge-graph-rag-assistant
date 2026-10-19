"""Demo: the same multi-hop question, answered two ways.

Chunk-only RAG retrieves passages about Priya Nair, returns handling, and
suppliers — but no single passage connects them, so it abstains. GraphRAG
walks the graph (leader -> team -> portal -> product -> supplier) and answers
with the connecting path as its citation.

Usage:  python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag import build_assistant  # noqa: E402

QUESTION = (
    "Which supplier supplies parts for the product whose returns are "
    "handled by the team led by Priya Nair?"
)


def show(title: str, answer) -> None:
    print(f"\n=== {title} ===")
    print(f"Answer   : {answer.answer or '(abstained — no grounded answer)'}")
    print(f"Strategy : {answer.strategy}")
    print(f"Why      : {answer.explanation}")
    print("Citations:")
    for cite in answer.citations:
        print(f"  - {cite}")


def main() -> None:
    assistant = build_assistant()
    stats = assistant.kg.stats()
    print(f"Knowledge graph: {stats['entities']} entities, {stats['edges']} edges, "
          f"relations: {', '.join(sorted(stats['relations']))}")
    print(f"\nQuestion: {QUESTION}")

    plan = assistant.plan(QUESTION)
    print(f"Planner route: {plan.route} | seeds: {plan.seed_entities} | "
          f"target type: {plan.target_type} | rationale: {'; '.join(plan.rationale)}")

    show("Chunk-only RAG (baseline)", assistant.ask(QUESTION, mode="chunks"))
    show("GraphRAG (hybrid: chunks + graph paths)", assistant.ask(QUESTION, mode="hybrid"))

    print("\nFactoid sanity check (both modes should agree):")
    factoid = "Who leads the Data Platform Team?"
    print(f"  Q: {factoid}")
    print(f"  chunk-only -> {assistant.ask(factoid, mode='chunks').answer}")
    print(f"  hybrid     -> {assistant.ask(factoid, mode='hybrid').answer}")


if __name__ == "__main__":
    main()
