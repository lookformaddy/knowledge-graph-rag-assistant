"""Comparison eval: chunk-only RAG vs hybrid GraphRAG.

Scores exact-answer accuracy of the committed answer entity for both modes
over ``eval_questions.json``, split by factoid vs multi-hop questions.

Usage:
    python eval/run_eval.py            # human-readable table
    python eval/run_eval.py --json     # machine-readable results

Expected shape of the result: both modes answer single-hop factoids; only the
hybrid mode answers multi-hop questions, because no single chunk contains the
connecting chain — the graph path does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag import build_assistant  # noqa: E402


def run(questions: list[dict]) -> dict:
    assistant = build_assistant()
    rows: list[dict] = []
    for q in questions:
        row = {"id": q["id"], "kind": q["kind"], "question": q["question"]}
        for mode in ("chunks", "hybrid"):
            ans = assistant.ask(q["question"], mode=mode)
            row[mode] = {
                "answer": ans.answer,
                "correct": ans.answer in q["expected_answers"],
                "strategy": ans.strategy,
            }
        rows.append(row)

    def accuracy(mode: str, kind: str | None = None) -> float:
        subset = [r for r in rows if kind is None or r["kind"] == kind]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r[mode]["correct"]) / len(subset)

    return {
        "rows": rows,
        "accuracy": {
            kind: {mode: accuracy(mode, kind if kind != "overall" else None)
                   for mode in ("chunks", "hybrid")}
            for kind in ("overall", "factoid", "multihop")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON results")
    args = parser.parse_args()

    questions = json.loads(
        (Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8")
    )
    results = run(questions)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{'ID':<12} {'Kind':<9} {'Chunk-only':<22} {'Hybrid GraphRAG':<22}")
    print("-" * 72)
    for row in results["rows"]:
        def fmt(cell: dict) -> str:
            mark = "OK " if cell["correct"] else "MISS"
            return f"{mark} {cell['answer'] or '(abstain)'}"
        print(f"{row['id']:<12} {row['kind']:<9} {fmt(row['chunks']):<22} {fmt(row['hybrid']):<22}")
    print("-" * 72)
    for kind in ("factoid", "multihop", "overall"):
        acc = results["accuracy"][kind]
        print(f"{kind:<10} chunk-only accuracy: {acc['chunks']:.0%}   "
              f"hybrid accuracy: {acc['hybrid']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
