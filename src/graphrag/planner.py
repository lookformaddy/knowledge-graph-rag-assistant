"""Query planner: routes questions to chunk search, graph traversal, or both.

The planner is a transparent heuristic router (in production this is where a
small classifier or an LLM with a structured-output schema would sit):

* **Entity linking** — canonical entity names and aliases are matched against
  the question to find graph seed entities.
* **Target typing** — question words ("which supplier", "who", "which team")
  determine the type of entity the answer must be.
* **Multi-hop detection** — relative clauses ("whose", "that owns", "led by"),
  chained possessives, or several linked entities signal that no single chunk
  can contain the answer, so the graph must be traversed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extraction import EntityResolver, normalise

_MULTI_HOP_CUES = (
    "whose", "that owns", "that leads", "that operates", "that depends",
    "led by", "owned by", "operated by", "handled by", "supplied by",
    "the team that", "the product that", "the system that", "connected to",
    "through", "via",
)

_RELATION_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("lead", "led"), "leads"),
    (("own",), "owns"),
    (("depend", "relies", "rely"), "depends_on"),
    (("suppl", "provid", "vendor"), "supplies_part_for"),
    (("return",), "handles_returns_for"),
    (("operat", "runs", "run by"), "operates"),
    (("part of", "category", "belong"), "part_of"),
    (("report",), "reports_to"),
    (("store",), "stores_for"),
]

_TARGET_TYPES: list[tuple[re.Pattern[str], str]] = [
    # "Who supplies...?" asks for a supplier (an organisation), not a person,
    # so the supply cue must win over the generic "who" rule below.
    (re.compile(r"suppl|\bvendor\b"), "supplier"),
    (re.compile(r"\bwho\b|\bperson\b|\bmanager\b|\bleader\b"), "person"),
    (re.compile(r"\bteam\b"), "team"),
    (re.compile(r"\bsystem\b|\bservice\b|\bportal\b|\bgateway\b|\bwarehouse\b"), "system"),
    (re.compile(r"\bproduct\b"), "product"),
    (re.compile(r"\bcategor"), "category"),
]


@dataclass
class QueryPlan:
    question: str
    route: str                              # "chunks" | "graph" | "hybrid"
    seed_entities: list[str] = field(default_factory=list)
    target_type: str | None = None
    relation_hints: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


def link_entities(question: str, resolver: EntityResolver,
                  known_entities: list[str]) -> list[str]:
    """Find canonical entities mentioned in the question (longest first so
    'Atlas Warehouse' wins over the 'Atlas' alias)."""
    q = f" {normalise(question)} "
    found: list[str] = []
    candidates = sorted(
        set(known_entities) | set(resolver.aliases.values()),
        key=lambda name: -len(normalise(name)),
    )
    for name in candidates:
        key = normalise(name)
        if key and f" {key} " in q.replace("?", " ").replace(",", " "):
            if name not in found:
                found.append(name)
    # Alias mentions resolve to their canonical entity.
    for alias, canon in sorted(resolver.aliases.items(), key=lambda kv: -len(kv[0])):
        if f" {alias} " in q and canon not in found:
            found.append(canon)
    return found


def plan_query(question: str, resolver: EntityResolver,
               known_entities: list[str]) -> QueryPlan:
    q_lower = question.lower()
    seeds = link_entities(question, resolver, known_entities)

    target_type: str | None = None
    for pattern, ttype in _TARGET_TYPES:
        if pattern.search(q_lower):
            target_type = ttype
            break

    hints: list[str] = []
    for words, relation in _RELATION_HINTS:
        if any(w in q_lower for w in words) and relation not in hints:
            hints.append(relation)

    rationale: list[str] = []
    multi_hop = False
    cues_hit = [c for c in _MULTI_HOP_CUES if c in q_lower]
    if cues_hit:
        multi_hop = True
        rationale.append(f"multi-hop cues: {', '.join(cues_hit)}")
    if len(seeds) >= 2 and hints:
        multi_hop = True
        rationale.append("multiple linked entities with relation hints")
    if re.search(r"\b(whose|that|which)\b.*\b(whose|that|which)\b", q_lower):
        multi_hop = True
        rationale.append("nested relative clauses")

    if multi_hop and seeds:
        route = "graph"
    elif seeds and target_type:
        route = "hybrid"
        rationale.append("single-hop question: chunks suffice, graph verifies")
    elif seeds:
        route = "hybrid"
        rationale.append("entity mentioned but no clear target type")
    else:
        route = "chunks"
        rationale.append("no entities linked: pure text question")

    return QueryPlan(question=question, route=route, seed_entities=seeds,
                     target_type=target_type, relation_hints=hints,
                     rationale=rationale)
