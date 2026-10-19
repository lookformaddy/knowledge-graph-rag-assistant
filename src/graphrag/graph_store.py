"""Knowledge graph store backed by NetworkX (MultiDiGraph).

Nodes are canonical entities with a guessed type; edges carry the relation,
the supporting sentence, and the source document id — provenance is a
first-class citizen because every retrieved graph fact must be citable.

Multi-hop retrieval is a bounded BFS over the *undirected view* used for
neighbourhood expansion, plus directed shortest-path search used to render
the exact connecting path into a citation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from .extraction import ExtractionResult, Triple, guess_type


@dataclass
class PathResult:
    nodes: list[str]              # entity names along the path
    steps: list[Triple]           # triples realising each hop

    def render(self) -> str:
        if not self.nodes:
            return ""
        parts = [self.nodes[0]]
        current = self.nodes[0]
        for step, nxt in zip(self.steps, self.nodes[1:]):
            if step.subject == current:
                parts.append(f"-[{step.relation}]-> {nxt}")
            else:  # hop traversed against the stored edge direction
                parts.append(f"<-[{step.relation}]- {nxt}")
            current = nxt
        return " ".join(parts)

    @property
    def hops(self) -> int:
        return len(self.steps)


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_extraction(cls, result: ExtractionResult) -> "KnowledgeGraph":
        kg = cls()
        for name in result.resolver.canonical.values():
            kg.graph.add_node(name, type=guess_type(name))
        for t in result.triples:
            kg.graph.add_edge(
                t.subject,
                t.object,
                relation=t.relation,
                doc_id=t.doc_id,
                sentence=t.sentence,
                via=t.via,
            )
        kg._refine_types(result.triples)
        return kg

    # Relation participation is stronger evidence than name shape:
    # whatever supplies parts is a supplier, whatever is supplied is a product.
    _TYPE_BY_ROLE = {
        ("leads", "subject"): "person",
        ("leads", "object"): "team",
        ("reports_to", "subject"): "person",
        ("reports_to", "object"): "person",
        ("supplies_part_for", "subject"): "supplier",
        ("supplies_part_for", "object"): "product",
        ("handles_returns_for", "object"): "product",
        ("part_of", "subject"): "product",
        ("part_of", "object"): "category",
        ("operates", "subject"): "team",
        ("operates", "object"): "system",
        ("owns", "object"): "system",
    }

    def _refine_types(self, triples: list[Triple]) -> None:
        for t in triples:
            for (relation, role), etype in self._TYPE_BY_ROLE.items():
                if t.relation != relation:
                    continue
                node = t.subject if role == "subject" else t.object
                if node in self.graph:
                    self.graph.nodes[node]["type"] = etype

    # -- basics -------------------------------------------------------------
    def __contains__(self, entity: str) -> bool:
        return entity in self.graph

    def entities(self) -> list[str]:
        return sorted(self.graph.nodes)

    def triples(self) -> list[Triple]:
        out: list[Triple] = []
        for u, v, data in self.graph.edges(data=True):
            out.append(
                Triple(subject=u, relation=data["relation"], object=v,
                       doc_id=data["doc_id"], sentence=data["sentence"],
                       via=data.get("via"))
            )
        return out

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            types[data.get("type", "entity")] = types.get(data.get("type", "entity"), 0) + 1
        relations: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            relations[data["relation"]] = relations.get(data["relation"], 0) + 1
        return {
            "entities": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_types": types,
            "relations": relations,
            "weakly_connected_components": nx.number_weakly_connected_components(self.graph),
            "avg_degree": round(
                sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1), 2
            ),
        }

    # -- traversal ----------------------------------------------------------
    def neighbours(self, entity: str, max_hops: int = 2) -> dict[str, int]:
        """Undirected BFS neighbourhood: entity -> hop distance."""
        if entity not in self.graph:
            return {}
        undirected = self.graph.to_undirected()
        dist = {entity: 0}
        queue: deque[str] = deque([entity])
        while queue:
            node = queue.popleft()
            if dist[node] >= max_hops:
                continue
            for nbr in undirected.neighbors(node):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        return dist

    def triples_around(self, entity: str, max_hops: int = 2) -> list[Triple]:
        """All triples touching the bounded neighbourhood of ``entity``."""
        hood = set(self.neighbours(entity, max_hops))
        return [t for t in self.triples() if t.subject in hood and t.object in hood]

    def find_path(self, source: str, target: str, max_hops: int = 4) -> PathResult | None:
        """Shortest directed-or-reverse path between two entities.

        Edges may be traversed backwards (relations read naturally in either
        direction for navigation, e.g. following ``supplies_part_for`` from a
        product back to its supplier); each step keeps its true direction in
        the stored Triple so citations stay honest.
        """
        if source not in self.graph or target not in self.graph:
            return None
        undirected = self.graph.to_undirected()
        try:
            node_path = nx.shortest_path(undirected, source, target)
        except nx.NetworkXNoPath:
            return None
        if len(node_path) - 1 > max_hops:
            return None
        steps: list[Triple] = []
        for a, b in zip(node_path, node_path[1:]):
            step = self._step_between(a, b)
            if step is None:
                return None
            steps.append(step)
        return PathResult(nodes=node_path, steps=steps)

    def _step_between(self, a: str, b: str) -> Triple | None:
        if self.graph.has_edge(a, b):
            data = self.graph.get_edge_data(a, b)[0]
            return Triple(a, data["relation"], b, data["doc_id"], data["sentence"], data.get("via"))
        if self.graph.has_edge(b, a):
            data = self.graph.get_edge_data(b, a)[0]
            return Triple(b, data["relation"], a, data["doc_id"], data["sentence"], data.get("via"))
        return None

    def entity_facts(self, entity: str) -> list[Triple]:
        return [t for t in self.triples() if t.subject == entity or t.object == entity]
