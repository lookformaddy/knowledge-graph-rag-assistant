"""Deterministic, offline entity/relation extraction.

Two layers, both rule-based and reproducible (no LLM, no network):

1. **Pattern extraction** — a small grammar of enterprise relation phrases
   ("leads", "owns", "depends on", "supplies X for Y", "handles returns for",
   ...) maps sentences to typed triples ``(subject, relation, object)``.
2. **Entity spotting** — a gazetteer is bootstrapped from the subjects/objects
   the patterns find, plus known multiword aliases (e.g. "Atlas" for
   "Atlas Warehouse"), so later passes and query-time entity linking can
   resolve mentions to canonical entities.

Entity resolution is deliberately explicit and inspectable: every canonical
entity carries the aliases that were merged into it, and the merge rules
(normalisation + alias table + containment) are unit-tested. In production you
would swap the alias table for an embedding/blocking-based resolver behind the
same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Relation patterns: (compiled regex, canonical relation, group order)
# Group order maps regex groups -> (subject, object) or (subject, object, via)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], str, tuple[int, ...]]] = [
    # "Helix Components supplies motors for the Aero Blender Pro."
    (re.compile(r"(.+?) supplies (.+?) for (?:the )?(.+?)\.?$"), "supplies_part_for", (1, 3, 2)),
    # "NordLogistics supplies LED panels for the Lumen Desk Lamp."
    (re.compile(r"(.+?) is part of (?:the )?(.+?)\.?$"), "part_of", (1, 2)),
    (re.compile(r"(.+?) handles returns for (?:the )?(.+?)\.?$"), "handles_returns_for", (1, 2)),
    (re.compile(r"(.+?) depends on (?:the )?(.+?)\.?$"), "depends_on", (1, 2)),
    (re.compile(r"(.+?) reports to (.+?)\.?$"), "reports_to", (1, 2)),
    (re.compile(r"(.+?) leads (?:the )?(.+?)\.?$"), "leads", (1, 2)),
    (re.compile(r"(.+?) owns (?:the )?(.+?)\.?$"), "owns", (1, 2)),
    (re.compile(r"(.+?) operates (?:the )?(.+?)\.?$"), "operates", (1, 2)),
    (re.compile(r"(.+?) stores (.+?) for (?:the )?(.+?)\.?$"), "stores_for", (1, 3, 2)),
    (re.compile(r"(.+?) uses (.+?)\.?$"), "uses", (1, 2)),
]

# Multiword types that help classify entities after extraction.
_TYPE_HINTS: list[tuple[str, str]] = [
    (" Team", "team"),
    (" Service", "system"),
    (" Portal", "system"),
    (" Gateway", "system"),
    (" Warehouse", "system"),
]

# Known short-form aliases -> canonical (bootstraps entity resolution).
_ALIAS_TABLE: dict[str, str] = {
    "atlas": "Atlas Warehouse",
    "beacon gateway": "Beacon API Gateway",
    "beacon": "Beacon API Gateway",
    "checkout": "Checkout Service",
    "returns": "Returns Portal",
    "stocksync": "StockSync Service",
}


def normalise(name: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    name = name.strip().lower()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    name = re.sub(r"[^a-z0-9 ]+", "", name)
    return re.sub(r"\s+", " ", name).strip()


def guess_type(name: str) -> str:
    for suffix, etype in _TYPE_HINTS:
        if name.endswith(suffix):
            return etype
    if re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", name):
        return "person"
    if any(w in name for w in ("Components", "Pack", "Logistics")):
        return "supplier"
    return "entity"


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    doc_id: str
    sentence: str
    via: str | None = None  # e.g. the supplied part in "supplies X for Y"

    def as_path_step(self) -> str:
        return f"{self.subject} -[{self.relation}]-> {self.object}"


@dataclass
class EntityResolver:
    """Resolves surface mentions to canonical entity names.

    Resolution order: exact canonical -> alias table -> token containment
    (e.g. "Atlas Warehouse" mentioned as part of a longer phrase). Every
    decision is deterministic and recorded in ``merges`` for auditability.
    """

    canonical: dict[str, str] = field(default_factory=dict)  # normalised -> display
    aliases: dict[str, str] = field(default_factory=dict)    # normalised alias -> canonical display
    merges: list[tuple[str, str, str]] = field(default_factory=list)  # (mention, canonical, rule)

    def register(self, name: str) -> str:
        """Register a canonical entity (first-seen display form wins)."""
        key = normalise(name)
        if key not in self.canonical:
            display = re.sub(r"^(The|A|An)\s+", "", name.strip(), flags=re.IGNORECASE)
            self.canonical[key] = display
        return self.canonical[key]

    def add_alias(self, alias: str, canonical_name: str) -> None:
        canon = self.register(canonical_name)
        self.aliases[normalise(alias)] = canon

    def resolve(self, mention: str) -> str | None:
        key = normalise(mention)
        if key in self.canonical:
            return self.canonical[key]
        if key in self.aliases:
            target = self.aliases[key]
            self.merges.append((mention, target, "alias-table"))
            return target
        # Token containment: mention's tokens are a strict subset of exactly
        # one canonical entity (e.g. "Atlas" inside "Atlas Warehouse").
        mention_tokens = set(key.split())
        candidates = [
            display
            for norm, display in self.canonical.items()
            if mention_tokens and mention_tokens < set(norm.split())
        ]
        if len(candidates) == 1:
            self.merges.append((mention, candidates[0], "containment"))
            return candidates[0]
        return None

    def bootstrap_aliases(self) -> None:
        for alias, canon in _ALIAS_TABLE.items():
            if normalise(canon) in self.canonical:
                self.aliases[normalise(alias)] = self.canonical[normalise(canon)]


@dataclass
class ExtractionResult:
    triples: list[Triple]
    resolver: EntityResolver


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def extract_from_text(doc_id: str, text: str, resolver: EntityResolver) -> list[Triple]:
    """Extract triples from one document, registering entities as found."""
    triples: list[Triple] = []
    for sentence in split_sentences(text):
        for pattern, relation, order in _PATTERNS:
            m = pattern.match(sentence)
            if not m:
                continue
            groups = [m.group(i).strip(" .") for i in order]
            subj_raw, obj_raw = groups[0], groups[1]
            via = groups[2] if len(groups) > 2 else None
            if relation == "part_of":
                obj_raw = re.sub(r"\s+category$", "", obj_raw, flags=re.IGNORECASE)
            subj = resolver.resolve(subj_raw) or resolver.register(subj_raw)
            obj = resolver.resolve(obj_raw) or resolver.register(obj_raw)
            triples.append(
                Triple(subject=subj, relation=relation, object=obj,
                       doc_id=doc_id, sentence=sentence, via=via)
            )
            break  # one relation per sentence keeps extraction predictable
    return triples


def extract_corpus(documents) -> ExtractionResult:
    """Two-pass extraction: pass 1 builds the gazetteer, pass 2 re-resolves
    every mention against it so early documents benefit from later aliases."""
    resolver = EntityResolver()
    all_triples: list[Triple] = []
    for doc in documents:
        all_triples.extend(extract_from_text(doc.doc_id, doc.text, resolver))
    resolver.bootstrap_aliases()
    # Pass 2: canonicalise subjects/objects through the finished resolver.
    for t in all_triples:
        t.subject = resolver.resolve(t.subject) or t.subject
        t.object = resolver.resolve(t.object) or t.object
    # Deduplicate identical triples, keeping the first provenance.
    seen: set[tuple[str, str, str, str | None]] = set()
    unique: list[Triple] = []
    for t in all_triples:
        key = (t.subject, t.relation, t.object, t.via)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return ExtractionResult(triples=unique, resolver=resolver)
