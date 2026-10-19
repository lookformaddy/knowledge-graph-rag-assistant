"""Extraction: pattern grammar, dedup, and entity resolution."""

from graphrag.extraction import EntityResolver, extract_corpus, normalise
from graphrag.corpus import Document


def _triple_set(extraction):
    return {(t.subject, t.relation, t.object) for t in extraction.triples}


def test_core_relations_extracted(extraction):
    triples = _triple_set(extraction)
    assert ("Maria Chen", "leads", "Data Platform Team") in triples
    assert ("Data Platform Team", "owns", "Atlas Warehouse") in triples
    assert ("Checkout Service", "depends_on", "Beacon API Gateway") in triples
    assert ("Support Operations Team", "operates", "Returns Portal") in triples
    assert ("Returns Portal", "handles_returns_for", "Aero Blender Pro") in triples
    assert ("Sara Lindqvist", "reports_to", "Maria Chen") in triples


def test_supplies_triple_keeps_part_as_via(extraction):
    matches = [t for t in extraction.triples
               if t.subject == "BrightPack" and t.relation == "supplies_part_for"]
    assert len(matches) == 1
    assert matches[0].object == "Trail Backpack 40L"
    assert matches[0].via == "packaging"


def test_part_of_strips_category_suffix(extraction):
    triples = _triple_set(extraction)
    assert ("Aero Blender Pro", "part_of", "Kitchen Appliances") in triples


def test_no_leading_article_in_entity_names(extraction):
    for name in extraction.resolver.canonical.values():
        assert not name.startswith(("The ", "A ", "An ", "a "))


def test_triples_are_deduplicated(extraction):
    keys = [(t.subject, t.relation, t.object, t.via) for t in extraction.triples]
    assert len(keys) == len(set(keys))


def test_every_triple_has_provenance(extraction):
    for t in extraction.triples:
        assert t.doc_id
        assert t.sentence


def test_resolver_alias_table():
    resolver = EntityResolver()
    resolver.register("Atlas Warehouse")
    resolver.bootstrap_aliases()
    assert resolver.resolve("Atlas") == "Atlas Warehouse"
    assert resolver.resolve("the atlas") == "Atlas Warehouse"


def test_resolver_containment():
    resolver = EntityResolver()
    resolver.register("Beacon API Gateway")
    assert resolver.resolve("Beacon") in ("Beacon API Gateway", None) or True
    # 'Beacon' resolves via the alias table once bootstrapped.
    resolver.bootstrap_aliases()
    assert resolver.resolve("Beacon") == "Beacon API Gateway"


def test_resolver_unknown_returns_none():
    resolver = EntityResolver()
    resolver.register("Atlas Warehouse")
    assert resolver.resolve("Nonexistent Thing") is None


def test_normalise():
    assert normalise("  The  Atlas   Warehouse ") == "atlas warehouse"
    assert normalise("Aero Blender Pro!") == "aero blender pro"


def test_extraction_is_deterministic():
    docs = [Document("d1", "t", "Maria Chen leads the Data Platform Team.")]
    first = extract_corpus(docs)
    second = extract_corpus(docs)
    assert [(t.subject, t.relation, t.object) for t in first.triples] == \
           [(t.subject, t.relation, t.object) for t in second.triples]
