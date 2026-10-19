"""Graph store: construction, stats, traversal, and path rendering."""


def test_stats_shape(kg):
    stats = kg.stats()
    assert stats["entities"] >= 20
    assert stats["edges"] >= 25
    assert stats["relations"]["leads"] == 4
    assert stats["relations"]["supplies_part_for"] == 5
    assert stats["entity_types"].get("supplier") == 3
    assert stats["entity_types"].get("person") == 6
    assert stats["entity_types"].get("product") == 3


def test_entity_types_refined_by_relation(kg):
    assert kg.graph.nodes["Helix Components"]["type"] == "supplier"
    assert kg.graph.nodes["Aero Blender Pro"]["type"] == "product"
    assert kg.graph.nodes["Kitchen Appliances"]["type"] == "category"
    assert kg.graph.nodes["Data Platform Team"]["type"] == "team"
    assert kg.graph.nodes["Maria Chen"]["type"] == "person"


def test_neighbours_respect_hop_budget(kg):
    one = kg.neighbours("Priya Nair", max_hops=1)
    assert "Support Operations Team" in one
    assert "Returns Portal" not in one  # 2 hops away
    two = kg.neighbours("Priya Nair", max_hops=2)
    assert two["Returns Portal"] == 2


def test_neighbours_unknown_entity(kg):
    assert kg.neighbours("Nobody") == {}


def test_find_path_multihop(kg):
    path = kg.find_path("Priya Nair", "Helix Components", max_hops=4)
    assert path is not None
    assert path.hops == 4
    assert path.nodes[0] == "Priya Nair"
    assert path.nodes[-1] == "Helix Components"
    rendered = path.render()
    assert "leads" in rendered and "supplies_part_for" in rendered


def test_find_path_reverse_traversal_is_marked(kg):
    path = kg.find_path("Atlas Warehouse", "Maria Chen", max_hops=3)
    assert path is not None
    assert path.hops == 2
    assert "<-[owns]-" in path.render()  # walked against the stored direction


def test_find_path_respects_max_hops(kg):
    assert kg.find_path("Priya Nair", "Helix Components", max_hops=2) is None


def test_find_path_unknown_entity(kg):
    assert kg.find_path("Priya Nair", "Ghost") is None


def test_entity_facts(kg):
    facts = kg.entity_facts("Atlas Warehouse")
    relations = {t.relation for t in facts}
    assert "owns" in relations and "depends_on" in relations
