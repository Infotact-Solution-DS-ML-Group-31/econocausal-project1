from econocausal.causal.graph import (
    build_pricing_causal_graph,
    graph_to_mermaid,
)


def test_graph_contains_required_nodes():
    graph = build_pricing_causal_graph()

    node_names = {node.name for node in graph.nodes}

    assert {
        "customer_features",
        "product_features",
        "campaign_context",
        "treatment",
        "outcome",
    }.issubset(node_names)


def test_treatment_causes_outcome_edge_exists():
    graph = build_pricing_causal_graph()

    assert ("treatment", "outcome") in graph.edges


def test_confounders_affect_treatment_and_outcome():
    graph = build_pricing_causal_graph()

    for confounder in [
        "customer_features",
        "product_features",
        "campaign_context",
    ]:
        assert (confounder, "treatment") in graph.edges
        assert (confounder, "outcome") in graph.edges


def test_graph_has_assumptions():
    graph = build_pricing_causal_graph()

    assert len(graph.assumptions) >= 4


def test_mermaid_export_contains_edges():
    graph = build_pricing_causal_graph()
    mermaid = graph_to_mermaid(graph)

    assert "flowchart LR" in mermaid
    assert "treatment --> outcome" in mermaid