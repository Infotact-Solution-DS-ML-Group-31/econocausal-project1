from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalNode:
    name: str
    description: str


@dataclass(frozen=True)
class CausalGraph:
    nodes: tuple[CausalNode, ...]
    edges: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]


def build_pricing_causal_graph() -> CausalGraph:
    """
    Define a conceptual causal graph for discount-based pricing.

    This is a proposed causal structure, not a graph learned
    or verified from the historical retail transaction data.
    """

    nodes = (
        CausalNode(
            "customer_features",
            "Pre-treatment customer characteristics and purchase history",
        ),
        CausalNode(
            "product_features",
            "Product characteristics and historical product demand",
        ),
        CausalNode(
            "campaign_context",
            "Campaign timing, channel, and promotion context",
        ),
        CausalNode(
            "treatment",
            "Discount or offer assigned to a customer",
        ),
        CausalNode(
            "outcome",
            "Purchase or conversion after offer assignment",
        ),
    )

    edges = (
        ("customer_features", "treatment"),
        ("customer_features", "outcome"),
        ("product_features", "treatment"),
        ("product_features", "outcome"),
        ("campaign_context", "treatment"),
        ("campaign_context", "outcome"),
        ("treatment", "outcome"),
    )

    assumptions = (
        "Customer, product, and campaign features are measured before treatment.",
        "Treatment represents an assigned offer, not merely an observed transaction price.",
        "Outcome includes exposed customers who did not purchase.",
        "All important common causes of treatment and outcome are measured.",
        "Treatment and outcome are defined for the same customer and decision period.",
    )

    return CausalGraph(
        nodes=nodes,
        edges=edges,
        assumptions=assumptions,
    )


def graph_to_mermaid(graph: CausalGraph) -> str:
    """Convert the causal graph into Mermaid flowchart syntax."""

    lines = ["flowchart LR"]

    for node in graph.nodes:
        lines.append(f"    {node.name}[{node.description}]")

    for source, target in graph.edges:
        lines.append(f"    {source} --> {target}")

    return "\n".join(lines)