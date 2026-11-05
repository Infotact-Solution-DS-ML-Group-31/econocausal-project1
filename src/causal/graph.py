"""Causal DAG specification and identification engine via DoWhy."""

from typing import Any, Optional
import networkx as nx
import pandas as pd

# Apply NetworkX 3+ compatibility patch for DoWhy
if not hasattr(nx.algorithms, "d_separated"):
    import networkx.algorithms.d_separation as ds

    nx.algorithms.d_separated = ds.is_d_separator

from dowhy import CausalModel


def build_causal_dag() -> str:
    """Build the formal causal Directed Acyclic Graph (DAG) in Graphviz DOT format.

    The DAG specifies:
      - Confounders W: {income, loyalty_score, historical_spend, browse_minutes}
      - Confounders bias treatment: W -> T (treatment)
      - Confounders affect outcome: W -> Y (conversion)
      - Direct causal effect of promotional pricing: T -> Y (conversion)

    Returns
    -------
    str
        DOT format string representing the causal graph.
    """
    dot_graph = (
        "digraph {\n"
        "  income -> treatment;\n"
        "  loyalty_score -> treatment;\n"
        "  historical_spend -> treatment;\n"
        "  browse_minutes -> treatment;\n"
        "  income -> conversion;\n"
        "  loyalty_score -> conversion;\n"
        "  historical_spend -> conversion;\n"
        "  browse_minutes -> conversion;\n"
        "  treatment -> conversion;\n"
        "}"
    )
    return dot_graph


def identify_causal_effect(
    df: pd.DataFrame,
    treatment: str = "treatment",
    outcome: str = "conversion",
    graph: Optional[str] = None,
) -> Any:
    """Instantiate a DoWhy CausalModel and identify the causal estimand via the Backdoor Criterion.

    Parameters
    ----------
    df : pd.DataFrame
        Customer observational dataset containing treatment, outcome, and confounders.
    treatment : str, default='treatment'
        Column name of the treatment variable.
    outcome : str, default='conversion'
        Column name of the outcome variable.
    graph : str, optional
        DOT graph representation. Defaults to build_causal_dag().

    Returns
    -------
    Any
        The identified causal estimand with backdoor adjustment sets, with the
        underlying CausalModel accessible via `.causal_model`.
    """
    causal_dag = graph or build_causal_dag()

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=causal_dag,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Attach the instantiated model for downstream estimation & refutations
    identified_estimand.causal_model = model

    return identified_estimand

