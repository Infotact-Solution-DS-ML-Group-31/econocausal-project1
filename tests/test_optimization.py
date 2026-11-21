import pandas as pd
import pytest

from econocausal.optimization.allocate import optimize_allocation


def sample_offers():
    return pd.DataFrame({
        "customer_id": ["A", "A", "B", "C"],
        "offer_id": ["A1", "A2", "B1", "C1"],
        "cost": [5.0, 4.0, 6.0, 3.0],
        "expected_incremental_value": [10.0, 9.0, 12.0, 5.0],
    })


def test_optimizer_respects_budget():
    result = optimize_allocation(sample_offers(), budget=9.0)

    assert result.total_cost <= 9.0


def test_optimizer_selects_at_most_one_offer_per_customer():
    result = optimize_allocation(sample_offers(), budget=20.0)

    selected = result.allocations.query("selected == 1")
    assert not selected["customer_id"].duplicated().any()


def test_optimizer_maximizes_value_for_sample():
    result = optimize_allocation(sample_offers(), budget=9.0)

    # A1 + C1 costs 8 and yields value 15.
    # A2 + B1 costs 10 and is over budget.
    # B1 + C1 costs 9 and yields value 17.
    assert result.total_expected_value == pytest.approx(17.0)


def test_optimizer_returns_selection_flags():
    result = optimize_allocation(sample_offers(), budget=9.0)

    assert set(result.allocations["selected"].unique()).issubset({0, 1})


def test_missing_required_column_fails():
    df = sample_offers().drop(columns=["cost"])

    with pytest.raises(ValueError, match="Missing required columns"):
        optimize_allocation(df, budget=10.0)


def test_negative_budget_fails():
    with pytest.raises(ValueError, match="budget"):
        optimize_allocation(sample_offers(), budget=-1.0)


def test_nonpositive_cost_fails():
    df = sample_offers()
    df.loc[0, "cost"] = 0

    with pytest.raises(ValueError, match="greater than zero"):
        optimize_allocation(df, budget=10.0)


def test_duplicate_offer_id_fails():
    df = sample_offers()
    df.loc[1, "offer_id"] = "A1"

    with pytest.raises(ValueError, match="unique"):
        optimize_allocation(df, budget=10.0)
