import pandas as pd

from econocausal.causal.data_contract import validate_causal_data


def valid_dataset():
    return pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "product_id": ["A", "A", "B", "B"],
        "treatment": [1, 0, 1, 0],
        "outcome": [1, 0, 0, 1],
        "assignment_date": [
            "2025-01-01",
            "2025-01-01",
            "2025-01-02",
            "2025-01-02",
        ],
        "previous_spend": [100, 80, 120, 90],
    })


def test_valid_dataset_passes():
    report = validate_causal_data(
        valid_dataset(),
        covariates=["previous_spend"],
    )

    assert report.valid
    assert report.rows_checked == 4
    assert report.errors == []


def test_missing_required_column_fails():
    df = valid_dataset().drop(columns=["treatment"])

    report = validate_causal_data(df)

    assert not report.valid
    assert any("Missing required columns" in error for error in report.errors)


def test_non_binary_treatment_fails():
    df = valid_dataset()
    df.loc[0, "treatment"] = 2

    report = validate_causal_data(df)

    assert not report.valid
    assert any("binary values" in error for error in report.errors)


def test_missing_customer_id_fails():
    df = valid_dataset()
    df.loc[0, "customer_id"] = None

    report = validate_causal_data(df)

    assert not report.valid
    assert any("customer_id" in error for error in report.errors)


def test_invalid_assignment_date_fails():
    df = valid_dataset()
    df.loc[0, "assignment_date"] = "not-a-date"

    report = validate_causal_data(df)

    assert not report.valid
    assert any("invalid date" in error for error in report.errors)


def test_missing_requested_covariate_fails():
    report = validate_causal_data(
        valid_dataset(),
        covariates=["age"],
    )

    assert not report.valid
    assert any("covariates" in error for error in report.errors)


def test_single_treatment_group_fails():
    df = valid_dataset()
    df["treatment"] = 1

    report = validate_causal_data(df)

    assert not report.valid
    assert any("Both treated and untreated" in error for error in report.errors)


def test_empty_dataset_fails():
    report = validate_causal_data(pd.DataFrame())

    assert not report.valid
    assert any("empty" in error.lower() for error in report.errors)
