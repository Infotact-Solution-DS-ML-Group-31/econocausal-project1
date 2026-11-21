
import numpy as np
import pandas as pd
import pytest

from econocausal.monitoring.drift import detect_drift


def test_no_drift_for_similar_numeric_data():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "age": rng.normal(35, 8, 2000),
    })
    current = pd.DataFrame({
        "age": rng.normal(35, 8, 2000),
    })

    report = detect_drift(reference, current)

    assert report.status == "no_drift"
    assert report.features_checked == 1


def test_detects_numeric_distribution_shift():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "income": rng.normal(30000, 5000, 2000),
    })
    current = pd.DataFrame({
        "income": rng.normal(60000, 5000, 2000),
    })

    report = detect_drift(reference, current)

    assert report.status == "drift_detected"
    assert "income" in report.drifted_features


def test_detects_categorical_distribution_shift():
    reference = pd.DataFrame({
        "segment": ["A"] * 900 + ["B"] * 100,
    })
    current = pd.DataFrame({
        "segment": ["A"] * 100 + ["B"] * 900,
    })

    report = detect_drift(reference, current)

    assert report.status == "drift_detected"
    assert report.results[0].feature_type == "categorical"


def test_reports_missing_value_distribution():
    reference = pd.DataFrame({
        "age": [20, 25, 30, 35, 40] * 20,
    })
    current = pd.DataFrame({
        "age": [20, 25, 30, 35, 40] * 10 + [np.nan] * 50,
    })

    report = detect_drift(reference, current)

    assert report.results[0].current_missing_pct == 50.0
    assert report.status == "drift_detected"


def test_rejects_empty_dataset():
    with pytest.raises(ValueError, match="must not be empty"):
        detect_drift(
            pd.DataFrame({"age": [20, 30]}),
            pd.DataFrame(),
        )


def test_rejects_missing_feature():
    with pytest.raises(ValueError, match="Missing features"):
        detect_drift(
            pd.DataFrame({"age": [20, 30]}),
            pd.DataFrame({"income": [100, 200]}),
            features=["age"],
        )


def test_rejects_invalid_bin_count():
    with pytest.raises(ValueError, match="bins"):
        detect_drift(
            pd.DataFrame({"age": [20, 30]}),
            pd.DataFrame({"age": [21, 31]}),
            bins=1,
        )


def test_detects_shift_from_constant_reference():
    reference = pd.DataFrame({"score": [10.0] * 100})
    current = pd.DataFrame({"score": [50.0] * 100})

    report = detect_drift(reference, current)

    assert report.status == "drift_detected"
