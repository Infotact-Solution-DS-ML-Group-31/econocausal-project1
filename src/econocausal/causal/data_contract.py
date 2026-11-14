from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "product_id",
    "treatment",
    "outcome",
    "assignment_date",
}


@dataclass
class ContractReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_checked: int = 0


def validate_causal_data(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> ContractReport:
    """
    Validate the structural requirements of a customer-level
    discount-assignment dataset.

    treatment: 1 = assigned discount, 0 = not assigned
    outcome: 1 = purchased/converted, 0 = did not purchase/convert

    Passing this validation does not establish causal identification.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(df, pd.DataFrame):
        return ContractReport(
            valid=False,
            errors=["Input must be a pandas DataFrame."],
        )

    if df.empty:
        return ContractReport(
            valid=False,
            errors=["Dataset is empty."],
        )

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        return ContractReport(
            valid=False,
            errors=[f"Missing required columns: {missing_columns}"],
            rows_checked=len(df),
        )

    covariates = covariates or []
    missing_covariates = sorted(set(covariates) - set(df.columns))

    if missing_covariates:
        errors.append(
            f"Missing requested pre-treatment covariates: {missing_covariates}"
        )

    for column in ["customer_id", "product_id", "assignment_date"]:
        if df[column].isna().any():
            errors.append(f"Column '{column}' contains missing values.")

    for column in ["treatment", "outcome"]:
        if df[column].isna().any():
            errors.append(f"Column '{column}' contains missing values.")
            continue

        invalid = ~df[column].isin([0, 1])
        if invalid.any():
            errors.append(
                f"Column '{column}' must contain only binary values 0 and 1."
            )

    parsed_dates = pd.to_datetime(
    df["assignment_date"],
    format="mixed",
    errors="coerce",
)
    if parsed_dates.isna().any():
        errors.append(
            "Column 'assignment_date' contains invalid date values."
        )

    for column in covariates:
        if column in df.columns and df[column].isna().any():
            warnings.append(
                f"Covariate '{column}' contains missing values."
            )

    if df["treatment"].nunique(dropna=True) < 2:
        errors.append(
            "Both treated and untreated observations are required."
        )

    if df["outcome"].nunique(dropna=True) < 2:
        warnings.append(
            "Only one outcome class is present; outcome modelling may be limited."
        )

    if df["customer_id"].nunique(dropna=True) < 2:
        warnings.append(
            "Fewer than two distinct customers are present."
        )

    warnings.append(
        "Structural validation does not establish random assignment, "
        "no unmeasured confounding, overlap, or causal identification."
    )

    return ContractReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        rows_checked=len(df),
    )
