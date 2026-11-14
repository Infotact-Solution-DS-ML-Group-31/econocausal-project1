from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from econocausal.causal.data_contract import validate_causal_data
from econocausal.optimization.allocate import optimize_allocation
from econocausal.causal.uplift import evaluate_uplift
from econocausal.monitoring.drift import detect_drift
from econocausal.causal.psm import (
    estimate_propensity_scores,
    check_common_support,
    nearest_neighbour_match,
    evaluate_matched_sample,
)


app = FastAPI(
    title="EconoCausal API",
    description=(
        "API for causal-data readiness checks and "
        "budget-constrained offer allocation."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OfferInput(BaseModel):
    customer_id: str = Field(min_length=1)
    offer_id: str = Field(min_length=1)
    cost: float = Field(gt=0, allow_inf_nan=False)
    expected_incremental_value: float = Field(allow_inf_nan=False)


class AllocationRequest(BaseModel):
    budget: float = Field(ge=0, allow_inf_nan=False)
    offers: list[OfferInput] = Field(min_length=1)


class CausalReadinessRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)
    covariates: list[str] = Field(default_factory=list)

class PSMRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)
    covariates: list[str] = Field(min_length=1)
    caliper: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    replacement: bool = False

class UpliftRequest(BaseModel):
    treatment: list[float] = Field(min_length=1)
    outcome: list[float] = Field(min_length=1)
    predicted_uplift: list[float] = Field(min_length=1)

class DriftRequest(BaseModel):
    reference_records: list[dict[str, Any]] = Field(min_length=1)
    current_records: list[dict[str, Any]] = Field(min_length=1)
    features: list[str] | None = None
    bins: int = Field(default=10, ge=2, le=100)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "econocausal-api",
    }


@app.get("/causal/readiness")
def causal_readiness() -> dict:
    """
    Report the known readiness status of the current project dataset.

    The current historical retail dataset does not contain assigned
    treatment or exposed non-purchaser records.
    """
    return {
        "causal_estimation_ready": False,
        "status": "not_ready",
        "reason": (
            "The current transaction dataset does not contain "
            "explicit discount assignment or exposed non-purchaser outcomes."
        ),
        "required_data": [
            "customer_id",
            "product_id",
            "treatment",
            "outcome",
            "assignment_date",
            "pre-treatment covariates",
        ],
        "warning": (
            "Observed transaction prices must not be treated as "
            "randomized discount assignments."
        ),
    }


@app.post("/causal/validate")
def validate_causal_records(request: CausalReadinessRequest) -> dict:
    """
    Validate the structure of a supplied customer-level causal dataset.

    Passing this check does not prove causal identification.
    """
    try:
        df = pd.DataFrame(request.records)
        report = validate_causal_data(
            df,
            covariates=request.covariates,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "valid": report.valid,
        "rows_checked": report.rows_checked,
        "errors": report.errors,
        "warnings": report.warnings,
        "causal_identification_established": False,
    }


@app.post("/causal/psm")
def evaluate_psm_endpoint(request: PSMRequest) -> dict:
    """
    Estimate propensity scores, identify common support, perform
    nearest-neighbour matching, and evaluate the matched sample.

    This endpoint reports an observational matched-sample estimate.
    It does not establish causal identification.
    """
    try:
        df = pd.DataFrame(request.records)

        propensity_result = estimate_propensity_scores(
            df=df,
            covariates=request.covariates,
        )

        support_result = check_common_support(
            propensity_scores=propensity_result.propensity_scores,
            treatment=df["treatment"].to_numpy(),
        )

        match_result = nearest_neighbour_match(
            propensity_scores=propensity_result.propensity_scores,
            treatment=df["treatment"].to_numpy(),
            caliper=request.caliper,
            replacement=request.replacement,
            in_support=support_result.in_support,
        )

        report = evaluate_matched_sample(
            df=df,
            match_result=match_result,
            covariates=request.covariates,
        )

    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "n_observations": propensity_result.n_observations,
        "n_treated": propensity_result.n_treated,
        "n_control": propensity_result.n_control,
        "common_support": {
            "lower_bound": support_result.lower_bound,
            "upper_bound": support_result.upper_bound,
            "n_in_support": support_result.n_in_support,
            "n_outside_support": support_result.n_outside_support,
            "n_treated_in_support": support_result.n_treated_in_support,
            "n_control_in_support": support_result.n_control_in_support,
        },
        "matching": {
            "n_matched": match_result.n_matched,
            "n_unmatched_treated": match_result.n_unmatched_treated,
            "n_unique_controls_used": match_result.n_unique_controls_used,
            "replacement": match_result.replacement,
            "caliper": match_result.caliper,
            "mean_matching_distance": report.mean_matching_distance,
            "treated_indices": match_result.treated_indices.tolist(),
            "control_indices": match_result.control_indices.tolist(),
        },
        "att": report.att,
        "smd_before": report.smd_before,
        "smd_after": report.smd_after,
        "balance_improved": report.balance_improved,
        "causal_identification_established": False,
        "warning": (
            "The ATT is an observational matched-sample estimate. "
            "Matching does not eliminate unobserved confounding or "
            "establish causal identification. Covariates must be "
            "measured before treatment assignment."
        ),
    }

@app.post("/causal/uplift")
def evaluate_uplift_endpoint(request: UpliftRequest) -> dict:
    """
    Evaluate predicted uplift rankings using Qini and AUUC metrics.

    Requires valid treatment assignment and outcome data.
    This endpoint does not establish causal identification.
    """
    try:
        result = evaluate_uplift(
            treatment=request.treatment,
            outcome=request.outcome,
            predicted_uplift=request.predicted_uplift,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "auuc": result.auuc,
        "qini_coefficient": result.qini_coefficient,
        "random_auuc": result.random_auuc,
        "n_observations": result.n_observations,
        "n_treated": result.n_treated,
        "n_control": result.n_control,
        "curve": result.curve,
        "random_curve": result.random_curve,
        "warnings": result.warnings,
        "causal_identification_established": False,
    }

@app.post("/monitoring/drift")
def detect_drift_endpoint(request: DriftRequest) -> dict:
    """
    Compare reference and current feature distributions.

    PSI measures distribution changes. It does not establish
    causal validity or prove model performance degradation.
    """
    try:
        reference_df = pd.DataFrame(request.reference_records)
        current_df = pd.DataFrame(request.current_records)

        report = detect_drift(
            reference=reference_df,
            current=current_df,
            features=request.features,
            bins=request.bins,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return report.to_dict()


@app.post("/optimization/allocate")
def allocate_offers(request: AllocationRequest) -> dict:
    """
    Allocate offers using externally supplied expected incremental values.

    This endpoint does not estimate treatment effects or uplift.
    """
    try:
        offers_df = pd.DataFrame(
            [offer.model_dump() for offer in request.offers]
        )

        result = optimize_allocation(
            offers=offers_df,
            budget=request.budget,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    selected = result.allocations[
        result.allocations["selected"] == 1
    ]

    return {
        "budget": result.budget,
        "total_cost": result.total_cost,
        "remaining_budget": result.budget - result.total_cost,
        "total_expected_incremental_value": result.total_expected_value,
        "selected_count": result.selected_count,
        "solver_status": result.solver_status,
        "allocations": result.allocations.to_dict(orient="records"),
        "selected_offers": selected.to_dict(orient="records"),
        "causal_estimates_generated": False,
    }


@app.post("/data/inspect-csv")
def inspect_csv(file: UploadFile = File(...)) -> dict:
    """
    Inspect a CSV in chunks without loading the entire dataset
    into memory. Reports structural causal-schema availability,
    not causal identification.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a file with a .csv extension.",
        )

    required_fields = {
        "customer_id",
        "product_id",
        "treatment",
        "outcome",
        "assignment_date",
    }

    total_rows = 0
    columns = None
    missing_counts = None
    inferred_types = {}

    try:
        for chunk in pd.read_csv(
            file.file,
            chunksize=100_000,
            low_memory=False,
        ):
            if columns is None:
                columns = list(chunk.columns)
                missing_counts = {
                    column: 0 for column in columns
                }
                inferred_types = {
                    column: str(chunk[column].dtype)
                    for column in columns
                }

            total_rows += len(chunk)

            chunk_missing = chunk.isna().sum()
            for column in columns:
                missing_counts[column] += int(
                    chunk_missing[column]
                )

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV is empty or has no header.",
        )
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {str(exc)}",
        )
    finally:
        file.file.close()

    if not columns or total_rows == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV contains no data rows.",
        )

    missing_fields = sorted(required_fields - set(columns))

    missing_summary = {
        column: {
            "missing_count": count,
            "missing_percentage": round(
                (count / total_rows) * 100, 2
            ),
        }
        for column, count in missing_counts.items()
    }

    return {
        "filename": file.filename,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "column_types_sample": inferred_types,
        "missing_values": missing_summary,
        "causal_schema_present": len(missing_fields) == 0,
        "missing_causal_fields": missing_fields,
        "causal_identification_established": False,
        "warning": (
            "Column availability and structural inspection do not "
            "establish random assignment, absence of confounding, "
            "overlap, or causal identification."
        ),
    }
