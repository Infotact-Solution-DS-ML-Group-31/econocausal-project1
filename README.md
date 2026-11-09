# EconoCausal: Dynamic Pricing via Double Machine Learning

A data science and machine learning project exploring causal inference, uplift evaluation, budget-constrained offer allocation, and data drift monitoring through a Python backend and React dashboard.

## 1. Project Overview

EconoCausal is designed to help marketing and business teams move beyond conventional predictive analytics toward treatment-effect analysis and budget-aware promotional decisions.

Traditional predictive models estimate which customers are likely to purchase. Causal machine learning instead aims to estimate how an intervention, such as a discount or promotional offer, changes an outcome.

The project combines:

- Data engineering and exploratory data analysis
- Double Machine Learning (DML)
- Causal-data validation and placebo refutation
- Uplift and Qini evaluation
- Budget-constrained offer allocation
- Data drift monitoring
- FastAPI REST services
- A React and Plotly dashboard

**Important:** The supplied historical retail dataset is suitable for descriptive sales and price analysis. It does not contain the explicit treatment assignment, exposure, and outcome information required to identify causal treatment effects. The project therefore separates descriptive retail analysis from causal estimation and validation using appropriately structured data.

## 2. Project Objectives

1. Build a reproducible data preparation and validation pipeline.
2. Analyze historical retail transactions and identify products with changing observed prices.
3. Implement a cross-fitted Double Machine Learning estimator.
4. Validate the structure of data submitted for causal analysis.
5. Evaluate supplied uplift rankings using Qini-related metrics.
6. Allocate promotional offers under a fixed budget and customer-level constraints.
7. Monitor changes in incoming data distributions.
8. Expose backend functionality through REST APIs.
9. Provide an interactive dashboard for analysis and visualization.

## 3. System Architecture

The project has four major layers:

### Data Engineering
- Raw retail workbook ingestion
- Data cleaning and preprocessing
- Exploratory data analysis
- Product-month candidate panel construction
- Dataset validation and reporting

### Causal and Optimization Modules
- Causal data contract validation
- Cross-fitted DML estimation
- Placebo-based refutation checks
- Uplift and Qini evaluation
- SciPy-based budget optimization
- Feature-level data drift detection

### Backend
- FastAPI REST API
- Causal readiness and validation endpoints
- Uplift evaluation endpoint
- Budget allocation endpoint
- Drift monitoring endpoint
- Chunked CSV inspection endpoint

### Frontend
- React dashboard
- Plotly uplift visualization
- Budget allocation interface
- Data drift monitoring
- CSV upload and inspection
- Causal schema validation feedback

## 4. Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python, JavaScript |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Causal estimation | Cross-fitted Double Machine Learning |
| Optimization | SciPy |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Visualization | Plotly.js |
| CSV processing | PapaParse |
| Testing | Pytest, HTTPX |
| Version control | Git, GitHub |

## 5. Repository Structure

econocausal-dynamic-pricing/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── figures/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── data_handoff.md
│   ├── eda_report.md
│   ├── limitations.md
│   ├── methodology.md
│   └── results.md
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── src/
│   └── econocausal/
│       ├── api/
│       ├── causal/
│       ├── data/
│       ├── evaluation/
│       ├── monitoring/
│       └── optimization/
│
├── tests/
├── pyproject.toml
└── README.md

## 6. Dataset and Data Pipeline

The data engineering workflow uses the Online Retail II transaction dataset.

Expected input location:

```text
data/raw/online_retail_II.xlsx
```

The preprocessing pipeline:
1. Loads the retail workbook.
2. Removes cancellation transactions.
3. Excludes nonpositive quantities and prices.
4. Excludes specified non-product or special stock codes.
5. Produces a cleaned transaction dataset.
6. Builds a monthly product-level candidate panel.
7. Validates duplicate product-month keys, calendar coverage, and numeric fields.

Run the complete pipeline from the repository root:

```bash
python -m econocausal.data.run_pipeline
```

The raw and processed datasets are excluded from Git because of their size and data-management considerations. Obtain the source workbook separately and place it in the expected input directory.

See `docs/data_dictionary.md`, `docs/data_handoff.md`, and `docs/eda_report.md` for additional details.

## 7. Causal Inference and Evaluation

### Double Machine Learning

The causal module implements a cross-fitted partially linear DML estimator using machine-learning nuisance models and residualization.

The estimator requires structured observations containing:
- Customer identifier
- Product identifier
- Binary treatment assignment
- Binary outcome
- Assignment date
- Pre-treatment covariates

The estimator's output depends on the partially linear treatment-effect assumptions and the quality of the supplied data.

A structurally valid dataset does not automatically establish causal identification. Randomization, confounding control, overlap, and other identification assumptions require separate justification.

### Propensity Score Matching

The causal module implements Propensity Score Matching (PSM) using logistic regression to estimate propensity scores from observed covariates.

The workflow:
1. Estimates each observation's propensity score.
2. Checks common support between treated and control groups.
3. Performs nearest-neighbour matching, with configurable replacement and an optional caliper.
4. Calculates the Average Treatment Effect on the Treated (ATT) for the matched sample.
5. Reports standardized mean differences (SMD) before and after matching to assess observed covariate balance.

Matching can improve balance on observed covariates, but it cannot eliminate unobserved confounding. The matched-sample ATT must therefore be interpreted as an observational estimate, not proof of causal identification.

### Uplift Evaluation

The uplift module evaluates supplied treatment-effect rankings and calculates cumulative Qini-related metrics, including AUUC and a Qini coefficient.

It evaluates supplied uplift scores; it does not itself train an uplift model.

### Refutation

The causal module includes placebo-based refutation functionality for checking whether estimated results remain sensitive to an artificial placebo treatment.

These checks are diagnostics, not proof that all unobserved confounding has been eliminated.

## 8. Budget Optimization

The optimization module uses SciPy-based mixed-integer optimization to select offers under constraints.

The objective is to maximize the supplied expected incremental value while respecting:
- Total budget constraints
- At most one selected offer per customer

The optimizer consumes expected incremental values supplied to it. It does not independently estimate causal uplift or guarantee that those values are causal.

## 9. Data Drift Monitoring

The monitoring module compares reference and current feature distributions using Population Stability Index (PSI)-based diagnostics.

It supports numerical and categorical feature comparisons and reports drift or warning statuses using configured thresholds.

These thresholds are monitoring heuristics, not universal statistical significance tests.

## 10. REST API

Start the backend from the repository root:

```bash
uvicorn econocausal.api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Implemented endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health check |
| GET | `/causal/readiness` | Report causal-data readiness requirements |
| POST | `/causal/validate` | Validate submitted causal data structure |
| POST | `/causal/psm` | Estimate propensity scores, check common support, perform nearest-neighbour matching, and report matched-sample balance and ATT |
| POST | `/causal/uplift` | Evaluate supplied uplift scores |
| POST | `/optimization/allocate` | Optimize offer allocation |
| POST | `/monitoring/drift` | Detect feature distribution drift |
| POST | `/data/inspect-csv` | Inspect uploaded CSV files |

## 11. Frontend Setup

Open a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

The frontend expects the backend at:

```text
http://127.0.0.1:8000
```

For a production frontend build:

```bash
npm run build
```

Run frontend lint checks:

```bash
npm run lint
```

## 12. Testing

Run backend tests from the repository root:

```bash
pytest -q
```

The current verified backend test result is **117 passed**.

Run frontend linting and production build:

```bash
cd frontend
npm run lint
npm run build
```

## 13. Current Limitations

The following limitations are important when interpreting results:

1. The historical retail dataset contains transaction records, not explicit promotional treatment assignments.
2. Missing customer identifiers limit customer-level analysis.
3. Purchase quantity is not equivalent to a conversion outcome for all exposed customers, including non-purchasers.
4. Product-month price variation does not establish randomized or exogenous price assignment.
5. Causal validation checks data structure but cannot prove causal identification.
6. The uplift evaluation consumes supplied scores and does not train an uplift model.
7. Budget optimization relies on supplied expected incremental values.
8. PSI-based drift thresholds are configurable monitoring heuristics.
9. The conceptual causal graph should not be interpreted as an empirically validated causal structure for the retail dataset.

Consequently, causal effect estimates should only be interpreted when suitable treatment assignment, outcome, covariate, and identification information is available.

## 14. Future Improvements

- Integrate a validated customer-level promotional campaign dataset.
- Add uncertainty intervals and sensitivity analysis for matched-sample ATT estimates.
- Strengthen causal identification and overlap diagnostics.
- Evaluate estimator robustness using additional refutation methods.
- Add validated heterogeneous treatment-effect estimation.
- Improve uplift-model training and evaluation.
- Add persistent experiment and model-result storage.
- Introduce authentication, deployment configuration, and production monitoring.

## 15. Project Deliverables

The project is developed as part of the Infotact Solutions Data Science and Machine Learning internship assignment.

The expected deliverables include:
- Source code
- Project documentation
- Project report
- Presentation
- Model evaluation results, where applicable
- Final project demonstration

---

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning
**Domain:** Economics and Causal Machine Learning
**Backend:** FastAPI
**Frontend:** React and Plotly
**Status:** Integrated development version
