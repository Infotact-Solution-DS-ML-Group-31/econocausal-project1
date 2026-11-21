# EconoCausal: Dynamic Pricing via Double Machine Learning

## Project Report

**Organization:** Infotact Solutions
**Project Domain:** Data Science and Machine Learning
**Project:** EconoCausal – Dynamic Pricing via Double Machine Learning
**Project Period:** 10 October 2026 – 10 November 2026
**Technology:** Python, Pandas, NumPy, Scikit-learn, SciPy, FastAPI, React, Plotly
**Repository:** `Infotact-Solution-DS-ML-Group-31/econocausal-project1`

---

## 1. Executive Summary

EconoCausal is a project focused on applying causal machine learning concepts to dynamic pricing and promotional budget allocation. The central motivation is that conventional predictive machine learning identifies patterns and correlations but does not, by itself, establish the effect of a particular intervention, such as offering a discount to a customer.

The project combines data engineering, exploratory data analysis, causal-data validation, a Double Machine Learning (DML) estimator, uplift evaluation, budget optimization, data drift monitoring, a REST API, and a React-based dashboard.

The data engineering work uses the UCI Online Retail II dataset. The original workbook contains 1,067,371 transaction rows. Following documented preparation rules, a cleaned transaction dataset containing 1,040,749 rows was generated. Exploratory analysis and product-level screening identified four products for focused descriptive analysis. A product-month panel containing 100 records was constructed and passed the implemented structural and numeric validation checks.

The software also includes a causal-data contract, a cross-fitted DML estimator, placebo refutation utilities, uplift evaluation, a constrained budget allocation module, and data drift monitoring. These components are exposed through a FastAPI backend and a React dashboard.

An important limitation is that the historical retail dataset records completed transactions rather than randomized promotional assignments and all exposed customer purchase opportunities. Consequently, the current retail dataset does not establish causal identification or support defensible customer-level individual treatment effect claims. The DML and uplift components must be interpreted according to their input assumptions and test data, not as proof of causal effects in the retail dataset.

## 2. Introduction

Businesses frequently use discounts and promotional campaigns to influence purchasing behaviour. A customer who is likely to purchase a product without a discount may not need an incentive. Giving that customer a discount can reduce revenue without creating an additional purchase.

The relevant business question is therefore not simply whether a customer is likely to purchase. It is whether the customer's behaviour would change as a result of receiving a particular intervention.

EconoCausal explores this problem through causal inference and prescriptive optimization. It aims to connect data preparation, treatment-effect estimation, uplift evaluation, and budget-constrained offer allocation in a single software workflow.

## 3. Problem Statement

Traditional predictive models estimate outcomes using observed relationships between input variables and target variables. These relationships do not automatically reveal what would happen if a business changed its pricing or promotional strategy.

For example, a model may predict that a customer has a high probability of purchasing. That prediction does not establish whether offering the customer a discount would increase the probability of purchase.

The project investigates a system that can validate causal-analysis inputs, estimate treatment effects under explicit modelling assumptions, evaluate uplift scores, and allocate a limited promotional budget using expected incremental value.

## 4. Project Objectives

The objectives are:

1. Understand and prepare historical retail transaction data.
2. Perform exploratory data analysis and document data-quality issues.
3. Identify products suitable for further price-pattern analysis.
4. Construct and validate a product-month analytical panel.
5. Implement a causal-data validation contract.
6. Implement a cross-fitted Double Machine Learning estimator.
7. Provide uplift evaluation and placebo-refutation utilities.
8. Develop a budget-constrained offer allocation module using SciPy.
9. Implement data drift monitoring.
10. Expose backend functionality through REST API endpoints.
11. Build a React dashboard with Plotly-based visualizations.
12. Document the methodology, results, limitations, and future requirements.

## 5. Dataset Description

### 5.1 Dataset source

The project uses the UCI Online Retail II workbook. The workbook contains transaction records from an online retail business.

The source data includes:

- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

The two workbook sheets were combined to create a transaction-level dataset containing 1,067,371 source rows.

### 5.2 Data preparation

The preparation pipeline excludes:

- Cancellation invoices.
- Transactions with nonpositive quantities.
- Transactions with nonpositive unit prices.
- Special stock codes that do not represent ordinary products.

The raw workbook is preserved. The resulting prepared dataset contains **1,040,749 rows**.

Missing customer identifiers and exact duplicate transaction rows are retained in the primary prepared dataset. Their effects are documented through separate analyses rather than silently removing them.

### 5.3 Initial data-quality findings

The initial inspection identified:

| Data-quality measure | Count |
|---|---:|
| Source transaction rows | 1,067,371 |
| Exact duplicate source rows | 12,133 |
| Rows with missing Customer IDs | 243,007 |
| Rows with missing descriptions | 4,382 |
| Nonpositive quantities | 22,950 |
| Nonpositive prices | 6,207 |
| Cancellation invoice rows | 19,494 |

The prepared dataset contains 235,941 rows with missing Customer IDs, representing 22.67% of prepared records.

### 5.4 Exploratory data analysis

The EDA process examined transaction quality, dates, quantities, prices, customer identifiers, and product descriptions.

The following visualizations were generated:

- Monthly gross line value.
- Unit-price distribution.
- Transaction-line quantity distribution.

For readability, the distribution visualizations display observations up to the 99th percentile. The underlying dataset was not truncated for this purpose. The 99th percentile was 18.00 for unit price and 100 units for transaction-line quantity.

## 6. Product-Level Price Analysis

### 6.1 Candidate screening

The prepared dataset was further filtered for valid product codes, invoice dates, positive prices, positive quantities, and specified special stock codes.

Transactions were grouped by product code and calendar month. The analysis calculated transaction-row counts, monthly median unit prices, distinct transaction prices, and total quantity sold.

A product was eligible for further investigation if it met all three conditions:

1. At least 100 transaction rows across the analysis period.
2. Activity in at least six distinct calendar months.
3. More than one distinct monthly median price.

The screening produced:

| Measure | Result |
|---|---:|
| Transaction rows after analysis exclusions | 1,037,391 |
| Product codes analysed | 4,908 |
| Product-month records | 66,958 |
| Products meeting screening criteria | 1,570 |

### 6.2 Shortlisted products

Four products were selected for focused descriptive analysis.

| Stock code | Active months | Transaction rows | Distinct monthly median prices |
|---|---:|---:|---:|
| 85099B | 25 | 4,124 | 2 |
| 47566 | 25 | 2,740 | 3 |
| 20685 | 25 | 1,769 | 4 |
| 48138 | 25 | 1,701 | 5 |

All four products have observations in each of the 25 calendar months from December 2009 through December 2011.

These products were selected for focused descriptive analysis. They were not automatically selected as the four highest-ranked products by the screening script, and their observed price variation does not establish causal price effects.

### 6.3 Product-month panel

A product-month panel was constructed for the four shortlisted products.

The panel contains 100 records: four products across 25 calendar months.

Its fields include transaction counts, median and mean unit prices, distinct transaction prices, total quantity sold, gross line value, distinct descriptions, and consecutive-month price-change measures.

### 6.4 Panel validation

The validation process checked:

- Duplicate product-month keys.
- Missing calendar months.
- Missing core numeric values.
- Nonpositive core numeric values.

The validation found:

| Validation measure | Result |
|---|---:|
| Product-month records | 100 |
| Duplicate product-month keys | 0 |
| Missing calendar months | 0 |
| Missing values in checked core measures | 0 |
| Nonpositive values in checked core measures | 0 |

The checks establish basic structural and numeric consistency. They do not establish that the data is free from all measurement errors, selection bias, or confounding.

## 7. System Architecture

The system is organized into data, causal-analysis, optimization, monitoring, API, and frontend components.

### 7.1 Data engineering layer

The data layer handles workbook ingestion, transaction cleaning, exploratory analysis, product screening, candidate-panel construction, and validation.

The processing pipeline is designed to preserve the original source workbook and produce reproducible processed outputs.

### 7.2 Causal analysis layer

The causal package includes:

- Causal-data contract validation.
- A cross-fitted Double Machine Learning estimator.
- Uplift and Qini evaluation.
- Placebo-refutation utilities.
- A conceptual causal graph.

The causal-data contract checks required columns, missing values, binary treatment and outcome fields, treatment-group availability, and basic structural conditions.

### 7.3 Optimization layer

The optimization module uses SciPy-based constrained optimization to select offers subject to a total budget and a maximum of one offer per customer.

The objective is to maximize the sum of expected incremental values while satisfying the allocation constraints.

The optimizer uses supplied expected incremental values. It does not independently establish that those values are causal.

### 7.4 Monitoring layer

The monitoring module compares reference and current datasets using Population Stability Index (PSI) calculations for numeric and categorical features.

It reports features that exceed configured drift thresholds. These thresholds are operational heuristics and should not be interpreted as universal statistical significance thresholds.

### 7.5 Backend layer

The backend is implemented using FastAPI. It exposes the project functionality through REST endpoints.

### 7.6 Frontend layer

The frontend uses React and Plotly to provide a dashboard for backend status, causal readiness, data inspection, uplift evaluation, budget allocation, and drift monitoring.

## 8. Methodology and Implemented Modules

### 8.1 Causal-data validation

The causal-data contract expects the following fields:

- `customer_id`
- `product_id`
- `treatment`
- `outcome`
- `assignment_date`

The validator checks whether the supplied dataset meets basic structural requirements. It also reports warnings for missing values, insufficient customer records, or limited outcome variation.

Passing structural validation does not establish random assignment, absence of confounding, overlap, or causal identification.

### 8.2 Double Machine Learning

The DML implementation uses cross-fitting to estimate nuisance relationships for treatment and outcome.

The implemented workflow includes:

1. Validate the supplied treatment, outcome, and covariate columns.
2. Estimate the outcome relationship using a Random Forest regression model.
3. Estimate treatment propensity using logistic regression.
4. Generate cross-fitted nuisance predictions.
5. Residualize treatment and outcome.
6. Estimate a partially linear treatment-effect coefficient.
7. Calculate a standard error and confidence interval.

The implementation is intended for data that satisfies the estimator's assumptions. Its output should not be interpreted as an individual treatment effect for each customer. The partially linear coefficient also relies on modelling and identification assumptions that must be assessed for the application dataset.

The current Online Retail II transaction data does not contain the required promotional assignment and exposed non-purchaser information needed to establish those assumptions.

### 8.3 Uplift evaluation

The uplift module accepts treatment indicators, observed outcomes, and predicted uplift scores.

It ranks observations by predicted uplift and calculates cumulative treated-versus-control outcome differences, a random baseline, an area-under-the-uplift-curve measure, and a Qini coefficient.

This module evaluates supplied scores. It does not itself train a customer-level uplift model. Results from synthetic or otherwise suitable evaluation data must not be presented as evidence of causal performance on the historical retail dataset.

### 8.4 Placebo refutation

The project includes placebo-refutation utilities intended to support robustness checks.

A placebo test can help identify suspicious model behaviour under a specified test design. It does not, by itself, prove that all confounding has been removed or that a causal estimate is valid.

### 8.5 Budget optimization

The optimization module uses the following inputs:

- Customer identifier.
- Offer identifier.
- Offer cost.
- Expected incremental value.

The solver seeks to maximize total expected incremental value while respecting the available budget and the one-offer-per-customer constraint.

The quality of the allocation depends on the validity of the supplied incremental-value estimates.

### 8.6 Data drift monitoring

The monitoring module calculates PSI-based drift measures for numeric and categorical features.

It identifies features with warning-level or drift-detected status according to configured thresholds. Missing values and newly observed categories are also considered.

Drift detection indicates changes in feature distributions. It does not establish why those changes occurred or whether model performance has deteriorated.

## 9. REST API

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health check |
| GET | `/causal/readiness` | Report causal-data readiness |
| POST | `/causal/validate` | Validate causal input structure |
| POST | `/causal/uplift` | Calculate uplift evaluation metrics |
| POST | `/optimization/allocate` | Optimize offer allocation |
| POST | `/monitoring/drift` | Detect feature distribution drift |
| POST | `/data/inspect-csv` | Inspect uploaded CSV files |

The causal validation and readiness responses explicitly distinguish structural checks from causal identification.

The CSV inspection endpoint processes uploaded CSV files in chunks, reports dimensions and missing-value statistics, and checks for the expected causal schema.

## 10. Frontend Dashboard

The React dashboard provides a user interface for interacting with the backend.

The implemented interface includes:

- Backend connection and health status.
- Causal readiness information.
- CSV upload and data inspection.
- Uplift and Qini visualization.
- Budget allocation interface.
- Data drift monitoring results.

Plotly is used to render the uplift visualization. The frontend communicates with the FastAPI backend through HTTP requests.

Large CSV uploads use the backend inspection endpoint rather than requiring the entire file to be parsed in the browser.

## 11. Testing and Evaluation

### 11.1 Automated testing

The project includes automated tests for the causal-data contract, DML estimator, uplift evaluation, placebo-refutation utilities, optimization module, drift monitoring, and API functionality.

The latest recorded full test-suite result is:

**80 tests passed.**

This confirms that the implemented test cases passed in the recorded environment. It does not establish production readiness or causal validity on the historical retail dataset.

### 11.2 Frontend validation

The frontend was checked using the available lint and production-build commands.

The dashboard was also exercised against the running backend. The recorded checks included the uplift visualization, CSV inspection, and backend functionality.

### 11.3 Large CSV inspection

The processed retail CSV contains 1,040,749 rows and nine columns.

The inspection endpoint successfully reported the dataset dimensions and missing-value statistics. It identified that the retail dataset does not contain the required causal schema.

In particular, the dataset uses fields such as `Customer ID` and `StockCode`, while the causal contract expects fields such as `customer_id` and `product_id`. More importantly, renaming columns would not create the missing treatment assignment, exposure, or outcome information.

### 11.4 Drift monitoring

The drift endpoint was tested with numeric and categorical features. The recorded test returned a drift-detected result for the numeric `age` feature and a warning for the categorical `segment` feature.

These results demonstrate the monitoring workflow on the supplied test data, not on a validated live customer campaign.

## 12. Key Results

The main outcomes of the project are summarized below.

| Area | Recorded result |
|---|---|
| Source data | 1,067,371 transaction rows |
| Prepared dataset | 1,040,749 rows |
| Product codes analysed | 4,908 |
| Product-month records screened | 66,958 |
| Products meeting screening criteria | 1,570 |
| Products shortlisted | 4 |
| Final candidate panel | 100 rows |
| Candidate-panel duplicate keys | 0 |
| Candidate-panel missing calendar months | 0 |
| Automated tests | 80 passed |
| Large CSV inspection | Successfully exercised |
| Dashboard uplift chart | Successfully rendered |
| Drift monitoring | Successfully exercised |

The data-engineering work produced a reproducible cleaned dataset and a validated descriptive product-month panel. The software work produced tested components for causal-data validation, estimation, uplift evaluation, optimization, monitoring, and dashboard interaction.

These results demonstrate implementation and testing progress. They do not establish that the historical retail data supports causal pricing conclusions.

## 13. Limitations

### 13.1 Transaction data does not capture all demand

The dataset records completed transactions. It does not record every product viewed, every purchase opportunity, or every price offered to customers.

Observed quantity sold should therefore not be interpreted as complete customer demand.

### 13.2 Missing customer identifiers

The prepared dataset contains 235,941 rows with missing Customer IDs, representing 22.67% of prepared records.

This restricts reliable customer-level histories, segmentation, and causal modelling.

### 13.3 Duplicate transaction rows

The prepared dataset retains 12,051 exact duplicate rows. A sensitivity analysis of the four shortlisted products found 39 exact duplicate rows among 10,334 candidate transaction rows.

Removing these duplicates did not change monthly median unit prices in the 100 product-month records, but it reduced total quantity by 214 units and gross line value by £571.51.

### 13.4 Product-description variation

Some product codes have multiple descriptions. Product-code aggregation does not automatically resolve these differences.

### 13.5 Observational price variation

Monthly median-price changes are observational. They may reflect changes in product mix, customer composition, transaction composition, or other factors. They do not establish that price changes caused changes in sales.

### 13.6 Monthly aggregation

Product-month aggregation hides transaction-level variation in timing, prices, quantities, and customer behaviour.

### 13.7 Causal identification requirements

The historical retail dataset does not explicitly record promotional treatment assignment, all exposed customers, non-purchaser outcomes, or the information required to establish a defensible causal identification strategy.

A stronger causal study would require:

- Customer and product identifiers.
- Assigned treatment or discount amount.
- Treatment assignment and exposure dates.
- Purchase or conversion outcomes for treated and control populations, including exposed non-purchasers.
- Relevant pre-treatment customer covariates and purchase history.
- Campaign and pricing context.
- A defensible identification strategy and overlap assessment.

The current analysis should therefore be treated as data-engineering and software preparation for causal research, not as a completed causal pricing study.

## 14. Future Scope

The following improvements would strengthen the project:

1. Obtain campaign-level data containing treatment assignment and customer exposure.
2. Define treatment, outcome, and pre-treatment covariates before estimation.
3. Evaluate the causal identification assumptions and overlap.
4. Add and validate a propensity-score matching workflow if required by the project specification.
5. Assess the DML estimator's assumptions and clarify the interpretation of its partially linear coefficient.
6. Add appropriate heterogeneous treatment-effect estimation if individual treatment-effect predictions are required.
7. Validate uplift evaluation against a clearly specified experimental or observational evaluation design.
8. Add sensitivity analysis for duplicate transactions and missing customer identifiers.
9. Improve API upload limits, error handling, and input-schema documentation.
10. Evaluate model performance and monitoring behaviour on representative campaign data.

## 15. Team Responsibilities

The project work was organized across the following areas:

| Team member | Assigned responsibility |
|---|---|
| Kuldeep Katre | Data engineering, data pipeline, drift detection, and final integration |
| Ayush | Causal machine learning methodology, DML, propensity-score matching, and validation/refutation |
| Aniket | Optimization, API contracts, backend integration, and testing |
| Prince | React frontend, Plotly dashboard, upload flow, and frontend QA |

This table records the planned responsibility allocation. Individual completion status should be confirmed with the relevant team members before submission.

## 16. Conclusion

EconoCausal brings together data engineering, causal machine learning components, optimization, monitoring, API development, and frontend visualization in a single project.

The data pipeline produces a prepared retail transaction dataset and a validated four-product monthly panel. The software includes a causal-data contract, a cross-fitted DML estimator, uplift evaluation, placebo-refutation utilities, budget optimization, drift monitoring, REST endpoints, and a React dashboard.

Automated tests and recorded dashboard checks demonstrate that the implemented components can be exercised in the development environment.

However, the historical Online Retail II dataset does not provide the treatment-assignment and exposure information required to establish causal pricing effects. The project therefore provides a software and analytical foundation for further causal research rather than evidence that a particular discount causes an increase in customer purchases.

The next stage is to connect the system to suitable campaign data, assess causal identification assumptions, and validate treatment-effect estimates before using them to make real promotional allocation decisions.

---

## Appendix A: Reproducibility Commands

Run commands from the repository root.

**Install the Python project:**

```powershell
pip install -e .
```

**Run the data pipeline:**

```powershell
python -m econocausal.data.run_pipeline
```

**Run automated tests:**

```powershell
pytest -q
```

**Start the backend:**

```powershell
uvicorn econocausal.api.main:app --reload
```

**Start the frontend in a separate terminal:**

```powershell
cd frontend
npm install
npm run dev
```

**Build the frontend:**

```powershell
npm run build
```

**Run frontend lint checks:**

```powershell
npm run lint
```

## Appendix B: Project Deliverables

The project submission should include:

- Source code and version-controlled repository.
- Project documentation.
- Project report.
- Presentation.
- Model evaluation and testing results, where applicable.
- Final project demonstration.

The report should be submitted alongside the final repository state and supporting documentation.
