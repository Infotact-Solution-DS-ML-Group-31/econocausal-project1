# EconoCausal â€” Data Pipeline Architecture

## 1. Overview

This document describes the architecture of the Data Engineering and Exploratory Data Analysis (EDA) workflow developed for the EconoCausal project.

The workflow transforms the original retail transaction workbook into a prepared transaction dataset and a validated product-month analytical panel.

The pipeline consists of three sequential stages:

1. Prepared transaction dataset creation.
2. Candidate product-month panel creation.
3. Candidate panel validation.

## 2. Architecture Workflow

```text
          UCI Online Retail II
          Original Excel Workbook
                     |
                     v
        data/raw/online_retail_II.xlsx
                     |
                     v
        +--------------------------+
        | Stage 1: Data Preparation|
        |      build_dataset.py    |
        +--------------------------+
                     |
                     v
        data/processed/retail_sales.csv
                     |
                     v
        +-----------------------------+
        | Stage 2: Candidate Panel    |
        |         Creation            |
        |    build_candidate_panel.py |
        +-----------------------------+
                     |
                     v
 data/processed/candidate_product_month_panel.csv
                     |
                     v
        +-----------------------------+
        | Stage 3: Panel Validation   |
        |   validate_candidate_panel.py|
        +-----------------------------+
                     |
                     v
             Validation Results
```

## 3. Pipeline Components

### 3.1 Input Dataset

**Source:** UCI Online Retail II

**Location:** `data/raw/online_retail_II.xlsx`

The source workbook contains historical retail transaction records, including invoice identifiers, product information, quantities, transaction dates, prices, customer identifiers, and countries.

The raw workbook is preserved and is not modified by the preparation pipeline.

### 3.2 Stage 1 â€” Transaction Dataset Preparation

**Module:** `src/econocausal/data/build_dataset.py`

This module reads the original Excel workbook and applies the documented data preparation rules.

The preparation process excludes:

- Cancellation invoices.
- Nonpositive quantities.
- Nonpositive prices.
- Selected special stock codes.

Missing Customer IDs and exact duplicate transaction rows are retained in the primary prepared dataset.

**Output:**

`data/processed/retail_sales.csv`

The prepared dataset contains:

- 1,040,749 transaction rows.
- 9 columns.

### 3.3 Stage 2 â€” Candidate Product-Month Panel Creation

**Module:** `src/econocausal/data/build_candidate_panel.py`

This module reads the prepared transaction dataset and constructs a monthly analytical panel for four shortlisted products:

- 20685
- 47566
- 48138
- 85099B

The panel aggregates transaction records by product and calendar month.

It includes:

- Transaction row count.
- Median unit price.
- Mean unit price.
- Number of distinct unit prices.
- Total quantity sold.
- Gross line value.
- Number of distinct descriptions.
- Previous month's median price.
- Monthly price change.
- Percentage price change.
- Transaction and description indicators.

**Output:**

`data/processed/candidate_product_month_panel.csv`

The resulting panel contains 100 observations, representing four products across 25 consecutive months from December 2009 to December 2011.

### 3.4 Stage 3 â€” Candidate Panel Validation

**Module:** `src/econocausal/data/validate_candidate_panel.py`

This module checks the structural and numerical validity of the candidate product-month panel.

The validation process checks:

1. Duplicate product-month keys.
2. Missing calendar months.
3. Missing core numeric values.
4. Nonpositive core numeric values.

The latest recorded validation results show:

- 100 product-month observations.
- Zero duplicate product-month keys.
- No missing calendar months.
- No missing or nonpositive values in validated core numeric fields.

The first month for each product naturally has no previous-month price comparison.

## 4. Pipeline Orchestration

**Main module:** `src/econocausal/data/run_pipeline.py`

The `run_pipeline.py` module coordinates the three preparation stages.

It imports the main functions from:

- `build_dataset.py`
- `build_candidate_panel.py`
- `validate_candidate_panel.py`

The stages execute sequentially through the pipeline's `main()` function.

To execute the complete workflow from the repository root with the project environment activated:

```powershell
python -m econocausal.data.run_pipeline
```

The pipeline prints stage headings and reports completion after all three stages finish successfully.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Pipeline implementation |
| Pandas | Data preparation and aggregation |
| OpenPyXL | Excel workbook reading |
| Pytest | Automated testing |
| Matplotlib | Exploratory data visualization |
| Git | Version control |
| GitHub | Repository hosting |

## 6. Supporting Analysis Modules

In addition to the core pipeline, the data engineering package contains supporting modules for:

- Initial dataset inspection.
- Data classification and preprocessing.
- Exploratory data analysis.
- Product price variation analysis.
- Monthly price analysis.
- Candidate quality assessment.
- Price-change calendar analysis.
- Price-change robustness analysis.
- Duplicate sensitivity analysis.
- Missing Customer ID sensitivity analysis.
- EDA visualization.

These modules support the investigation and documentation of historical retail data.

They are not all automatically executed by `run_pipeline.py`.

## 7. Testing and Quality Assurance

The project includes automated tests under the `tests/` directory.

The data-related tests cover dataset preparation and candidate panel validation.

The last verified test execution reported 19 passing tests.

The pipeline and validation checks support reproducibility and help identify data-quality problems before downstream analysis.

## 8. Scope and Limitations

The implemented architecture supports historical transaction preparation, descriptive analysis, product-month aggregation, and data validation.

It does not currently implement:

- A complete causal inference engine.
- Double Machine Learning estimation.
- Propensity score matching.
- Discount optimization.
- React/Plotly user interface.
- API integration.
- Drift detection.

The historical retail dataset also lacks explicit discount assignment and complete offer-exposed customer outcomes. Therefore, the prepared data should not independently be treated as sufficient for customer-level causal effect estimation.

## 9. Conclusion

The implemented data architecture provides a reproducible workflow from the original retail workbook to a prepared transaction dataset and validated candidate product-month panel.

It establishes the data preparation and analytical foundation for subsequent EconoCausal project modules.
