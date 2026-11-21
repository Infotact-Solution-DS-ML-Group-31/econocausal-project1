# EconoCausal Data Handoff

## 1. Purpose

This document defines the data handoff from the Data Engineering and Exploratory Data Analysis (EDA) work to the downstream modelling and evaluation modules.

It describes the available datasets, their structure, loading instructions, validation status, limitations, and additional information required for causal modelling.

**Handoff status:** Ready for descriptive analysis and downstream data exploration. Not independently ready for customer-level causal effect estimation.

## 2. Available Datasets

### 2.1 Prepared Transaction-Level Dataset

**Path:** `data/processed/retail_sales.csv`

**Purpose:** Provides cleaned transaction-level retail records for descriptive price, quantity, product, customer, and geographic analysis.

**Observed structure:**
- 1,040,749 rows.
- 9 columns.
- Source: UCI Online Retail II.

**Row granularity:** One prepared transaction line.

**Columns:**

| Column | Description |
|---|---|
| Invoice | Invoice identifier |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Units purchased in the transaction line |
| InvoiceDate | Transaction date and time |
| Price | Recorded unit price |
| Customer ID | Customer identifier, when available |
| Country | Customer country |
| SourceSheet | Original workbook sheet |

### 2.2 Candidate Product-Month Panel

**Path:** `data/processed/candidate_product_month_panel.csv`

**Purpose:** Provides monthly aggregated price and sales metrics for four shortlisted products.

**Observed structure:**
- 100 rows.
- 14 columns.
- Four products.
- 25 consecutive months per product.
- Period: December 2009 to December 2011.

**Row granularity:** One product-month observation.

**Candidate products:**
- `20685`
- `47566`
- `48138`
- `85099B`

**Columns:**

| Column | Description |
|---|---|
| StockCode | Product identifier |
| Month | Calendar month |
| transaction_rows | Number of transaction rows |
| median_unit_price | Median observed unit price |
| mean_unit_price | Mean observed unit price |
| distinct_unit_prices | Number of distinct observed unit prices |
| total_quantity_sold | Total units sold |
| gross_line_value | Aggregated gross transaction-line value |
| distinct_descriptions | Number of distinct descriptions |
| previous_month_price | Previous month's median unit price |
| price_change | Change in median unit price |
| price_change_pct | Percentage change in median unit price |
| at_least_10_rows | Whether the month has at least 10 transaction rows |
| multiple_descriptions | Whether multiple descriptions occur |

## 3. How to Load the Datasets

Run commands from the repository root with the project virtual environment activated.

```python
import pandas as pd

transactions = pd.read_csv(
    "data/processed/retail_sales.csv"
)

candidate_panel = pd.read_csv(
    "data/processed/candidate_product_month_panel.csv",
    dtype={"StockCode": str}
)

print(transactions.shape)
print(candidate_panel.shape)
```

The candidate panel's `StockCode` should be loaded as a string to preserve alphanumeric identifiers such as `85099B`.

## 4. Pipeline and Validation

The main data preparation pipeline is:

```powershell
python -m econocausal.data.run_pipeline
```

It runs three stages:

1. Builds the prepared transaction-level dataset.
2. Builds the candidate product-month panel.
3. Validates the candidate panel.

The panel validator checks:
- Duplicate product-month keys.
- Missing calendar months.
- Missing or nonpositive core numeric values.

The latest recorded validation results are:
- 100 product-month observations.
- Zero duplicate product-month keys.
- No missing calendar months.
- No missing or nonpositive values in validated core numeric fields.

The first month for each product has no previous-month comparison. Therefore, the four missing values in each of `previous_month_price`, `price_change`, and `price_change_pct` are expected.

## 5. Data Preparation Decisions

The prepared transaction dataset excludes:
- Cancellation invoices.
- Nonpositive quantities.
- Nonpositive prices.
- Selected special stock codes.

Missing Customer IDs and exact duplicate transaction rows are retained in the primary prepared dataset. Their impact was investigated separately through sensitivity analyses.

The raw source workbook is not modified by the pipeline.

## 6. Important Analytical Limitations

### 6.1 Historical Price Is Not Discount Assignment

The `Price` column represents the observed transaction unit price. It does not establish:
- The discount offered to a customer.
- The discount amount or percentage.
- Whether a customer was assigned to a treatment or control group.
- The price a customer would have faced under an alternative treatment.

Historical price must not automatically be interpreted as a randomized discount treatment.

### 6.2 Purchase Quantity Is Not a Complete Conversion Outcome

`Quantity` records units purchased in observed transaction lines. The dataset does not contain a complete record of customers who were exposed to an offer but did not purchase.

It therefore cannot directly measure customer-level conversion across an offer-exposed population.

### 6.3 Customer Identification

Some transactions have missing Customer IDs. Customer-level analyses must account for incomplete identification and should not assume that every transaction can be linked to a known customer.

### 6.4 Product-Month Aggregation

The candidate panel is aggregated at product-month level. It does not preserve individual customer treatment and response observations.

It is suitable for descriptive product-level price and sales analysis, not individual treatment effect estimation.

## 7. Additional Inputs Required for DML

A defensible customer-level Double Machine Learning analysis requires an appropriate treatment, outcome, and pre-treatment covariate structure.

A separate campaign or experimental dataset should ideally provide:

| Required information | Purpose |
|---|---|
| Customer identifier | Link observations to customers |
| Discount offered or assigned | Define treatment |
| Discount amount or percentage | Measure treatment intensity |
| Treatment assignment date | Establish treatment timing |
| Purchase or conversion outcome | Define the response |
| Non-purchasing exposed customers | Represent outcomes where no purchase occurred |
| Pre-treatment customer characteristics | Support confounder adjustment |
| Pre-treatment purchase history | Capture prior customer behaviour |
| Campaign and product context | Define the relevant intervention and population |

The exact treatment and outcome definitions must be agreed upon by the causal-modelling team before model development.

## 8. Handoff Recommendations

Downstream project modules should:

1. Use the transaction dataset for descriptive retail analysis and appropriate feature exploration.
2. Use the candidate panel for historical product-month price analysis.
3. Preserve product identifiers as strings.
4. Account for missing Customer IDs.
5. Avoid treating missing historical price comparisons as zero.
6. Obtain suitable treatment and outcome data before estimating causal effects.
7. Document the assumptions behind any causal identification strategy.

## 9. Handoff Conclusion

The Data Engineering and EDA work provides reproducible transaction preparation, a validated candidate product-month panel, documented data-quality findings, and sensitivity analyses.

These outputs support historical retail analysis and downstream data exploration.

They do not independently establish the causal effect of discounts on customer purchasing behaviour. Additional campaign-level treatment, outcome, and pre-treatment customer information is required before customer-level DML estimation can be justified.