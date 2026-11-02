# EconoCausal Data Dictionary

## 1. Purpose

This document describes the structure, data types, meanings, and analytical roles of the two processed datasets created as part of the Data Engineering and Exploratory Data Analysis (EDA) work.

The datasets are:

1. `data/processed/retail_sales.csv` — transaction-level retail data.
2. `data/processed/candidate_product_month_panel.csv` — monthly aggregated data for four shortlisted products.

The source is the UCI Online Retail II dataset.

## 2. Transaction-Level Dataset

**File:** `data/processed/retail_sales.csv`

**Observed structure:** 1,040,749 rows and 9 columns.

| Column | Data type | Description | Analytical role |
|---|---|---|---|
| Invoice | int64 | Invoice identifier | Transaction reference |
| StockCode | string | Product identifier | Product grouping |
| Description | string | Product description | Product metadata |
| Quantity | int64 | Number of units in the transaction line | Observed sales outcome |
| InvoiceDate | string | Transaction date and time | Time variable |
| Price | float64 | Recorded unit price | Observed price variable |
| Customer ID | float64 | Customer identifier, when available | Customer grouping |
| Country | string | Customer's country | Geographic attribute |
| SourceSheet | string | Original workbook sheet | Data provenance |

### Important considerations

- Customer IDs are missing for a portion of the transactions.
- The dataset contains completed transaction records, not records of every customer who was exposed to an offer.
- `Price` represents the observed transaction unit price. It does not identify an assigned discount or the customer's counterfactual price.
- `Quantity` records units purchased in observed transactions. It does not capture customers who were offered a product but did not purchase it.
- `Invoice` identifies an invoice, not necessarily a unique customer or independent experimental unit.

## 3. Candidate Product-Month Panel

**File:** `data/processed/candidate_product_month_panel.csv`

**Observed structure:** 100 rows and 14 columns.

The panel contains four shortlisted products observed across 25 consecutive months, from December 2009 to December 2011.

| Column | Data type | Description | Analytical role |
|---|---|---|---|
| StockCode | string | Product identifier | Panel entity |
| Month | string | Calendar month | Time index |
| transaction_rows | int64 | Number of transaction rows in the product-month | Activity measure |
| median_unit_price | float64 | Median observed unit price | Monthly price metric |
| mean_unit_price | float64 | Mean observed unit price | Monthly price metric |
| distinct_unit_prices | int64 | Number of distinct observed unit prices | Price variation metric |
| total_quantity_sold | int64 | Total units sold | Monthly sales metric |
| gross_line_value | float64 | Aggregated gross transaction-line value | Monthly sales-value metric |
| distinct_descriptions | int64 | Number of distinct product descriptions | Product consistency indicator |
| previous_month_price | float64 | Previous month's median unit price | Historical price feature |
| price_change | float64 | Difference from previous month's median price | Price movement metric |
| price_change_pct | float64 | Percentage change from previous month's median price | Relative price movement |
| at_least_10_rows | bool | Whether the month has at least 10 transaction rows | Activity indicator |
| multiple_descriptions | bool | Whether multiple descriptions occur in the product-month | Consistency indicator |

### Shortlisted products

- `20685`
- `47566`
- `48138`
- `85099B`

### Missing values

The panel contains four missing values in each of these columns:

- `previous_month_price`
- `price_change`
- `price_change_pct`

These are expected because the first month for each product has no preceding month within the panel.

The missing values should not automatically be imputed as zero. They represent unavailable historical comparisons.

## 4. Potential Causal Analysis Variables

The following table assesses whether the available variables can directly support the proposed Double Machine Learning (DML) analysis.

| Causal concept | Available variable | Assessment |
|---|---|---|
| Treatment (T) | None directly observed | No explicit discount assignment variable |
| Outcome (Y) | Quantity | Observed only for completed transactions |
| Covariates (X) | Country, product, time, historical price | Potential covariates, but insufficient to establish confounding control |
| Customer identifier | Customer ID | Available for identified transactions, but incomplete |
| Treatment assignment mechanism | Not available | Cannot determine how discounts or prices were assigned |

### Why the existing variables are insufficient

**Treatment limitation:** The recorded unit price is not equivalent to a discount treatment. There is no explicit record of the discount offered, discount percentage, or treatment assignment.

**Outcome limitation:** Quantity captures purchases that occurred. The dataset does not include a complete population of customers who received an offer but made no purchase.

**Confounding limitation:** Customer characteristics, purchase history before treatment, campaign exposure, and other possible drivers of pricing and purchasing decisions are not comprehensively represented.

**Customer identification limitation:** Missing Customer IDs prevent complete customer-level tracking.

## 5. Causal Readiness Assessment

The transaction-level dataset is suitable for descriptive analysis of historical product prices, quantities, transaction values, and product-level activity.

The candidate product-month panel is suitable for examining historical price variation and monthly sales patterns.

However, neither dataset independently provides the treatment assignment and customer-level response information required for a defensible customer-level DML estimate.

A separate campaign or experimental dataset would be needed, containing:

- Customer identifier.
- Discount offered or assigned.
- Discount amount or percentage.
- Treatment assignment date.
- Purchase or conversion outcome.
- Customers who received offers but did not purchase.
- Pre-treatment customer and behavioural covariates.
- Campaign and product context.

## 6. Recommended Data Handoff

The Data Engineering and EDA work provides the following outputs for subsequent project modules:

1. Cleaned transaction-level dataset.
2. Candidate product-month panel.
3. Data validation results.
4. Historical price variation analysis.
5. Data-quality and sensitivity analysis.
6. Documentation of missing information and causal limitations.

Downstream causal modelling should not interpret historical observed prices as randomized discount treatments without additional supporting evidence.

## 7. Conclusion

The current datasets provide a validated foundation for descriptive retail price and sales analysis. They identify historical price variation and provide a shortlist of products for further investigation.

They do not, by themselves, establish the causal effect of discounts on customer purchasing behaviour. Additional treatment, outcome, and pre-treatment customer information is required before a defensible Double Machine Learning analysis can be performed.