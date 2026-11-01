# EconoCausal — Exploratory Data Analysis Report

## 1. Project overview

This report documents the data engineering and exploratory data analysis work completed for the EconoCausal Dynamic Pricing project.

The work focuses on preparing the Online Retail II dataset, investigating data quality, analysing product-level price patterns, constructing a candidate product-month panel, and evaluating the sensitivity of descriptive results to duplicate records and missing customer identifiers.

The analysis is descriptive. It does not estimate causal price effects.

## 2. Dataset overview

The project uses the UCI Online Retail II workbook, which contains two sheets covering retail transactions.

The two sheets were combined into a transaction-level dataset containing:

- Source transaction rows: 1,067,371
- Prepared transaction rows: 1,040,749
- Analysis period: December 2009 to December 2011

The original raw workbook was preserved.

## 3. Data quality assessment

Initial inspection identified the following issues in the source dataset:

| Data-quality issue | Observations |
|---|---:|
| Exact duplicate rows | 12,133 |
| Missing Customer IDs | 243,007 |
| Missing descriptions | 4,382 |
| Nonpositive quantities | 22,950 |
| Nonpositive prices | 6,207 |
| Cancellation invoice rows | 19,494 |

These counts describe the initial source-data inspection. They should not be added together to calculate excluded rows because different conditions can overlap.

## 4. Data preparation

The preparation process excluded:

- Cancellation invoices.
- Transactions with nonpositive quantities.
- Transactions with nonpositive unit prices.
- Special stock codes that do not represent ordinary products.

The resulting prepared dataset contains 1,040,749 rows.

Exact duplicate rows and transactions with missing Customer IDs were retained. Their effects were investigated separately through sensitivity analyses.

## 5. Exploratory data analysis

Three visualizations were generated:

1. Monthly gross line value.
2. Unit-price distribution.
3. Transaction-line quantity distribution.

The distribution charts display values up to the 99th percentile for readability, without removing observations from the underlying dataset.

| Measure | 99th percentile |
|---|---:|
| Unit price | 18.00 |
| Transaction-line quantity | 100 |

The monthly gross line value series covers 25 calendar months.

Generated visualizations:

- `docs/figures/monthly_gross_line_value.png`
- `docs/figures/unit_price_distribution.png`
- `docs/figures/quantity_distribution.png`

## 6. Product candidate screening

The prepared dataset was further filtered for valid product codes, invoice dates, positive prices, and positive quantities.

The following special stock codes were excluded from candidate screening:

`DOT`, `POST`, `ADJUST`, `B`, `M`, `BANK CHARGES`, and `AMAZONFEE`.

Transactions were aggregated by product code and calendar month.

The screening required:

- At least 100 transaction rows.
- Activity in at least six distinct months.
- More than one distinct monthly median price.

Screening results:

| Measure | Result |
|---|---:|
| Transaction rows after analysis exclusions | 1,037,391 |
| Product codes analysed | 4,908 |
| Product-month records | 66,958 |
| Eligible candidate products | 1,570 |

## 7. Shortlisted products

Four products were shortlisted for focused descriptive analysis.

| Stock code | Active months | Transaction rows | Distinct monthly median prices |
|---|---:|---:|---:|
| 85099B | 25 | 4,124 | 2 |
| 47566 | 25 | 2,740 | 3 |
| 20685 | 25 | 1,769 | 4 |
| 48138 | 25 | 1,701 | 5 |

All four products have observations in each of the 25 calendar months from December 2009 through December 2011.

The shortlist was intended to provide products with different transaction volumes and levels of observed monthly median-price variation. The screening script identified 1,570 eligible products; it did not automatically select these four as the top four candidates.

## 8. Candidate product-month panel

A product-month panel was constructed for the four shortlisted products.

| Panel characteristic | Result |
|---|---:|
| Products | 4 |
| Calendar months per product | 25 |
| Total product-month records | 100 |
| Period | December 2009 – December 2011 |

The panel contains:

- Transaction-row count.
- Median unit price.
- Mean unit price.
- Number of distinct unit prices.
- Total quantity sold.
- Gross line value.
- Number of distinct product descriptions.
- Consecutive-month price changes and percentage changes.

## 9. Panel validation

The panel was validated for structural and numeric consistency.

| Validation check | Result |
|---|---:|
| Duplicate product-month keys | 0 |
| Missing calendar months | 0 |
| Missing values in checked core measures | 0 |
| Nonpositive values in checked core measures | 0 |

The checked core measures were transaction-row count, median unit price, total quantity sold, and gross line value.

The panel passed these validation checks. This establishes basic structural consistency, not causal validity.

## 10. Observed monthly median-price changes

The number of nonzero consecutive-month median-price changes was:

| Stock code | Number of changes |
|---|---:|
| 20685 | 3 |
| 47566 | 5 |
| 48138 | 4 |
| 85099B | 1 |

These are descriptive changes in observed monthly median prices. They do not establish that price changes caused changes in sales quantities.

## 11. Duplicate-row sensitivity analysis

The prepared dataset contained 12,051 exact duplicate rows, which were retained in the primary analysis.

A separate analysis examined the four shortlisted products:

| Measure | Result |
|---|---:|
| Candidate transaction rows | 10,334 |
| Exact duplicate rows | 39 |
| Rows after deduplication | 10,295 |
| Product-month median prices changed | 0 of 100 |
| Quantity difference | 214 units |
| Gross line-value difference | £571.51 |

No monthly median unit prices changed after removing exact duplicate rows.

The findings indicate that the candidate products' monthly median-price patterns are robust to this deduplication exercise, while quantity and gross line-value measures are affected.

The analysis does not establish whether the repeated records are data-entry errors or legitimate repeated transaction lines.

## 12. Missing Customer ID analysis

The prepared dataset contained:

- 235,941 rows with missing Customer IDs.
- 22.67% of all prepared transaction rows.

For the four shortlisted products, 1,793 of 10,334 rows had missing Customer IDs, representing 17.35%.

Product-level missing-ID rates:

| Stock code | Total rows | Missing Customer IDs | Missing percentage |
|---|---:|---:|---:|
| 20685 | 1,769 | 254 | 14.36% |
| 47566 | 2,740 | 619 | 22.59% |
| 48138 | 1,701 | 156 | 9.17% |
| 85099B | 4,124 | 764 | 18.53% |

The missing-ID rate varies across products. Product 47566 has the highest rate, while product 48138 has the lowest.

Transactions with missing Customer IDs were retained in the product-level analysis because the product code, transaction date, price, and quantity were available.

## 13. Missing Customer ID sensitivity analysis

A separate comparison evaluated monthly product-level summaries using:

1. All candidate transactions.
2. Only transactions with identified Customer IDs.

Three of the 100 product-month records had different monthly median prices after excluding transactions with missing Customer IDs.

| Stock code | Month | All-transactions median | Identified-customer median | Difference |
|---|---|---:|---:|---:|
| 47566 | 2010-08 | 8.47 | 4.65 | 3.82 |
| 47566 | 2010-12 | 8.47 | 4.65 | 3.82 |
| 48138 | 2009-12 | 10.31 | 6.75 | 3.56 |

The affected observations have limited identified-customer transaction support:

- Product 47566, August 2010: 3 identified-customer rows out of 34.
- Product 47566, December 2010: 35 identified-customer rows out of 72.
- Product 48138, December 2009: 1 identified-customer row out of 2.

The overall differences between all transactions and identified-customer transactions were:

| Measure | Difference |
|---|---:|
| Total quantity sold | 8,767 units |
| Gross line value | £66,959.05 |

The results show that excluding transactions with missing Customer IDs can change some monthly median-price observations and substantially change sales aggregates.

Consequently, missing-ID transactions remain included in the primary product-level analysis.

This comparison does not establish why Customer IDs are missing or whether the identified-customer subset is representative.

## 14. Limitations

The analysis has several important limitations:

- The dataset records completed transactions rather than all purchase opportunities or prices offered.
- Missing Customer IDs restrict reliable customer-level analysis and customer segmentation.
- Exact duplicate rows affect transaction counts, quantity totals, and gross line value.
- Product-description variation may affect product interpretation.
- Monthly aggregation hides transaction-level variation.
- Candidate screening does not guarantee suitability for causal estimation.
- The observed monthly median-price changes are not causal effects.
- Sensitivity analyses measure the effect of specific exclusions but do not resolve the underlying missing-data mechanism or possible selection bias.

## 15. Conclusion and next steps

The data engineering and exploratory analysis produced:

- A prepared transaction dataset containing 1,040,749 rows.
- Three descriptive visualizations.
- A monthly product-price screening dataset.
- A shortlist of four products.
- A validated 100-row product-month panel.
- Automated checks for panel structure and numeric consistency.
- Duplicate-row and missing-Customer-ID sensitivity analyses.
- Reproducible analysis scripts and project documentation.

The work provides a foundation for further descriptive analysis and future causal research.

Before causal estimation, further work is required to define the price treatment and sales outcome, identify and justify potential confounders, assess identification assumptions, and conduct appropriate robustness checks.

No causal price effect is claimed by this report.