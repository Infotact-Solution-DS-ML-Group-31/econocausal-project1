# Results

## 1. Dataset preparation

The Online Retail II workbook contained 1,067,371 transaction rows across its two sheets.

After excluding cancellation invoices, nonpositive quantities, nonpositive prices, and special stock codes, the prepared transaction dataset contained 1,040,749 rows.

## 2. Initial data-quality findings

The initial dataset inspection identified:

- 12,133 exact duplicate source rows.
- 243,007 rows with missing customer IDs.
- 4,382 rows with missing product descriptions.
- 22,950 rows with nonpositive quantities.
- 6,207 rows with nonpositive prices.
- 19,494 cancellation invoice rows.

These findings informed the data-preparation and exploratory-analysis process.

## 3. Exploratory data analysis

Three visualizations were generated:

- Monthly gross line value.
- Unit-price distribution.
- Transaction-line quantity distribution.

The 99th percentile of unit price was 18.00, while the 99th percentile of transaction-line quantity was 100 units. Distribution charts were limited to these percentiles for readability; the underlying dataset was not truncated for this purpose.

The monthly gross line value visualization covers 25 calendar months.

## 4. Product candidate screening

The monthly product-price analysis produced:

| Measure | Result |
|---|---:|
| Transaction rows after analysis exclusions | 1,037,391 |
| Product codes analysed | 4,908 |
| Product-month records | 66,958 |
| Products meeting screening criteria | 1,570 |

The screening criteria required at least 100 transaction rows, at least six active months, and more than one distinct monthly median price.

## 5. Shortlisted products

Four products were selected for focused descriptive analysis:

| Stock code | Active months | Transaction rows | Distinct monthly median prices |
|---|---:|---:|---:|
| 85099B | 25 | 4,124 | 2 |
| 47566 | 25 | 2,740 | 3 |
| 20685 | 25 | 1,769 | 4 |
| 48138 | 25 | 1,701 | 5 |

Each shortlisted product has observations in all 25 calendar months from December 2009 through December 2011.

The products show different levels of monthly median-price variation, making them useful candidates for comparative descriptive analysis.

## 6. Candidate panel validation

The final candidate product-month panel contains 100 rows, representing four products across 25 months.

Validation results:

- Duplicate product-month keys: 0.
- Missing calendar months: 0 for all four products.
- Missing values in checked core measures: 0.
- Nonpositive values in checked core measures: 0.

The checked core measures were transaction-row count, median unit price, total quantity sold, and gross line value.

## 7. Consecutive-month median-price changes

The panel contains the following nonzero consecutive-month changes in monthly median unit price:

| Stock code | Number of changes |
|---|---:|
| 20685 | 3 |
| 47566 | 5 |
| 48138 | 4 |
| 85099B | 1 |

These counts describe observed changes in monthly median prices. They do not measure causal price effects.

## 8. Overall outcome

The data-engineering and exploratory-analysis work produced a cleaned transaction dataset, descriptive visualizations, a monthly product-price screening dataset, a four-product candidate panel, and automated structural and numeric validation.

The resulting panel provides a reproducible foundation for further descriptive analysis and future causal research, subject to the limitations documented separately.