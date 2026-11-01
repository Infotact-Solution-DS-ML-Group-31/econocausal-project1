# Methodology

## 1. Dataset preparation

The analysis uses the UCI Online Retail II workbook. The two workbook sheets were combined into a transaction-level dataset containing 1,067,371 source rows.

The dataset preparation process excluded:
- Cancellation invoices.
- Transactions with nonpositive quantities.
- Transactions with nonpositive unit prices.
- Special stock codes that do not represent ordinary products.

The resulting prepared dataset contains 1,040,749 rows. The original raw workbook was preserved.

## 2. Exploratory data analysis

Initial data-quality checks examined:
- Exact duplicate transaction rows.
- Missing customer IDs and product descriptions.
- Nonpositive quantities and prices.
- Cancellation invoices.
- Transaction dates, quantities, and unit prices.

Three visualizations were generated:
- Monthly gross line value.
- Unit-price distribution.
- Transaction-line quantity distribution.

The distribution charts display observations up to the 99th percentile for readability. The underlying dataset retains its observations.

The 99th percentiles were 18.00 for unit price and 100 units for transaction-line quantity.

## 3. Monthly product-price analysis

For candidate screening, the prepared dataset was further checked for valid product codes, invoice dates, positive prices, and positive quantities. The following special stock codes were excluded:

`DOT`, `POST`, `ADJUST`, `B`, `M`, `BANK CHARGES`, and `AMAZONFEE`.

Transactions were grouped by product code and calendar month. The analysis calculated:
- Transaction-row count.
- Monthly median unit price.
- Number of distinct transaction prices.
- Total quantity sold.

A product was considered eligible for further investigation if it met all three conditions:

1. At least 100 transaction rows across the analysis period.
2. Activity in at least six distinct calendar months.
3. More than one distinct monthly median price.

This screening identified 1,570 eligible products from 4,908 product codes, producing 66,958 product-month records.

## 4. Candidate product shortlisting

Four products were shortlisted for more detailed analysis:

| Stock code | Active months | Transaction rows | Distinct monthly median prices |
|---|---:|---:|---:|
| 85099B | 25 | 4,124 | 2 |
| 47566 | 25 | 2,740 | 3 |
| 20685 | 25 | 1,769 | 4 |
| 48138 | 25 | 1,701 | 5 |

All four candidates have observations in each of the 25 calendar months from December 2009 through December 2011.

The shortlist provides a comparison of products with different transaction volumes and different levels of observed monthly median-price variation. The four products were shortlisted for focused descriptive analysis; they were not automatically selected as the top four products by the screening script.

## 5. Candidate product-month panel

A product-month panel was constructed for the four shortlisted products. It contains 100 records: 25 months for each of four products.

The panel includes transaction counts, median and mean unit prices, distinct transaction prices, total quantity sold, gross line value, distinct descriptions, and consecutive-month price-change measures.

## 6. Validation

The panel was checked for:
- Duplicate product-month keys.
- Missing calendar months.
- Missing core numeric values.
- Nonpositive core numeric values.

The validation found no duplicate keys, missing months, or invalid values in the checked core measures.

## 7. Analytical scope

This work is descriptive and focuses on data engineering, exploratory analysis, candidate screening, panel construction, and data validation. Observed price changes are not interpreted as causal effects.