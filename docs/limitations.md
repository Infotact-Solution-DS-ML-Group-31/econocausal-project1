# Limitations

## 1. Transaction data does not capture all demand

The Online Retail II dataset records completed transactions. It does not record every product viewed, every purchase opportunity, or every price offered to customers.

Therefore, observed quantity sold should not be interpreted as a complete measure of customer demand.

## 2. Missing customer information

A substantial number of transaction rows have missing customer IDs. This limits customer-level analysis, customer segmentation, and the ability to construct reliable individual purchase histories for all transactions.

## 3. Duplicate transaction rows

The initial inspection identified 12,133 exact duplicate source rows. These rows were retained in the prepared dataset. Their presence may affect transaction counts and aggregated sales measures.

The impact of these duplicates should be considered when interpreting descriptive results.

## 4. Product-description variation

Some product codes have multiple descriptions. This may reflect changes in descriptions or inconsistencies in product records. Product-code-based aggregation does not automatically resolve these differences.

## 5. Price variation does not establish causality

Changes in monthly median unit prices are observational. They may be associated with changes in product mix, transaction composition, customer composition, or other factors.

The current analysis does not establish that price changes caused changes in quantities sold.

## 6. Monthly aggregation hides transaction-level variation

Aggregating transactions by product and month simplifies analysis but hides differences in transaction timing, individual prices, quantities, and customer behavior within each month.

Monthly median prices and total quantities therefore provide summaries rather than a complete representation of transaction-level behavior.

## 7. Candidate selection is a screening process

The candidate-screening criteria identify products with sufficient transaction volume, activity across multiple months, and variation in monthly median prices.

Meeting these criteria does not guarantee that a product is suitable for causal estimation. The four shortlisted products were chosen for focused descriptive analysis, not through a validated causal identification procedure.

## 8. Panel validation has a limited scope

The candidate panel passed checks for duplicate product-month keys, calendar continuity, and missing or nonpositive values in selected core numeric measures.

These checks establish basic structural and numeric consistency. They do not establish that the data is free from all errors, measurement problems, selection bias, or confounding.

## 9. Further work required for causal estimation

A causal analysis would require additional decisions and investigation, including:

- A defensible definition of price treatment and sales outcome.
- Identification and justification of relevant confounding variables.
- Assessment of whether the available data supports the required identification assumptions.
- Appropriate causal estimation and robustness checks.

The current data-engineering and exploratory-analysis work should therefore be treated as preparation for further research, not as a completed causal pricing model.

## 10. Duplicate-row sensitivity analysis

The prepared transaction dataset contains 12,051 exact duplicate rows, which were retained in the primary analysis.

A separate sensitivity analysis was conducted on the four shortlisted products. Of 10,334 candidate transaction rows, 39 were exact duplicates.

Comparing monthly summaries before and after removing exact duplicate rows showed:

- No change in monthly median unit prices across all 100 product-month records.
- A reduction of 214 units in total quantity sold.
- A reduction of £571.51 in gross line value.

The results indicate that the shortlisted products' monthly median-price patterns are robust to the removal of exact duplicate rows. However, quantity and gross line-value measures are affected.

The sensitivity analysis does not establish whether the duplicate records represent data-entry errors or legitimate repeated transaction lines. Therefore, the primary prepared dataset retains them, and the duplicate impact is documented separately.

## 11. Missing customer identifier analysis

The prepared transaction dataset contains 235,941 rows with missing Customer IDs, representing 22.67% of all 1,040,749 prepared rows.

Among the four shortlisted products, 1,793 of 10,334 transaction rows have missing Customer IDs, representing 17.35%.

The missing-ID rates vary across shortlisted products:

| Stock code | Total rows | Missing Customer IDs | Missing percentage |
|---|---:|---:|---:|
| 20685 | 1,769 | 254 | 14.36% |
| 47566 | 2,740 | 619 | 22.59% |
| 48138 | 1,701 | 156 | 9.17% |
| 85099B | 4,124 | 764 | 18.53% |

The missing customer identifier rate is not uniform across the four products. Product 47566 has the highest rate, while product 48138 has the lowest.

Transactions with missing Customer IDs were retained in the product-level analysis because product code, transaction date, price, and quantity are available for the relevant records.

However, missing customer identifiers restrict reliable customer-level analysis, customer segmentation, and customer-level causal modelling. The reasons for the missing identifiers have not been established, so the missingness mechanism should not be assumed to be random.

## 12. Missing Customer ID sensitivity analysis

A sensitivity analysis compared monthly product-level summaries using all candidate transactions against summaries using only transactions with identified Customer IDs.

Among the 100 product-month records, three had different monthly median unit prices after excluding transactions with missing Customer IDs.

The affected records were:

| Stock code | Month | All-transactions median | Identified-customer median | Difference |
|---|---|---:|---:|---:|
| 47566 | 2010-08 | 8.47 | 4.65 | 3.82 |
| 47566 | 2010-12 | 8.47 | 4.65 | 3.82 |
| 48138 | 2009-12 | 10.31 | 6.75 | 3.56 |

For product 47566, only 3 of 34 transaction rows in August 2010 and 35 of 72 rows in December 2010 had identified Customer IDs.

For product 48138 in December 2009, only two transaction rows were available, and just one had an identified Customer ID. This makes the monthly median particularly sensitive to individual observations.

The analysis also found differences of 8,767 units in total quantity sold and £66,959.05 in gross line value when comparing all transactions against identified-customer transactions only.

These results show that excluding transactions with missing Customer IDs can alter some monthly median-price observations and substantially change sales aggregates. Therefore, transactions with missing Customer IDs remain included in the primary product-level analysis.

The sensitivity analysis does not establish the mechanism behind missing customer identifiers or prove that identified-customer transactions are representative of all transactions.