## Exploratory Data Analysis

   The Online Retail II workbook was combined into a dataset containing 1,067,371 source rows. The positive-sales filtering step produced 1,040,749 rows.

   Initial diagnostics identified 12,133 exact duplicate source rows, 243,007 rows with missing customer IDs, 4,382 rows with missing descriptions, and records with nonpositive quantities or prices. These issues were investigated before defining the positive-sales analysis subset.

   Visual EDA generated three charts: monthly gross line value, unit-price distribution, and transaction-line quantity distribution. The distribution charts display values up to the 99th percentile for readability, while the underlying dataset retains its observations.

   The price and quantity 99th percentiles were 18.00 and 100 units, respectively. The monthly series contains 25 periods.

   **Limitation:** These are descriptive results. They do not establish a causal effect of price on demand. A causal analysis requires a defensible treatment definition, appropriate confounders, and a justified identification strategy.

   ## Candidate Product-Month Panel Validation

### Dataset construction

A candidate product-month panel was constructed from the prepared Online Retail II transaction dataset for four products: `20685`, `47566`, `48138`, and `85099B`.

The panel contains **100 rows**, covering 25 consecutive months from December 2009 through December 2011 for each product.

### Validation results

- Duplicate product-month keys: 0
- Missing calendar months: 0 for all four products
- Missing values in the checked core measures: 0
- Nonpositive values in the checked core measures: 0

Core measures checked were transaction count, median unit price, total quantity sold, and gross line value.

### Consecutive-month median-price changes

| Stock code | Number of changes |
|---|---:|
| 20685 | 3 |
| 47566 | 5 |
| 48138 | 4 |
| 85099B | 1 |

A price change is counted only when a product's monthly median price differs from the previous consecutive calendar month's median.

### Interpretation and limitations

The panel passes basic structural and numeric validation. This does not establish that the observed price changes are exogenous or that the panel identifies a causal price effect.

Online Retail II records completed transactions rather than all purchase opportunities and prices offered to customers. Consequently, total quantity sold is an observed sales outcome, not a direct measure of all customer demand. Product-description variation and within-month price dispersion also warrant further investigation.

The panel is suitable for candidate screening and descriptive analysis. Causal estimation requires a defensible identification strategy, a justified treatment and outcome definition, and appropriate confounder selection.