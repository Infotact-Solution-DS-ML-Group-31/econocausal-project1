## Exploratory Data Analysis

   The Online Retail II workbook was combined into a dataset containing 1,067,371 source rows. The positive-sales filtering step produced 1,040,749 rows.

   Initial diagnostics identified 12,133 exact duplicate source rows, 243,007 rows with missing customer IDs, 4,382 rows with missing descriptions, and records with nonpositive quantities or prices. These issues were investigated before defining the positive-sales analysis subset.

   Visual EDA generated three charts: monthly gross line value, unit-price distribution, and transaction-line quantity distribution. The distribution charts display values up to the 99th percentile for readability, while the underlying dataset retains its observations.

   The price and quantity 99th percentiles were 18.00 and 100 units, respectively. The monthly series contains 25 periods.

   **Limitation:** These are descriptive results. They do not establish a causal effect of price on demand. A causal analysis requires a defensible treatment definition, appropriate confounders, and a justified identification strategy.