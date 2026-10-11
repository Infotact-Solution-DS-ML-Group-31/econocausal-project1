
# Dataset Guide

## Purpose

This directory is reserved for datasets used by the EconoCausal dynamic-pricing project.

## Supported input

The current data loader supports CSV files readable by pandas.

## Data loading

Use `load_csv()` from `econocausal.data.load` to load a dataset. The loader checks that the file exists, contains data rows, and includes any specified required columns.

## Preprocessing

Use `preprocess_dataframe()` from `econocausal.data.preprocess` to validate required columns and optionally remove exact duplicate rows.

Missing values are preserved by default. Imputation, outlier handling, and variable transformations must be chosen based on the dataset and documented methodology.

## Dataset provenance

Record the dataset name, source URL or provider, retrieval date, license, and any applicable usage restrictions before using a real dataset.

## Privacy and version control

Do not commit confidential, personally identifiable, or otherwise restricted data. Follow the repository's data-handling rules before adding datasets.


## Exploratory Data Analysis

The source workbook contains two sheets:
- Year 2009-2010
- Year 2010-2011

The combined workbook contains 1,067,371 rows and eight original columns. The EDA process adds a `SourceSheet` column to preserve sheet provenance.

### Initial data-quality findings

- 12,133 exact duplicate rows in the combined source.
- 243,007 rows with missing customer IDs.
- 4,382 rows with missing descriptions.
- 22,950 rows with nonpositive quantities.
- 6,207 rows with nonpositive prices.
- 19,494 invoice rows whose invoice identifier begins with `C`.

These categories can overlap and must not be summed to calculate total excluded rows.

### Positive-sales analysis subset

The dataset builder retains rows with positive quantities and positive prices, excludes cancellation-marked invoices, and excludes the special stock codes B, M, BANK CHARGES, and AMAZONFEE.

Missing customer IDs are retained. Exact duplicates are not automatically removed.

The current output contains 1,040,749 rows, including 12,051 exact duplicate rows.

### Duplicate sensitivity

Removing exact duplicate rows for diagnostic purposes reduces the dataset to 1,028,698 rows. The quantity total changes from 11,410,210 to 11,375,575, and gross line value changes from 20,586,257.81 to 20,527,943.36.

These results demonstrate sensitivity to duplicate handling; they do not establish that every repeated row is erroneous.

### Causal inference limitation

This transaction dataset records completed transactions and does not provide a complete record of purchase opportunities or randomized price assignments. Descriptive price and quantity patterns alone cannot establish causal price effects.
