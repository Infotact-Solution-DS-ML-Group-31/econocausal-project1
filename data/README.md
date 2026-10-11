
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
