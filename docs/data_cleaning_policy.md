# Vantara Customer Intelligence Platform
## Data Cleaning Policy

**Status:** Baseline policy — pending final modeling validation  
**Dataset:** Online Retail II  
**Source file:** `data/raw/online_retail_II.xlsx`

## 1. Purpose

This document records the initial data-quality findings and defines conservative rules for preparing transaction data for customer-level analysis.

The raw workbook must remain unchanged. Cleaning decisions must be reproducible and auditable.

## 2. Raw Data Preservation

- Never overwrite the original Excel workbook.
- Save intermediate outputs separately under `data/interim/`.
- Preserve all source transaction rows in the initial cleaning baseline.
- Do not remove rows without documenting the reason and measuring the impact.

## 3. Initial Data Quality Findings

| Check | Observed result |
|---|---:|
| Total source rows | 1,067,371 |
| Missing Customer IDs | 243,007 |
| Missing descriptions | 4,382 |
| Exact duplicate occurrences | 34,335 |
| Rows participating in duplicate groups | 67,242 |
| Negative quantity rows | 22,950 |
| Zero quantity rows | 0 |
| Negative price rows | 5 |
| Zero price rows | 6,202 |
| Rows with a C-prefixed invoice | 19,494 |

These counts describe the current dataset and are not automatic deletion instructions.

## 4. Duplicate Handling

- Keep all rows in the initial cleaning baseline.
- Flag exact duplicate occurrences and audit duplicate groups.
- Do not assume identical transaction fields prove that a row is erroneous.
- Test the effect of deduplication separately before selecting a modeling policy.

## 5. Missing Customer IDs

- Preserve transactions with missing Customer IDs in the baseline.
- Do not invent or impute customer identities.
- Exclude unidentified transactions from customer-level modeling where a reliable customer identity is required.
- Report the excluded volume and its potential impact.

## 6. Returns and Cancellations

- Preserve negative-quantity rows and C-prefixed invoice signals.
- Treat these as review signals rather than definitive labels on their own.
- Analyze return and cancellation behavior separately from positive purchases.
- Define revenue and CLV treatment explicitly before creating model features.

## 7. Price and Quantity Rules

- Flag negative and zero prices separately.
- Flag negative quantities separately from zero quantities.
- Do not silently convert invalid values to positive values or delete them without review.
- Define positive-sale eligibility explicitly for purchase-based analyses.

## 8. Preliminary Positive-Sale Eligibility

The current exploratory eligibility rule requires:

- A non-missing Customer ID
- Quantity greater than zero
- Price greater than zero
- An invoice number that does not begin with `C`

The audit identified 805,549 eligible rows, 5,878 eligible customers, and 36,969 eligible invoices.

This is a preliminary analysis filter, not the final churn or CLV training dataset.

## 9. Validation and Audit Trail

- Validate the saved baseline after every cleaning change.
- Record row counts before and after each transformation.
- Keep quality flags and review outputs available for investigation.
- Do not include raw datasets or generated CSV outputs in Git commits.

## 10. Modeling Readiness

Before training, define customer observation dates, prediction horizons, churn labels, and CLV targets. Ensure features use only information available at the relevant prediction cutoff.

Revisit duplicate, return, cancellation, and missing-customer rules after exploratory analysis.

## 11. Current Limitations

- Duplicate rows have not been removed.
- Missing Customer IDs have not been imputed.
- Returns and cancellations have not been definitively classified.
- The eligibility filter has not been validated as a final modeling population.
- No churn or CLV performance claims are made at this stage.