# Technical Report: Data Engineering Pipeline
## Phase 1 — Data Preprocessing, Alignment & Feature Engineering

**Project:** Deep Learning-Based Rainfall Data Imputation and Prediction System  
**Author:** Arinn Danish  
**Date:** June 2026  
**Notebook:** `01_e2e_data_engineering.ipynb`

---

## 1. Introduction

This report documents the end-to-end data engineering pipeline that transforms two raw multi-frequency rainfall telemetry exports into a unified, feature-rich dataset suitable for GAN-based imputation and TCN-based multi-scale prediction.

### 1.1 Data Sources

| Parameter | Dataset A (Sub-Hourly) | Dataset B (Daily) |
|---|---|---|
| Frequency | 15-minute intervals | Daily totals |
| Rows | 578,304 | 6,024 |
| Date Range | 2009-01-01 to 2025-06-29 | 2009-01-01 to 2025-06-29 |
| Stations | 3 | 3 |
| Format | BulkExport CSV (4-row metadata header) | BulkExport CSV (4-row metadata header) |

### 1.2 Telemetry Stations

| Station ID | Name | Code |
|---|---|---|
| 0570061RF | Ldg. Nada at Pahang | 3931013 |
| 0570051RF | Sg. Lembing Pahang Consolidated Company Limited Mill | 3930012 |
| 0570021RF | Ldg. Kuala Reman at Pahang | 3931014 |

All three stations are located in the Pahang region of Peninsular Malaysia, an area characterized by tropical monsoon climate with distinct wet (November-March) and dry (May-September) seasons.

---

## 2. Data Ingestion

Both CSV files share an identical structure: a 4-row metadata header (export tag, station IDs, station names, parameter names) followed by a column-name row with `Start of Interval`, `End of Interval`, and `Total (mm)` columns per station.

Pandas `read_csv` with `skiprows=4` and `parse_dates=[0, 1]` correctly parses both files. The generic column names (`Total (mm)`, `Total (mm).1`, `Total (mm).2`) are renamed to descriptive station identifiers: `nada_3931013`, `lembing_3930012`, `reman_3931014`.

---

## 3. Data Quality Assessment

### 3.1 Temporal Continuity

Both datasets are temporally complete with **zero gaps**:

- **15-min dataset:** All 578,304 expected 15-minute intervals present from 2009-01-01 00:00 to 2025-06-29 23:45. No duplicate timestamps.
- **Daily dataset:** All 6,024 expected days present. No duplicate dates.

### 3.2 Missing Values

Missing values are encoded as NaN (not as sentinel values or empty strings).

**Daily Dataset:**

| Station | Missing Count | Missing % |
|---|---|---|
| Nada (3931013) | 158 | 2.62% |
| Lembing (3930012) | 210 | 3.49% |
| Reman (3931014) | 127 | 2.11% |

**15-Minute Dataset:**

| Station | Missing Count | Missing % |
|---|---|---|
| Nada (3931013) | 15,203 | 2.63% |
| Lembing (3930012) | 20,232 | 3.50% |
| Reman (3931014) | 12,235 | 2.12% |

The proportional consistency between the two datasets (e.g., Lembing has the highest missing rate in both) confirms that missingness originates from sensor-level outages rather than data export artefacts.

### 3.3 Value Range Validation

| Station | Min (mm) | Max (mm) | Mean (mm) | Std (mm) |
|---|---|---|---|---|
| Nada (daily) | 0.0 | 333.5 | 8.12 | 19.69 |
| Lembing (daily) | 0.0 | 338.1 | 8.91 | 19.47 |
| Reman (daily) | 0.0 | 359.9 | 7.85 | 19.00 |

No negative values were found in either dataset, confirming physical plausibility. The high standard deviation relative to the mean (coefficient of variation ~2.4) reflects the characteristic heavy-tailed distribution of tropical rainfall.

### 3.4 Cleaning Actions

- **No rows dropped** — missing values are preserved and flagged with binary `_missing` columns for the GAN imputation phase.
- **No duplicates removed** — none found.
- **No outlier clipping** — extreme values (>300mm/day) are plausible for tropical monsoon events and are retained.

---

## 4. Multi-Frequency Alignment

### 4.1 Aggregation: 15-Minute to Daily

The 15-minute dataset was aggregated to daily resolution using three statistics per station:

| Metric | Aggregation | Rationale |
|---|---|---|
| Daily Total (mm) | `sum()` | Total precipitation volume |
| Daily Peak Intensity (mm/15min) | `max()` | Maximum instantaneous rainfall rate |
| Daily Std Dev (mm) | `std()` | Within-day temporal variability |

This produces 9 new columns (3 stations x 3 statistics), yielding a daily-resolution aggregated table of 6,024 rows x 10 columns.

### 4.2 Deterministic Left Join

The daily dataset (left) is joined with the aggregated 15-minute metrics (right) on the date key:

```
df_merged = df_daily.merge(df_15min_daily, on='date_start', how='left', validate='1:1')
```

The `validate='1:1'` parameter ensures no row explosion. The result preserves all 6,024 rows with 17 columns (original 8 + 9 aggregated metrics).

---

## 5. Temporal Feature Engineering

### 5.1 Cyclical Time Features

Day-of-year and month are encoded as sine/cosine pairs to capture seasonal periodicity without discontinuities:

```
doy_sin = sin(2pi * day_of_year / 365.25)
doy_cos = cos(2pi * day_of_year / 365.25)
month_sin = sin(2pi * month / 12)
month_cos = cos(2pi * month / 12)
```

This encoding ensures that December 31 and January 1 are adjacent in feature space, which is critical for the TCN to learn monsoon transition patterns.

### 5.2 Rolling Window Statistics

For each station, moving averages are computed at three temporal scales:

| Window | Feature Name | Purpose |
|---|---|---|
| 3-day | `{station}_total_rm3` | Short-term rainfall trend |
| 7-day | `{station}_total_rm7` | Weekly-scale pattern |
| 14-day | `{station}_total_rm14` | Bi-weekly accumulation |

Rolling means of 15-minute peak intensity are also computed at the same windows, providing multi-scale intensity context. `min_periods=1` ensures no NaN at sequence boundaries.

### 5.3 Historical Lags

Lagged values at 3, 7, and 14 days provide the model with explicit autoregressive features:

- `{station}_total_lag3`: Rainfall 3 days ago
- `{station}_total_lag7`: Rainfall 1 week ago
- `{station}_total_lag14`: Rainfall 2 weeks ago

These features introduce NaN in the first 14 rows, which propagate intentionally (acceptable since these are a negligible fraction of the dataset).

### 5.4 Consecutive Dry Days

A custom feature counts consecutive days with rainfall below 0.1mm:

```
is_dry = (rainfall < 0.1).astype(int)
groups = (is_dry != is_dry.shift()).cumsum()
consec_dry = is_dry.groupby(groups).cumsum()
```

Maximum consecutive dry spells observed: station-dependent, reflecting local microclimate differences. This feature helps models learn dry-spell dynamics and rainfall onset patterns.

---

## 6. Final Dataset Summary

### 6.1 Schema

The final engineered dataset contains **6,024 rows x 51 columns**:

| Category | Columns | Count |
|---|---|---|
| Date fields | date_start, date_end | 2 |
| Raw daily rainfall | 3 stations | 3 |
| Missing flags | 3 stations | 3 |
| 15-min aggregates | total, peak, std x 3 stations | 9 |
| Cyclical time | sin/cos x DoY + Month | 4 |
| Rolling means (rainfall) | 3 windows x 3 stations | 9 |
| Rolling means (peak) | 3 windows x 3 stations | 9 |
| Lag features | 3 lags x 3 stations | 9 |
| Consecutive dry days | 3 stations | 3 |
| **Total** | | **51** |

### 6.2 Data Integrity

- **Temporal gaps:** 0 (continuous daily coverage)
- **Core engineered features NaN-free:** Confirmed (cyclical, dry days, missing flags)
- **Intentional NaN retained:** Raw station columns (158-210 per station), lag features (first 14 rows), rolling features propagating from NaN source values
- **File size:** 2.9 MB (`engineered_rainfall_data.csv`)

---

## 7. Visualizations

The notebook includes three key visualizations:

1. **Missing Data Heatmap** — Monthly missing-value fraction for both daily and 15-minute datasets, revealing temporal clustering of sensor outages.
2. **Feature Correlation Matrix** — Lower-triangular heatmap of 22 selected features showing strong cross-station correlations (r > 0.7) and expected high correlations between rolling windows of different scales.
3. **Rainfall Time Series** — Monthly total rainfall bar charts with 12-month moving average overlay for each station, confirming the monsoon seasonal pattern.

---

## 8. Output Artifacts

| Artifact | Description |
|---|---|
| `engineered_rainfall_data.csv` | Production-ready feature-engineered dataset (6,024 x 51) |
| `01_e2e_data_engineering.ipynb` | Fully documented, executable Jupyter notebook |
| `missing_data_heatmap.png` | Missing value visualization |
| `feature_correlation_matrix.png` | Feature correlation heatmap |
| `rainfall_timeseries.png` | Monthly rainfall overview |
