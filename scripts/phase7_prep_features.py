"""
Phase 7 — Feature Preparation
Extracts two new feature sets and saves to data/processed/:

  1. imerg_stations_monthly.csv
     Columns: Date (month-start), {slug}_imerg_mm, {slug}_imerg_3x3_mm
     for every station in all_rivers_station_meta.json.
     Covers 2015-01 to 2025-09 (IMERG V07B coverage).

  2. climate_indices_monthly.csv
     Columns: Date, oni, dmi, oni_lag3m, oni_lag6m, dmi_lag3m, dmi_lag6m
     Aggregated from the existing daily climate_indices_daily.csv.

Run (standard Python, no TF needed):
  python scripts/phase7_prep_features.py
"""

import os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from calendar import monthrange

BASE     = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data" / "processed"
IMERG_DIR = BASE / "data" / "imerg_monthly"

# ── load station metadata ──────────────────────────────────────────────────────
with open(PROC_DIR / "all_rivers_station_meta.json") as f:
    stations = json.load(f)
print(f"Stations loaded: {len(stations)}")

# ── 1. IMERG extraction ───────────────────────────────────────────────────────
print("\n=== Extracting IMERG per station ===")

# regex to pull YYYYMM from filename
_RE = re.compile(r"3B-MO\.MS\.MRG\.3IMERG\.(\d{4})(\d{2})\d{2}-")

hdf_files = sorted(IMERG_DIR.glob("*.HDF5"))
print(f"Found {len(hdf_files)} IMERG monthly files")

rows = []
for hdf in hdf_files:
    m = _RE.search(hdf.name)
    if not m:
        continue
    year, mon = int(m.group(1)), int(m.group(2))
    date = pd.Timestamp(year=year, month=mon, day=1)
    days_in_month = monthrange(year, mon)[1]
    hours = days_in_month * 24

    with h5py.File(hdf, "r") as h:
        lat_arr = h["Grid/lat"][:]       # shape (1800,) -89.95..89.95 step 0.1
        lon_arr = h["Grid/lon"][:]       # shape (3600,) -179.95..179.95 step 0.1
        prec    = h["Grid/precipitation"][0]  # shape (3600, 1800) mm/hr; fill=-9999.9

    row = {"Date": date}
    for sta in stations:
        slug = sta["slug"]
        slat, slon = sta["lat"], sta["lon"]

        # nearest pixel index
        lon_idx = int(np.argmin(np.abs(lon_arr - slon)))
        lat_idx = int(np.argmin(np.abs(lat_arr - slat)))

        # single pixel
        v = prec[lon_idx, lat_idx]
        mm_pt = float(v * hours) if v > -9000 else np.nan

        # 3×3 spatial mean (±1 pixel = ±0.1°)
        lo0 = max(0, lon_idx - 1); lo1 = min(3600, lon_idx + 2)
        la0 = max(0, lat_idx - 1); la1 = min(1800, lat_idx + 2)
        patch = prec[lo0:lo1, la0:la1]
        valid = patch[patch > -9000]
        mm_3x3 = float(valid.mean() * hours) if len(valid) > 0 else np.nan

        row[f"{slug}_imerg_mm"]      = round(mm_pt, 2)   if not np.isnan(mm_pt)   else np.nan
        row[f"{slug}_imerg_3x3_mm"]  = round(mm_3x3, 2)  if not np.isnan(mm_3x3)  else np.nan

    rows.append(row)
    if mon == 1:
        print(f"  {year} done")

imerg_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
out_imerg = PROC_DIR / "imerg_stations_monthly.csv"
imerg_df.to_csv(out_imerg, index=False)
print(f"Written: {out_imerg}  shape={imerg_df.shape}")
print(f"Date range: {imerg_df['Date'].min()} to {imerg_df['Date'].max()}")

# ── 2. Climate indices (monthly) ──────────────────────────────────────────────
print("\n=== Aggregating climate indices to monthly ===")

ci = pd.read_csv(PROC_DIR / "climate_indices_daily.csv", parse_dates=["Date"])
ci = ci.set_index("Date")

# keep only base columns (recompute lags from scratch)
base_cols = [c for c in ["oni", "dmi"] if c in ci.columns]
ci_mon = ci[base_cols].resample("MS").mean().reset_index()
ci_mon.rename(columns={"Date": "Date"}, inplace=True)

# compute lags on the monthly series
for col in base_cols:
    ci_mon[f"{col}_lag3m"] = ci_mon[col].shift(3)
    ci_mon[f"{col}_lag6m"] = ci_mon[col].shift(6)

out_ci = PROC_DIR / "climate_indices_monthly.csv"
ci_mon.to_csv(out_ci, index=False)
print(f"Written: {out_ci}  shape={ci_mon.shape}")
print(ci_mon.head(8).to_string())
print(f"Date range: {ci_mon['Date'].min()} to {ci_mon['Date'].max()}")
print("\nDone.")
