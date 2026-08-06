"""
Phase 7 — Enhanced Ridge Regression (all rivers)
Adds IMERG satellite precipitation + climate teleconnection indices + lag-2
on top of the Phase 6 ERA5 feature set.

Features (24 total per station):
  ERA5 (15): log_tp, t2m, d2m, sp, u10, v10, tcwv, rh, ws,
             sin_m, cos_m, sin_2m, cos_2m, lag1, log_lag1
  IMERG (2): log_imerg, log_imerg_3x3
  Climate (5): oni, dmi, oni_lag3m, oni_lag6m, dmi_lag3m
  Lag-2 (2):  lag2, log_lag2

Test period: 2024-05 to 2025-09 (17 months — full IMERG V07B coverage).

Run (standard Python, no TF needed):
  python scripts/phase7_ridge_enhanced.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent
PROC_DIR = BASE / "data" / "processed"
PRED_DIR = BASE / "predictions"
PRED_DIR.mkdir(exist_ok=True)

VAL_END  = pd.Period("2024-04", freq="M")
TEST_END = pd.Period("2025-09", freq="M")   # IMERG coverage limit
MIN_DAYS = 15

RIVERS = {
    "johor":   {"imputed": "johor_daily_imputed.csv",   "era5": "era5_all_rivers_mapped.csv"},
    "kedah":   {"imputed": "kedah_daily_imputed.csv",   "era5": "era5_all_rivers_mapped.csv"},
    "klang":   {"imputed": "klang_daily_imputed.csv",   "era5": "era5_all_rivers_mapped.csv"},
    "kuantan": {"imputed": "kuantan_daily_imputed.csv", "era5": "era5_kuantan_station_mapped.csv"},
}

ERA5_VARS = ["era5_t2m","era5_d2m","era5_sp","era5_u10","era5_v10","era5_tcwv","era5_rh","era5_ws"]
ERA5_FEATS = ["log_tp"] + ERA5_VARS + ["sin_m","cos_m","sin_2m","cos_2m","lag1","log_lag1"]
IMERG_FEATS = ["log_imerg","log_imerg_3x3"]
CLIM_FEATS  = ["oni","dmi","oni_lag3m","oni_lag6m","dmi_lag3m"]
LAG2_FEATS  = ["lag2","log_lag2"]
ALL_FEATS   = ERA5_FEATS + IMERG_FEATS + CLIM_FEATS + LAG2_FEATS  # 24 total

# ── load shared monthly feature tables ────────────────────────────────────────
print("Loading shared feature tables...")

imerg_df = pd.read_csv(PROC_DIR / "imerg_stations_monthly.csv", parse_dates=["Date"])
imerg_df["ym"] = imerg_df["Date"].dt.to_period("M")

ci_df = pd.read_csv(PROC_DIR / "climate_indices_monthly.csv", parse_dates=["Date"])
ci_df["ym"] = ci_df["Date"].dt.to_period("M")
# ensure we have all expected climate cols; fill missing with 0
for c in CLIM_FEATS:
    if c not in ci_df.columns:
        ci_df[c] = 0.0
ci_df[CLIM_FEATS] = ci_df[CLIM_FEATS].fillna(0)

print(f"IMERG: {imerg_df.shape}, CI: {ci_df.shape}")


def build_monthly(river):
    rain = pd.read_csv(PROC_DIR / RIVERS[river]["imputed"], parse_dates=["Date"])
    rain = rain[(rain["Date"] >= "2015-01-01") & (rain["Date"] <= "2025-12-31")].copy()
    rain["ym"] = rain["Date"].dt.to_period("M")

    era5 = pd.read_csv(PROC_DIR / RIVERS[river]["era5"], parse_dates=["Date"])
    era5 = era5.ffill().bfill()
    era5["ym"] = era5["Date"].dt.to_period("M")

    era5_var_map = {"t2m":"era5_t2m","d2m":"era5_d2m","sp":"era5_sp",
                    "u10":"era5_u10","v10":"era5_v10","tcwv":"era5_tcwv",
                    "rh":"era5_rh","ws":"era5_ws","tp":"_tp_raw"}

    station_cols = [c for c in rain.columns
                    if c not in ("Date","ym") and not c.endswith("_imputed")]

    result = {}
    for sta in station_cols:
        flag_col = f"{sta}_imputed"
        if flag_col not in rain.columns:
            continue

        # monthly rainfall (real observations only)
        mrows = []
        for ym, grp in rain.groupby("ym"):
            real = grp[grp[flag_col] == 0][sta].dropna()
            total = real.sum() if len(real) >= MIN_DAYS else np.nan
            mrows.append({"ym": ym, "rainfall_mm": total})
        mdf = pd.DataFrame(mrows)

        # ERA5 monthly aggregate
        sta_cols = {v: f"{sta}_{v}" for v in era5_var_map if f"{sta}_{v}" in era5.columns}
        if "tp" not in sta_cols:
            continue

        era5_agg = era5.groupby("ym").agg(
            **{"_tp_monthly": (sta_cols["tp"], "sum")},
            **{era5_var_map[v]: (col, "mean") for v, col in sta_cols.items() if v != "tp"}
        ).reset_index()
        era5_agg["log_tp"] = np.log1p(np.clip(era5_agg["_tp_monthly"] * 1000, 0, None))
        era5_agg = era5_agg.drop(columns=["_tp_monthly"])

        mdf = mdf.merge(era5_agg, on="ym", how="left").sort_values("ym").reset_index(drop=True)

        # seasonal harmonics
        month = mdf["ym"].apply(lambda p: p.month).astype(float)
        mdf["sin_m"]  = np.sin(2 * np.pi * month / 12)
        mdf["cos_m"]  = np.cos(2 * np.pi * month / 12)
        mdf["sin_2m"] = np.sin(4 * np.pi * month / 12)
        mdf["cos_2m"] = np.cos(4 * np.pi * month / 12)

        # lag-1 and lag-2
        mdf["lag1"]     = mdf["rainfall_mm"].shift(1).fillna(0)
        mdf["log_lag1"] = np.log1p(np.clip(mdf["lag1"], 0, None))
        mdf["lag2"]     = mdf["rainfall_mm"].shift(2).fillna(0)
        mdf["log_lag2"] = np.log1p(np.clip(mdf["lag2"], 0, None))

        # IMERG features for this station
        slug = sta if sta.startswith(river + "_") else f"{river}_{sta}"
        imm_col   = f"{slug}_imerg_mm"
        imm3_col  = f"{slug}_imerg_3x3_mm"
        if imm_col in imerg_df.columns:
            sta_imerg = imerg_df[["ym", imm_col, imm3_col]].copy()
            mdf = mdf.merge(sta_imerg, on="ym", how="left")
            mdf["log_imerg"]     = np.log1p(np.clip(mdf[imm_col].fillna(0), 0, None))
            mdf["log_imerg_3x3"] = np.log1p(np.clip(mdf[imm3_col].fillna(0), 0, None))
        else:
            mdf["log_imerg"]     = 0.0
            mdf["log_imerg_3x3"] = 0.0

        # climate indices
        mdf = mdf.merge(ci_df[["ym"] + CLIM_FEATS], on="ym", how="left")

        # fill any remaining NaN in features with 0
        for c in ALL_FEATS:
            if c not in mdf.columns:
                mdf[c] = 0.0
        mdf[ALL_FEATS] = mdf[ALL_FEATS].fillna(0)

        result[sta] = mdf

    return result


def train_river(river):
    mdfs = build_monthly(river)
    print(f"\n--- {river.upper()}: {len(mdfs)} stations ---")

    station_results = []

    for sta, mdf in sorted(mdfs.items()):
        slug = sta if sta.startswith(river + "_") else f"{river}_{sta}"

        train_mask = mdf["ym"] <= VAL_END
        test_mask  = (mdf["ym"] > VAL_END) & (mdf["ym"] <= TEST_END)

        train_df = mdf[train_mask].dropna(subset=["rainfall_mm"])
        test_df  = mdf[test_mask].dropna(subset=["rainfall_mm"])

        if len(train_df) < 12 or len(test_df) < 3:
            continue

        Xtr = train_df[ALL_FEATS].values
        ytr = np.log1p(train_df["rainfall_mm"].values)
        Xte = test_df[ALL_FEATS].values
        yte_mm = test_df["rainfall_mm"].values

        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc = sc.transform(Xte)

        alphas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(Xtr_sc, ytr)

        pred_log = ridge.predict(Xte_sc)
        pred_mm  = np.expm1(pred_log)
        pred_mm  = np.clip(pred_mm, 0, None)

        r2   = r2_score(yte_mm, pred_mm)
        r, _ = pearsonr(yte_mm, pred_mm)
        rmse = float(np.sqrt(np.mean((yte_mm - pred_mm) ** 2)))

        station_results.append({
            "river": river,
            "station": slug,
            "r2": round(r2, 4),
            "r":  round(r, 4),
            "rmse": round(rmse, 1),
            "n_train": len(train_df),
            "n_test":  len(test_df),
            "best_alpha": ridge.alpha_,
            "predictions": [round(float(p), 1) for p in pred_mm],
        })

        flag = "✓" if r2 >= 0.80 else ("~" if r2 >= 0.50 else " ")
        print(f"  {flag} {slug:<55} R²={r2*100:6.1f}%  r={r:.3f}  RMSE={rmse:6.0f}  n_train={len(train_df)}  α={ridge.alpha_}")

    return station_results


# ── run all rivers ──────────────────────────────────────────────────────────
all_results = []
for river in ["johor","kedah","klang","kuantan"]:
    all_results.extend(train_river(river))

# ── summary ─────────────────────────────────────────────────────────────────
print("\n\n=== Phase 7 Summary ===\n")
df = pd.DataFrame([{k: v for k, v in r.items() if k != "predictions"} for r in all_results])
df = df.sort_values("r2", ascending=False)

for river in ["kuantan","klang","kedah","johor"]:
    sub = df[df["river"] == river]
    if sub.empty:
        continue
    n_pos = (sub["r2"] > 0).sum()
    n_50  = (sub["r2"] >= 0.5).sum()
    n_80  = (sub["r2"] >= 0.8).sum()
    best  = sub.iloc[0]
    print(f"{river.upper()}: best={best['station']} R²={best['r2']*100:.1f}%  "
          f"positive={n_pos}/{len(sub)}  ≥50%={n_50}  ≥80%={n_80}")

print("\nTop 10 stations:")
print(df.head(10)[["river","station","r2","r","rmse","n_train","n_test"]].to_string(index=False))

# ── save ──────────────────────────────────────────────────────────────────────
import json as _json

output = {
    "description": "Phase 7 enhanced Ridge — ERA5 + IMERG + climate indices + lag2",
    "features": ALL_FEATS,
    "n_features": len(ALL_FEATS),
    "train_period": "2015-01 to 2024-04",
    "test_period":  "2024-05 to 2025-09",
    "n_test_months": 17,
    "stations": all_results,
}

out_json = PRED_DIR / "phase7_ridge_enhanced_results.json"
with open(out_json, "w") as f:
    _json.dump(output, f, indent=2)
print(f"\nWritten: {out_json}")

# CSV: one row per station (summary, no per-month predictions)
df.to_csv(PRED_DIR / "phase7_ridge_enhanced_summary.csv", index=False)
print(f"Written: {PRED_DIR / 'phase7_ridge_enhanced_summary.csv'}")
