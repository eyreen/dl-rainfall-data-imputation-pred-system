"""
Phase 8 — East Malaysia Expansion: Padas (Sabah) + Sarawak
RidgeCV monthly prediction using IMERG satellite + climate teleconnection indices.
No ERA5 (East Malaysian stations lie outside the Pahang bounding box used in
Phases 6/7). IMERG point + 3x3 spatial mean substitutes for ERA5 precipitation.

Feature set (15 total):
  IMERG (2):    log_imerg, log_imerg_3x3
  Seasonal (4): sin_m, cos_m, sin_2m, cos_2m
  Climate (5):  oni, dmi, oni_lag3m, oni_lag6m, dmi_lag3m
  Lags (4):     lag1, log_lag1, lag2, log_lag2

Model: RidgeCV, StandardScaler, log1p target transform.
  alpha grid: [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
  cv folds:   max(2, min(5, n_train // 6))

Train period: Jan 2015 – Apr 2024
Test  period: May 2024 – Sep 2025 (17 months, IMERG V07B coverage limit)

Run:
  python scripts/phase8_ridge_east_malaysia.py

Outputs:
  predictions/phase7_ridge_padas_sarawak_results.json  (full per-station results)
  predictions/phase7_ridge_padas_sarawak_summary.csv   (summary table)
"""

import json
import warnings
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
TEST_END = pd.Period("2025-09", freq="M")
MIN_DAYS = 15

RIVERS = {
    "padas":   "padas_daily_imputed.csv",
    "sarawak": "sarawak_daily_imputed.csv",
}

IMERG_FEATS = ["log_imerg", "log_imerg_3x3"]
SEASON_FEATS = ["sin_m", "cos_m", "sin_2m", "cos_2m"]
CLIM_FEATS   = ["oni", "dmi", "oni_lag3m", "oni_lag6m", "dmi_lag3m"]
LAG_FEATS    = ["lag1", "log_lag1", "lag2", "log_lag2"]
ALL_FEATS    = IMERG_FEATS + SEASON_FEATS + CLIM_FEATS + LAG_FEATS  # 15

ALPHAS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

# ── load shared tables ─────────────────────────────────────────────────────────
print("Loading IMERG and climate index tables...")
imerg_df = pd.read_csv(PROC_DIR / "imerg_stations_monthly.csv", parse_dates=["Date"])
imerg_df["ym"] = imerg_df["Date"].dt.to_period("M")

ci_df = pd.read_csv(PROC_DIR / "climate_indices_monthly.csv", parse_dates=["Date"])
ci_df["ym"] = ci_df["Date"].dt.to_period("M")
for c in CLIM_FEATS:
    if c not in ci_df.columns:
        ci_df[c] = 0.0
ci_df[CLIM_FEATS] = ci_df[CLIM_FEATS].fillna(0)
print(f"IMERG: {imerg_df.shape}  CI: {ci_df.shape}")


def build_monthly(river):
    rain = pd.read_csv(PROC_DIR / RIVERS[river], parse_dates=["Date"])
    rain = rain[(rain["Date"] >= "2015-01-01") & (rain["Date"] <= "2025-12-31")].copy()
    rain["ym"] = rain["Date"].dt.to_period("M")

    station_cols = [c for c in rain.columns
                    if c not in ("Date", "ym") and not c.endswith("_imputed")]

    result = {}
    for sta in station_cols:
        flag_col = f"{sta}_imputed"
        if flag_col not in rain.columns:
            continue

        # monthly rainfall (real observations only — _imputed == 0)
        mrows = []
        for ym, grp in rain.groupby("ym"):
            real = grp[grp[flag_col] == 0][sta].dropna()
            total = real.sum() if len(real) >= MIN_DAYS else np.nan
            mrows.append({"ym": ym, "rainfall_mm": total})
        mdf = pd.DataFrame(mrows).sort_values("ym").reset_index(drop=True)

        # seasonal harmonics
        month = mdf["ym"].apply(lambda p: p.month).astype(float)
        mdf["sin_m"]  = np.sin(2 * np.pi * month / 12)
        mdf["cos_m"]  = np.cos(2 * np.pi * month / 12)
        mdf["sin_2m"] = np.sin(4 * np.pi * month / 12)
        mdf["cos_2m"] = np.cos(4 * np.pi * month / 12)

        # lag features (use 0 when prior month is unknown)
        mdf["lag1"]     = mdf["rainfall_mm"].shift(1).fillna(0)
        mdf["log_lag1"] = np.log1p(np.clip(mdf["lag1"], 0, None))
        mdf["lag2"]     = mdf["rainfall_mm"].shift(2).fillna(0)
        mdf["log_lag2"] = np.log1p(np.clip(mdf["lag2"], 0, None))

        # IMERG: use river_station slug
        slug = sta if sta.startswith(river + "_") else f"{river}_{sta}"
        imm_col  = f"{slug}_imerg_mm"
        imm3_col = f"{slug}_imerg_3x3_mm"
        if imm_col in imerg_df.columns:
            mdf = mdf.merge(imerg_df[["ym", imm_col, imm3_col]], on="ym", how="left")
            mdf["log_imerg"]     = np.log1p(np.clip(mdf[imm_col].fillna(0), 0, None))
            mdf["log_imerg_3x3"] = np.log1p(np.clip(mdf[imm3_col].fillna(0), 0, None))
        else:
            mdf["log_imerg"]     = 0.0
            mdf["log_imerg_3x3"] = 0.0

        # climate indices
        mdf = mdf.merge(ci_df[["ym"] + CLIM_FEATS], on="ym", how="left")

        # ensure all features present
        for c in ALL_FEATS:
            if c not in mdf.columns:
                mdf[c] = 0.0
        mdf[ALL_FEATS] = mdf[ALL_FEATS].fillna(0)

        result[sta] = (mdf, slug)

    return result


def train_river(river):
    station_map = build_monthly(river)
    print(f"\n--- {river.upper()}: {len(station_map)} stations ---")

    station_results = []
    for sta, (mdf, slug) in sorted(station_map.items()):

        train_mask = mdf["ym"] <= VAL_END
        test_mask  = (mdf["ym"] > VAL_END) & (mdf["ym"] <= TEST_END)

        train_df = mdf[train_mask].dropna(subset=["rainfall_mm"])
        test_df  = mdf[test_mask].dropna(subset=["rainfall_mm"])

        if len(train_df) < 12 or len(test_df) < 3:
            print(f"  SKIP {slug:<50} n_train={len(train_df)} n_test={len(test_df)}")
            continue

        Xtr = train_df[ALL_FEATS].values
        ytr = np.log1p(train_df["rainfall_mm"].values)
        Xte = test_df[ALL_FEATS].values
        yte = test_df["rainfall_mm"].values

        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc = sc.transform(Xte)

        cv = max(2, min(5, len(train_df) // 6))
        ridge = RidgeCV(alphas=ALPHAS, cv=cv)
        ridge.fit(Xtr_sc, ytr)

        pred_log = ridge.predict(Xte_sc)
        pred_mm  = np.clip(np.expm1(pred_log), 0, None)

        r2    = r2_score(yte, pred_mm)
        r, _  = pearsonr(yte, pred_mm)
        rmse  = float(np.sqrt(np.mean((yte - pred_mm) ** 2)))

        flag = "✓" if r2 >= 0.50 else ("~" if r2 >= 0 else " ")
        print(f"  {flag} {slug:<52} R²={r2*100:6.1f}%  r={r:.3f}  "
              f"RMSE={rmse:6.0f}  α={ridge.alpha_}  n_train={len(train_df)}")

        station_results.append({
            "river":       river,
            "station":     slug,
            "r2":          round(r2, 4),
            "r":           round(r, 4),
            "rmse":        round(rmse, 1),
            "n_train":     len(train_df),
            "n_test":      len(test_df),
            "best_alpha":  ridge.alpha_,
            "predictions": [round(float(v), 1) for v in pred_mm],
        })

    return station_results


# ── run both East Malaysian rivers ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 8 — East Malaysia: Padas (Sabah) + Sarawak")
print("Feature set: IMERG + Seasonal + ONI/DMI lags + Lag1/Lag2  (15 features)")
print("=" * 70)

all_results = []
for river in ["padas", "sarawak"]:
    all_results.extend(train_river(river))

# ── river-level summary ────────────────────────────────────────────────────────
print("\n\n=== Phase 8 Summary ===\n")
df = pd.DataFrame([{k: v for k, v in r.items() if k != "predictions"}
                   for r in all_results])

for river in ["padas", "sarawak"]:
    sub = df[df["river"] == river]
    if sub.empty:
        continue
    n_pos = (sub["r2"] > 0).sum()
    n_50  = (sub["r2"] >= 0.5).sum()
    best  = sub.sort_values("r2", ascending=False).iloc[0]
    median_r2 = sub["r2"].median()
    print(f"{river.upper():10s}: best={best['station']}  R²={best['r2']*100:.1f}%")
    print(f"           median R²={median_r2*100:.1f}%  "
          f"positive={n_pos}/{len(sub)}  >=50%={n_50}")

# ── save ───────────────────────────────────────────────────────────────────────
output = {
    "description": "Phase 8 East Malaysia Ridge — Padas (Sabah) + Sarawak; IMERG + Climate Indices, no ERA5",
    "features":    ALL_FEATS,
    "n_features":  len(ALL_FEATS),
    "train_period": "2015-01 to 2024-04",
    "test_period":  "2024-05 to 2025-09",
    "n_test_months": 17,
    "stations": all_results,
}

out_json = PRED_DIR / "phase7_ridge_padas_sarawak_results.json"
with open(out_json, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nWritten: {out_json}")

df_sorted = df.sort_values(["river", "r2"], ascending=[True, False])
out_csv = PRED_DIR / "phase7_ridge_padas_sarawak_summary.csv"
df_sorted.to_csv(out_csv, index=False)
print(f"Written: {out_csv}")
