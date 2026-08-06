"""
Phase 6 — Per-River Ridge Regression Baseline
Ridge(alpha sweep) on 15 ERA5 features, per river, per station.
Establishes achievable R² ceiling for the new rivers before TCN.

Run (standard Python, no TF needed):
  python scripts/phase6_ridge_allrivers.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
PRED_DIR = os.path.join(ROOT, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

TRAIN_END = pd.Period("2022-09", freq="M")
VAL_END   = pd.Period("2024-04", freq="M")
TEST_END  = pd.Period("2025-12", freq="M")
MIN_DAYS  = 15

RIVERS = {
    "johor":   {"imputed": "johor_daily_imputed.csv",   "era5": "era5_all_rivers_mapped.csv"},
    "kedah":   {"imputed": "kedah_daily_imputed.csv",   "era5": "era5_all_rivers_mapped.csv"},
    "klang":   {"imputed": "klang_daily_imputed.csv",   "era5": "era5_all_rivers_mapped.csv"},
    "kuantan": {"imputed": "kuantan_daily_imputed.csv", "era5": "era5_kuantan_station_mapped.csv"},
}

ERA5_GENERIC = ["era5_t2m","era5_d2m","era5_sp","era5_u10","era5_v10",
                "era5_tcwv","era5_rh","era5_ws"]
COMMON_FEATS = ["log_tp"] + ERA5_GENERIC + \
               ["sin_m","cos_m","sin_2m","cos_2m","lag1","log_lag1"]  # 15 total


def build_monthly(river):
    rain = pd.read_csv(os.path.join(PROC_DIR, RIVERS[river]["imputed"]),
                       parse_dates=["Date"])
    rain = rain[(rain["Date"] >= "2015-01-01") & (rain["Date"] <= "2025-12-31")].copy()
    rain["ym"] = rain["Date"].dt.to_period("M")

    era5 = pd.read_csv(os.path.join(PROC_DIR, RIVERS[river]["era5"]),
                       parse_dates=["Date"]).ffill().bfill()
    era5["ym"] = era5["Date"].dt.to_period("M")

    era5_var_map = {"t2m":"era5_t2m","d2m":"era5_d2m","sp":"era5_sp",
                    "u10":"era5_u10","v10":"era5_v10","tcwv":"era5_tcwv",
                    "rh":"era5_rh","ws":"era5_ws","tp":"_tp_raw"}

    stations = [c for c in rain.columns
                if c not in ("Date","ym") and not c.endswith("_imputed")]

    result = {}
    for sta in stations:
        flag_col = f"{sta}_imputed"
        if flag_col not in rain.columns:
            continue

        mrows = []
        for ym, grp in rain.groupby("ym"):
            real = grp[grp[flag_col] == 0][sta].dropna()
            total = real.sum() if len(real) >= MIN_DAYS else np.nan
            mrows.append({"ym": ym, "rainfall_mm": total})
        mdf = pd.DataFrame(mrows)

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
        month = mdf["ym"].apply(lambda p: p.month).astype(float)
        mdf["sin_m"]  = np.sin(2 * np.pi * month / 12)
        mdf["cos_m"]  = np.cos(2 * np.pi * month / 12)
        mdf["sin_2m"] = np.sin(4 * np.pi * month / 12)
        mdf["cos_2m"] = np.cos(4 * np.pi * month / 12)
        mdf["lag1"]     = mdf["rainfall_mm"].shift(1).fillna(0)
        mdf["log_lag1"] = np.log1p(np.clip(mdf["lag1"], 0, None))
        mdf[COMMON_FEATS] = mdf[COMMON_FEATS].fillna(0)
        result[sta] = mdf

    return result


def train_river_ridge(river):
    mdfs = build_monthly(river)
    print(f"  {river}: {len(mdfs)} stations")

    station_results = {}
    pred_rows = []

    for sta, mdf in sorted(mdfs.items()):
        # train on pre-test period (train + val combined)
        train_mask = mdf["ym"] <= VAL_END
        test_mask  = (mdf["ym"] > VAL_END) & (mdf["ym"] <= TEST_END)

        train_df = mdf[train_mask].dropna(subset=["rainfall_mm"])
        test_df  = mdf[test_mask].dropna(subset=["rainfall_mm"])

        if len(train_df) < 12 or len(test_df) < 3:
            continue

        Xtr = train_df[COMMON_FEATS].values
        ytr = np.log1p(train_df["rainfall_mm"].values)
        Xte = test_df[COMMON_FEATS].values
        yte_mm = test_df["rainfall_mm"].values

        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc = sc.transform(Xte)

        model = RidgeCV(alphas=[0.1, 0.5, 1, 5, 10, 50, 100, 500])
        model.fit(Xtr_sc, ytr)

        pred_log = model.predict(Xte_sc)
        pred_mm  = np.expm1(np.clip(pred_log, 0, 8))
        r2   = float(r2_score(yte_mm, pred_mm))
        rmse = float(np.sqrt(np.mean((yte_mm - pred_mm) ** 2)))
        r_corr = float(np.corrcoef(yte_mm, pred_mm)[0, 1]) if len(yte_mm) > 2 else 0.0

        station_results[sta] = {
            "r2": r2, "r": r_corr, "rmse_mm": rmse,
            "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "ridge_alpha": float(model.alpha_),
        }
        for ym, actual, pred in zip(test_df["ym"].values, yte_mm, pred_mm):
            pred_rows.append({"river": river, "station": sta, "period": str(ym),
                              "actual_mm": float(actual), "pred_mm": float(pred)})

    return station_results, pred_rows


if __name__ == "__main__":
    print("=" * 72)
    print("Phase 6 — Per-River Ridge Regression (train on 2015-2024-04)")
    print("=" * 72)

    all_results  = {}
    all_pred_rows = []
    river_summary = {}

    for river in RIVERS:
        print(f"\n{river.upper()}")
        sta_res, pred_rows = train_river_ridge(river)
        all_results[river] = sta_res
        all_pred_rows.extend(pred_rows)

        r2s = [v["r2"] for v in sta_res.values()]
        valid = [v for v in r2s if not np.isnan(v)]
        if not valid:
            continue
        river_summary[river] = {
            "n_stations": len(valid),
            "median_r2": float(np.median(valid)),
            "mean_r2":   float(np.mean(valid)),
            "max_r2":    float(np.max(valid)),
            "min_r2":    float(np.min(valid)),
            "n_positive_r2": sum(1 for v in valid if v > 0),
            "n_r2_ge50pct":  sum(1 for v in valid if v >= 0.5),
            "n_r2_ge80pct":  sum(1 for v in valid if v >= 0.8),
        }

    print("\n\n" + "=" * 72)
    print("PER-STATION TEST RESULTS  (test: 2024-05 to 2025-12)")
    print("=" * 72)
    print(f"\n{'River':<10} {'Station':<48} {'R²':>7} {'r':>6} {'RMSE':>8} {'N_tr':>5}")
    print("-" * 82)
    for river, sta_results in sorted(all_results.items()):
        for sta, res in sorted(sta_results.items()):
            tag = " ✓" if res["r2"] >= 0.80 else (" ●" if res["r2"] >= 0.50 else ("  " if res["r2"] >= 0 else "  ↓"))
            print(f"  {river:<10} {sta:<46} {res['r2']*100:>+.1f}%{tag}  "
                  f"{res['r']:>+.3f}  {res['rmse_mm']:>6.0f}mm  {res['n_train']:>4}")

    print("\n\n" + "=" * 72)
    print("RIVER SUMMARY")
    print("=" * 72)
    print(f"{'River':<12} {'N':>4} {'Median R²':>11} {'Max R²':>9} {'≥0%':>5} {'≥50%':>6} {'≥80%':>6}")
    print("-" * 56)
    for river, rs in sorted(river_summary.items()):
        print(f"  {river:<12} {rs['n_stations']:>4}   "
              f"{rs['median_r2']*100:>+.1f}%     {rs['max_r2']*100:>+.1f}%   "
              f"{rs['n_positive_r2']:>3}   {rs['n_r2_ge50pct']:>4}   {rs['n_r2_ge80pct']:>4}")

    out = {
        "description": "Phase 6 — per-river Ridge regression baseline",
        "model": "RidgeCV (alpha sweep), StandardScaler, 15 ERA5 features",
        "training_period": "2015-01 to 2024-04 (train+val combined)",
        "test_period": "2024-05 to 2025-12",
        "river_summary": river_summary,
        "per_station": {
            f"{river}::{sta}": v
            for river, sta_results in all_results.items()
            for sta, v in sta_results.items()
        },
    }
    res_path  = os.path.join(PRED_DIR, "phase6_ridge_allrivers_results.json")
    pred_path = os.path.join(PRED_DIR, "phase6_ridge_allrivers_predictions.csv")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(all_pred_rows).to_csv(pred_path, index=False)
    print(f"\nSaved → predictions/phase6_ridge_allrivers_results.json")
    print(f"Saved → predictions/phase6_ridge_allrivers_predictions.csv")
