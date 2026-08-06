"""
Phase 6 — Dual-Path TCN  (TF 2.15, CPU)
=========================================
Architecture: linear baseline + TCN residual
  - LINEAR PATH  : current-month 24 ERA5 feats → Dense(1, L2)   [Ridge-equivalent]
  - TCN RESIDUAL : 6-step sequence → 2 causal conv blocks → GlobalMaxPool+AvgPool
                   → Dense(8) → Dense(1, strong L2)              [starts ≈ 0]
  - OUTPUT       : linear + residual

Guarantee: if TCN residual ≈ 0 (L2 drives it there), model == linear regression.

Feature space: own ERA5 (8 vars) + river-mean ERA5 (8 vars)
               + log_tp + log_tp_river + 4 seasonals + lag1 + log_lag1 = 24 features
ERA5 dual-coverage is analogous to Phase 5's dual-grid that drove R²=81.7% for Kuantan.

Training: per-river, all pre-test data, 100 epochs cosine LR.
Test    : 2024-05 to 2025-12 (19 months).
"""

import sys, os, warnings
os.environ.update({"TF_CPP_MIN_LOG_LEVEL": "3",
                   "CUDA_VISIBLE_DEVICES": "-1",
                   "TF_ENABLE_ONEDNN_OPTS": "0"})
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import math

tf.random.set_seed(42)
np.random.seed(42)

BASE = Path(__file__).parent.parent

ERA5_ALLRIVERS = BASE / "data/processed/era5_all_rivers_mapped.csv"
ERA5_KUANTAN   = BASE / "data/processed/era5_kuantan_station_mapped.csv"
RAIN_FILES = {
    "johor":   BASE / "data/processed/johor_daily_imputed.csv",
    "kedah":   BASE / "data/processed/kedah_daily_imputed.csv",
    "klang":   BASE / "data/processed/klang_daily_imputed.csv",
    "kuantan": BASE / "data/processed/kuantan_daily_imputed.csv",
}
OUT_DIR = BASE / "predictions"
MDL_DIR = BASE / "models"
OUT_DIR.mkdir(exist_ok=True)
MDL_DIR.mkdir(exist_ok=True)

ERA5_VARS     = ["t2m", "d2m", "sp", "u10", "v10", "tcwv", "rh", "ws", "tp"]
ERA5_NONTP    = [v for v in ERA5_VARS if v != "tp"]   # 8 vars
TRAIN_END     = "2024-04"
TEST_START    = "2024-05"
TEST_END      = "2025-12"
T_PAST        = 6
MIN_TRAIN     = 20
N_EPOCHS      = 100

FEAT_COLS = (
    [f"era5_{v}" for v in ERA5_NONTP] +   # 8 own ERA5 vars (no tp, handled via log)
    [f"mean_{v}" for v in ERA5_NONTP] +   # 8 river-mean ERA5 vars
    ["log_tp", "log_tp_river",             # 2 log-transformed tp
     "sin_m", "cos_m", "sin_2m", "cos_2m",# 4 seasonal
     "lag1", "log_lag1"]                   # 2 lag
)  # total = 8 + 8 + 2 + 4 + 2 = 24 features


def flush(*args):
    print(*args)
    sys.stdout.flush()


print("=" * 72)
flush("Phase 6 — Dual-Path TCN (Linear + TCN Residual, no Attention)")
flush(f"TF {tf.__version__} | T_PAST={T_PAST} | Features={len(FEAT_COLS)} (dual ERA5)")
flush(f"Train: 2015-01 → {TRAIN_END} | Test: {TEST_START} → {TEST_END}")
print("=" * 72)
sys.stdout.flush()


# ── ERA5: load once ───────────────────────────────────────────────────────

flush("Loading ERA5 files...")

def _aggregate_era5_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily ERA5 to monthly.
    tp (precipitation) columns are summed; all other ERA5 vars are averaged.
    """
    df = df.copy()
    df.index = df.index.to_period("M").to_timestamp()  # collapse to month-start
    tp_cols  = [c for c in df.columns if c.endswith("_tp")]
    avg_cols = [c for c in df.columns if c not in tp_cols]
    monthly  = pd.concat([
        df[avg_cols].resample("MS").mean() if avg_cols else pd.DataFrame(),
        df[tp_cols ].resample("MS").sum()  if tp_cols  else pd.DataFrame(),
    ], axis=1)
    return monthly[df.columns]   # restore original column order

_era5_all = _aggregate_era5_monthly(
    pd.read_csv(ERA5_ALLRIVERS, parse_dates=["Date"], index_col="Date"))
_era5_kun = _aggregate_era5_monthly(
    pd.read_csv(ERA5_KUANTAN, parse_dates=["Date"], index_col="Date"))
flush(f"ERA5 loaded — {len(_era5_all)} monthly rows.")


def get_river_era5(river: str, station_names: list[str]) -> pd.DataFrame:
    """
    Returns monthly ERA5 DataFrame with:
      {station}_{var}   for each station's own ERA5 (prefixed or bare)
      mean_{var}        river-mean across all stations
    """
    if river == "kuantan":
        df = _era5_kun.copy()
        bare_names = [s.replace("kuantan_", "") for s in station_names]
    else:
        df = _era5_all.copy()
        bare_names = station_names  # already have river prefix in column names

    for v in ERA5_VARS:
        cols = [f"{b}_{v}" for b in bare_names if f"{b}_{v}" in df.columns]
        df[f"mean_{v}"] = df[cols].mean(axis=1) if cols else np.nan

    return df


def monthly_rain(rain_path: Path, col: str) -> pd.Series:
    """Read daily CSV (lazily), resample one column to monthly MS."""
    df = pd.read_csv(rain_path, parse_dates=["Date"], index_col="Date",
                     usecols=["Date", col])
    s = df[col].copy()
    s[s < 0] = np.nan
    return s.resample("MS").sum(min_count=15)


def build_features(era5_df: pd.DataFrame, river: str,
                   station: str, rain_monthly: pd.Series) -> pd.DataFrame:
    """
    Build 24-column feature DataFrame (+ target column).
    station is the full name e.g. 'johor_lepau' or 'kuantan_sg_cherating'.
    """
    bare = station.replace(f"{river}_", "")
    feat = pd.DataFrame(index=era5_df.index)

    # own ERA5 vars (non-tp)
    for v in ERA5_NONTP:
        col = f"{bare}_{v}" if river == "kuantan" else f"{station}_{v}"
        feat[f"era5_{v}"] = era5_df[col].values if col in era5_df.columns else np.nan

    # river-mean ERA5 (non-tp)
    for v in ERA5_NONTP:
        feat[f"mean_{v}"] = era5_df[f"mean_{v}"].values

    # log-tp
    own_tp_col  = f"{bare}_tp" if river == "kuantan" else f"{station}_tp"
    feat["log_tp"] = np.log1p(
        era5_df[own_tp_col].clip(lower=0).values * 1000
        if own_tp_col in era5_df.columns else 0)
    feat["log_tp_river"] = np.log1p(
        era5_df["mean_tp"].clip(lower=0).values * 1000)

    # seasonal
    m = feat.index.month
    feat["sin_m"]  = np.sin(2 * np.pi * m / 12)
    feat["cos_m"]  = np.cos(2 * np.pi * m / 12)
    feat["sin_2m"] = np.sin(4 * np.pi * m / 12)
    feat["cos_2m"] = np.cos(4 * np.pi * m / 12)

    # rainfall lags
    rain = rain_monthly.reindex(feat.index)
    feat["lag1"]     = rain.shift(1)
    feat["log_lag1"] = np.log1p(feat["lag1"].clip(lower=0))
    feat["target"]   = rain.values

    feat.dropna(subset=["log_tp"], inplace=True)
    return feat


def make_sequences(feat_df: pd.DataFrame):
    """
    Split into train / test sequences of shape (T_PAST, N_FEATS).
    Crucially: the LAST step of each sequence IS the current month (month i).
    So inp[:, -1, :] = current month's ERA5 + lag1=rain(i-1) — same features as Ridge.
    Sequence covers months i-T_PAST+1 through i (inclusive).
    """
    tr_cutoff = pd.Period(TRAIN_END, "M")
    te_start  = pd.Period(TEST_START, "M")
    te_end    = pd.Period(TEST_END, "M")

    feat_df = feat_df.sort_index()
    periods = feat_df.index.to_period("M")
    n = len(feat_df)
    valid_cols = [c for c in FEAT_COLS if c in feat_df.columns]

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i in range(T_PAST - 1, n):
        period_i = periods[i]
        # sequence: rows i-T_PAST+1 to i (inclusive); last row = current month
        seq = feat_df.iloc[i - T_PAST + 1:i + 1]

        if seq[valid_cols].isnull().any().any():
            continue
        if pd.isna(feat_df.iloc[i]["target"]) or feat_df.iloc[i]["target"] < 0:
            continue

        x = seq[valid_cols].values.astype(np.float32)
        y = np.float32(np.log1p(feat_df.iloc[i]["target"]))

        if period_i <= tr_cutoff:
            X_tr.append(x); y_tr.append(y)
        elif te_start <= period_i <= te_end:
            X_te.append(x); y_te.append(y)

    return (np.array(X_tr), np.array(y_tr, np.float32),
            np.array(X_te), np.array(y_te, np.float32))


# ── model ─────────────────────────────────────────────────────────────────

def build_dualpath_tcn(t_past, n_feats, filters=8, dropout=0.2,
                        l2_lin=1e-3, l2_tcn=5e-2):
    """
    Dual-path:
      A: current month features → Dense(1, L2)        [linear/Ridge-like]
      B: full sequence → TCN(causal, dilated) → pool  [residual correction, L2]
      out = A + B
    """
    inp = keras.Input(shape=(t_past, n_feats))

    # ── Path A: linear ────────────────────────────────────────────────────
    current = inp[:, -1, :]
    linear  = layers.Dense(1, name="linear",
                           kernel_regularizer=regularizers.l2(l2_lin))(current)

    # ── Path B: TCN residual ──────────────────────────────────────────────
    r = regularizers.l2(l2_tcn * 0.05)

    skip = layers.Conv1D(filters, 1, padding="same", kernel_regularizer=r)(inp)

    x = layers.Conv1D(filters, 3, padding="causal", dilation_rate=1,
                      activation="relu", kernel_regularizer=r)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, 3, padding="causal", dilation_rate=2,
                      activation="relu", kernel_regularizer=r)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, skip])

    # pool both max and avg (no attention needed for 6-step sequences)
    gmax = layers.GlobalMaxPooling1D()(x)
    gavg = layers.GlobalAveragePooling1D()(x)
    pool = layers.Concatenate()([gmax, gavg])

    res = layers.Dense(8, activation="relu",
                       kernel_regularizer=regularizers.l2(l2_tcn * 0.3))(pool)
    res = layers.Dropout(dropout)(res)
    res = layers.Dense(1, name="residual",
                       kernel_regularizer=regularizers.l2(l2_tcn))(res)

    out = layers.Add(name="pred")([linear, res])
    return keras.Model(inp, out, name="DualPath_TCN")


def cosine_lr(epoch, lr_max=5e-4, lr_min=1e-5, total=N_EPOCHS):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total))


# ── per-river pipeline ────────────────────────────────────────────────────

def run_river(river: str) -> list[dict]:
    flush(f"\n{'─'*72}")
    flush(f"  River: {river.upper()}")
    flush(f"{'─'*72}")

    rain_path = RAIN_FILES[river]

    # discover stations
    rain_cols = pd.read_csv(rain_path, nrows=0).columns.tolist()
    if river == "kuantan":
        bare_stns = [c for c in rain_cols if c not in ("Date",) and "_imputed" not in c]
        stations  = [f"kuantan_{s}" for s in bare_stns]
    else:
        stations = [c for c in rain_cols
                    if c.startswith(f"{river}_") and "_imputed" not in c]

    era5_df = get_river_era5(river, stations)

    results = []
    n_valid = 0

    for stn in stations:
        bare = stn.replace(f"{river}_", "")
        col  = bare if river == "kuantan" else stn
        if col not in rain_cols:
            col = next((c for c in rain_cols if bare in c and "_imputed" not in c), None)
        if col is None:
            continue

        rain_m  = monthly_rain(rain_path, col)
        feat_df = build_features(era5_df, river, stn, rain_m)

        if FEAT_COLS[0] not in feat_df.columns:
            continue

        X_tr, y_tr, X_te, y_te = make_sequences(feat_df)

        rec = {"river": river, "station": stn,
               "n_train": len(X_tr), "n_test": len(X_te)}

        if len(X_tr) < MIN_TRAIN or len(X_te) < 3:
            rec.update({"r2": None, "r": None, "rmse": None,
                        "status": "insufficient_data"})
            results.append(rec)
            continue

        n_feats = X_tr.shape[2]
        actuals_mm = np.expm1(y_te)

        # ── per-station model ──────────────────────────────────────────
        scaler = StandardScaler()
        shape_tr = X_tr.shape
        X_tr_s   = scaler.fit_transform(X_tr.reshape(-1, n_feats)).reshape(shape_tr)
        X_te_s   = scaler.transform(X_te.reshape(-1, n_feats)).reshape(X_te.shape)

        model = build_dualpath_tcn(T_PAST, n_feats, filters=8, dropout=0.15,
                                    l2_lin=1e-3, l2_tcn=5e-2)
        model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")

        lr_cb = keras.callbacks.LearningRateScheduler(
            lambda ep, _: cosine_lr(ep), verbose=0)

        model.fit(X_tr_s, y_tr,
                  epochs=N_EPOCHS, batch_size=min(16, len(X_tr)),
                  shuffle=True, callbacks=[lr_cb], verbose=0)

        y_pred_log = model.predict(X_te_s, verbose=0).flatten()
        y_pred_mm  = np.expm1(y_pred_log).clip(min=0)

        keras.backend.clear_session()   # free memory between station models

        r2   = float(r2_score(actuals_mm, y_pred_mm))
        corr = np.corrcoef(actuals_mm, y_pred_mm)[0, 1] if len(actuals_mm) > 1 else np.nan
        rmse = float(np.sqrt(np.mean((actuals_mm - y_pred_mm) ** 2)))

        mark = ("✓" if r2 >= 0.8 else "●" if r2 >= 0.5
                else "↑" if r2 >= 0 else "↓")
        flush(f"    {stn:<54} {r2*100:+6.1f}% {mark}  r={corr:+.3f}")

        rec.update({"r2": round(r2, 4),
                    "r":  round(float(corr), 4) if np.isfinite(corr) else None,
                    "rmse": round(rmse, 1), "status": "ok",
                    "predictions": [round(float(v), 1) for v in y_pred_mm]})
        results.append(rec)
        n_valid += 1

    flush(f"  {n_valid} stations trained per-station (dual-path TCN)")
    return results


# ── main ──────────────────────────────────────────────────────────────────

all_results = []
for river in ["johor", "kedah", "klang", "kuantan"]:
    all_results.extend(run_river(river))

# ── summary table ─────────────────────────────────────────────────────────

flush("\n\n" + "=" * 72)
flush("PER-STATION TEST RESULTS  (test: 2024-05 to 2025-12)")
flush("=" * 72)
flush(f"\n{'River':<10}  {'Station':<52}  {'R²':>7}  {'r':>7}  {'RMSE':>7}  N_tr")
flush("-" * 90)

for rec in sorted(all_results, key=lambda x: (x["river"], x.get("r2") or -999)):
    r2_val = rec.get("r2")
    r_val  = rec.get("r")
    r2_s  = f"{r2_val*100:+.1f}%" if r2_val is not None else "  N/A"
    r_s   = f"{r_val:+.3f}" if r_val is not None else "  N/A"
    rmse_s= f"{rec['rmse']:.0f}mm" if rec.get("rmse") is not None else "  N/A"
    mark  = ("✓" if (r2_val or -1) >= 0.8 else
             "●" if (r2_val or -1) >= 0.5 else
             "↑" if (r2_val or -1) >= 0 else "↓")
    flush(f"  {rec['river']:<8}  {rec['station']:<52}  {r2_s:>8} {mark}  "
          f"{r_s:>7}  {rmse_s:>7}  {rec['n_train']}")

flush("\n\n" + "=" * 72)
flush("RIVER SUMMARY")
flush("=" * 72)
flush(f"{'River':<12}  {'N':>4}  {'Median R²':>10}  {'Max R²':>8}  {'≥0%':>5}  {'≥50%':>5}  {'≥80%':>5}")
flush("-" * 60)
ridge_best = {"kuantan": 0.807, "kedah": 0.507, "johor": 0.340, "klang": 0.189}

for river in ["johor", "kedah", "klang", "kuantan"]:
    sub = [r for r in all_results if r["river"] == river and r.get("r2") is not None]
    if not sub:
        continue
    r2s = [r["r2"] for r in sub]
    flush(f"  {river:<10}  {len(sub):>4}  {np.median(r2s)*100:>9.1f}%  "
          f"{max(r2s)*100:>7.1f}%  "
          f"{sum(1 for v in r2s if v >= 0):>5}  "
          f"{sum(1 for v in r2s if v >= 0.5):>5}  "
          f"{sum(1 for v in r2s if v >= 0.8):>5}")

flush("\n── vs Phase 6 Ridge baseline ─────────────────────────────────────────")
for river in ["johor", "kedah", "klang", "kuantan"]:
    sub = [r for r in all_results if r["river"] == river and r.get("r2") is not None]
    if not sub:
        continue
    best  = max(r["r2"] for r in sub)
    base  = ridge_best.get(river, 0)
    delta = best - base
    sign  = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "≈")
    flush(f"  {river:<10}: DualPath best={best*100:.1f}%  Ridge={base*100:.1f}%  "
          f"{sign} {abs(delta)*100:.1f}pp")

# save
out_json = {
    "model": "DualPath_TCN",
    "architecture": "Linear(current_month_feats) + TCN_Residual(6mo_causal_conv)",
    "features": len(FEAT_COLS),
    "feature_detail": "own_ERA5x8 + river_mean_ERA5x8 + log_tp + log_tp_river + 4_seasonals + lag1 + log_lag1",
    "t_past": T_PAST, "n_epochs": N_EPOCHS,
    "train_end": TRAIN_END, "test_start": TEST_START, "test_end": TEST_END,
    "stations": all_results
}
with open(OUT_DIR / "phase6_dualpath_tcn_results.json", "w") as f:
    json.dump(out_json, f, indent=2)

pred_rows = [{"river": r["river"], "station": r["station"], "pred_mm": mm}
             for r in all_results if r.get("predictions")
             for mm in r["predictions"]]
pd.DataFrame(pred_rows).to_csv(OUT_DIR / "phase6_dualpath_tcn_predictions.csv", index=False)

flush("\nSaved → phase6_dualpath_tcn_results.json")
flush("Saved → phase6_dualpath_tcn_predictions.csv")
flush("Phase 6 dual-path TCN complete.")
