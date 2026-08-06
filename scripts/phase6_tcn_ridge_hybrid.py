"""
Phase 6 — TCN Backbone + Ridge Head Hybrid  (TF 2.15, CPU)
============================================================
Architecture (proven by Phase 5 for Kuantan):
  1. TF TCN backbone: 6-month sequence → 2 causal conv blocks → pooled embedding
  2. sklearn RidgeCV head: (TCN_embedding || current_month_24_features) → rainfall

Why this works where end-to-end TCN fails:
  - Ridge head is a closed-form optimal solution → cannot degrade below Ridge-only
  - TCN adds complementary non-linear temporal features if they exist
  - Per-station Ridge handles cross-station interference

Feature expansion vs Phase 6 Ridge (15→24 features):
  own ERA5 (8) + river-mean ERA5 (8) + log_tp + log_tp_river + 4 seasonals + lag1 + log_lag1
  The river-mean ERA5 is the dual-grid equivalent that drove Phase 5's 81.7%.
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
from sklearn.linear_model import RidgeCV
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
OUT_DIR.mkdir(exist_ok=True); MDL_DIR.mkdir(exist_ok=True)

ERA5_VARS  = ["t2m", "d2m", "sp", "u10", "v10", "tcwv", "rh", "ws", "tp"]
ERA5_NONTP = [v for v in ERA5_VARS if v != "tp"]

TRAIN_END  = "2024-04"
TEST_START = "2024-05"
TEST_END   = "2025-12"
T_PAST     = 6
MIN_TRAIN  = 20
N_EPOCHS   = 80     # backbone training epochs
EMB_DIM    = 16     # TCN embedding dimension (GlobalAvgPool + GlobalMaxPool each 8)

FEAT_COLS = (
    [f"era5_{v}" for v in ERA5_NONTP] +
    [f"mean_{v}" for v in ERA5_NONTP] +
    ["log_tp", "log_tp_river", "sin_m", "cos_m", "sin_2m", "cos_2m", "lag1", "log_lag1"]
)  # 24 features


def flush(*a): print(*a); sys.stdout.flush()


print("=" * 72)
flush("Phase 6 — TCN Backbone + Ridge Head Hybrid")
flush(f"TF {tf.__version__} | T_PAST={T_PAST} | ERA5 features=24 (dual) | EMB={EMB_DIM}")
flush(f"Train: 2015-01 → {TRAIN_END} | Test: {TEST_START} → {TEST_END}")
print("=" * 72)


# ── ERA5 loading ──────────────────────────────────────────────────────────

flush("Loading ERA5 files...")

def _agg_era5(path):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.index = df.index.to_period("M").to_timestamp()
    tp_cols  = [c for c in df.columns if c.endswith("_tp")]
    avg_cols = [c for c in df.columns if c not in tp_cols]
    m = pd.concat([df[avg_cols].resample("MS").mean(),
                   df[tp_cols ].resample("MS").sum()], axis=1)
    return m[df.columns]

_era5_all = _agg_era5(ERA5_ALLRIVERS)
_era5_kun = _agg_era5(ERA5_KUANTAN)
flush(f"ERA5 loaded — {len(_era5_all)} monthly rows.")


def get_river_era5(river, stations):
    df = _era5_kun.copy() if river == "kuantan" else _era5_all.copy()
    bare = [s.replace(f"{river}_", "") for s in stations] if river == "kuantan" else stations
    for v in ERA5_VARS:
        cols = [f"{b}_{v}" for b in bare if f"{b}_{v}" in df.columns]
        df[f"mean_{v}"] = df[cols].mean(axis=1) if cols else np.nan
    return df


def monthly_rain(rain_path, col):
    df = pd.read_csv(rain_path, parse_dates=["Date"], index_col="Date",
                     usecols=["Date", col])
    s = df[col].copy(); s[s < 0] = np.nan
    return s.resample("MS").sum(min_count=15)


def build_features(era5_df, river, station, rain_monthly):
    bare = station.replace(f"{river}_", "")
    feat = pd.DataFrame(index=era5_df.index)
    for v in ERA5_NONTP:
        col = f"{bare}_{v}" if river == "kuantan" else f"{station}_{v}"
        feat[f"era5_{v}"] = era5_df[col].values if col in era5_df.columns else np.nan
    for v in ERA5_NONTP:
        feat[f"mean_{v}"] = era5_df[f"mean_{v}"].values
    tp_col  = f"{bare}_tp" if river == "kuantan" else f"{station}_tp"
    feat["log_tp"] = np.log1p(era5_df[tp_col].clip(0).values * 1000
                               if tp_col in era5_df.columns else 0)
    feat["log_tp_river"] = np.log1p(era5_df["mean_tp"].clip(0).values * 1000)
    m = feat.index.month
    feat["sin_m"] = np.sin(2*np.pi*m/12); feat["cos_m"]  = np.cos(2*np.pi*m/12)
    feat["sin_2m"]= np.sin(4*np.pi*m/12); feat["cos_2m"] = np.cos(4*np.pi*m/12)
    rain = rain_monthly.reindex(feat.index)
    feat["lag1"]     = rain.shift(1)
    feat["log_lag1"] = np.log1p(feat["lag1"].clip(0))
    feat["target"]   = rain.values
    feat.dropna(subset=["log_tp"], inplace=True)
    return feat


def make_sequences(feat_df):
    """
    Sequence: rows i-T_PAST+1 to i (inclusive).
    Last step = current month i → linear path has current ERA5 + lag1=rain(i-1).
    """
    tr_cut = pd.Period(TRAIN_END, "M")
    te_s   = pd.Period(TEST_START, "M"); te_e = pd.Period(TEST_END, "M")
    feat_df = feat_df.sort_index()
    periods = feat_df.index.to_period("M")
    n = len(feat_df)
    vc = [c for c in FEAT_COLS if c in feat_df.columns]

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i in range(T_PAST - 1, n):
        p  = periods[i]
        seq = feat_df.iloc[i - T_PAST + 1:i + 1]
        if seq[vc].isnull().any().any(): continue
        if pd.isna(feat_df.iloc[i]["target"]) or feat_df.iloc[i]["target"] < 0: continue
        x = seq[vc].values.astype(np.float32)
        y = np.float32(np.log1p(feat_df.iloc[i]["target"]))
        if p <= tr_cut: X_tr.append(x); y_tr.append(y)
        elif te_s <= p <= te_e: X_te.append(x); y_te.append(y)

    return (np.array(X_tr), np.array(y_tr, np.float32),
            np.array(X_te), np.array(y_te, np.float32))


# ── TCN backbone ──────────────────────────────────────────────────────────

def build_tcn_backbone(t_past, n_feats, filters=8, l2=5e-3):
    """Returns (model, embedding_dim). Backbone extracts temporal features."""
    inp  = keras.Input(shape=(t_past, n_feats))
    reg  = regularizers.l2(l2)

    skip = layers.Conv1D(filters, 1, padding="same", kernel_regularizer=reg)(inp)
    x    = layers.Conv1D(filters, 3, padding="causal", dilation_rate=1,
                         activation="relu", kernel_regularizer=reg)(inp)
    x    = layers.LayerNormalization()(x)
    x    = layers.Conv1D(filters, 3, padding="causal", dilation_rate=2,
                         activation="relu", kernel_regularizer=reg)(x)
    x    = layers.LayerNormalization()(x)
    x    = layers.Add()([x, skip])

    gavg = layers.GlobalAveragePooling1D()(x)   # shape (filters,)
    gmax = layers.GlobalMaxPooling1D()(x)        # shape (filters,)
    emb  = layers.Concatenate()([gavg, gmax])   # shape (2*filters,) = 16

    # regression head for backbone pre-training
    out  = layers.Dense(1)(emb)
    model = keras.Model(inp, out, name="TCN_backbone")
    return model, 2 * filters


def cosine_lr(ep, n=N_EPOCHS, lo=1e-5, hi=5e-4):
    return lo + 0.5*(hi-lo)*(1+math.cos(math.pi*ep/n))


# ── feature extractor (intermediate model after training) ─────────────────

def get_embedding_model(trained_model):
    """Strip off the final Dense head → returns embeddings."""
    return keras.Model(trained_model.input,
                       trained_model.get_layer("concatenate").output,
                       name="TCN_emb")


# ── per-river pipeline ────────────────────────────────────────────────────

def run_river(river):
    flush(f"\n{'─'*72}")
    flush(f"  River: {river.upper()}")
    flush(f"{'─'*72}")

    rain_path = RAIN_FILES[river]
    rain_cols = pd.read_csv(rain_path, nrows=0).columns.tolist()
    if river == "kuantan":
        bare_stns = [c for c in rain_cols if c != "Date" and "_imputed" not in c]
        stations  = [f"kuantan_{s}" for s in bare_stns]
    else:
        stations = [c for c in rain_cols
                    if c.startswith(f"{river}_") and "_imputed" not in c]

    era5_df = get_river_era5(river, stations)

    # build all station sequences
    station_seqs = {}
    for stn in stations:
        bare = stn.replace(f"{river}_", "")
        col  = bare if river == "kuantan" else stn
        if col not in rain_cols:
            col = next((c for c in rain_cols if bare in c and "_imputed" not in c), None)
        if col is None: continue

        rain_m  = monthly_rain(rain_path, col)
        feat_df = build_features(era5_df, river, stn, rain_m)
        if not FEAT_COLS[0] in feat_df.columns: continue

        X_tr, y_tr, X_te, y_te = make_sequences(feat_df)
        if len(X_tr) < MIN_TRAIN or len(X_te) < 3: continue
        station_seqs[stn] = (X_tr, y_tr, X_te, y_te, np.expm1(y_te))

    if not station_seqs:
        flush("  No stations with sufficient data — skipping")
        return []

    # ── Step 1: scale features pooled across river ────────────────────────
    X_tr_all = np.concatenate([v[0] for v in station_seqs.values()], axis=0)
    y_tr_all = np.concatenate([v[1] for v in station_seqs.values()], axis=0)
    n_feats  = X_tr_all.shape[2]

    scaler = StandardScaler()
    shape  = X_tr_all.shape
    X_tr_s = scaler.fit_transform(X_tr_all.reshape(-1, n_feats)).reshape(shape)

    flush(f"  {len(station_seqs)} stations | {len(X_tr_all)} pooled train seqs | {n_feats} features")

    # ── Step 2: pre-train TCN backbone on pooled river data ───────────────
    backbone, emb_dim = build_tcn_backbone(T_PAST, n_feats, filters=8, l2=5e-3)
    backbone.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
    lr_cb = keras.callbacks.LearningRateScheduler(lambda ep, _: cosine_lr(ep), verbose=0)
    backbone.fit(X_tr_s, y_tr_all,
                 epochs=N_EPOCHS, batch_size=32, shuffle=True,
                 callbacks=[lr_cb], verbose=0)

    emb_model = get_embedding_model(backbone)
    flush(f"  TCN backbone trained | params={backbone.count_params():,} | emb_dim={emb_dim}")

    # ── Step 3: per-station Ridge on (TCN_emb || current_ERA5_feats) ──────
    results = []
    for stn, (X_tr, y_tr, X_te, y_te, actuals_mm) in station_seqs.items():
        # scale station sequences using pooled scaler
        X_tr_s2 = scaler.transform(X_tr.reshape(-1, n_feats)).reshape(X_tr.shape)
        X_te_s2 = scaler.transform(X_te.reshape(-1, n_feats)).reshape(X_te.shape)

        # extract TCN embeddings (shape: N × emb_dim)
        emb_tr = emb_model.predict(X_tr_s2, verbose=0)   # N_tr × emb_dim
        emb_te = emb_model.predict(X_te_s2, verbose=0)   # N_te × emb_dim

        # current-month features from last step of each sequence (N × n_feats)
        curr_tr = X_tr_s2[:, -1, :]  # current month ERA5 features (already scaled)
        curr_te = X_te_s2[:, -1, :]

        # combine embeddings with current-month features
        Z_tr = np.concatenate([emb_tr, curr_tr], axis=1)  # N × (emb_dim + n_feats)
        Z_te = np.concatenate([emb_te, curr_te], axis=1)

        # fit per-station RidgeCV
        alphas = [0.1, 0.5, 1, 5, 10, 50, 100]
        ridge  = RidgeCV(alphas=alphas, cv=5 if len(Z_tr) >= 10 else 3)
        ridge.fit(Z_tr, y_tr)

        y_pred_log = ridge.predict(Z_te)
        y_pred_mm  = np.expm1(y_pred_log).clip(min=0)

        r2   = float(r2_score(actuals_mm, y_pred_mm))
        corr = np.corrcoef(actuals_mm, y_pred_mm)[0, 1] if len(actuals_mm) > 1 else np.nan
        rmse = float(np.sqrt(np.mean((actuals_mm - y_pred_mm)**2)))

        mark = "✓" if r2 >= 0.8 else "●" if r2 >= 0.5 else "↑" if r2 >= 0 else "↓"
        flush(f"    {stn:<52} {r2*100:+6.1f}% {mark}  r={corr:+.3f}")

        results.append({"river": river, "station": stn,
                         "n_train": len(X_tr), "n_test": len(X_te),
                         "r2": round(r2, 4),
                         "r":  round(float(corr), 4) if np.isfinite(corr) else None,
                         "rmse": round(rmse, 1), "status": "ok",
                         "best_alpha": round(float(ridge.alpha_), 3),
                         "predictions": [round(float(v), 1) for v in y_pred_mm]})

    keras.backend.clear_session()
    return results


# ── main ──────────────────────────────────────────────────────────────────

all_results = []
for river in ["johor", "kedah", "klang", "kuantan"]:
    all_results.extend(run_river(river))

# ── summary ───────────────────────────────────────────────────────────────

flush("\n\n" + "=" * 72)
flush("PER-STATION TEST RESULTS  (test: 2024-05 to 2025-12)")
flush("=" * 72)
flush(f"\n{'River':<10}  {'Station':<52}  {'R²':>7}  {'r':>7}  {'RMSE':>7}  N_tr")
flush("-" * 90)

for rec in sorted(all_results, key=lambda x: (x["river"], x.get("r2") or -999)):
    r2 = rec.get("r2"); r = rec.get("r")
    r2s = f"{r2*100:+.1f}%" if r2 is not None else "  N/A"
    rs  = f"{r:+.3f}" if r is not None else "  N/A"
    rms = f"{rec['rmse']:.0f}mm" if rec.get("rmse") is not None else "  N/A"
    mark= "✓" if (r2 or -1) >= 0.8 else "●" if (r2 or -1) >= 0.5 else "↑" if (r2 or -1) >= 0 else "↓"
    flush(f"  {rec['river']:<8}  {rec['station']:<52}  {r2s:>8} {mark}  {rs:>7}  {rms:>7}  {rec['n_train']}")

flush("\n\n" + "=" * 72)
flush("RIVER SUMMARY")
flush("=" * 72)
flush(f"{'River':<12}  {'N':>4}  {'Median R²':>10}  {'Max R²':>8}  {'≥0%':>5}  {'≥50%':>5}  {'≥80%':>5}")
flush("-" * 60)

ridge_best = {"kuantan": 0.807, "kedah": 0.507, "johor": 0.340, "klang": 0.189}
for river in ["johor", "kedah", "klang", "kuantan"]:
    sub = [r for r in all_results if r["river"] == river and r.get("r2") is not None]
    if not sub: continue
    r2s = [r["r2"] for r in sub]
    flush(f"  {river:<10}  {len(sub):>4}  {np.median(r2s)*100:>9.1f}%  "
          f"{max(r2s)*100:>7.1f}%  "
          f"{sum(1 for v in r2s if v>=0):>5}  "
          f"{sum(1 for v in r2s if v>=0.5):>5}  "
          f"{sum(1 for v in r2s if v>=0.8):>5}")

flush("\n── vs Phase 6 Ridge baseline ─────────────────────────────────────────")
for river in ["johor", "kedah", "klang", "kuantan"]:
    sub = [r for r in all_results if r["river"] == river and r.get("r2") is not None]
    if not sub: continue
    best = max(r["r2"] for r in sub); base = ridge_best.get(river, 0)
    delta = best - base
    sign = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "≈")
    flush(f"  {river:<10}: TCN+Ridge best={best*100:.1f}%  Ridge={base*100:.1f}%  "
          f"{sign} {abs(delta)*100:.1f}pp")

# ── save ──────────────────────────────────────────────────────────────────
out = {"model": "TCN_backbone_Ridge_head",
       "architecture": "TF TCN backbone (2 causal conv, 8 filters) → pooled embedding → per-station RidgeCV",
       "feature_detail": "TCN_emb_16d + own_ERA5x8 + river_mean_ERA5x8 + log_tp x2 + 4_seasonals + lag1 + log_lag1",
       "n_era5_features": 24, "tcn_emb_dim": EMB_DIM, "t_past": T_PAST,
       "train_end": TRAIN_END, "test_start": TEST_START, "test_end": TEST_END,
       "stations": all_results}

out_path = OUT_DIR / "phase6_tcn_ridge_hybrid_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

rows = [{"river": r["river"], "station": r["station"], "pred_mm": mm}
        for r in all_results if r.get("predictions") for mm in r["predictions"]]
pd.DataFrame(rows).to_csv(OUT_DIR / "phase6_tcn_ridge_hybrid_predictions.csv", index=False)

flush(f"\nSaved → {out_path.name}")
flush("Saved → phase6_tcn_ridge_hybrid_predictions.csv")
flush("Phase 6 TCN+Ridge hybrid complete.")
