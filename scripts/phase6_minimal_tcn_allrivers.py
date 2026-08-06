"""
Phase 6 — Per-River Minimal TCN + Attention
Trains on ALL pre-test data (2015-01 to 2024-04, no val split).
Uses weight decay + cosine LR decay as implicit regularization.
Much smaller model: 1 TCN block × 8 filters (~3K params) to match data size.

Run:
  $env:TF_CPP_MIN_LOG_LEVEL="3"; $env:CUDA_VISIBLE_DEVICES="-1"; $env:TF_ENABLE_ONEDNN_OPTS="0"
  T:\\.venv_tf\\Scripts\\python.exe -W ignore "T:\\scripts\\phase6_minimal_tcn_allrivers.py"
"""

import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
tf.random.set_seed(42)
np.random.seed(42)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
PRED_DIR = os.path.join(ROOT, "predictions")
MDL_DIR  = os.path.join(ROOT, "models")
os.makedirs(PRED_DIR, exist_ok=True)

TRAIN_END = pd.Period("2024-04", freq="M")   # use ALL pre-test data for training
TEST_END  = pd.Period("2025-12", freq="M")
TEST_START = pd.Period("2024-05", freq="M")
MIN_DAYS  = 15
T_PAST    = 6

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


def make_sequences(mdf, include_test=False):
    rows = mdf.reset_index(drop=True)
    X_seq, y_log, y_mm_list, periods = [], [], [], []
    for i in range(T_PAST, len(rows)):
        ym_i = rows.iloc[i]["ym"]
        if include_test:
            if ym_i > TEST_END:
                continue
        else:
            if ym_i > TRAIN_END:
                continue
        target = rows.iloc[i]["rainfall_mm"]
        if pd.isna(target):
            continue
        seq = rows.iloc[i-T_PAST:i][COMMON_FEATS].values.astype("float32")
        if seq.shape[0] != T_PAST:
            continue
        X_seq.append(seq)
        y_log.append(np.log1p(max(target, 0)))
        y_mm_list.append(target)
        periods.append(ym_i)

    if not X_seq:
        return (np.empty((0, T_PAST, len(COMMON_FEATS)), "float32"),
                np.array([]), np.array([]), [])
    return np.array(X_seq), np.array(y_log), np.array(y_mm_list), periods


def build_minimal_tcn(t_past, n_feats, filters=8, key_dim=4, dropout=0.2, l2=1e-4):
    """Minimal TCN: 1 block × 8 filters + lightweight attention."""
    reg = regularizers.l2(l2)
    inp = keras.Input(shape=(t_past, n_feats), name="monthly_seq")
    # single TCN block, dilation=1
    skip = layers.Conv1D(filters, 1, padding="same", kernel_regularizer=reg)(inp)
    x = layers.Conv1D(filters, 3, padding="causal", dilation_rate=1,
                      activation="relu", kernel_regularizer=reg)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, 3, padding="causal", dilation_rate=2,
                      activation="relu", kernel_regularizer=reg)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, skip])
    # lightweight self-attention (1 head)
    attn = layers.MultiHeadAttention(num_heads=1, key_dim=key_dim,
                                     kernel_regularizer=reg,
                                     name="temporal_attention")(x, x)
    x = layers.Add(name="attn_res")([x, attn])
    x = layers.LayerNormalization(name="attn_norm")(x)
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=reg, name="fc1")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, name="pred")(x)
    return keras.Model(inp, out, name="MinimalTCN_Attn")


def cosine_lr(epoch, lr, total_epochs=120, min_lr=1e-5, initial_lr=3e-3):
    progress = epoch / total_epochs
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train_river(river, river_mdfs):
    TOTAL_EPOCHS = 120

    tr_X, tr_y = [], []
    test_sets   = {}

    for sta, mdf in river_mdfs.items():
        Xtr, ytr, _, _ = make_sequences(mdf, include_test=False)
        Xte, _, y_mm_te, p_te = make_sequences(mdf, include_test=True)

        # filter test sequences to actual test window
        test_idx = [i for i, p in enumerate(p_te) if p >= TEST_START]
        if not test_idx:
            continue
        Xte_t = Xte[test_idx]
        y_mm_t = np.array(y_mm_te)[test_idx]
        p_te_t = [p_te[i] for i in test_idx]

        if len(Xtr) > 0: tr_X.append(Xtr); tr_y.append(ytr)
        if len(Xte_t) >= 3:
            test_sets[sta] = (Xte_t, y_mm_t, p_te_t)

    if not tr_X:
        print(f"  [{river}] No training data — skipping.")
        return {}, []

    Xtr_all = np.vstack(tr_X)
    ytr_all = np.concatenate(tr_y)

    print(f"\n  Train: {len(Xtr_all)} seqs | Test stations: {len(test_sets)}")

    model = build_minimal_tcn(T_PAST, Xtr_all.shape[2])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3), loss="mse")

    # print param count
    total_params = model.count_params()
    print(f"  Model params: {total_params:,}")

    lr_sched = keras.callbacks.LearningRateScheduler(
        lambda ep, lr: cosine_lr(ep, lr, TOTAL_EPOCHS), verbose=0)

    model.fit(
        Xtr_all, ytr_all,
        epochs=TOTAL_EPOCHS, batch_size=32,
        callbacks=[lr_sched], verbose=0,
    )
    print(f"  Trained {TOTAL_EPOCHS} epochs (cosine LR, no early stopping)")

    model_path = os.path.join(MDL_DIR, f"phase6_{river}_mintcn.keras")
    model.save(model_path)
    print(f"  Saved → models/phase6_{river}_mintcn.keras")

    station_results = {}
    pred_rows = []
    for sta, (Xte, y_mm_te, p_te) in sorted(test_sets.items()):
        p_log  = model.predict(Xte, verbose=0).flatten()
        p_mm   = np.expm1(np.clip(p_log, 0, 8))
        r2     = float(r2_score(y_mm_te, p_mm))
        rmse   = float(np.sqrt(np.mean((y_mm_te - p_mm) ** 2)))
        r_corr = float(np.corrcoef(y_mm_te, p_mm)[0, 1]) if len(y_mm_te) > 2 else 0.0

        station_results[sta] = {"r2": r2, "r": r_corr, "rmse_mm": rmse,
                                "n_test": int(len(y_mm_te))}
        for period, actual, pred in zip(p_te, y_mm_te, p_mm):
            pred_rows.append({"river": river, "station": sta, "period": str(period),
                              "actual_mm": float(actual), "pred_mm": float(pred)})

    tf.keras.backend.clear_session()
    return station_results, pred_rows


if __name__ == "__main__":
    print("=" * 72)
    print("Phase 6 — Per-River Minimal TCN+Attention (all pre-test training)")
    print(f"TF {tf.__version__} | T_PAST={T_PAST} | Features={len(COMMON_FEATS)}")
    print("Train period: 2015-01 to 2024-04 | Test: 2024-05 to 2025-12")
    print("=" * 72)

    all_results  = {}
    all_pred_rows = []
    river_summary = {}

    for river in RIVERS:
        print(f"\n{'─'*72}")
        print(f"  River: {river.upper()}")
        print(f"{'─'*72}")
        print(f"  Loading {river}...")
        river_mdfs = build_monthly(river)
        print(f"  {river}: {len(river_mdfs)} stations loaded")

        sta_results, pred_rows = train_river(river, river_mdfs)
        if not sta_results:
            continue

        all_results[river] = sta_results
        all_pred_rows.extend(pred_rows)

        r2s   = [v["r2"] for v in sta_results.values()]
        valid = [v for v in r2s if not np.isnan(v)]
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
    print("PER-STATION RESULTS")
    print("=" * 72)
    print(f"\n{'River':<10} {'Station':<48} {'R²':>7} {'r':>6} {'RMSE':>8}")
    print("-" * 75)
    for river, sta_results in sorted(all_results.items()):
        for sta, res in sorted(sta_results.items()):
            tag = " ✓" if res["r2"] >= 0.80 else (" ●" if res["r2"] >= 0.50 else ("  " if res["r2"] >= 0 else "  ↓"))
            print(f"  {river:<10} {sta:<46} {res['r2']*100:>+.1f}%{tag}  "
                  f"{res['r']:>+.3f}  {res['rmse_mm']:>6.0f}mm")

    print("\n\n" + "=" * 72)
    print("RIVER SUMMARY")
    print("=" * 72)
    print(f"{'River':<12} {'N':>4} {'Median R²':>11} {'Max R²':>9} {'≥0%':>5} {'≥50%':>6} {'≥80%':>6}")
    print("-" * 55)
    for river, rs in sorted(river_summary.items()):
        print(f"  {river:<12} {rs['n_stations']:>4}   "
              f"{rs['median_r2']*100:>+.1f}%     {rs['max_r2']*100:>+.1f}%   "
              f"{rs['n_positive_r2']:>3}   {rs['n_r2_ge50pct']:>4}   {rs['n_r2_ge80pct']:>4}")

    out = {
        "description": "Phase 6 — per-river minimal TCN+Attention, trained on all pre-test data",
        "model": "Minimal TCN (1 block, 8 filters, dilation 1+2) + 1-head self-attention",
        "training_period": "2015-01 to 2024-04 (no val split; cosine LR, L2 regularization)",
        "test_period": "2024-05 to 2025-12",
        "river_summary": river_summary,
        "per_station": {
            f"{river}::{sta}": v
            for river, sta_results in all_results.items()
            for sta, v in sta_results.items()
        },
    }
    res_path  = os.path.join(PRED_DIR, "phase6_mintcn_results.json")
    pred_path = os.path.join(PRED_DIR, "phase6_mintcn_predictions.csv")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(all_pred_rows).to_csv(pred_path, index=False)
    print(f"\nSaved → predictions/phase6_mintcn_results.json")
    print(f"Saved → predictions/phase6_mintcn_predictions.csv")
    print("Phase 6 complete.")
