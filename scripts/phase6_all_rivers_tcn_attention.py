"""
Phase 6 — All-Rivers Monthly Prediction: TCN + Multi-Head Attention
Rivers: Johor (18 stations), Kedah (11), Klang (19), Kuantan (5 active)
Padas / Sarawak: ERA5 mapping not yet available — skipped.

Architecture
------------
Input  : (batch, T_PAST=6, 15 features)
TCN-1  : Conv1D dilated causal (filters=32, dilation=1) × 2 + skip
TCN-2  : Conv1D dilated causal (filters=32, dilation=2) × 2 + skip
TCN-3  : Conv1D dilated causal (filters=32, dilation=4) × 2 + skip
Attn   : MultiHeadAttention (heads=4, key_dim=8) over time axis + Add+LN
Pool   : GlobalAveragePooling1D
Head   : Dense(32, relu) → Dropout(0.15) → Dense(1) [log1p rainfall]

Features (15, consistent across all rivers):
  log_tp, era5_t2m, era5_d2m, era5_sp, era5_u10, era5_v10,
  era5_tcwv, era5_rh, era5_ws,
  sin_m, cos_m, sin_2m, cos_2m, lag1, log_lag1

Run:  T:\\.venv_tf\\Scripts\\python.exe scripts/phase6_all_rivers_tcn_attention.py
"""

import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"   # CPU only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
tf.random.set_seed(42)
np.random.seed(42)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
PRED_DIR = os.path.join(ROOT, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# ── Splits ──────────────────────────────────────────────────────────────────
TRAIN_END = pd.Period("2022-09", freq="M")
VAL_END   = pd.Period("2024-04", freq="M")
TEST_END  = pd.Period("2025-12", freq="M")
MIN_DAYS  = 15      # minimum real obs to count a month valid
T_PAST    = 6       # look-back months for each sequence

# ── River configs ────────────────────────────────────────────────────────────
RIVERS = {
    "johor":   {"imputed": "johor_daily_imputed.csv",   "era5_src": "all"},
    "kedah":   {"imputed": "kedah_daily_imputed.csv",   "era5_src": "all"},
    "klang":   {"imputed": "klang_daily_imputed.csv",   "era5_src": "all"},
    "kuantan": {"imputed": "kuantan_daily_imputed.csv", "era5_src": "kuantan"},
}

# Shared 15-feature space (generic names — same across all rivers/stations)
ERA5_GENERIC = ["era5_t2m","era5_d2m","era5_sp","era5_u10","era5_v10",
                "era5_tcwv","era5_rh","era5_ws"]
COMMON_FEATS = ["log_tp"] + ERA5_GENERIC + \
               ["sin_m","cos_m","sin_2m","cos_2m","lag1","log_lag1"]  # 15 total


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data loading — build monthly DataFrames per station
# ═══════════════════════════════════════════════════════════════════════════

def load_era5_df(river: str) -> pd.DataFrame:
    fname = ("era5_kuantan_station_mapped.csv" if river == "kuantan"
             else "era5_all_rivers_mapped.csv")
    era5 = pd.read_csv(os.path.join(PROC_DIR, fname), parse_dates=["Date"])
    return era5.ffill().bfill()


def build_monthly_all(river: str) -> dict:
    """Return {station_name: monthly_df} with COMMON_FEATS + rainfall_mm."""
    rain = pd.read_csv(os.path.join(PROC_DIR, RIVERS[river]["imputed"]),
                       parse_dates=["Date"])
    rain = rain[(rain["Date"] >= "2015-01-01") & (rain["Date"] <= "2025-12-31")].copy()
    rain["ym"] = rain["Date"].dt.to_period("M")

    stations = [c for c in rain.columns
                if c not in ("Date","ym") and not c.endswith("_imputed")]

    era5 = load_era5_df(river)
    era5["ym"] = era5["Date"].dt.to_period("M")

    era5_var_map = {
        "t2m":"era5_t2m","d2m":"era5_d2m","sp":"era5_sp",
        "u10":"era5_u10","v10":"era5_v10","tcwv":"era5_tcwv",
        "rh":"era5_rh","ws":"era5_ws","tp":"_tp_raw",
    }

    result = {}
    for sta in stations:
        flag_col = f"{sta}_imputed"
        if flag_col not in rain.columns:
            continue

        # --- monthly real-obs rainfall ---
        mrows = []
        for ym, grp in rain.groupby("ym"):
            real = grp[grp[flag_col] == 0][sta].dropna()
            total = real.sum() if len(real) >= MIN_DAYS else np.nan
            mrows.append({"ym": ym, "rainfall_mm": total})
        mdf = pd.DataFrame(mrows)

        # --- monthly ERA5 aggregation for this station ---
        sta_cols_present = {v: f"{sta}_{v}" for v in era5_var_map
                            if f"{sta}_{v}" in era5.columns}
        if "tp" not in sta_cols_present:
            continue

        tp_col = sta_cols_present["tp"]
        mean_col_map = {v: sta_cols_present[v] for v in sta_cols_present if v != "tp"}

        era5_agg = era5.groupby("ym").agg(
            **{"_tp_monthly": (tp_col, "sum")},
            **{era5_var_map[v]: (col, "mean") for v, col in mean_col_map.items()}
        ).reset_index()

        era5_agg["log_tp"] = np.log1p(np.clip(era5_agg["_tp_monthly"] * 1000, 0, None))
        era5_agg = era5_agg.drop(columns=["_tp_monthly"])

        mdf = mdf.merge(era5_agg, on="ym", how="left").sort_values("ym").reset_index(drop=True)

        # --- seasonal harmonics ---
        month = mdf["ym"].apply(lambda p: p.month).astype(float)
        mdf["sin_m"]  = np.sin(2 * np.pi * month / 12)
        mdf["cos_m"]  = np.cos(2 * np.pi * month / 12)
        mdf["sin_2m"] = np.sin(4 * np.pi * month / 12)
        mdf["cos_2m"] = np.cos(4 * np.pi * month / 12)

        # --- lag-1 ---
        mdf["lag1"]     = mdf["rainfall_mm"].shift(1).fillna(0)
        mdf["log_lag1"] = np.log1p(np.clip(mdf["lag1"], 0, None))

        # fill any remaining NaN in features
        mdf[COMMON_FEATS] = mdf[COMMON_FEATS].fillna(0)

        result[sta] = mdf

    print(f"  {river}: {len(result)} stations loaded")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sequence builder for TCN
# ═══════════════════════════════════════════════════════════════════════════

def make_sequences(mdf: pd.DataFrame, split: str):
    """
    split: 'train' | 'val' | 'test'
    Returns X (N, T_PAST, 15), y_log (N,), y_mm (N,), periods (N,)
    """
    if split == "train":
        mask = mdf["ym"] <= TRAIN_END
    elif split == "val":
        mask = (mdf["ym"] > TRAIN_END) & (mdf["ym"] <= VAL_END)
    else:
        # include T_PAST months of context before test window
        mask = mdf["ym"] <= TEST_END   # filter to test after building sequences

    rows = mdf[mask].reset_index(drop=True) if split != "test" else mdf.reset_index(drop=True)

    X_seq, y_log, y_mm_list, periods = [], [], [], []
    for i in range(T_PAST, len(rows)):
        ym_i = rows.iloc[i]["ym"]
        if split == "test" and not (VAL_END < ym_i <= TEST_END):
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
        return (np.empty((0, T_PAST, len(COMMON_FEATS)), dtype="float32"),
                np.array([]), np.array([]), [])
    return np.array(X_seq), np.array(y_log), np.array(y_mm_list), periods


# ═══════════════════════════════════════════════════════════════════════════
# 3. TCN + Attention model
# ═══════════════════════════════════════════════════════════════════════════

def tcn_block(x, filters, kernel_size, dilation, dropout=0.1):
    skip = x
    for _ in range(2):
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding="same")(skip)
    return layers.Add()([x, skip])


def build_tcn_attention_model(t_past: int, n_feats: int,
                               filters=32, n_heads=4, key_dim=8,
                               dense_units=32, dropout=0.15):
    inp = keras.Input(shape=(t_past, n_feats), name="monthly_seq")
    x = tcn_block(inp, filters, kernel_size=3, dilation=1)
    x = tcn_block(x,   filters, kernel_size=3, dilation=2)
    x = tcn_block(x,   filters, kernel_size=3, dilation=4)
    # temporal self-attention: each month weighs all past months
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim,
                                     name="temporal_attention")(x, x)
    x = layers.Add(name="attn_residual")([x, attn])
    x = layers.LayerNormalization(name="attn_norm")(x)
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, name="pred")(x)
    return keras.Model(inp, out, name="TCN_Attention")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Train + evaluate
# ═══════════════════════════════════════════════════════════════════════════

def run(all_mdfs: dict):
    """all_mdfs = {(river, sta): mdf}"""
    print("\nBuilding sequences (pooled across all rivers)...")

    tr_X, tr_y = [], []
    vl_X, vl_y = [], []
    test_sets   = {}   # (river, sta) -> (X_te, y_mm_te, periods_te)

    for (river, sta), mdf in all_mdfs.items():
        Xtr, ytr, _, _       = make_sequences(mdf, "train")
        Xvl, yvl, _, _       = make_sequences(mdf, "val")
        Xte, _, y_mm_te, p_te = make_sequences(mdf, "test")

        if len(Xtr) > 0: tr_X.append(Xtr); tr_y.append(ytr)
        if len(Xvl) > 0: vl_X.append(Xvl); vl_y.append(yvl)
        if len(Xte) >= 3:
            test_sets[(river, sta)] = (Xte, y_mm_te, p_te)

    Xtr_all = np.vstack(tr_X);  ytr_all = np.concatenate(tr_y)
    Xvl_all = np.vstack(vl_X);  yvl_all = np.concatenate(vl_y)

    print(f"  Train sequences : {len(Xtr_all)}")
    print(f"  Val   sequences : {len(Xvl_all)}")
    print(f"  Test  stations  : {len(test_sets)}")
    print(f"  Features        : {Xtr_all.shape[2]}  |  Lookback: {T_PAST} months")

    # build model
    model = build_tcn_attention_model(T_PAST, Xtr_all.shape[2])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    model.summary(print_fn=lambda s: print(" ", s))

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                      restore_best_weights=True, min_delta=1e-5),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10,
                                          factor=0.5, min_lr=1e-5),
    ]

    print("\nTraining TCN+Attention model...")
    hist = model.fit(
        Xtr_all, ytr_all,
        validation_data=(Xvl_all, yvl_all),
        epochs=300, batch_size=64,
        callbacks=cbs, verbose=1,
    )
    best_ep = int(np.argmin(hist.history["val_loss"])) + 1
    best_vl = float(min(hist.history["val_loss"]))
    print(f"\n  Stopped at epoch {len(hist.history['val_loss'])} | "
          f"best epoch {best_ep} | best val_loss {best_vl:.4f}")

    # save model
    model_path = os.path.join(ROOT, "models", "phase6_tcn_attention.keras")
    model.save(model_path)
    print(f"  Model saved → models/phase6_tcn_attention.keras")

    # ── Per-station evaluation ────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("Per-Station Test Results  (19-month test: May 2024 – Dec 2025)")
    print("=" * 75)

    results   = {}
    pred_rows = []
    by_river  = {}

    for (river, sta), (Xte, y_mm_te, p_te) in sorted(test_sets.items()):
        p_log  = model.predict(Xte, verbose=0).flatten()
        p_mm   = np.expm1(np.clip(p_log, 0, 8))
        r2     = float(r2_score(y_mm_te, p_mm))
        rmse   = float(np.sqrt(np.mean((y_mm_te - p_mm) ** 2)))
        r_corr = float(np.corrcoef(y_mm_te, p_mm)[0, 1]) if len(y_mm_te) > 2 else 0.0

        results[(river, sta)] = {"r2": r2, "r": r_corr, "rmse_mm": rmse,
                                 "n_test": int(len(y_mm_te))}
        by_river.setdefault(river, []).append(r2)

        for period, actual, pred in zip(p_te, y_mm_te, p_mm):
            pred_rows.append({"river": river, "station": sta, "period": str(period),
                              "actual_mm": float(actual), "pred_mm": float(pred)})

    # print summary per river
    print(f"\n{'River':<10} {'Station':<48} {'R²':>7} {'r':>6} {'RMSE':>8}")
    print("-" * 75)
    for (river, sta), res in sorted(results.items()):
        tag = " ✓" if res["r2"] >= 0.80 else ("  " if res["r2"] >= 0.50 else "  ↓")
        print(f"  {river:<10} {sta:<46} {res['r2']*100:>+.1f}%{tag}  "
              f"{res['r']:>5.3f}  {res['rmse_mm']:>6.0f}mm")

    # river summary
    print("\n" + "=" * 75)
    print("River Summary")
    print("=" * 75)
    print(f"{'River':<12} {'Stations':>9} {'Median R²':>11} {'Mean R²':>9} {'Max R²':>9}")
    print("-" * 55)
    river_summary = {}
    for river, r2s in sorted(by_river.items()):
        valid = [v for v in r2s if not np.isnan(v)]
        med   = np.median(valid) * 100
        mean  = np.mean(valid)   * 100
        mx    = np.max(valid)    * 100
        print(f"  {river:<12} {len(valid):>9}   {med:>+.1f}%      {mean:>+.1f}%    {mx:>+.1f}%")
        river_summary[river] = {"n_stations": len(valid),
                                "median_r2": float(med/100),
                                "mean_r2":   float(mean/100),
                                "max_r2":    float(mx/100)}

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "description": "Phase 6 — all rivers monthly TCN+Attention (pooled training)",
        "model": "TCN (dilated causal conv, 3 blocks) + Multi-Head Self-Attention",
        "architecture": {
            "tcn_blocks": 3, "dilations": [1, 2, 4], "filters": 32,
            "attention_heads": 4, "key_dim": 8,
            "t_past_months": T_PAST, "n_features": int(Xtr_all.shape[2]),
            "feature_names": COMMON_FEATS,
        },
        "training": {
            "train_sequences": int(len(Xtr_all)),
            "val_sequences":   int(len(Xvl_all)),
            "best_epoch": best_ep,
            "best_val_loss": best_vl,
        },
        "test_period": "2024-05 to 2025-12",
        "river_summary": river_summary,
        "per_station": {
            f"{rv}::{sta}": v for (rv, sta), v in results.items()
        },
    }

    res_path  = os.path.join(PRED_DIR, "phase6_tcn_attention_results.json")
    pred_path = os.path.join(PRED_DIR, "phase6_tcn_attention_predictions.csv")
    with open(res_path, "w") as f:
        json.dump(output, f, indent=2)
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    print(f"\nSaved → {res_path}")
    print(f"Saved → {pred_path}")
    print("\nNote: Padas & Sarawak require ERA5 mapping before they can be modelled.")
    print("Phase 6 complete.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 75)
    print("Phase 6 — All-Rivers Monthly: TCN + Multi-Head Self-Attention")
    print(f"TensorFlow {tf.__version__}  |  T_PAST={T_PAST}  |  Features={len(COMMON_FEATS)}")
    print("=" * 75)

    all_mdfs = {}
    for river in RIVERS:
        print(f"\nLoading {river}...")
        for sta, mdf in build_monthly_all(river).items():
            all_mdfs[(river, sta)] = mdf

    total = len(all_mdfs)
    print(f"\nTotal stations: {total}  across {len(RIVERS)} rivers")

    run(all_mdfs)
