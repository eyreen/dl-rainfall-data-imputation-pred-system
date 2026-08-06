"""
Phase 6 — Best-of Results Aggregator
For each station, picks the higher R² between Phase 6 Ridge and TCN+Ridge hybrid.
Outputs:
  predictions/phase6_best_results.json   — per-station best model + metrics
  predictions/phase6_best_predictions.csv — predictions from whichever model won
"""
import json
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PRED_DIR = BASE / "predictions"

# --- load inputs -----------------------------------------------------------
with open(PRED_DIR / "phase6_ridge_allrivers_results.json") as f:
    ridge_raw = json.load(f)

with open(PRED_DIR / "phase6_tcn_ridge_hybrid_results.json") as f:
    hybrid_raw = json.load(f)

# Build ridge lookup: station_key -> {r2, r, rmse, predictions}
ridge_by_station = {}
for key, v in ridge_raw["per_station"].items():
    # key like "johor::johor_lepau" or "kuantan::sg_cherating"
    river   = key.split("::")[0]
    station_raw = key.split("::")[-1]
    # Normalize: ensure station name carries the river prefix (matches hybrid naming)
    if not station_raw.startswith(river + "_"):
        station = river + "_" + station_raw
    else:
        station = station_raw
    ridge_by_station[station] = {
        "river": river,
        "station": station,
        "r2": v.get("r2", float("-inf")),
        "r":  v.get("r", float("nan")),
        "rmse": v.get("rmse", float("nan")),
        "n_train": v.get("n_train_months", "?"),
        "n_test": v.get("n_test_months", "?"),
        "predictions": v.get("predictions", []),
        "model": "Ridge",
    }

# Build hybrid lookup
hybrid_by_station = {}
for entry in hybrid_raw["stations"]:
    s = entry["station"]
    hybrid_by_station[s] = {
        "river": entry["river"],
        "station": s,
        "r2": entry.get("r2", float("-inf")),
        "r":  entry.get("r", float("nan")),
        "rmse": entry.get("rmse", float("nan")),
        "n_train": entry.get("n_train", "?"),
        "n_test": entry.get("n_test", "?"),
        "predictions": entry.get("predictions", []),
        "model": "TCN+Ridge",
    }

# Collect all station keys from both
all_stations = sorted(set(list(ridge_by_station.keys()) + list(hybrid_by_station.keys())))

best_stations = []
for s in all_stations:
    r = ridge_by_station.get(s)
    h = hybrid_by_station.get(s)
    if r is None and h is None:
        continue
    if r is None:
        best = dict(h, model="TCN+Ridge_only")
    elif h is None:
        best = dict(r, model="Ridge_only")
    else:
        # Both exist — pick higher R² (guard against sentinel -inf and astronomical negatives)
        r2_r = r["r2"] if r["r2"] > -1e10 else float("-inf")
        r2_h = h["r2"] if h["r2"] > -1e10 else float("-inf")
        if r2_h > r2_r:
            best = dict(h, model="TCN+Ridge")
        else:
            best = dict(r, model="Ridge")
    best_stations.append(best)

# --- river summary -----------------------------------------------------------
from collections import defaultdict
river_summary = defaultdict(lambda: {"stations": [], "best_r2": float("-inf"),
                                      "best_station": None, "n_positive": 0, "n_ge50": 0, "n_ge80": 0})
for b in best_stations:
    rv = b["river"]
    r2 = b["r2"]
    river_summary[rv]["stations"].append(b["station"])
    if r2 > -1e10:
        river_summary[rv]["n_positive"] += int(r2 > 0)
        river_summary[rv]["n_ge50"]     += int(r2 >= 0.5)
        river_summary[rv]["n_ge80"]     += int(r2 >= 0.8)
        if r2 > river_summary[rv]["best_r2"]:
            river_summary[rv]["best_r2"]     = r2
            river_summary[rv]["best_station"] = b["station"]

# --- write JSON --------------------------------------------------------------
output_json = {
    "description": "Phase 6 best-of — per-station max R² across Ridge and TCN+Ridge hybrid",
    "train_period": "2015-01 to 2024-04",
    "test_period": "2024-05 to 2025-12",
    "river_summary": {
        rv: {
            "best_station": d["best_station"],
            "best_r2": round(d["best_r2"], 4),
            "n_stations": len(d["stations"]),
            "n_positive_r2": d["n_positive"],
            "n_r2_ge50pct": d["n_ge50"],
            "n_r2_ge80pct": d["n_ge80"],
        }
        for rv, d in sorted(river_summary.items())
    },
    "per_station": [
        {
            "river": b["river"],
            "station": b["station"],
            "best_model": b["model"],
            "r2": round(b["r2"], 4) if b["r2"] > -1e10 else None,
            "r":  round(b["r"], 4),
            "rmse": round(b["rmse"], 1),
            "n_train": b["n_train"],
            "n_test":  b["n_test"],
        }
        for b in best_stations
    ]
}

out_json = PRED_DIR / "phase6_best_results.json"
with open(out_json, "w") as f:
    json.dump(output_json, f, indent=2)
print(f"Written: {out_json}")

# --- write CSV ---------------------------------------------------------------
# One row per (station, month); period column, actual, predicted, from-model
# We only write stations that have predictions
csv_rows = []
for b in best_stations:
    preds = b.get("predictions", [])
    if not preds:
        continue
    river   = b["river"]
    station = b["station"]
    model   = b["model"]
    for i, pred in enumerate(preds):
        csv_rows.append({
            "river": river,
            "station": station,
            "best_model": model,
            "test_month_idx": i + 1,
            "predicted_mm": round(pred, 1),
        })

out_csv = PRED_DIR / "phase6_best_predictions.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["river", "station", "best_model", "test_month_idx", "predicted_mm"])
    w.writeheader()
    w.writerows(csv_rows)
print(f"Written: {out_csv}")

# --- print summary table -----------------------------------------------------
print("\n=== Phase 6 Best-of Summary ===\n")
print(f"{'River':<10} {'Best station':<50} {'R²':>7} {'Model':<14}")
print("-" * 90)
for rv, d in sorted(river_summary.items()):
    best_s = d["best_station"]
    best_b = next((b for b in best_stations if b["station"] == best_s), None)
    if best_b:
        print(f"{rv:<10} {best_s:<50} {d['best_r2']*100:>6.1f}% {best_b['model']:<14}")

print("\n=== Per-station Best-of (all, sorted by R²) ===\n")
sorted_best = sorted(best_stations, key=lambda b: b["r2"] if b["r2"] > -1e10 else float("-inf"), reverse=True)
print(f"{'River':<10} {'Station':<50} {'R²':>7} {'r':>6} {'RMSE':>7} {'Model':<14}")
print("-" * 100)
for b in sorted_best:
    r2_str = f"{b['r2']*100:>6.1f}%" if b["r2"] > -1e10 else "   N/A"
    print(f"{b['river']:<10} {b['station']:<50} {r2_str} {b['r']:>6.3f} {b['rmse']:>7.1f} {b['model']:<14}")
