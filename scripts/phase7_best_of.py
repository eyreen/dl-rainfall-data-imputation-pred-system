"""
Phase 7 — Best-of Aggregator
Combines Phase 6 best-of (Ridge + TCN+Ridge hybrid) with Phase 7 enhanced Ridge.
For each station, keeps the higher R².

Note on test periods:
  Phase 6: May 2024 – Dec 2025 (19–20 months)
  Phase 7: May 2024 – Sep 2025 (13–17 months, IMERG cutoff)

Run: python scripts/phase7_best_of.py
"""

import json, re
from pathlib import Path
from collections import defaultdict

BASE     = Path(__file__).resolve().parent.parent
PRED_DIR = BASE / "predictions"

# ── load Phase 6 best-of ──────────────────────────────────────────────────────
with open(PRED_DIR / "phase6_best_results.json") as f:
    p6 = json.load(f)

p6_by_station = {}
for entry in p6["per_station"]:
    s = entry["station"]
    p6_by_station[s] = entry

# ── load Phase 7 Ridge enhanced ───────────────────────────────────────────────
with open(PRED_DIR / "phase7_ridge_enhanced_results.json") as f:
    p7 = json.load(f)

p7_by_station = {}
for entry in p7["stations"]:
    s = entry["station"]
    p7_by_station[s] = entry

# ── merge ─────────────────────────────────────────────────────────────────────
all_stations = sorted(set(list(p6_by_station.keys()) + list(p7_by_station.keys())))

best_stations = []
comparison_rows = []

for s in all_stations:
    p6e = p6_by_station.get(s)
    p7e = p7_by_station.get(s)

    river = (p6e or p7e)["river"]

    r2_p6 = (p6e["r2"] if p6e and p6e["r2"] is not None else float("-inf"))
    r2_p7 = (p7e["r2"] if p7e and p7e["r2"] is not None else float("-inf"))

    # guard against astronomical negatives from offline stations
    if r2_p6 < -1e10: r2_p6 = float("-inf")
    if r2_p7 < -1e10: r2_p7 = float("-inf")

    if r2_p7 > r2_p6:
        winner = dict(p7e, best_phase="Phase7_Ridge_Enhanced", best_model=p7e.get("best_alpha","Ridge+IMERG+CI"))
        win_tag = "P7"
    else:
        winner = dict(p6e, best_phase="Phase6", best_model=p6e.get("best_model","Ridge"))
        win_tag = "P6"

    best_stations.append(winner)

    comparison_rows.append({
        "river": river,
        "station": s,
        "p6_r2": round(r2_p6 * 100, 1) if r2_p6 > -1e10 else None,
        "p7_r2": round(r2_p7 * 100, 1) if r2_p7 > -1e10 else None,
        "best_r2": round(max(r2_p6, r2_p7) * 100, 1) if max(r2_p6, r2_p7) > -1e10 else None,
        "winner": win_tag,
        "delta_pp": round((max(r2_p6, r2_p7) - r2_p6) * 100, 1) if r2_p6 > -1e10 else None,
    })

# ── print comparison table ─────────────────────────────────────────────────────
print("=== Phase 6 vs Phase 7 — Station Comparison ===\n")
print(f"{'Station':<55}  {'P6 R²':>7}  {'P7 R²':>7}  {'Best':>7}  {'Δ pp':>6}  {'Win'}")
print("-" * 95)

for row in sorted(comparison_rows, key=lambda x: x["best_r2"] or -999, reverse=True):
    p6_str = f"{row['p6_r2']:6.1f}%" if row["p6_r2"] is not None else "    N/A"
    p7_str = f"{row['p7_r2']:6.1f}%" if row["p7_r2"] is not None else "    N/A"
    be_str = f"{row['best_r2']:6.1f}%" if row["best_r2"] is not None else "    N/A"
    de_str = f"{row['delta_pp']:+5.1f}" if row["delta_pp"] is not None else "  N/A"
    print(f"  {row['station']:<53}  {p6_str}  {p7_str}  {be_str}  {de_str}  {row['winner']}")

# ── river summary ─────────────────────────────────────────────────────────────
print("\n\n=== Best-of River Summary (Phase 6 ∪ Phase 7) ===\n")
from collections import defaultdict
river_agg = defaultdict(list)
for b in best_stations:
    r2 = b.get("r2")
    if r2 is None or r2 < -1e10:
        continue
    river_agg[b["river"]].append((r2, b["station"], b.get("best_phase","?")))

for river in ["kuantan","klang","kedah","johor"]:
    entries = sorted(river_agg[river], reverse=True)
    if not entries:
        continue
    best_r2, best_sta, best_phase = entries[0]
    n_pos = sum(1 for r2, _, _ in entries if r2 > 0)
    n_50  = sum(1 for r2, _, _ in entries if r2 >= 0.5)
    n_80  = sum(1 for r2, _, _ in entries if r2 >= 0.8)
    print(f"{river.upper():10s}  best={best_sta}  R²={best_r2*100:.1f}%  [{best_phase}]")
    print(f"           positive={n_pos}  ≥50%={n_50}  ≥80%={n_80}")
    print(f"           Top 3:")
    for r2, sta, phase in entries[:3]:
        print(f"             {sta}  {r2*100:.1f}%  [{phase}]")
    print()

# ── save ─────────────────────────────────────────────────────────────────────
output = {
    "description": "Phase 7 final best-of — max R² per station across Phase 6 (Ridge + TCN+Ridge) and Phase 7 (Ridge + IMERG + climate)",
    "note": "Phase 6 test: May 2024–Dec 2025 (19–20 months). Phase 7 test: May 2024–Sep 2025 (13–17 months).",
    "per_station": [
        {
            "river": b["river"],
            "station": b["station"],
            "best_phase": b.get("best_phase","?"),
            "best_model": str(b.get("best_model","?")),
            "r2": round(b["r2"], 4) if b.get("r2") and b["r2"] > -1e10 else None,
            "r":  round(b["r"], 4)  if b.get("r") else None,
            "rmse": round(b.get("rmse",0), 1),
        }
        for b in best_stations if b.get("r2") and b["r2"] > -1e10
    ]
}

out_json = PRED_DIR / "phase7_final_best_results.json"
with open(out_json, "w") as f:
    json.dump(output, f, indent=2)
print(f"Written: {out_json}")
