# Data Setup Guide — External Data Sources

This project uses three external datasets that are **not included in the GitHub repository** due to file size. This guide explains how to download and place each one correctly before running any scripts or notebooks.

---

## What is NOT in the repo

| Folder / File | Size (approx) | What it is |
|---|---|---|
| `data/era5/` | ~2–5 GB | Daily ERA5 weather reanalysis files (.nc) |
| `data/imerg_monthly/` | ~500 MB | NASA IMERG monthly satellite rainfall files (.HDF5) |
| `data/raw/Permohonan Data YBTM - KE/` | varies | Raw JPS/DID station data — all 6 rivers including Padas and Sarawak |

Everything in `data/processed/` is already in the repo (pre-processed CSVs ready to use). Climate index data (`climate_indices_daily.csv`, `climate_indices_monthly.csv`) is also already included.

> **East Malaysia note (Padas & Sarawak):** The ERA5 files in `data/era5/` that were used during development only covered the Pahang bounding box. The IMERG + climate indices model was used instead for Padas and Sarawak (see Phase 7 / East Malaysia results). If you download fresh ERA5 using the wider bounding box in Section 1 below (which covers all of Malaysia including Sabah and Sarawak), ERA5 features can also be applied to East Malaysian stations.

---

## 1. ERA5 Weather Reanalysis Data

ERA5 is a global weather dataset produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). It provides daily temperature, humidity, wind, and rainfall estimates at ~28 km resolution.

### Step 1 — Register for CDS

1. Go to: **https://cds.climate.copernicus.eu**
2. Click **"Register"** and create a free account
3. After logging in, go to your profile page and copy your **API Key**

### Step 2 — Install the CDS API client

```bash
pip install cdsapi
```

### Step 3 — Create the API credentials file

Create a file named `.cdsapirc` in your home directory (`C:\Users\<your-name>\.cdsapirc` on Windows):

```
url: https://cds.climate.copernicus.eu/api
key: <paste-your-API-key-here>
```

### Step 4 — Download ERA5 data

Create a folder `data/era5/` in the project root, then run the following Python script. It downloads daily ERA5 for Malaysia from 2015 to 2025.

```python
import cdsapi
from pathlib import Path

client = cdsapi.Client()
output_dir = Path("data/era5")
output_dir.mkdir(exist_ok=True)

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_water_vapour",
    "total_precipitation",
]

# Download year by year to keep file sizes manageable
for year in range(2015, 2026):
    out_file = output_dir / f"era5_malaysia_{year}.nc"
    if out_file.exists():
        print(f"Already exists: {out_file.name}, skipping.")
        continue
    print(f"Downloading {year}...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": "12:00",          # daily noon snapshot
            "area": [8, 99, 0, 120],  # [N, W, S, E] — covers Peninsular Malaysia
            "format": "netcdf",
        },
        str(out_file),
    )
    print(f"Saved: {out_file.name}")
```

**Expected output:** 11 files named `era5_malaysia_2015.nc` through `era5_malaysia_2025.nc` in `data/era5/`.

> **Note:** Downloads may take 10–30 minutes per year depending on queue wait time. ERA5 requests are queued — the script will wait automatically.

---

## 2. NASA IMERG Monthly Satellite Rainfall

IMERG (Integrated Multi-satellitE Retrievals for GPM) is NASA's global monthly rainfall product at 0.1° resolution (~11 km). It is used in Phase 7 to provide finer-scale rainfall estimates than ERA5.

### Step 1 — Register for NASA Earthdata

1. Go to: **https://urs.earthdata.nasa.gov**
2. Click **"Register"** and create a free account
3. After registering, go to **"Applications → Authorized Apps"** and approve **"NASA GESDISC DATA ARCHIVE"**

### Step 2 — Create Earthdata credentials file

Create a file named `.netrc` in your home directory (`C:\Users\<your-name>\.netrc`):

```
machine urs.earthdata.nasa.gov
    login <your-earthdata-username>
    password <your-earthdata-password>
```

On Windows, also create `C:\Users\<your-name>\_netrc` with the same content (Windows uses `_netrc` as the fallback).

### Step 3 — Download IMERG V07B monthly files

Create a folder `data/imerg_monthly/` in the project root.

The files to download are from the **GPM_3IMERGM v07** product on NASA GES DISC:

**Product page:** https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGM_07/summary

**File naming pattern:**
```
3B-MO.MS.MRG.3IMERG.YYYYMMDD-S000000-E235959.YYYYMM.V07B.HDF5
```

You can download using this Python script (requires `requests` and a valid `.netrc`):

```python
import requests, calendar
from pathlib import Path

output_dir = Path("data/imerg_monthly")
output_dir.mkdir(exist_ok=True)

BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.07"

for year in range(2015, 2026):
    for month in range(1, 13):
        # IMERG V07B available up to approximately Sep 2025
        if year == 2025 and month > 9:
            break
        days = calendar.monthrange(year, month)[1]
        filename = (
            f"3B-MO.MS.MRG.3IMERG.{year}{month:02d}01"
            f"-S000000-E235959.{year}{month:02d}.V07B.HDF5"
        )
        url = f"{BASE_URL}/{year}/{filename}"
        out = output_dir / filename
        if out.exists():
            print(f"Already exists: {filename}, skipping.")
            continue
        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Saved ({out.stat().st_size // 1024} KB)")
        else:
            print(f"  ERROR {r.status_code}: {url}")
```

**Expected output:** ~129 HDF5 files (Jan 2015 – Sep 2025) in `data/imerg_monthly/`.

> **Tip:** If the download script fails with a 401 error, make sure your Earthdata account has approved the GES DISC application (Step 1 above) and that your `.netrc` / `_netrc` file is saved correctly.

---

## 3. Climate Indices (ONI & DMI) — Already Included

The climate index data (`data/processed/climate_indices_daily.csv` and `climate_indices_monthly.csv`) is **already committed to the repository** — you do not need to download it again.

If you need to **update it** in the future (e.g. to extend beyond 2026), here is where each index comes from:

### ONI — Oceanic Niño Index (El Niño/La Niña)

- **Source:** NOAA Climate Prediction Center
- **URL:** https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
- Monthly values; positive = El Niño (less rain in Malaysia), negative = La Niña (more rain)

```python
import pandas as pd
oni = pd.read_csv(
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    sep=r"\s+", skiprows=1,
    names=["year","jan","feb","mar","apr","may","jun",
           "jul","aug","sep","oct","nov","dec"]
)
```

### DMI — Indian Ocean Dipole (Dipole Mode Index)

- **Source:** JAMSTEC (Japan Agency for Marine-Earth Science and Technology)
- **URL:** https://www.jamstec.go.jp/aplinfo/sintexf/iod/Data/dmi.monthly.txt
- Monthly values; positive IOD = reduced rainfall over Malay Peninsula

```python
dmi = pd.read_csv(
    "https://www.jamstec.go.jp/aplinfo/sintexf/iod/Data/dmi.monthly.txt",
    sep=r"\s+", comment="#"
)
```

---

## 4. Raw Station Rainfall Data

The raw station data from JPS/DID (Department of Irrigation and Drainage Malaysia) is stored in `data/raw/`. These files are not in the repo for confidentiality reasons.

The project uses two sets of raw data:

| Dataset | File(s) | Covers | Used in |
|---|---|---|---|
| Pahang 3 stations | `BulkExport-0570061RF,0570051RF,0570021RF-*.csv` (×2) | Ldg. Nada, Sg. Lembing, Ldg. Kuala Reman | Phases 1–3 notebook |
| Multi-river 6 rivers | `Permohonan Data YBTM - KE/` folder (ZIP → folder) | Johor, Kedah, Klang, Kuantan, Padas, Sarawak (96 stations) | Phases 4–7 (processed CSVs already in repo) |

The ZIP/folder should contain six Excel files, one per river:

| File | River | Stations |
|---|---|---|
| `RF Sg. Johor 1.5.2015 - 1.6.2026.xlsx` | Sg. Johor | 18 |
| `RF Sg. Kedah 1.5.2015 - 1.6.2026.xlsx` | Sg. Kedah | 11 |
| `RF Sg. Klang 1.5.2015 - 1.6.2026.xlsx` | Sg. Klang | 19 |
| `RF Sg. Kuantan 1.5.2015 - 1.6.2026.xlsx` | Sg. Kuantan | 5 |
| `RF Sg. Padas 1.5.2015 - 1.6.2026.xlsx` | Sg. Padas (Sabah) | 3 |
| `RF Sg. Sarawak 1.5.2015 - 1.6.2026.xlsx` | Sg. Sarawak | 40 |

> **Important — Padas & Sarawak data format:** These two files use a **cumulative counter** format (not incremental like the other four). The processing notebook `01b_data_engineering_multiriver.ipynb` handles this automatically using `diff()`-based aggregation. Do not apply the same processing as the peninsular rivers.

> **Processed CSVs already in repo:** `johor_daily_raw.csv`, `kedah_daily_raw.csv`, `klang_daily_raw.csv`, `kuantan_daily_raw.csv`, `padas_daily_raw.csv`, `sarawak_daily_raw.csv` are all in `data/processed/`. You only need the raw Excel files if you want to re-run data cleaning from scratch.

---

## 5. Verify Setup

After downloading, your `data/` folder should look like this:

```
data/
├── era5/
│   ├── era5_malaysia_2015.nc
│   ├── era5_malaysia_2016.nc
│   ├── ...
│   └── era5_malaysia_2025.nc
├── imerg_monthly/
│   ├── 3B-MO.MS.MRG.3IMERG.20150101-S000000-E235959.201501.V07B.HDF5
│   ├── 3B-MO.MS.MRG.3IMERG.20150201-S000000-E235959.201502.V07B.HDF5
│   ├── ...
│   └── 3B-MO.MS.MRG.3IMERG.20250901-S000000-E235959.202509.V07B.HDF5
├── processed/       ← already in repo, no download needed
│   ├── johor_daily_raw.csv          ← 18 stations
│   ├── kedah_daily_raw.csv          ← 11 stations
│   ├── klang_daily_raw.csv          ← 19 stations
│   ├── kuantan_daily_raw.csv        ← 5 stations
│   ├── padas_daily_raw.csv          ← 3 stations (East Malaysia)
│   ├── sarawak_daily_raw.csv        ← 40 stations (East Malaysia)
│   ├── imerg_stations_monthly.csv   ← all 96 stations, 129 months
│   ├── climate_indices_daily.csv
│   ├── climate_indices_monthly.csv
│   └── all_rivers_station_meta.json ← lat/lon for all 96 stations
└── raw/
    ├── BulkExport-*.csv                      ← Pahang original data
    └── Permohonan Data YBTM - KE/            ← 6-river raw Excel files
```

Once `data/era5/` and `data/imerg_monthly/` are populated, all scripts in `scripts/` and all notebooks will run without modification — the paths are already configured.

---

## 6. Python Dependencies

Install all required packages:

```bash
pip install tensorflow scikit-learn xgboost pandas numpy h5py cdsapi requests netCDF4
```

Or if using the project virtual environment:

```bash
# Windows
.venv_tf\Scripts\activate
pip install -r requirements.txt
```
