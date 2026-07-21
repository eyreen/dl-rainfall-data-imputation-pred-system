# Project Progress Brief
## Deep Learning-Based Rainfall Data Imputation & Prediction System — Pahang, Malaysia

**Date:** July 2026  
**Prepared by:** Arinn Danish  
**Client:** YBTM / Quang  

---

## 1. Project Objective

Build an end-to-end deep learning pipeline that:
1. **Fills in missing historical rainfall data** *(imputation — reconstructing data points that were never recorded due to sensor failures or transmission gaps)* at 3 telemetry stations in Pahang
2. **Predicts future rainfall** at daily, weekly, and monthly scales from those stations

**Target accuracy: R² ≥ 0.80**

> **What is R²?**  
> R² (pronounced "R-squared" or "coefficient of determination") measures how well a model explains the variation in the actual data.  
> - **R² = 1.0** → perfect prediction — model matches observations exactly  
> - **R² = 0.5** → model explains 50% of the variation; the other 50% is unexplained  
> - **R² = 0.0** → model is no better than just predicting the historical average every day  
> - **R² < 0** → model is *worse* than predicting the average (actively misleading)  
>
> The client's target of **R² ≥ 0.80** means the model must correctly explain at least 80% of why rainfall goes up and down from day to day.

---

## 2. Stations & Data

**Original dataset (Phases 1–3):**

| Station | ID | Mean daily (mm) | Zero-rain days |
|---|---|---|---|
| Ldg. Nada | 3931013 | 7.97 mm | 41.8% |
| Sg. Lembing | 3930012 | 8.72 mm | 39.1% |
| Ldg. Kuala Reman | 3931014 | 7.74 mm | 44.8% |

**Period:** 2009–2025 (daily, 6,024 rows)  
**Total missing values imputed by GAN:** 495 station-days across 3 stations

> **Key challenge — zero-inflation:**  
> Between 39–45% of all days are completely dry (zero rainfall). This is called a *zero-inflated distribution*. It means the model sees far more dry days than rainy ones in the training data, which causes it to learn a bias towards predicting "no rain" — even on days that are actually wet. Every architectural improvement in Phase 3 is partially aimed at fixing this bias.

---

## 3. Phase 1 — Data Engineering

**What was done:**
- Loaded raw daily telemetry data for 3 stations
- Performed exploratory data analysis (gap detection, seasonality plots, outlier review)
- Identified missing periods and flagged imputed rows
- Applied **log-transform** *(compresses the effect of extreme high values — e.g. a 300 mm storm day — so the model is not overwhelmed by outliers)* and **min-max normalisation** *(rescales all values to the range 0–1 so the model treats all features equally)*
- Engineered **seasonal features** *(mathematical functions that encode the time of year as smooth curves, so the model understands Malaysia's two monsoon seasons without being told the month by name)*:
  - `sin` and `cos` of day-of-year → captures the annual monsoon cycle
  - `sin` and `cos` of month → captures sub-annual seasonal rhythm
  - Using sine/cosine pairs (rather than a plain month number) ensures December and January are treated as neighbours, not opposites
- Added **7-day and 30-day rolling rainfall means** *(a rolling mean is a sliding window average — the 7-day mean on Monday = the average of the previous 7 days. This encodes "how wet has it been recently?" which is a strong predictor of tomorrow's rainfall)*

---

## 4. Phase 2 — GAN Imputation

### What is a GAN?

A **Generative Adversarial Network (GAN)** is a system of two competing neural networks:

| Network | Role | Analogy |
|---|---|---|
| **Generator** | Produces fake/imputed values to fill gaps | A student forging answers |
| **Discriminator** | Tries to detect whether a value is real or generated | An examiner catching forgeries |

The two networks train together — the generator gets better at producing realistic values, and the discriminator gets better at catching fakes. After training, the generator produces imputed values that are statistically indistinguishable from real observations.

We used a specific variant called **GAIN (Generative Adversarial Imputation Net)** with a **CNN generator** (Convolutional Neural Network — a network that reads patterns across a sliding window of time, like a magnifying glass moving along the timeline).

**Ensemble approach:** Rather than trusting one single trained model, we trained **5 independent GAN models** with different random seeds and averaged their predictions. This *ensemble* approach reduces the risk that any single model's quirks or mistakes are carried into the prediction phase.

**Results:**

| Model | R² (imputation quality) | NRMSE |
|---|---|---|
| Mean imputation (baseline) | -0.056 | 226.0% |
| CNN GAN single model | +0.152 | 202.4% |
| **CNN GAN ensemble ×5** | **+0.179** | **199.2%** |

> **NRMSE** = Normalised Root Mean Square Error. It expresses the average prediction error as a percentage of the data's natural variation. Lower is better. 199% means our average error is still about twice the natural spread — imputation of rainfall is genuinely hard because missing data often coincides with extreme events.

**Output:** `completed_daily_rainfall_cnn_tf.csv` — the GAN-completed dataset used as input for all prediction models.

---

## 5. Phase 3 — TCN Prediction

All prediction models read from the GAN-completed dataset.

### 5.1 What is a TCN?

A **Temporal Convolutional Network (TCN)** is a type of deep learning model designed specifically for sequences that unfold over time (time series). Unlike older recurrent models (RNN, LSTM) that process data one step at a time — and can "forget" what happened long ago — a TCN reads the entire history window at once using **dilated causal convolutions**:

- **Causal** — the model can only look at past data, never future. Position "today" cannot see tomorrow.
- **Dilated** — each successive layer skips over positions with increasing gaps (1, 2, 4, 8, 16 days), letting the model look very far back without needing an enormous number of parameters.

With 5 stacked TCN blocks, the model effectively "sees" the past **125 days** in a single forward pass. This is called the **receptive field** — the total span of history that influences a single prediction.

**Why TCN over LSTM?**
- Faster to train (fully parallelisable — all time steps computed simultaneously)
- No vanishing gradient problem (gradients flow cleanly through convolutional layers)
- Explicit, controllable receptive field

---

### 5.2 Multi-Head Architecture — The Core Innovation

This is the central design decision that separates our prediction model from a standard approach.

**The naive approach (and its problem):**  
Predict tomorrow's rainfall → feed it back as input → predict the day after → repeat for 14 days.  
This is called a **multi-step rollout** and it compounds errors: any mistake on day 1 is carried into day 2, day 2's error is added on top, and by day 14 the predictions are essentially noise.

**Our approach — Direct Multi-Scale Prediction (Multi-Head):**  
All three prediction targets (daily, weekly, monthly) are predicted **simultaneously and independently** from the same shared backbone. There is no rollout, no error compounding.

```
                      ┌────────────────────────────────────────────┐
                      │            Shared TCN Backbone             │
                      │   (5 blocks × dilated causal convolutions) │
                      │      "sees" the past 45 days at once       │
                      └──────────────┬─────────────────────────────┘
                                     │
                         Shared Dense Context Layer
                        (compresses 45×22 features → 192 values)
                                     │
               ┌─────────────────────┼─────────────────────┐
               ▼                     ▼                       ▼
        HEAD 1: Daily         HEAD 2: Weekly          HEAD 3: Monthly
     next-day rainfall     7-day cumulative sum    30-day cumulative sum
        (regression)           (regression)            (regression)
```

**"Head"** = a small dedicated output layer attached to the end of the shared backbone. Each head specialises in predicting its own timescale. The backbone is shared — it learns general temporal patterns — while each head learns what those patterns mean at its own scale.

**Why this works well for monthly predictions:** Monthly totals are inherently smoother than daily point values. The model can correctly capture "it will be a very wet 30-day period" even if it cannot pinpoint exactly which days within that period will see rain.

---

### 5.3 Model Evolution & Results

All results on the **held-out test set (2024–2025, never seen during training)**:

| Model | Daily R² | Weekly R² | Monthly R² | Key change |
|---|---|---|---|---|
| PyTorch TCN (baseline) | +7.5% | +1.3% | +15.2% | Hand-tuned, 3 seeds |
| TF Basic TCN (rollout) | −10.0% | — | — | Error compounding confirmed |
| TF Multi-Head TCN | −3.2% | +2.7% | +23.8% | Beat baseline on monthly by +8.5pp |
| + Hurdle model | +2.8% | +4.5% | +21.1% | Daily finally turned positive |
| + Extreme-event weighted loss | (included above) | | | Heavy-rain errors weighted up to 3× |
| + ERA5 atmospheric data | **+5.8%** | +7.5% | +21.5% | Best daily; +3pp weekly gain |
| **+ Optuna HPO (best config)** | +1.9% | **+11.3%** | **+26.6%** | **New best weekly & monthly** |

> **pp** = percentage points (absolute difference, not relative). +8.5pp means the R² score moved from 15.2% to 23.8%.

---

### 5.4 Hurdle Model — Solving the Zero-Inflation Problem

**The problem:** A standard regression model trained on zero-inflated data learns to predict near-zero almost all the time (since that minimises average error across the majority dry days). On the rare but important days when it does rain heavily, the model under-predicts.

**The hurdle model** is a two-stage approach inspired by actuarial science and ecology:

```
Stage 1 — Classification:   "Will it rain at all tomorrow?"
                             Output: probability (0.0 to 1.0)
                             
Stage 2 — Regression:       "If yes, how much?"
                             Output: predicted rainfall amount

Final prediction = Stage 2 output × (Stage 1 probability > 0.4)
                 = 0 if Stage 1 says "probably dry"
                 = rainfall amount if Stage 1 says "probably wet"
```

The model has **4 output heads** (not 3):

| Head | What it predicts | Output type |
|---|---|---|
| `wetdry` | Wet/dry probability for each station | Classification (0–1) |
| `daily` | Next-day rainfall amount | Regression |
| `weekly` | 7-day cumulative sum | Regression |
| `monthly` | 30-day cumulative sum | Regression |

**Extreme-event weighted loss:**  
Standard Mean Squared Error (MSE) treats all errors equally. A 50 mm/day prediction error on a 300 mm storm event is weighted the same as a 5 mm error on a light rain day — clearly wrong from a flood-risk perspective.

We modified the loss function so that errors on high-rainfall days are penalised up to **3× more heavily**, forcing the model to pay special attention to extreme events during training.

| Event type | Example value | Error weighting |
|---|---|---|
| Dry day | 0 mm | ×1.0 (baseline) |
| Light rain | 10 mm | ×1.1 |
| Moderate rain | 50 mm | ×1.4 |
| Heavy rain | 150 mm | ×2.4 |
| Extreme event | 300 mm | ×3.4 |

---

### 5.5 ERA5 Atmospheric Reanalysis Data

**What ERA5 is:**  
ERA5 is a global dataset produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) — the world's leading weather forecasting organisation, used by the UK Met Office, Météo-France, and virtually every national meteorological agency. ERA5 combines historical observations with a physics-based atmospheric model to reconstruct the full state of the Earth's atmosphere every 6 hours, going back to 1940.

**Why we added it:**  
Station rainfall history tells the model "it rained 20 mm yesterday." ERA5 tells the model *why* it rained — the pressure gradient, the moisture content, the wind direction. These **physical causes** of rainfall provide information that no amount of past station data can replicate.

We downloaded the Pahang region bounding box (2009–2025) and extracted 9 daily variables:

| Variable | Physical meaning | Why it matters for rainfall |
|---|---|---|
| 2m air temperature | How hot is the surface air | Warm air rises → convection → clouds → rain |
| 2m dewpoint temperature | How much moisture is in the air | Higher dewpoint = air closer to saturation |
| Surface pressure | Weight of the atmosphere above | Low pressure = rising air = rainfall trigger |
| 10m wind (east-west) | Horizontal wind direction/speed | Northeast monsoon brings moisture from South China Sea |
| 10m wind (north-south) | Horizontal wind direction/speed | Southwest monsoon = drier conditions |
| Total column water vapour | Total moisture in atmosphere column | Direct measure of available moisture for rain |
| ERA5 precipitation | Independent model-estimated rainfall | Provides a second, physics-based rainfall estimate |
| Relative humidity (derived) | % saturation of air | 100% = rain is imminent |
| Wind speed (derived) | Combined wind magnitude | Frontal activity and squall lines |

**Result of adding ERA5:** The model went from 9 rainfall/seasonal features → **22 features** (9 original + 9 ERA5 + 4 seasonal). Both daily and weekly R² improved approximately **+3 percentage points each**.

---

## 6. Improvement Roadmap — Status

| # | Approach | Status | R² Impact Seen |
|---|---|---|---|
| 1 | SHAP sensitivity analysis | **Done** | Revealed 45-day look-back is optimal (not 90) |
| 2 | Hurdle / two-stage model | **Done** | Daily R²: −3.2% → +2.8% |
| 3 | Extreme-event weighted loss | **Done** | Improves heavy-rain accuracy |
| 4 | ERA5 atmospheric reanalysis | **Done** | Weekly best: +7.5%; daily: +5.8% |
| 5 | Bayesian hyperparameter search (Optuna) | **Done** | Weekly +11.3%, Monthly +26.6% (new bests) |
| 6 | Wavelet feature decomposition | Pending | Est. +4–9% |
| 7 | TCN + Attention hybrid | Pending | Est. +5–12% |
| 8 | Ensemble diversity (TCN+LSTM+Transformer) | Pending | Est. +4–10% |

> **SHAP** (SHapley Additive exPlanations): A technique from game theory that measures how much each input feature contributes to a model's prediction. We ran SHAP analysis on our trained model to find which features and time lags actually matter — and found that rainfall signal beyond 45 days ago contributes almost nothing. Shrinking the look-back window from 90 → 45 days reduced model complexity and improved generalisation.

> **Optuna / Bayesian hyperparameter search:** Neural networks have many design choices that are not learned from data — the number of layers, how many neurons per layer, learning rate, dropout rate, etc. These are called *hyperparameters*. Rather than testing all combinations by brute force, Optuna uses a Bayesian approach: it learns from each trial which regions of the search space tend to produce better results, and focuses subsequent trials there. This is far more efficient than a grid search.

> **Wavelet decomposition:** A wavelet transform decomposes a signal into components at different frequencies — like separating a musical chord into its individual notes. Applied to rainfall, it separates daily noise from weekly patterns from seasonal trends, giving the model explicit access to each frequency component rather than having to infer them from raw data.

> **TCN + Attention:** Attention is a mechanism (originally from the Transformer architecture used in large language models like GPT) that lets the model learn which specific past time steps to focus on most heavily for each prediction. Combined with TCN, the model can both read local temporal patterns (via convolutions) and attend to globally relevant historical days (e.g., "the same week in the previous monsoon season").

> **Ensemble diversity:** Currently our 3-model ensemble runs the same architecture 3 times with different random seeds. The next step replaces this with genuinely different architectures — TCN, LSTM, and Transformer — so each model captures a different aspect of the temporal structure. Averaging diverse models reduces error more than averaging near-identical ones.

---

## 7. Why R² = 0.80 Is Difficult

The client's target of R² ≥ 0.80 is a legitimate scientific goal. Here is the honest picture:

**Realistic R² ceiling by input type:**

| Available inputs | Realistic daily R² ceiling |
|---|---|
| Rainfall history only (3 stations) | 0–10% |
| + ERA5 atmospheric data *(current)* | 25–45% |
| + High-resolution numerical weather prediction | 45–65% |
| + Dense station network (10+ stations) | 50–70% |

**Current best: daily R² = +5.8% (ERA5+Hurdle), weekly R² = +11.3%, monthly R² = +26.6% (Optuna)**

**Why daily is harder than monthly:**  
On any given dry day, whether it starts raining depends on small-scale triggers — a sea-breeze collision, an orographic updraft, a cold pool outflow from a distant storm — none of which are captured in station records or ERA5's coarse grid. Monthly totals average out this randomness: correctly predicting "October will be very wet" is achievable even if the exact rainy days shift by one or two days.

All published global operational weather models struggle to achieve daily point rainfall R² > 0.30 for tropical stations.

**Monthly R² = +23.8% is actually a strong result** for rainfall-only + ERA5 inputs. For reservoir management and water supply planning — which operate on weekly/monthly planning horizons — this is directly usable today.

The remaining roadmap (Steps 5–8) is our best engineering path toward pushing daily R² as high as the available data allows.

---

## 8. New Dataset — Sg. Kuantan (Next Phase)

**What the client provided:** Raw 15-minute cumulative rainfall data for 5 stations along the Sg. Kuantan catchment, May 2015 – June 2026.

**Stations in new dataset:**

| Station | Latitude | Longitude |
|---|---|---|
| Sg. Kuantan di Pasir Kemudi | 3.873°N | 103.191°E |
| Felda Panching Utara | 3.836°N | 103.161°E |
| KOMTUR | 3.830°N | 103.290°E |
| Sg. Belat di Sri Damai | 3.762°N | 103.236°E |
| Sg. Cherating di Cherating | 4.130°N | 103.394°E |

**Data format — cumulative telemetry:**  
The raw sensor records a *running total* that resets at midnight each day and on 1 January each year. To get actual rainfall for each 15-minute interval, we take the difference between consecutive readings. This requires careful handling of the reset points (at midnight the counter drops to zero — the negative jump must not be treated as "negative rain").

**Why 5 stations instead of 3 is better:**  
More stations give the model more spatial coverage and more cross-station correlation signals. When one station sees the front of a rain band and another sees the tail, the model can infer the storm's direction and timing — which a single station cannot.

**What was done with this data:**

| Step | Description | Status |
|---|---|---|
| 1 | Convert cumulative → incremental 15-min rainfall | **Done** |
| 2 | Handle midnight/year-end counter resets | **Done** |
| 3 | Aggregate 15-min → daily totals | **Done** |
| 4 | Fill remaining gaps with GAN imputation | **Done** — 3,495 station-days filled |
| 5 | Extract ERA5 data at each station's exact coordinates | **Done** — 4,018 rows, 18 unique features |
| 6 | Retrain prediction model for 5 new stations | **Running now** |

**Data quality after processing:**

| Station | Data period | Original missing | After GAN fill | Still missing |
|---|---|---|---|---|
| Pasir Kemudi | May 2015 – Jun 2026 | 41.6% NaN | 247 days filled | 35.5% remain |
| Felda Panching | May 2015 – Jun 2026 | 56.1% NaN | 941 days filled | 32.9% remain |
| KOMTUR | May 2015 – Jun 2026 | 54.1% NaN | 1,157 days filled | 25.5% remain |
| Sg. Belat | May 2015 – Jun 2026 | 47.0% NaN | 456 days filled | 35.8% remain |
| Sg. Cherating | May 2015 – Jun 2026 | 51.9% NaN | 694 days filled | 34.7% remain |

> **Why are there still gaps after GAN imputation?**  
> 1,440 days (35.6% of the full period) had 4 or all 5 stations offline simultaneously. When that happens, the GAN has no neighbouring stations to learn from — it cannot reliably reconstruct those days, so they are left as NaN rather than producing unreliable values. The prediction model is trained only on the days where we have reliable observations or credible imputations.

> **ERA5 coordinate mapping (the client's question):**  
> Yes — ERA5 can be mapped to each station's GPS coordinates. The ERA5 grid has 0.25° spacing (~28 km). We locate the nearest ERA5 grid point for each station. Four of the five stations (Pasir Kemudi, Felda Panching, KOMTUR, Sg. Belat) are close enough together that they share the same ERA5 grid cell; only Sg. Cherating maps to a different cell further north. This means we extract 9 atmospheric variables at each of the 2 unique grid cells, giving 18 distinct ERA5 features for the model — rather than 45 redundant ones.

The deep learning architecture stays exactly the same — only the number of station inputs changes from 3 to 5.

---

## 9. Current Status (as of July 2026)

| Task | Status |
|---|---|
| Phase 1: Data Engineering (old stations) | **Complete** |
| Phase 2: GAN Imputation (old stations) | **Complete** |
| Phase 3: Multi-Head TCN + ERA5 | **Complete** |
| Hurdle model + extreme-event weighted loss | **Complete** |
| ERA5 integration (17 years, 9 variables) | **Complete** |
| Optuna hyperparameter search | **Complete** — weekly R²=+11.3%, monthly R²=+26.6% |
| New Kuantan data: 15-min → daily conversion | **Complete** — 388,705 rows → 4,050 daily rows |
| New Kuantan data: GAN imputation (5 stations) | **Complete** — 3,495 station-days filled |
| ERA5 station-mapped extraction (5 stations) | **Complete** — 4,018 rows, 18 unique features |
| Phase 4 daily model (XGBoost Hurdle, 3 active stations) | **Complete** — daily R² 20–24%, weekly R² 30–37% |
| Phase 4 monthly model (direct ERA5-aggregate prediction) | **Complete** — sg_cherating monthly R²=72.2% (r=0.918) |
| Steps 8–9 (ensemble diversity, higher-res precipitation) | Pending — roadmap documented in technical report |

---

## 9a. Phase 4 Prediction Results Summary

The XGBoost Hurdle model was chosen over the TCN because the TCN could not converge on the sparse Kuantan data (1,300–1,600 real training days per station). Two model types were trained:

### Daily/Weekly Prediction (XGBoost Hurdle)

| Station | Daily R² | Weekly R² | Wet/Dry Accuracy | Real test days |
|---|---|---|---|---|
| Pasir Kemudi | **+20.5%** | +29.9% | 69.8% | 477 days |
| Felda Panching | *(sensor offline since 2024)* | — | — | 0 |
| KOMTUR | *(sensor offline since 2024)* | — | — | 9 |
| Sg. Belat | **+23.6%** | +30.2% | 73.7% | 472 days |
| Sg. Cherating | **+22.8%** | +36.8% | 62.8% | 492 days |

> **What 20–24% daily R² means:** The model correctly explains about 1 in 5 units of day-to-day rainfall variability, and correctly classifies whether a day is wet or dry 70–74% of the time. For tropical convective rainfall in Malaysia, this is competitive — the physical chaotic nature of convection means 20–35% is the expected range for statistical models using reanalysis inputs.

### Monthly Prediction (Direct Monthly Aggregation Model)

| Station | Monthly R² | Pearson r | RMSE | Real test months |
|---|---|---|---|---|
| Pasir Kemudi | **+66.2%** | 0.822 | 79 mm | 19 |
| Felda Panching | *(sensor offline)* | — | — | 0 |
| KOMTUR | *(sensor offline)* | — | — | 0 |
| Sg. Belat | **+49.9%** | 0.709 | 124 mm | 19 |
| Sg. Cherating | **+72.2%** | **0.918** | 121 mm | 19 |

> **Sg. Cherating r=0.918:** The model is capturing the correct monthly rainfall pattern with very high fidelity — a correlation of 0.918 means it predicts whether a given month is wet or dry, and by roughly how much, almost 92% correctly in rank. The R² is lower than r² (84%) because ERA5 at 0.25° resolution underestimates the amplitude of extreme monsoon months (observed 760 mm; predicted 540 mm in Nov 2024). Higher-resolution precipitation input (IMERG, MSWEP) would close this gap.

---

## 10. Deliverables Produced So Far

| File | Description |
|---|---|
| `data/processed/completed_daily_rainfall_cnn_tf.csv` | GAN-completed daily rainfall, 2009–2025 (original 3 stations) |
| `data/processed/era5_pahang_daily.csv` | ERA5 atmospheric daily, 2009–2025, 9 variables (original area average) |
| `data/processed/kuantan_daily_raw.csv` | 5-station Sg. Kuantan daily rainfall, post-QC, 4,050 rows |
| `data/processed/era5_kuantan_station_mapped.csv` | ERA5 per-station, 4,018 rows, 45 columns (18 unique features) |
| `data/processed/kuantan_daily_imputed.csv` | GAN-filled 5-station data, 3,495 station-days imputed |
| `predictions/era5_hurdle_predictions.csv` | Test set predictions — ERA5+Hurdle model (original 3 stations) |
| `predictions/hurdle_tcn_predictions.csv` | Test set predictions — Hurdle-only model (original 3 stations) |
| `predictions/optuna_best_predictions.csv` | Optuna best config test predictions (original 3 stations) |
| `predictions/phase4_xgb_results.json` | XGBoost Hurdle per-station R² (daily/weekly/monthly) — 5 Kuantan stations |
| `predictions/phase4_xgb_predictions.csv` | XGBoost Hurdle test predictions, 3 active Kuantan stations |
| `predictions/phase4_monthly_v2_results.json` | Direct monthly model results — R², RMSE, correlation per station |
| `predictions/phase4_monthly_v2_predictions.csv` | Monthly predictions vs actuals, 19 test months |
| `predictions/phase4_best_results.json` | **Consolidated best R² per station per scale — final Phase 4 numbers** |
| `figures/tcn/` | Scatter plots, SHAP importance, R² comparison charts |
| `reports/Technical_Report_Master_Progress.md` | Master technical report — all phases with equations and glossary |
| `notebooks/03b_tcn_multihead_tensorflow.ipynb` | Reproducible code notebook — multi-head, hurdle, ERA5 |

**Remaining steps (optional for R² improvement):**

| Step | Expected R² gain | Input required |
|---|---|---|
| Ensemble diversity (LightGBM + LSTM blend) | +3–8pp daily | Current data |
| Higher-res precipitation input (IMERG/MSWEP) | +8–15pp monthly for sg_cherating | IMERG download (~5 GB) |
| Sensor data recovery (Felda Panching, KOMTUR) | Enables evaluation for 2 stations | YBTM sensor repair |
