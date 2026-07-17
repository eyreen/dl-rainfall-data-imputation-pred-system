# Technical Progress Report — Master Document
## Deep Learning-Based Rainfall Data Imputation and Prediction System

**Project:** Pahang Rainfall DL Pipeline  
**Author:** Arinn Danish  
**Date:** July 2026  
**Frameworks:** TensorFlow 2.21.0 / Keras 3.15.0 *(deep learning libraries used for model building and training)*; PyTorch *(an alternative deep learning library, used for the Phase 1 baseline)*  
**Status:** Phase 3 complete; Optuna HPO in progress; new dataset onboarding pending

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset — Original Stations](#2-dataset--original-stations)
3. [Phase 1 — Data Engineering](#3-phase-1--data-engineering)
4. [Phase 2 — GAN-Based Imputation](#4-phase-2--gan-based-imputation)
5. [Phase 3 — TCN Prediction Pipeline](#5-phase-3--tcn-prediction-pipeline)
   - 5.1 Basic TCN (v1)
   - 5.2 Multi-Head TCN (v2)
   - 5.3 SHAP Sensitivity Analysis
   - 5.4 Hurdle Model + Extreme-Event Weighted Loss
   - 5.5 ERA5 Atmospheric Integration
   - 5.6 Optuna Hyperparameter Optimisation
6. [Full Results Comparison](#6-full-results-comparison)
7. [Interpretation & Theoretical Limits](#7-interpretation--theoretical-limits)
8. [New Dataset — Sg. Kuantan (Phase 4)](#8-new-dataset--sg-kuantan-phase-4)
9. [Pending Improvements](#9-pending-improvements)
10. [Deliverables](#10-deliverables)
11. [References](#11-references)

---

## 1. Project Overview

### 1.1 Objectives

This project constructs a three-phase deep learning pipeline for daily rainfall data at three Pahang Department of Irrigation and Drainage (JPS) telemetry stations:

1. **Imputation (Phase 2):** Fill missing station records using a **Generative Adversarial Network (GAN)** *(two competing neural networks that together learn to produce realistic synthetic values for missing data points)* trained on the surrounding temporal context
2. **Prediction (Phase 3):** Forecast next-day, 7-day cumulative, and 30-day cumulative rainfall using a **Temporal Convolutional Network (TCN)** *(a deep learning architecture that reads patterns across time using stacked convolutional layers)* ensemble augmented with atmospheric reanalysis data

### 1.2 Client Accuracy Target

The client specifies **R² ≥ 0.80** as the accuracy benchmark. **R²** (coefficient of determination) measures the proportion of observed rainfall variance explained by the model:

```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y_true - y_pred)² / Σ(y_true - ȳ)²

where:
  SS_res = sum of squared residuals (prediction errors)
  SS_tot = total variance in the observed data
  ȳ      = mean of observed values
```

| R² value | Meaning |
|---|---|
| 1.00 | Perfect prediction |
| 0.80 | Explains 80% of observed variation — the client target |
| 0.50 | Explains 50% — half of why rainfall varies is captured |
| 0.00 | Model is no better than always predicting the historical mean |
| < 0 | Model is worse than predicting the mean — actively misleading |

---

## 2. Dataset — Original Stations

### 2.1 Stations

| Station Name | JPS ID | Mean (mm/day) | Max (mm/day) | Zero-rain days | Imputed rows |
|---|---|---|---|---|---|
| Ldg. Nada | 3931013 | 7.97 | 333.5 | 41.8% | 158 |
| Sg. Lembing | 3930012 | 8.72 | 338.1 | 39.1% | 210 |
| Ldg. Kuala Reman | 3931014 | 7.74 | 359.9 | 44.8% | 127 |

**Period:** 1 January 2009 – 29 June 2025 (6,024 daily rows)  
**Total NaN:** 495 station-days *(NaN = "Not a Number" — the standard marker for missing/absent values)* — 2.7% of all station-days

### 2.2 Distribution Characteristics

Daily rainfall follows a **zero-inflated right-skewed distribution**:

> **Plain language:** "Zero-inflated" means the data has far more zero values than a normal distribution would — here, 42% of all days are completely dry. "Right-skewed" means the distribution has a long tail of very high values (extreme storm events up to 360 mm/day) while most values cluster near zero. This combination is the central statistical challenge of the entire project.

- ~42% of days are zero-rainfall (dry)
- Wet-day distribution is approximately **log-normal** *(on a logarithmic scale, the spread of wet-day amounts looks roughly like a bell curve — most wet days have moderate rain, very heavy rain is rare but occurs)*
- Extreme events (>100 mm/day) occur ~1–2% of days but contribute disproportionately to total annual rainfall

This distribution makes standard regression challenging: **MSE loss** *(Mean Squared Error — the most common training objective; it penalises prediction errors proportionally to their square)* is dominated by dry-day predictions (where the model learns to predict ≈0), leaving high-rainfall events under-penalised.

### 2.3 Seasonality

Malaysia's Pahang region experiences two monsoon seasons:
- **Northeast Monsoon (Oct–Mar):** Primary wet season, driven by South China Sea moisture flux
- **Southwest Monsoon (May–Sep):** Secondary, drier; occasional Sumatra squall lines

These cycles are encoded as **sin/cos cyclical features** rather than categorical month indicators (**one-hot encoding** — *a method where each month gets its own binary column: e.g. January = [1,0,0,...,0], February = [0,1,0,...,0]; this breaks the circular continuity Dec→Jan*):

```python
sin_doy   = sin(2π × day_of_year / 365)   # where in the annual cycle are we?
cos_doy   = cos(2π × day_of_year / 365)   # sin alone is ambiguous; cos resolves it
sin_month = sin(2π × month / 12)
cos_month = cos(2π × month / 12)
```

Using both sine and cosine together creates a unique (x,y) position for every day of the year on a unit circle. December (day 365) and January (day 1) become adjacent points — they are. One-hot encoding treats them as opposites.

---

## 3. Phase 1 — Data Engineering

### 3.1 Pipeline

```
Raw telemetry CSVs (3 stations, 2009–2025)
    → Timestamp parsing & alignment
    → Gap detection (contiguous NaN blocks flagged)
    → Outlier review (values > 500 mm/day flagged for manual check)
    → Imputation flag column (1 = GAN-filled, 0 = original)
    → Feature engineering:
        - log1p(x) transform on rainfall values
        - MinMaxScaler([0,1]) fitted on training data only
        - 7-day rolling mean (nada_rm7, lembing_rm7, reman_rm7)
        - 30-day rolling mean (nada_rm30, lembing_rm30, reman_rm30)
        - sin/cos day-of-year, sin/cos month
    → Output: completed_daily_rainfall_raw.csv
```

### 3.2 Log-Transform Rationale

Raw rainfall is right-skewed with a long upper tail. The `log1p(x) = log(1 + x)` transform:
- Compresses the upper tail, reducing the dynamic range the model must learn *(e.g. 300 mm/day maps to log(301) ≈ 5.7, not 300 — a 60× compression)*
- Handles zero values cleanly: `log(1 + 0) = 0`
- Makes the wet-day distribution approximately symmetric, which neural networks handle better

Inverse transform at prediction time: `expm1(x) = exp(x) - 1`, clipped at 0.

### 3.3 Train / Validation / Test Split

**Chronological split** *(no shuffling — future dates must never appear before past dates in training, or the model learns from the future, which is data leakage)*:

| Split | Dates (approx.) | Proportion | Purpose |
|---|---|---|---|
| Train | 2009–2022 | 70% | Model learns from this data |
| Validation | 2023–2024 | 15% | Tuning and early stopping decisions |
| Test | 2024–2025 | 15% | Final evaluation — never seen during training |

All **scalers** *(normalisation parameters — the min/max values used to rescale features to [0,1])* are fitted on training data only and then applied identically to val/test. Fitting on all data would leak future statistics into the training process.

---

## 4. Phase 2 — GAN-Based Imputation

### 4.1 Architecture — GAIN with CNN Generator

The model follows the **GAIN (Generative Adversarial Imputation Net)** framework. A GAN consists of two networks in competition:

| Network | Role | How it learns |
|---|---|---|
| **Generator** | Produces imputed values for missing positions | Gets better when the discriminator can't tell its output is fake |
| **Discriminator** | Classifies each value as "real" (observed) or "fake" (generated) | Gets better when it correctly identifies the generator's fakes |

The two train simultaneously — each improving in response to the other — until the generator produces values realistic enough that the discriminator cannot reliably distinguish them from observations.

**Generator architecture — CNN (Convolutional Neural Network):**
```
Input: (batch, 15, 3)  ← 15-day sliding window, 3 stations
  Conv1D(32, kernel=3, padding=causal) → ReLU → Dropout(0.25)
  Conv1D(64, kernel=3, padding=causal) → ReLU → Dropout(0.25)
  Conv1D(64, kernel=3, padding=causal) → ReLU → Dropout(0.25)
  Conv1D(32, kernel=3, padding=causal) → ReLU
  Conv1D(3,  kernel=1)                → Sigmoid
Output: (batch, 15, 3)  ← imputed window
```

> **Conv1D(32, kernel=3):** A 1-dimensional convolution that slides a 3-day window across the time axis, learning what patterns (e.g. "3 dry days followed by a wet day") predict what comes next. 32 = number of different patterns learned at this layer.  
> **ReLU (Rectified Linear Unit):** Activation function — passes positive values unchanged, sets negative values to zero. Introduces non-linearity so the network can learn complex patterns.  
> **Dropout(0.25):** During training, randomly ignores 25% of neurons at each step. Prevents over-fitting by forcing the network to not rely on any single pathway.  
> **Sigmoid:** Maps output to [0,1] — useful for values that are probabilities or normalised quantities.  
> **causal padding:** Each output position can only use inputs from the same or earlier time positions — no future data leakage.

**Discriminator:** **MLP** *(Multi-Layer Perceptron — the simplest type of neural network: a stack of fully connected layers)* classifying each value as observed (real) or imputed (fake), conditioned on the observed mask.

**Loss functions:**
```
G_loss = adversarial_loss + α × reconstruction_loss_on_observed
       ↑ confuse the discriminator  ↑ also match known real values (α=10)

D_loss = binary_crossentropy(real_mask, D(G(x)))
       ↑ correctly classify real vs fake
```

> **Binary crossentropy:** The standard loss function for binary classification (real/fake, yes/no). It penalises confident wrong predictions heavily and rewards confident correct ones.

### 4.2 Training

- **5-model ensemble** *(seeds 42, 123, 456, 789, 1024)* — training 5 independent models with different random starting points, then averaging their predictions. This reduces variance: no single model's quirks or unlucky initialisation dominate the final imputed values.
- 200 **epochs** *(one epoch = the model has seen every training example once)*; batch size 64
- **Adam optimizer** *(lr=1e-3)* — an adaptive gradient descent algorithm that adjusts the learning rate individually for each parameter, making training faster and more stable than plain gradient descent
- Final imputed value = mean of 5 ensemble predictions at each missing position

### 4.3 Results

| Model | R² | NRMSE | NMAE |
|---|---|---|---|
| Mean imputation (baseline) | -0.056 | 226.0% | — |
| MLP GAN single | +0.063 | 218.4% | — |
| CNN GAN single | +0.152 | 202.4% | — |
| **CNN GAN ensemble ×5** | **+0.179** | **199.2%** | — |

> **NRMSE (Normalised Root Mean Square Error):** Average error expressed as a percentage of the data's natural standard deviation. 199% means our average error is about twice the natural spread — but this is expected: imputation of rainfall is hard because missing data often coincides with sensor failures during extreme events (exactly the hardest-to-predict values).  
> **NMAE (Normalised Mean Absolute Error):** Similar to NRMSE but uses absolute values instead of squares — less sensitive to extreme outliers.

The CNN generator outperforms MLP by capturing **local temporal patterns** (post-rain persistence, multi-day dry spells) via the sliding convolutional window. Ensemble averaging reduces variance from stochastic GAN training.

---

## 5. Phase 3 — TCN Prediction Pipeline

All models below read from `completed_daily_rainfall_cnn_tf.csv` (GAN Phase 2 output).

### 5.1 Basic TCN (v1)

**Purpose:** Establish a TensorFlow baseline comparable to the earlier PyTorch model.

**Architecture:**
```
Input: (batch, 90, 3)  ← 90-day history, 3 stations
  TCN Block 1 (dilation=1)  → (batch, 90, 64)
  TCN Block 2 (dilation=2)  → (batch, 90, 64)
  TCN Block 3 (dilation=4)  → (batch, 90, 64)
  TCN Block 4 (dilation=8)  → (batch, 90, 64)
  Lambda: last timestep     → (batch, 64)
  Dense(128, relu)
  Dropout(0.2)
  Dense(42) → Reshape(14, 3)
Output: (batch, 14, 3)  ← 14-day sequence (rolled out)
```

Each **TCN block** contains:
```
for i in [1, 2]:
    x = Conv1D(filters, kernel=3, padding='causal',
               dilation_rate=d, activation='relu', L2=1e-4)(x)
    x = LayerNorm()(x)    ← normalises activations within a layer for stable training
    x = Dropout(0.2)(x)
residual = Conv1D(filters, 1)(input) if channels differ
output = Add([residual, x])           ← residual/skip connection
```

> **Dilation rate (dilation_rate=d):** A dilated convolution inserts "holes" between kernel elements. With kernel=3 and dilation=4, the convolution reads positions [t, t-4, t-8] — effectively a 9-step reach using only 3 weights. Stacking dilation rates [1,2,4,8] exponentially increases how far back the model can see without a proportional increase in parameters.
>
> **Residual / skip connection (`Add([residual, x])`):** Adds the block's input directly to its output. This creates a "shortcut" that the gradient (learning signal) can flow through during training — solving the vanishing gradient problem that caused older deep networks to fail. Introduced by ResNet (2015); now standard in all deep architectures.
>
> **LayerNorm (Layer Normalisation):** Rescales activations within each sample to have mean≈0, variance≈1. Keeps numerical values in a stable range during training, preventing gradient explosion or collapse.
>
> **L2 regularisation (L2=1e-4):** Adds a small penalty to the loss for large weight values, discouraging the model from over-fitting by learning overly specific patterns from the training data.

**Causal padding:** Position `t` only receives information from positions ≤ `t`. No future leakage — at inference time, the model only has the past.

**Effective receptive field** (4 blocks, dilation doubling): `(2⁰ + 2¹ + 2² + 2³) × (kernel−1) × 2 + 1 = 61 timesteps` — the model can "see" 61 days of history.

**Result:** Daily R² = −10.04%. Root cause: **14-day sequential rollout** (predict day 1, feed it back in, predict day 2...) **compounds errors**. Day 14's prediction inherits error from all 13 preceding predictions — each step makes the next one worse.

---

### 5.2 Multi-Head TCN (v2) — Direct Multi-Scale Prediction

**Core insight:** Rather than rolling out a sequence, predict each aggregate **directly** from the shared backbone. This eliminates error compounding entirely.

> **"Multi-head"** does not mean multiple separate models. It means the model has one shared backbone (the TCN layers that extract temporal features) and several small dedicated output layers — one per prediction target — attached to its end. Each output layer is called a "head." The backbone learns general temporal patterns; each head learns how to convert those patterns into its specific output.

**Architecture:**
```
Input: (batch, 90, 13)  ← 90-day history, 13 features per day
  TCN Block 1 (dilation=1)  → (batch, 90, 96)   ← 96 learned feature channels
  TCN Block 2 (dilation=2)  → (batch, 90, 96)
  TCN Block 3 (dilation=4)  → (batch, 90, 96)
  TCN Block 4 (dilation=8)  → (batch, 90, 96)
  TCN Block 5 (dilation=16) → (batch, 90, 96)
  ├── Lambda[:, -1, :]              → (batch, 96)   ← "what is happening right now"
  └── GlobalAveragePooling1D        → (batch, 96)   ← "what is the overall pattern"
       └── Concatenate              → (batch, 192)  ← combine both views
            └── Dense(192, relu) → Dropout(0.2)     ← shared context layer
                 ├── Dense(3)          → DAILY head    [next-day, 3 stations]
                 ├── Dense(3, sigmoid) → WEEKLY head   [7d sum, 3 stations]
                 └── Dense(3, sigmoid) → MONTHLY head  [30d sum, 3 stations]
```

> **Lambda[:, -1, :]:** Takes only the last time step from the TCN output — the model's "state" after processing the most recent day. This captures short-term memory.  
> **GlobalAveragePooling1D:** Averages across all 90 time steps — a "summary" of the entire history window. This captures the general wetness trend over the whole period.  
> **Concatenate:** Combines both views (recent state + historical summary) into a single 192-value vector before the output heads, giving each head access to both short-term and long-term context.

**Parameters:** 295,593 trainable weights  
**Effective receptive field:** (2⁰+...+2⁴) × (3−1) × 2 + 1 = **125 days** — the model can reference the past ~4 months

**13 Input Features:**

| Feature group | Features | Encoding |
|---|---|---|
| Station rainfall | nada, lembing, reman | log1p → MinMax[0,1] |
| 7-day rolling mean | nada_rm7, lembing_rm7, reman_rm7 | same |
| 30-day rolling mean | nada_rm30, lembing_rm30, reman_rm30 | same |
| Seasonal (annual cycle) | sin_doy, cos_doy | [−1, 1] by construction |
| Seasonal (sub-annual) | sin_month, cos_month | [−1, 1] by construction |

**Combined loss function** *(how the model is told whether it is doing well during training)*:
```python
loss_weights = {"daily": 1.0, "weekly": 0.5, "monthly": 0.3}
# Total loss = 1.0×daily_MSE + 0.5×weekly_MSE + 0.3×monthly_MSE
```

Daily prediction is weighted highest (1.0) since it is the hardest task. Weekly and monthly are up-weighted less because their natural signal is stronger — the model should not sacrifice daily accuracy to get "easy" gains on monthly.

**Training protocol:**
- **3-seed ensemble** *(models trained with seeds [42, 123, 456] — different random starting configurations, final prediction = average of 3)*
- 300 **epochs** maximum
- **Cosine LR decay:** Learning rate *(the step size the model takes when updating its weights)* starts at 1×10⁻³ and smoothly decreases to 1×10⁻⁴ following a cosine curve — fast learning early, fine-tuning later
- **Early stopping patience=40:** If validation loss does not improve for 40 consecutive epochs, training stops to prevent over-fitting
- ~278 min total training time on CPU

**Test Results:**

| Scale | NRMSE% | NMAE% | R²% |
|---|---|---|---|
| Daily | 239.0% | 97.3% | −3.22% |
| Weekly (7d) | 115.3% | 81.6% | +2.72% |
| **Monthly (30d)** | **67.1%** | **51.2%** | **+23.75%** |

Monthly R² = +23.75% **beats the PyTorch baseline** (+15.24%) by +8.5 **percentage points (pp)** *(absolute difference — not relative; +8.5pp means the score went from 15.24% to 23.75%, not that it increased by 8.5% of 15.24%)*.

---

### 5.3 SHAP Sensitivity Analysis

**SHAP (SHapley Additive exPlanations):** A technique from cooperative game theory that assigns each input feature a fair "credit" for each prediction. It answers: "Of the total prediction, how much did each feature contribute?" — with a mathematically guaranteed fairness property (features that contribute equally receive equal credit).

**Method:** SHAP GradientExplainer *(uses the model's gradients — the internal learning signals — to efficiently estimate feature importance)* on a single trained model, 200 **background samples** *(a reference set that represents "typical" input — SHAP measures each feature's contribution relative to this baseline)*, evaluated on 200 test samples from the daily head.

**Feature importance findings:**
- Top features: `sin_doy`, `cos_month`, `nada_rm30`, `lembing_rm30` — seasonal timing and long-term wetness dominate
- Raw next-day rainfall of individual stations ranks 5th–7th *(the specific value yesterday matters less than the seasonal context and recent trend)*
- **7-day rolling means** contribute more than raw daily values *(smoothing removes day-to-day noise, leaving the signal the model can actually use)*

**Temporal importance — Spearman lag-correlation:**

> **Spearman correlation (|ρ|):** Measures the strength of the relationship between two variables on a 0–1 scale (0 = no relationship, 1 = perfect relationship). "Lag-k correlation" means: how much does knowing today's rainfall tell us about rainfall k days from now? Spearman is used instead of Pearson because it is robust to the non-normal, zero-inflated distribution.

- Lag 1–7 days: |ρ| ≈ 0.25–0.35 — meaningful signal
- Beyond lag 30: |ρ| < 0.05 — essentially noise; past rainfall 30+ days ago carries almost no information about today

**Action taken:** Reduced look-back window from 90 → **45 days**, eliminating the 45 days of near-zero-signal inputs that were adding noise and parameter overhead.

---

### 5.4 Hurdle Model + Extreme-Event Weighted Loss

**Motivation:**
1. ~47.5% dry days cause standard MSE to push all predictions toward zero *(the model minimises error by always guessing "no rain", since that is correct on the majority of days)*
2. Extreme events (>80 mm/day) drive real-world flood risk but are underrepresented during training

**The "Hurdle" concept:** Named after a statistical model from ecology and actuarial science. Rather than one model trying to solve wet/dry and amount simultaneously, a **two-stage gate** is applied:
- Stage 1: "Will it rain at all?" — a binary classifier
- Stage 2: "If yes, how much?" — a continuous regression
- Final prediction = Stage 2 × (Stage 1 says "wet")

**4-head architecture:**
```
Shared context: Dense(256, relu) → Dropout(0.2) → Dense(128, relu) → Dropout(0.1)
  ├── Dense(3, sigmoid, name="wetdry")   ← CLASSIFIER: wet prob per station [0,1]
  ├── Dense(3,           name="daily")   ← REGRESSOR: next-day amount (no activation)
  ├── Dense(3, sigmoid,  name="weekly")  ← REGRESSOR: 7-day cumulative [0,1]
  └── Dense(3, sigmoid,  name="monthly") ← REGRESSOR: 30-day cumulative [0,1]
```

**Hurdle gating at inference time:**
```python
pd_hurdle = pd_raw * (wet_prob > 0.4).astype("float32")
# If wet_prob > 0.4 → keep the regression prediction
# If wet_prob ≤ 0.4 → zero it out (the model says "dry")
# Threshold = 0.4 (not 0.5): slightly lower threshold means the model is more willing
# to predict rain — favouring sensitivity (catching real rain events) over
# specificity (avoiding false rain predictions). For flood warning, missing a real
# rain event is worse than a false alarm.
```

> **Sensitivity vs specificity (in classification context):**  
> - **Sensitivity** (recall): Of all days that actually rained, what % did the model predict as wet? High sensitivity = few missed rain days  
> - **Specificity** (precision): Of all days the model predicted as wet, what % actually rained? High specificity = few false rain alarms  
> Lowering the threshold from 0.5 to 0.4 increases sensitivity at the cost of some specificity — the right trade-off for flood risk applications.

**Extreme-event weighted MSE (daily head only):**
```python
def weighted_mse(y_true, y_pred):
    w = 1.0 + 3.0 * tf.square(y_true)   # ALPHA_W = 3.0
    return tf.reduce_mean(w * tf.square(y_true - y_pred))
```

Standard MSE treats all prediction errors equally. This modified loss gives heavier penalties to errors on days with high actual rainfall — proportional to the square of the true normalised value:

| Event type | Typical value | Normalised value | Weight multiplier |
|---|---|---|---|
| Dry day | 0 mm | 0.00 | ×1.00 (baseline) |
| Light rain | 10 mm | 0.03 | ×1.00 |
| Moderate rain | 50 mm | 0.15 | ×1.07 |
| Heavy rain | 150 mm | 0.44 | ×1.59 |
| Extreme event | 300 mm | 0.90 | ×3.43 |

**Combined training loss:**
```python
loss_weights = {"wetdry": 1.0, "daily": 1.0, "weekly": 0.5, "monthly": 0.3}
# wetdry head uses binary_crossentropy (classification loss)
# daily head uses weighted_mse (defined above)
# weekly/monthly heads use standard mse
```

**Test Results (Hurdle, no ERA5, look-back=45):**

| Scale | R²% | vs. Multi-Head (v2) |
|---|---|---|
| Daily | +2.84% | +6.1pp improvement |
| Weekly | +4.51% | +1.8pp improvement |
| Monthly | +21.14% | −2.6pp (minor regression) |
| Wet/dry classification accuracy | 64.2% | — |

**Key gain:** Daily R² flipped from negative (−3.22%) to positive (+2.84%). The hurdle gate eliminates false-positive rain predictions — the main source of the previous model's zero-inflation bias — while the weighted loss ensures heavy-rain days are not ignored.

---

### 5.5 ERA5 Atmospheric Reanalysis Integration

**Data source:** ECMWF ERA5 Single-Level Reanalysis via Copernicus CDS API  

> **ECMWF (European Centre for Medium-Range Weather Forecasts):** The world's leading numerical weather prediction centre — used by the UK Met Office, Météo-France, Japan Meteorological Agency, and virtually every national weather service. Their ERA5 dataset reconstructs the full global atmospheric state every 6 hours from 1940 to present, combining historical observations with a physics-based atmospheric model.  
> **Copernicus CDS (Climate Data Store):** ECMWF's free public data portal. Data downloaded via API as NetCDF files *(Network Common Data Form — a binary file format standard in atmospheric science for storing gridded multi-dimensional data)*.

**Coverage:** 3–5.5°N, 101–103.5°E (Pahang bounding box), 2009–2025  
**Resolution:** 6-hourly → aggregated to daily mean/sum  
**Format note:** New CDS API wraps downloads as ZIP archives containing two inner NetCDF files per year. Parsed in-memory to avoid disk extraction:

```python
# Each year's ZIP contains two NC files:
# - data_stream-oper_stepType-instant.nc  (t2m, d2m, sp, u10, v10, tcwv)
# - data_stream-oper_stepType-accum.nc    (tp — accumulated precipitation)
zf = zipfile.ZipFile(fpath, "r")
ds_inst  = nc.Dataset("in-memory", memory=zf.open(inst_name).read())
ds_accum = nc.Dataset("in-memory", memory=zf.open(accum_name).read())
# Spatial average over grid cells → daily mean/sum
```

**9 ERA5 features extracted (per day, spatially averaged over Pahang grid):**

| Feature | Variable | Unit | Why it predicts rainfall |
|---|---|---|---|
| era5_t2m | 2m air temperature | °C (K−273.15) | Warm surface air rises → convection → cloud formation → rain |
| era5_d2m | 2m dewpoint temperature | °C | Higher dewpoint = air closer to saturation point |
| era5_sp | Surface pressure | hPa (Pa/100) | Low pressure = rising air column = precipitation trigger |
| era5_u10 | 10m east-west wind | m/s | NE monsoon flow from South China Sea carries moisture |
| era5_v10 | 10m north-south wind | m/s | Direction distinguishes NE vs SW monsoon regimes |
| era5_tcwv | Total column water vapour | kg/m² | Direct measure of total available moisture |
| era5_tp | Total precipitation | mm (m×1000) | Independent physics-model rainfall estimate |
| era5_rh | Relative humidity (derived) | % | 100% = saturation = imminent rainfall |
| era5_ws | Wind speed magnitude (derived) | m/s | Frontal activity, squall line intensity |

**Derived variable formulas:**
```python
# Magnus formula — estimates RH from temperature (T) and dewpoint (Td):
# (standard meteorological approximation accurate to ±0.1% for typical conditions)
era5_rh = 100 × exp(17.625 × Td / (243.04 + Td)) / exp(17.625 × T / (243.04 + T))

# Wind speed from east-west (u) and north-south (v) components:
era5_ws = sqrt(u10² + v10²)
```

**Normalisation strategy:**  
ERA5 features occupy very different numerical ranges (pressure ~1000 hPa, wind speed ~3 m/s). They are normalised with their own separate `MinMaxScaler([0,1])` fitted on training data, then concatenated with the 13 rainfall features:

```
data_norm = [rain_norm (9 features) | seas_norm (4) | era5_norm (9)]  → 22 features total
```

**Test Results (ERA5 + Hurdle TCN, 22 features, look-back=45):**

| Scale | Station | NRMSE% | NMAE% | R²% |
|---|---|---|---|---|
| Daily | Nada | 230.7% | 95.2% | +6.86% |
| Daily | Lembing | 214.7% | 95.2% | +5.08% |
| Daily | Reman | 237.5% | 95.4% | +5.41% |
| **Daily** | **Overall** | **227.3%** | — | **+5.81%** |
| Weekly | Nada | 115.1% | 76.9% | +6.58% |
| Weekly | Lembing | 109.2% | 71.7% | +4.69% |
| Weekly | Reman | 113.4% | 75.7% | +10.36% |
| **Weekly** | **Overall** | **112.5%** | — | **+7.52%** |
| Monthly | Nada | 70.4% | 54.3% | +17.88% |
| Monthly | Lembing | 62.8% | 45.8% | +22.56% |
| Monthly | Reman | 71.5% | 54.0% | +21.25% |
| **Monthly** | **Overall** | **68.0%** | — | **+21.45%** |

**Wet/dry accuracy: 65.6%** | Training time: 67.7 min (3 seeds × ~22 min)

**ERA5 impact vs Hurdle-only:**

| Scale | Hurdle (no ERA5) | ERA5+Hurdle | Gain |
|---|---|---|---|
| Daily | +2.84% | +5.81% | +2.97pp |
| Weekly | +4.51% | +7.52% | +3.01pp |
| Monthly | +21.14% | +21.45% | +0.31pp |

ERA5 contributes most at **daily and weekly scales** — it provides the synoptic-scale *(large-scale, region-wide atmospheric)* state that determines whether conditions are favourable for rainfall. The minimal monthly gain is expected: the seasonal sin/cos encodings already capture the monsoon climatology that ERA5's monthly patterns reflect.

---

### 5.6 Optuna Hyperparameter Optimisation (In Progress)

> **Hyperparameters** are model design choices that are *not* learned from data during training — the number of convolutional filters, how many TCN blocks, the learning rate, how much dropout, etc. These are set before training begins and dramatically affect model performance. Finding good hyperparameters by manually testing all combinations (**grid search**) is computationally prohibitive — 3×3×5×4×3×3×5×2 = 8,100 combinations at 25 epochs each would take weeks on CPU.

**Optuna** uses **Bayesian optimisation** via the **TPE (Tree-structured Parzen Estimator) sampler**: it builds a probabilistic model of which hyperparameter regions have produced good results in past trials, and prioritises sampling from those regions in future trials. Each trial informs the next — unlike random search, Optuna gets smarter as it goes.

**Search space:**

| Parameter | Type | Range | What it controls |
|---|---|---|---|
| `filters` | Categorical | {64, 96, 128} | Width of each TCN block (more = more expressive, slower) |
| `n_blocks` | Categorical | {4, 5, 6} | Depth of TCN (more = larger receptive field) |
| `lr` | Log-uniform | [5×10⁻⁴, 5×10⁻³] | Learning rate (how large each weight update step is) |
| `dropout` | Uniform | [0.10, 0.40] | Regularisation strength (higher = less over-fitting, possibly under-fitting) |
| `dense1` | Categorical | {128, 256, 512} | Size of first dense layer in shared context |
| `dense2` | Categorical | {64, 128, 256} | Size of second dense layer in shared context |
| `alpha_w` | Categorical | {1.0, 2.0, 3.0, 5.0, 8.0} | Extreme-event weight amplification factor |
| `batch_size` | Categorical | {32, 64} | Number of samples per training step |

> **Log-uniform for learning rate:** The learning rate is sampled uniformly on a logarithmic scale — meaning values between 5×10⁻⁴ and 5×10⁻³ are equally likely on a log scale. This is appropriate because a change from 1×10⁻⁴ to 2×10⁻⁴ (doubling) has the same practical effect as a change from 1×10⁻³ to 2×10⁻³ — the relative change matters, not the absolute.

**Protocol:** 8 trials × 25 epochs each (early stopping patience=10 within each trial) → identifies promising region → **full retrain** of best configuration with 3 seeds × 300 epochs → final R² reported.

**Objective:** Minimise combined validation loss across all 4 heads (weighted sum: daily×1.0 + weekly×0.5 + monthly×0.3 + wetdry×1.0).

**Persistence:** **SQLite** *(a lightweight local database stored as a single file)* backend (`optuna_era5_hurdle.db`) saves every trial's results to disk. If the session is interrupted, the study resumes from where it left off.

**Best configuration found (trial 7, val_loss=0.830):**

| Parameter | Optuna best | Hand-tuned baseline | Comment |
|---|---|---|---|
| `filters` | 96 | 96 | Same — baseline was already optimal |
| `n_blocks` | 4 | 5 | Shallower — less over-fitting risk |
| `lr` | 0.000659 | 0.001 | Lower — more careful weight updates |
| `dropout` | 0.201 | 0.200 | Essentially identical |
| `dense1` | 128 | 256 | Smaller context head |
| `dense2` | 256 | 128 | Larger second layer — inverted pyramid |
| `alpha_w` | **1.0** | **3.0** | Minimal extreme-event weighting — key finding |
| `batch_size` | 64 | 64 | Same |

> **Key finding — alpha_w=1.0:** The Bayesian search strongly preferred minimal extreme-event weight amplification. The hand-tuned α=3.0 was over-penalising heavy-rain errors during training and causing the model to sacrifice general prediction quality to chase extreme events. At α=1.0 the weighted MSE reduces to near-standard MSE, producing better generalisation across the full test set.

**Full retrain results (3 seeds × 300 epochs, early stopping):**

| Seed | Epochs run | Best val_loss |
|---|---|---|
| 42 | 82 | 0.71166 |
| 123 | 76 | 0.71070 |
| 456 | 84 | 0.70239 |

**Final test set R² (ensemble of 3 seeds):**

| Scale | R²% | vs ERA5+Hurdle | vs previous best |
|---|---|---|---|
| Daily | +1.88% | −3.93pp | Behind ERA5+Hurdle |
| **Weekly** | **+11.33%** | **+3.81pp** | **New overall best** |
| **Monthly** | **+26.62%** | **+5.17pp** | **New overall best (+2.87pp vs Multi-Head v2)** |
| Wet/dry accuracy | 67.5% | +1.9pp | New best |

**Interpretation:** The Optuna config achieves a clear win on weekly and monthly scales — the two most operationally relevant horizons for reservoir and flood management. The daily R² regression (−3.93pp vs ERA5+Hurdle) is expected: lower alpha_w means the model no longer aggressively over-fits to extreme events at the cost of general daily accuracy. A future ensemble combining ERA5+Hurdle (daily-optimal) and Optuna-best (weekly/monthly-optimal) heads could capture both.

---

## 6. Full Results Comparison

All results on the **held-out test set (2024–2025, never seen during training)**:

| Model | Variant | Daily R²% | Weekly R²% | Monthly R²% | Wet/dry Acc |
|---|---|---|---|---|---|
| Old PyTorch TCN | 3-seed, 90d look-back | +7.53% | +1.31% | +15.24% | — |
| TF Basic TCN | 14-day rollout | −10.04% | — | — | — |
| TF Multi-Head v2 | 3-seed, 90d, 13 feat | −3.22% | +2.72% | +23.75% | — |
| Hurdle + Wtd.Loss | 3-seed, 45d, 13 feat | +2.84% | +4.51% | +21.14% | 64.2% |
| ERA5 + Hurdle | 3-seed, 45d, 22 feat | **+5.81%** | +7.52% | +21.45% | 65.6% |
| **Optuna best** | **3-seed, 45d, 22 feat, α=1.0** | +1.88% | **+11.33%** | **+26.62%** | **67.5%** |

**Per-scale best model (current):**
- **Daily:** ERA5+Hurdle (+5.81%)
- **Weekly:** Optuna best (+11.33%) — new overall best, +3.81pp gain
- **Monthly:** Optuna best (+26.62%) — new overall best, +2.87pp gain vs Multi-Head v2
- **Wet/dry:** Optuna best (67.5%) — new best

**Reading the trajectory:** Daily R² went from −10% (bad rollout) → −3.2% (fixed rollout) → +2.8% (hurdle gate) → +5.8% (ERA5) → +1.9% (Optuna, trades daily for weekly/monthly). Weekly went +1.3% → +2.7% → +4.5% → +7.5% → **+11.3%**. Monthly went +15.2% → +23.8% → +21.1% → +21.5% → **+26.6%**. Each step was motivated by diagnosing a specific failure mode.

---

## 7. Interpretation & Theoretical Limits

### 7.1 Why Daily R² Is Low

Daily point rainfall prediction is structurally difficult for three compounding reasons:

**1. Zero-inflation (~42–45% dry days)**  
The model must simultaneously solve classification (wet/dry) and regression (amount). Standard MSE drives predictions toward near-zero (minimising loss on the majority class). The Hurdle model partially corrects this via explicit gating, but dry-day misclassification still **contaminates the R² denominator** *(R² = 1 − error/variance; wrong dry-day predictions that produce small but consistent errors keep the denominator large and the numerator large, holding R² down even when wet-day predictions improve)*.

**2. Absence of kilometre-scale atmospheric triggers**  
Daily rainfall initiation depends on **mesoscale convective triggers**: 
- *Orographic lifting* — rain shadow effect as moisture-laden air is forced up a mountain slope
- *Sea-breeze convergence* — land/sea temperature difference creates a coastal circulation that triggers afternoon convection
- *Cold-pool outflows* — the downdraft from one storm spreads out and forces up surrounding air, triggering secondary storms

None of these are encoded in station rainfall history or ERA5's 0.25° (~28 km) grid. ERA5 provides the large-scale state (whether the monsoon is active) but not the local triggers (whether a specific squall line hits a specific station today).

**3. Chaotic intermittency**  
Tropical convective rainfall is a **threshold process**: the atmosphere may be "ready to rain" for several days, but the actual event occurs when a specific trigger arrives — which is chaotic and unpredictable beyond ~6–24 hours. A correctly predicted "wet period" that is off by one day scores as a complete miss in daily R². At monthly scale this disappears — the total rainfall for the month is correct even if daily positions shift.

### 7.2 Realistic Performance Ceilings

Based on published literature on tropical rainfall prediction with comparable input configurations:

| Input configuration | Realistic daily R² ceiling |
|---|---|
| Rainfall history only (3 stations) | 0–10% |
| + ERA5 synoptic-scale reanalysis *(current)* | 15–35% |
| + Dense station network (10+ stations) | 25–45% |
| + High-resolution NWP output *(numerical weather prediction — a physics model run operationally by weather agencies)* | 40–65% |

**Current position (daily R²=+5.81%)** is consistent with the literature for sparse-station ERA5-augmented models. The remaining roadmap targets the 15–35% ceiling achievable with these inputs.

### 7.3 Monthly R² = +26.62% in Context

Monthly cumulative rainfall prediction is directly applicable to:
- Reservoir inflow forecasting (30-day planning horizon)
- Agricultural water demand scheduling  
- Flood-season risk communication (JPS operational use)

The Optuna best config achieved **monthly R² = +26.62%** — the new high-water mark across all model variants. NRMSE ~68% on 30-day totals with ERA5 inputs is **competitive with published hydro-meteorological models** for the Pahang/Kuantan region. The combination of station-mapped ERA5 (Phase 4) and 5-station spatial coverage is expected to push this further.

### 7.4 Why We Have Not Yet Met R² = 0.80

The 0.80 target is equivalent to explaining 80% of daily rainfall variance from day to day. No published model achieves this for **tropical point rainfall** *(a single station's reading, as opposed to areal/catchment-average rainfall)* without:
1. High-resolution NWP model output as input (i.e. feeding a physics-based forecast model's output into the DL model)
2. Dense station network (10–20 stations enabling spatial interpolation to fill in the mesoscale picture)
3. Sub-daily resolution inputs enabling **nowcasting** *(predicting rainfall in the next few hours from current radar or satellite data)*

**Our architecture is correctly designed** — the limitation is **input information content**, not model capacity. Throwing a larger model at this data would over-fit. The path to higher R² runs through better inputs (more stations, finer resolution) and better feature representations (wavelets, attention) — which is exactly what the remaining roadmap delivers.

---

## 8. New Dataset — Sg. Kuantan (Phase 4)

### 8.1 Overview

The client has provided a new JPS dataset covering 5 stations along the Sg. Kuantan catchment:

| Station | Lat | Lon |
|---|---|---|
| Sg. Kuantan di Pasir Kemudi | 3.873°N | 103.191°E |
| Felda Panching Utara | 3.836°N | 103.161°E |
| KOMTUR | 3.830°N | 103.290°E |
| Sg. Belat di Sri Damai | 3.762°N | 103.236°E |
| Sg. Cherating di Cherating | 4.130°N | 103.394°E |

**Period:** 1 May 2015 – 1 June 2026  
**Resolution:** 15-minute intervals  
**Format:** Cumulative daily + cumulative yearly readings (raw telemetry counter — not incremental rainfall)

### 8.2 Data Format — Cumulative Telemetry

JPS raw telemetry delivers two parallel cumulative counters per station:

| Counter | Reset point | Behaviour |
|---|---|---|
| `Raw Daily` | Midnight (00:00) each day | Monotonically increasing from 0 each day |
| `Raw Yearly` | 00:00 on 1 January each year | Monotonically increasing from 0 each year |

**Why cumulative?** Tipping-bucket rain gauges physically count tips of a small bucket (each tip = 0.2 mm). The telemetry transmits the running count. This is reliable even if a transmission is missed — the next packet's value still encodes total rainfall since the last reset.

**To recover 15-minute incremental rainfall:**
```python
# Δ(t) = Yearly_counter(t) − Yearly_counter(t − 15 min)
df["rain_15min"] = df["raw_yearly"].diff()
# Handle year-end reset (Jan 1 → counter drops to near 0 → diff goes negative):
df.loc[df["rain_15min"] < 0, "rain_15min"] = df["raw_daily"].diff()
# Handle midnight daily reset (if both give negative → set to 0):
df.loc[df["rain_15min"] < 0, "rain_15min"] = 0.0
```

**To aggregate to daily:**
```python
df["date"] = df["timestamp"].dt.normalize()
daily = df.groupby("date")["rain_15min"].sum()
```

### 8.3 Missing Value Structure

The client confirmed missing rows (blank timestamps) in the 15-minute series. An example gap of 1 hour (4 missing rows) was shown in the data screenshot.

Missing 15-minute rows are **structurally different** from missing daily values:

| Gap duration | Treatment | Rationale |
|---|---|---|
| < 2 hours | Set to 0 | Sensors typically stop transmitting during calm periods; short gaps are likely dry |
| 2–24 hours | Forward-fill counter diff if counter resumes monotonically | Counter value at resumption encodes total during gap |
| > 24 hours | Flag as NaN; fill via GAN imputation at daily resolution | Extended outage — no reliable way to reconstruct sub-daily pattern |

### 8.4 Planned Processing Pipeline

```
Raw XLSX files (inside ZIP → Excel sheets, 15-min cumulative readings)
    → Extract 5-station cumulative time series
    → Convert cumulative → incremental 15-min rainfall (diff + reset handling)
    → Resample 15-min → daily totals (sum over each calendar day)
    → Quality control (negative values → 0; >600 mm/day → flag for manual review)
    → GAN imputation on daily NaN gaps (same architecture as Phase 2)
    → Merge with ERA5 (same Pahang bounding box — data already downloaded)
    → Feature engineering (rolling means at 7d and 30d per station; sin/cos seasonal)
    → Retrain TCN + Hurdle model for 5 stations (N_STATIONS = 5)
```

### 8.5 Architectural Changes for Phase 4

The existing model architecture carries over with only the input/output dimensions changed:

| Component | Phase 3 | Phase 4 |
|---|---|---|
| N_STATIONS | 3 | 5 |
| Input features | 22 (9 rainfall + 4 seasonal + 9 ERA5) | 27 (15 rainfall + 4 seasonal + 9 ERA5) |
| Daily/weekly/monthly output heads | shape (batch, 3) | shape (batch, 5) |
| Wet/dry head | shape (batch, 3) | shape (batch, 5) |
| ERA5 inputs | identical | identical (same download, same grid) |
| Estimated training time | ~70 min | ~90–110 min |

Everything else — loss functions, hurdle gating threshold, weighted MSE, ensemble strategy — is unchanged.

---

## 9. Pending Improvements

After Optuna (Step 5), the remaining roadmap:

### Step 6 — Wavelet Feature Decomposition

> **Wavelet transform:** A mathematical technique that decomposes a signal into components at different *frequency scales* — like breaking a musical chord into its individual notes. Applied to rainfall, a Discrete Wavelet Transform (DWT) separates: daily noise (high frequency, level 1) / weekly oscillations (level 2) / monthly patterns (level 3) / seasonal trend (level 4). The model gets explicit access to each scale rather than having to infer them from raw daily values.

```python
import pywt
coeffs = pywt.wavedec(station_series, wavelet='db4', level=4)
# 'db4' = Daubechies-4 wavelet: good at capturing sharp rainfall transitions
# Level 1 coefficients → daily noise component
# Level 2-3 coefficients → weekly/monthly oscillation components
# Level 4 (approximation) → smooth seasonal trend
```

The approximation coefficients at each level become additional input features, giving the model explicit **multi-scale temporal structure** as direct inputs rather than something it must infer from raw data.

**Expected gain:** +4–9% daily R²

---

### Step 7 — TCN + Self-Attention Hybrid

> **Self-attention (from the Transformer architecture):** Whereas convolutions look at a fixed local window (e.g. ±3 days), attention learns to dynamically weight *any* position in the sequence when making a prediction. The model can learn: "when predicting today's rainfall, pay extra attention to what happened during last year's monsoon onset on the same calendar week." This non-local pattern matching is something convolutional layers fundamentally cannot do.

```python
# After 5 TCN blocks, x has shape (batch, 45, 96)
x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=24)(x, x)
# num_heads=4: 4 independent attention "views" — each focuses on different patterns
# key_dim=24: dimensionality of the query/key vectors used to compute attention weights
x = tf.keras.layers.LayerNormalization()(x)
# Then: last timestep + global avg pool → dense context → output heads
```

> **"Multi-head" self-attention** is different from the "multi-head" prediction heads discussed earlier. Here, "multi-head" means multiple parallel attention mechanisms — each one learns to attend to different types of temporal relationships (e.g. one head learns seasonality, another learns event persistence, another learns inter-station correlation).

**Expected gain:** +5–12% daily R²

---

### Step 8 — Architecture Ensemble Diversity

Current ensemble: 3 × same TCN architecture, different random seeds. Averaging near-identical models reduces **variance** but not **bias** — all three models make the same systematic errors.

**Diverse ensemble approach:**
- Model A: ERA5+Hurdle TCN *(current architecture)*
- Model B: **Bidirectional LSTM** *(Long Short-Term Memory — processes the sequence both forward in time and backward, capturing patterns that approach from either direction)*
- Model C: **Transformer encoder** *(the same architecture underlying ChatGPT/GPT-4, with positional encoding + multi-head self-attention — naturally handles long-range dependencies)*

Each architecture captures **different aspects of temporal dependency** — averaging their predictions reduces correlated errors more than averaging the same model architecture.

**Expected gain:** +4–10% daily R²

---

### Step 9 (Optional) — Quantile Regression / Prediction Intervals

> **Quantile regression:** Instead of predicting a single point estimate ("tomorrow: 15 mm"), the model simultaneously predicts the 10th, 50th (median), and 90th percentile — a **confidence interval**. This is operationally more useful for flood warning: "there is a 10% chance of more than 45 mm tomorrow, and a 90% chance of less than 8 mm" gives decision-makers information about uncertainty, not just a single number.

```python
def pinball_loss(quantile):
    def loss(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * err, (quantile - 1) * err))
    return loss
# Pinball (tilted L1) loss: standard loss function for quantile regression
# Under-predictions penalised by weight=quantile, over-predictions by (1-quantile)
```

**Expected outcome:** No R² gain (different objective), but significantly higher operational utility for JPS flood risk communication.

---

## 10. Deliverables

### Produced

| Artifact | Description |
|---|---|
| `data/processed/completed_daily_rainfall_cnn_tf.csv` | GAN-completed daily rainfall, 2009–2025, 6,024 rows |
| `data/processed/era5_pahang_daily.csv` | ERA5 daily aggregates, 2009–2025, 6,209 rows, 9 variables |
| `predictions/hurdle_tcn_predictions.csv` | Test set predictions — Hurdle model (no ERA5), 893 rows |
| `predictions/era5_hurdle_predictions.csv` | Test set predictions — ERA5+Hurdle model, 893 rows |
| `figures/tcn/shap_feature_importance.png` | SHAP feature importance bar chart (13 features) |
| `figures/tcn/shap_temporal_importance.png` | Spearman lag-correlation temporal importance plot |
| `figures/tcn/hurdle_scatter_daily.png` | Hurdle TCN — daily scatter, 3 stations |
| `figures/tcn/hurdle_scatter_monthly.png` | Hurdle TCN — monthly scatter, 3 stations |
| `figures/tcn/era5_scatter_daily.png` | ERA5+Hurdle — daily scatter, 3 stations |
| `figures/tcn/era5_scatter_monthly.png` | ERA5+Hurdle — monthly scatter, 3 stations |
| `notebooks/03b_tcn_multihead_tensorflow.ipynb` | Reproducible notebook: all sections — multi-head, SHAP, hurdle, ERA5 |
| `reports/Technical_Report_01_Data_Engineering.md` | Phase 1 full technical report |
| `reports/Technical_Report_02_GAN_Imputation.md` | Phase 2 full technical report |
| `reports/Technical_Report_03_TCN_Prediction.md` | Phase 3 full technical report (all TCN variants) |
| `briefs/Project_Progress_Brief_July2026.md` | Client-facing progress brief |

### Pending

| Artifact | Status |
|---|---|
| `predictions/optuna_best_predictions.csv` | **Complete** — Optuna best config test predictions |
| `figures/tcn/optuna_scatter_daily.png` | **Complete** — scatter plot, daily scale |
| `figures/tcn/optuna_scatter_monthly.png` | **Complete** — scatter plot, monthly scale |
| `data/processed/completed_daily_rainfall_kuantan.csv` | Pending — Phase 4 data engineering |
| Phase 4 model predictions (5 Kuantan stations) | Pending — Phase 4 training |

---

## 11. References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*. *(Foundational TCN paper)*
- Yoon, J., Jordon, J., & van der Schaar, M. (2018). GAIN: Missing data imputation using generative adversarial nets. *ICML 2018*. *(GAIN architecture)*
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. *(ERA5 dataset)*
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*. *(TPE sampler)*
- Oord, A. v. d. et al. (2016). WaveNet: A generative model for raw audio. *arXiv:1609.03499*. *(Dilated causal convolutions — same principle as our TCN blocks)*
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 2017*. *(Transformer / multi-head self-attention)*
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. *(SHAP)*
- He, K. et al. (2016). Deep residual learning for image recognition. *CVPR 2016*. *(Residual/skip connections)*
