# TCN Prediction (TensorFlow) — What We Did and Why

A plain-English walkthrough of the TensorFlow rewrite of Phase 3 — `03_tcn_prediction_tensorflow.ipynb`.

---

## The Problem in One Sentence

We have a complete, gap-free rainfall record (from the GAN imputation phase). Now we want a model that looks at the **last 90 days** of rainfall across all 3 stations and predicts the **next 14 days**, all at once.

---

## How This Differs From the Initial Version

The first TCN notebook (`03_tcn_prediction.ipynb`) predicted three *separate* things — tomorrow, this week's total, this month's total — using 35 hand-built features. This TensorFlow version simplifies the target and the inputs considerably:

| | Initial version | This version |
|---|---|---|
| Input features | 35 engineered (lags, rolling means, cyclical time, dry-streaks...) | 3 (just raw rainfall per station) |
| Prediction target | 3 separate scales: next-day, next-week-sum, next-month-sum | 1 target: the next **14 individual days**, day-by-day |
| Output shape | 3 heads × 3 stations = 9 numbers | 14 days × 3 stations = 42 numbers |
| Framework | PyTorch, hand-written training loop | TensorFlow/Keras, built-in `.fit()` |
| Metrics | RMSE/MAE in mm | NRMSE%/NMAE%/R²% (percentage of the mean — easier to compare across stations) |
| Normalisation | StandardScaler (mean 0, std 1) | MinMaxScaler (0 to 1) — better suited to a naturally non-negative quantity like rainfall |

Why simplify the inputs? Feature engineering (rolling averages, lags, etc.) is really just a way to hand the model information it could, in principle, learn to extract itself from raw history — at the cost of more code and more chances for subtle bugs like data leakage. Since the TCN's whole design is built around learning from raw sequences, this version tests whether it can find those patterns unaided, using a longer, more literal forecast horizon (14 actual days instead of 3 aggregate numbers) as the target instead.

---

## What is a TCN? (Recap)

A **Temporal Convolutional Network** processes a sequence of days using stacked 1D convolutions with three key properties:

- **Causal padding** — the model at day *t* only ever sees days ≤ *t*. No peeking at the future.
- **Exponential dilation** — each of the 4 stacked blocks looks further back than the last (dilation 1 → 2 → 4 → 8), so the model accumulates a wide view of history without needing dozens of layers.
- **Residual connections** — shortcuts around each block that keep gradients flowing cleanly during training.

---

## Step-by-Step: What the Code Actually Does

### Step 1 — Load the GAN-Completed Dataset

We load `completed_daily_rainfall_cnn_tf.csv` — the output of the improved CNN-ensemble GAN imputation (see the companion brief on `02b_gan_cnn_tensorflow.ipynb`). Zero NaNs, 3 stations, full 2009–2025 range. We also plot the whole series with imputed points marked in red, just to visually sanity-check that the filled-in values look plausible next to the real ones.

### Step 2 — Log1p + MinMax Normalise

Same reasoning as the imputation phase: rainfall is heavily skewed (mostly 0–5mm, occasionally 200mm+ during a storm). We apply `log1p` to compress the scale, then squeeze everything into [0, 1] with a `MinMaxScaler`.

### Step 3 — Build 90-Day → 14-Day Sequences

```
Days 1–90   →  predict Days 91–104
Days 2–91   →  predict Days 92–105
Days 3–92   →  predict Days 93–106
...
```

Each sample is a sliding window: 90 days of history in, 14 days of forecast out, across all 3 stations simultaneously. Split chronologically: 70% train / 15% validation / 15% test — no shuffling across time, so the model is always evaluated on data that comes *after* what it trained on, just like real deployment.

### Step 4 — The TCN Architecture

```
Input (90 days × 3 stations)
        ↓
  4 × dilated causal Conv1D blocks (dilations 1, 2, 4, 8)
  each block: Conv1D → LayerNorm → Dropout → Conv1D → LayerNorm → Dropout → residual add
        ↓
  Take the representation at the very last timestep
        ↓
  Dense(128) → Dropout
        ↓
  Dense(42) → reshape to (14 days × 3 stations)
```

64 filters per block, kernel size 3. The receptive field (how far back the model can "see") works out to roughly 90 days — matched to the lookback window, so no history is wasted or starved.

### Step 5 — Train

- **Loss:** MSE on the normalised values, tracked alongside MAE.
- **Optimiser:** Adam with cosine-decay learning rate (starts at 1e-3, smoothly decays to 1e-4 over training).
- **Early stopping:** patience of 30 epochs on validation loss, restoring the best weights found. Up to 200 epochs total.
- Training uses Keras's built-in `.fit()` with callbacks — considerably less code than the initial version's hand-rolled PyTorch loop, and less room for bugs in things like checkpoint tracking.

### Step 6 — Evaluate With Percentage Metrics

Instead of raw RMSE/MAE in millimetres (where "is 18mm good?" isn't obvious without context), everything is reported as a **percentage of the mean observed rainfall**:

```
NRMSE% = (RMSE ÷ mean rainfall) × 100
NMAE%  = (MAE  ÷ mean rainfall) × 100
R²%    = R² × 100
```

This makes results comparable across stations with different average rainfall, and immediately readable — "22% error" needs no extra context, whereas "18mm error" does.

Metrics are computed separately for the validation set and the (unseen) test set.

### Step 7 — Accuracy vs. Forecast Horizon

Forecasting further into the future is inherently harder — day-1 predictions should be noticeably better than day-14 predictions, since more can change in two weeks. We plot NRMSE% for each of the 14 forecast days to see exactly how fast accuracy degrades with lead time. This is a diagnostic a single aggregate metric can't give you.

### Step 8 — Visual Checks

- **Scatter plots** of actual vs. predicted rainfall per station, with a red diagonal marking "perfect prediction."
- **Sample 14-day forecasts** for 5 randomly chosen points in the test set, showing the 90-day history, the actual next 14 days, and the predicted next 14 days side-by-side as bar charts — a concrete, human-checkable view of what the model is actually outputting.

### Step 9 — Save Everything

| File | What it is |
|---|---|
| `tcn_rainfall_tf.keras` | The trained model (Keras native format) |
| `tcn_test_predictions.csv` | Every test-set forecast, actual vs. predicted, per day per station |
| `tcn_input_timeseries.png`, `tcn_training_curves.png`, `tcn_nrmse_by_day.png`, `tcn_scatter.png`, `tcn_sample_forecasts.png` | Supporting plots for the report |

---

## Why NRMSE% Looks Large for Daily Rainfall

A quirk worth flagging in the report: daily rainfall NRMSE% often looks alarmingly high (30–60%+). This is expected, not a bug — it's a **zero-inflated** quantity, meaning most days are 0mm or near-0mm, punctuated by rare storm days of 100–360mm. The *mean* observed rainfall is small (~8mm), so even a modest absolute error becomes a large percentage of that small mean. The same absolute error would look far more reasonable expressed against, say, the standard deviation instead of the mean — but reporting relative to the mean is the more conservative, more standard choice.

---

## Key Terms Quick Reference

| Term | Plain English |
|---|---|
| TCN | A time-series model built from 1D convolutions instead of recurrence |
| Causal padding | The model can only see the past, never the future |
| Dilation | Skipping steps between inputs so a layer sees further back |
| Receptive field | How many days of history the model can actually use |
| Lookback window | Days of history fed in (90 here) |
| Horizon | Days ahead being forecast (14 here) |
| Sliding window | Overlapping chunks of consecutive days used as training samples |
| log1p | A math trick (`log(x+1)`) that compresses skewed data |
| MinMax normalisation | Rescaling values into the [0, 1] range |
| NRMSE% / NMAE% | Error size expressed as a percentage of mean rainfall — comparable across stations |
| R²% | How much better than a dumb average, as a percentage |
| Cosine LR decay | Learning rate that smoothly shrinks over training, following a cosine curve |
| Early stopping | Halting training once validation performance stalls |
