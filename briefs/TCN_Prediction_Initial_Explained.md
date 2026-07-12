# TCN Prediction (Initial Version) — What We Did and Why

A plain-English walkthrough of the first version of Phase 2/3 of the Pahang Rainfall project — `03_tcn_prediction.ipynb`.

---

## The Problem in One Sentence

Phase 1 gave us a complete rainfall record with no gaps. Now we want to **forecast the future** — given everything up to today, predict what's coming at three different zoom levels: tomorrow, this week, and this month.

That's what a **TCN (Temporal Convolutional Network)** is for.

---

## Why Predict Three Scales at Once?

A farmer or dam operator doesn't just want "will it rain tomorrow" — they want different answers for different decisions:

| Scale | Question it answers | Decision it supports |
|---|---|---|
| **Daily** | How much rain tomorrow? | Should I harvest today? |
| **Weekly** | How much rain this week? | Should I reschedule a task? |
| **Monthly** | How much rain this month? | Should I plan for flood risk? |

Rather than building three separate models, we built **one model with a shared "brain" and three separate output heads** — one per scale. The shared brain learns general rainfall patterns; each head specialises in translating that into a specific forecast horizon.

---

## What is a TCN?

**TCN = Temporal Convolutional Network.** It's an alternative to the more familiar LSTM/RNN for time series, built entirely from **1D convolutions** — the same building block used in image models, just applied to a sequence of days instead of a grid of pixels.

Three ideas make it work for time series:

| Idea | Plain English |
|---|---|
| **Causal convolution** | The model at day *t* can only see days ≤ *t*. It never accidentally peeks at the future while training. |
| **Dilated convolution** | Instead of only looking at neighbouring days, each layer looks further and further back (1 day, 2 days, 4 days, 8 days, 16 days...) by skipping steps. Stack a few of these and the model "sees" months of history without needing a huge number of layers. |
| **Residual connections** | Shortcuts that let information (and gradients) skip past a block if that block isn't helping. Keeps deep networks trainable. |

**Receptive field** = how many days into the past the model can actually "see." With 5 stacked layers and dilations [1,2,4,8,16], our receptive field works out to **125 days** — about 4 months of context — even though we only feed it a 90-day window.

---

## Step-by-Step: What the Code Actually Does

### Step 1 — Load the Completed Dataset

We load `completed_daily_rainfall.csv` — the zero-NaN output from Phase 1's GAIN imputation. 6,024+ rows, 3 stations, no missing values.

### Step 2 — Engineer 35 Features Per Day

A TCN only knows what you feed it. Instead of giving it just 3 raw rainfall numbers per day, we hand it a richer picture:

| Feature group | Count | What it captures |
|---|---|---|
| Raw rainfall (nada, lembing, reman) | 3 | The basic signal |
| Log-transformed rainfall | 3 | Same signal, compressed scale (see GAN brief) |
| Cyclical time (sin/cos of day-of-year, month) | 4 | "What season is it?" — monsoon vs. dry season |
| Rolling averages (3, 7, 14, 30-day) | 12 | "Has it been wet or dry lately?" |
| Lag features (1, 3, 7-day-ago values) | 9 | "What happened recently?" |
| Consecutive dry-day streaks | 3 | "How long since it last rained?" |
| Cross-station average | 1 | "What's the regional picture?" |
| **Total** | **35** | |

### Step 3 — Build the Three Targets

For every day *t*, we compute what actually happened next:

```
Daily target   = rainfall on day t+1
Weekly target  = total rainfall from t+1 to t+7
Monthly target = total rainfall from t+1 to t+30
```

Rows near the end of the dataset (where a full 30-day future doesn't exist yet) are dropped, since we can't score them.

### Step 4 — Chronological Train / Val / Test Split

We split by **time**, not randomly — 70% train (earliest years) → 15% validation → 15% test (most recent years). This matters: if we shuffled randomly, the model could "cheat" by learning from future data to predict the past, which is impossible in real deployment. A chronological split simulates the real situation of only ever knowing the past.

### Step 5 — Normalise Everything

- **Features:** scaled with `StandardScaler` (mean 0, std 1), fitted only on the training set.
- **Targets:** each scale (daily/weekly/monthly) gets its own scaler, also fitted only on training data.

Fitting only on training data prevents the validation/test sets from leaking information into the scaling.

### Step 6 — Build Sliding-Window Sequences

Each training example = 90 consecutive days of the 35 features, paired with the 3 targets (daily/weekly/monthly) for the last day in that window. This is what gets fed to the TCN.

### Step 7 — The Multi-Head TCN Architecture

```
Input (35 features × 90 days)
        ↓
  Shared backbone: 5 stacked dilated-causal-conv blocks
  (dilations 1, 2, 4, 8, 16 — GELU activation, BatchNorm, Dropout)
        ↓
  Take the representation at the most recent timestep
        ↓
   ┌────────────┬─────────────┬──────────────┐
   ↓            ↓             ↓
Daily head   Weekly head   Monthly head
(3 values)   (3 values)    (3 values)
```

~75K parameters total — deliberately kept compact because we only have a few thousand training sequences, and a bigger model would just memorise noise.

### Step 8 — Train

- **Loss:** Mean Squared Error, summed equally across all three scales, so the model doesn't neglect one horizon in favour of another.
- **Optimiser:** AdamW with cosine annealing (learning rate smoothly decays over training) and weight decay (a mild penalty that discourages overly large weights, reducing overfitting).
- **Early stopping:** if validation loss doesn't improve for 40 epochs, stop and keep the best checkpoint. Trained for up to 300 epochs.

### Step 9 — Evaluate on the Test Set

The test set is the most recent 15% of years — data the model has genuinely never seen, at any stage. We inverse-transform predictions back to millimetres and compute RMSE, MAE, and R² per station, per scale.

### Step 10 — Save Everything

| File | What it is |
|---|---|
| `tcn_multiscale_model.pth` | Trained weights + all scalers + config, in one file |
| `tcn_prediction_results.csv` | Every test-set prediction (daily/weekly/monthly, actual vs. predicted) |

---

## Why This Was the "Initial" Version

This notebook proved the multi-scale, multi-head concept works end-to-end — one shared backbone genuinely can serve three different forecast horizons. But it has some rough edges that motivated the later rewrite (`03_tcn_prediction_tensorflow.ipynb`):

- **Hand-engineered features (35 of them)** — more moving parts, more chances for subtle leakage (e.g., a rolling average computed carelessly could leak future info), and more to explain to a non-technical reader.
- **PyTorch training loop written by hand** — full control, but verbose and easy to introduce bugs in (manual gradient clipping, manual best-checkpoint tracking, etc.).
- **Absolute error metrics only (RMSE/MAE in mm)** — hard to judge "is 18mm good or bad?" without extra context, which is why the newer version switched to percentage-based metrics (NRMSE%, NMAE%) that are self-explanatory regardless of station.
- **StandardScaler on raw rainfall** — allows negative normalised values, which is a slightly awkward fit for a naturally non-negative quantity like rainfall.

The core TCN idea (causal + dilated + residual + multi-horizon) carried forward into later versions; what changed was the plumbing around it.

---

## Key Terms Quick Reference

| Term | Plain English |
|---|---|
| TCN | A time-series model built from 1D convolutions instead of recurrence |
| Causal convolution | Can only see the past, never the future |
| Dilated convolution | Skips steps to see further back without adding many layers |
| Residual connection | A shortcut that helps gradients flow through deep networks |
| Receptive field | How many days of history the model can actually "see" |
| Multi-head | One shared backbone, several separate output branches |
| Lookback window | How many past days are fed in as input (90 here) |
| Horizon | How far ahead we're forecasting (1, 7, or 30 days here) |
| Chronological split | Train/val/test split by time order, not randomly |
| Early stopping | Stop training once validation performance stalls, to avoid overfitting |
| RMSE / MAE | Average error size, in millimetres |
| R² | How much better than always predicting the average |
