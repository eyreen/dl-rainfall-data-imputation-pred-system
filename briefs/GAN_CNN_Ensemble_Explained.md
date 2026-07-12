# GAN Imputation v2 (CNN Ensemble) — What We Did and Why

A plain-English walkthrough of the improved imputation pipeline — `02b_gan_cnn_tensorflow.ipynb`. This is the second, tuned iteration of the imputation work described in `GAN_Imputation_Explained.md` — read that one first if you haven't; this brief focuses on what changed and why.

---

## The Problem, Recapped

Our rainfall dataset has gaps — days where a sensor didn't record anything (~2-3% of values). We need to fill those gaps with realistic values before the TCN prediction model can use the data, since it needs an unbroken daily series.

The original approach (see `GAN_Imputation_Explained.md`) used **GAIN** — a GAN adapted for filling in missing table values — with a 15-day temporal window and a simple feed-forward (MLP) Generator/Discriminator. It worked, but left room to improve: R² was slightly negative (-5.56%), meaning the model was, technically, doing worse than just guessing the average.

This notebook is the tuned, second attempt at the same problem, built in TensorFlow.

---

## What Changed, and Why

### 1. Convolutional Layers Instead of Plain Dense Layers

The original Generator/Discriminator were MLPs (simple stacks of fully-connected layers) that treated the 15-day window as a flat list of numbers, with no built-in sense of "day 5 is next to day 6." This version uses **Conv1D** layers instead — the same building block used in the TCN prediction models — which naturally understand that nearby days are related, without having to learn that from scratch.

```
Architecture: Conv1D 32 → 64 → 64 → 32 filters
              LeakyReLU + BatchNorm + Dropout after each layer
```

### 2. A First Attempt Overfit Badly — Then Got Fixed

Before landing on the final settings, an earlier CNN attempt (64→128→128→64 filters, no dropout, no weight decay, trained for a fixed 800 epochs) was tried and it **overfit catastrophically** — R² of -534.6%, far worse than the original MLP. A bigger, less-regularised model with more training time didn't help; it just memorised the training data and produced nonsense on unseen positions.

The fix — captured in the table below — was to make the model *smaller* and add several forms of regularisation (techniques that discourage the model from memorising noise):

| Setting | Original CNN (overfit) | Finetuned CNN (this notebook) | Why the change helps |
|---|---|---|---|
| Filters | 64→128→128→64 | **32→64→64→32** | Fewer parameters = less capacity to memorise |
| Dropout | 0.1 | **0.25** | Randomly "turns off" more neurons each step, forcing the network to not rely on any single path |
| L2 weight decay | None | **1e-4** | Penalises very large weights, which tend to correlate with overfitting |
| Alpha (reconstruction weight) | 10.0 | **5.0** | Less pressure to perfectly reproduce observed values, leaving more room to generalise |
| Training length | 800 fixed epochs | **Early stopping (patience=60)** | Stops automatically once validation performance stops improving, instead of forcing a fixed epoch count that runs well past the useful point |

### 3. An Ensemble of 5 Models Instead of 1

Rather than trusting a single trained Generator, this version trains **5 independent models** with different random seeds (42, 123, 456, 789, 1024) and **averages their predictions** at the end.

Why this helps: each individual model's training is somewhat random — different starting weights, different mini-batch orderings — so each one lands in a slightly different place, especially for ambiguous missing values. Averaging 5 independent "opinions" smooths out that randomness the same way averaging 5 different judges' scores is more reliable than trusting just one.

---

## Step-by-Step: What the Code Actually Does

### Step 1 — Load Raw Data

Same source as before: the raw daily CSV exported from the telemetry stations, ~2-3% missing per station.

### Step 2 — Log1p + MinMax Normalise

Identical reasoning to the original: `log1p` compresses the heavy skew (mostly small values, occasional large storms), then `MinMaxScaler` squeezes everything to [0, 1] — fit only on the observed (non-missing) values, so missing positions can't leak into the scale.

### Step 3 — 20% Validation Holdout

As before, we can't validate against real missing values (we don't know the true answer). So we artificially hide 20% of the *known* values, train without them, then compare predictions against the real numbers we hid. This is the only honest way to measure imputation accuracy.

### Step 4 — Build 15-Day Temporal Windows

Same windowing idea as the original GAIN: each sample is a 15-day window centred on the day being filled, giving the model both temporal context (nearby days) and spatial context (all 3 stations at once).

### Step 5 — Train Each Ensemble Member

For each of the 5 seeds:
- **Generator** takes the windowed data (missing positions zeroed) + mask, and outputs a filled version.
- **Discriminator** takes the filled data + a partial "hint" about which positions are real, and tries to guess which positions were actually observed vs. generated.
- The Generator is trained to (a) fool the Discriminator and (b) accurately reconstruct the positions it *does* know, weighted by Alpha=5.0.
- Training uses a **cosine-decaying learning rate** (starts at 5e-4, smoothly decays to 1e-5) and stops early once validation reconstruction error stops improving for 60 epochs straight.

### Step 6 — Validate: Single Model vs. Ensemble

Both a single model (seed=42 alone) and the full 5-model ensemble average are evaluated against the 20% holdout, so we can directly see the benefit the ensembling provides.

### Step 7 — Compare Against the Original MLP

| Model | NRMSE% | NMAE% | R²% |
|---|---|---|---|
| MLP v1 — PyTorch (original GAIN) | 18.00 | 7.62 | **-5.56%** |
| CNN overfit — TF (800 fixed epochs) | 44.13 | 16.99 | -534.6% |
| CNN finetuned — TF, single model | 16.13 | 6.68 | +15.23% |
| **CNN ensemble ×5 — TF (final)** | **15.87** | **6.49** | **+17.93%** |

The finetuned CNN — and especially the ensembled version — clearly beats the original MLP on every metric, and crucially flips R² from *negative* to *positive*, meaning the model now genuinely beats a naive "always guess the average" baseline.

### Step 8 — Visual Checks

- **Scatter plot** of actual vs. imputed values per station on the holdout set.
- **Distribution comparison** — histograms of observed vs. imputed non-zero rainfall values, to visually confirm the imputed values follow a realistic shape rather than clustering artificially.

### Step 9 — Final Imputation Pass

All 5 models are **retrained from scratch on the full observed dataset** (no holdout this time, since the holdout was only needed for validation) and their outputs averaged to produce the final filled-in values for every missing position.

### Step 10 — Export

| File | What it is |
|---|---|
| `completed_daily_rainfall_cnn_tf.csv` | Full dataset, zero NaN, plus `*_imputed` flag columns marking which values were filled |

The distribution shift after imputation is tiny — station means shift by only a fraction of a percent — confirming the filled values preserve the real statistical character of the data rather than distorting it.

---

## Understanding "R²% = 17.93%" — Is That Good?

Positive R² here means the ensemble genuinely outperforms the naive baseline of "always predict the station average" — a real, if modest, achievement for imputing daily rainfall from just 3 stations' own history, with no external weather data.

The notebook's own conclusion, worth keeping for the report: **R² in this range (~15–20%) is roughly the realistic ceiling** for this data configuration. Going meaningfully higher (R²>35%) would require additional inputs the current dataset doesn't have — more stations, or meteorological variables like humidity, pressure, or wind. The positive R² is the important signal, not the exact number.

---

## The Big Picture

```
Raw daily data          CNN GAIN ensemble         Completed dataset
(~2-3% missing)    →   (5 models, Conv1D,    →   (zero gaps, R²=+17.93%
                        early stopping,            vs -5.56% for MLP v1)
                        averaged)
                                ↓
                        Feeds into the TCN
                        prediction notebooks
                        (03_tcn_prediction*.ipynb)
```

---

## Key Terms Quick Reference

| Term | Plain English |
|---|---|
| Imputation | Filling in missing data |
| GAIN | GAN adapted for table-style missing data (see original GAN brief) |
| Conv1D | A convolution layer applied along a sequence — understands "nearby" |
| Overfitting | Model memorises training data instead of learning general patterns |
| Regularisation | Techniques (dropout, L2, early stopping) that discourage overfitting |
| Dropout | Randomly disabling neurons during training to prevent over-reliance on any one path |
| L2 weight decay | A penalty on large weights, encouraging simpler, more general solutions |
| Alpha | How strongly the Generator is pushed to reconstruct known values exactly |
| Ensemble | Training several models and averaging their outputs to reduce randomness |
| Cosine LR decay | Learning rate that smoothly shrinks over training |
| Early stopping | Halting training once validation performance stalls |
| NRMSE% / NMAE% | Error size as a percentage of mean rainfall — comparable across stations |
| R² | How much better than a dumb average (1=perfect, 0=same as average, <0=worse) |
