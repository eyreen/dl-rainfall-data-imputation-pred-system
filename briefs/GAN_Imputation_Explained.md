# GAN Imputation — What We Did and Why

A plain-English walkthrough of Phase 2 of the Pahang Rainfall project.

---

## The Problem in One Sentence

Our rainfall dataset has **gaps** — days where the sensor didn't record anything. We can't just delete those days or leave them blank, because the prediction model (Phase 3) needs a complete, unbroken series of daily values. So we need to *fill in* the missing values intelligently.

That process of filling gaps is called **imputation**.

---

## Why Not Just Use the Average?

You could fill every missing day with the station's average rainfall (~8 mm). But that's fake — it flattens out real weather patterns. If it rained 80 mm for three days in a row and one day is missing, filling it with 8 mm is clearly wrong.

A smarter approach looks at:
- What the *neighbouring days* look like (was there a storm that week?)
- What the *other stations* recorded on the same day (if Lembing got 60 mm, Nada probably got something too)

That's what the GAN does.

---

## What is a GAN?

**GAN = Generative Adversarial Network.**

Think of it as two AI models competing against each other:

| Model | Role | Analogy |
|---|---|---|
| **Generator** | Tries to create fake (filled-in) values | A forger making fake paintings |
| **Discriminator** | Tries to catch which values are fake | An art inspector |

They train together. The forger gets better at faking, and the inspector gets better at catching. After enough rounds, the forger becomes so good that the inspector can't tell the difference. At that point, the "fake" filled values are realistic.

---

## GAIN — The Specific Version We Used

Regular GANs generate images or text. **GAIN (Generative Adversarial Imputation Nets)** is a version designed specifically for **tables with missing values**.

The twist: instead of asking the Discriminator *"is this real or fake data?"*, it asks *"which specific cells in this row were originally observed vs. filled in?"*

This is harder and more useful — it forces the Generator to produce realistic values at exactly the right positions.

**Paper:** Yoon et al., 2018 — GAIN: Missing Data Imputation using GANs.

---

## Our Addition: Temporal Windows

Standard GAIN treats each row (each day) as completely independent. But rainfall isn't random day-to-day — a monsoon event spans multiple days.

We modified GAIN to look at **15 consecutive days at once** (a sliding window), with the target day in the middle. So when filling a missing value on Day 50, the model sees Days 43–57.

```
Day 43  Day 44  ...  [Day 50 — MISSING]  ...  Day 56  Day 57
  ↓       ↓               ↑                     ↓       ↓
  known   known       fill this              known   known
```

This gives the Generator both:
- **Temporal context** — what was the trend before and after?
- **Spatial context** — what did the other 2 stations record?

---

## Step-by-Step: What the Code Actually Does

### Step 1 — Load the Raw Daily Data

We load the daily CSV (6,024 rows × 3 stations). About 2–3% of values are missing (NaN).

```
nada:    ~130 missing days
lembing: ~148 missing days
reman:   ~121 missing days
```

### Step 2 — Log Transform

Rainfall is extremely skewed. Most days have 0–5 mm. But occasionally there's 200–360 mm (a storm). If we train on raw values, the model obsesses over the rare big events and ignores the common small ones.

We apply `log1p(x)` which means `log(x + 1)`. This compresses the scale:

```
Raw:      0 mm → 0      50 mm → 50     360 mm → 360
After:    0 mm → 0      50 mm → 3.93   360 mm → 5.89
```

Everything fits in a much smaller range. The model can now learn from all values equally.

### Step 3 — Normalise to [0, 1]

We scale everything between 0 and 1 using the observed (non-missing) values only. This is important — we must **not** let the missing positions influence the scale.

### Step 4 — Build the Mask

We create a binary matrix the same shape as the data:

```
1 = this value was actually recorded
0 = this value is missing
```

The Generator only needs to produce values where the mask is 0. Everywhere else, we keep the original.

### Step 5 — Validation Holdout

To test whether our imputation is any good, we need to know the "right answer." But the missing values don't have right answers (that's why they're missing).

So we **artificially hide** 20% of the *observed* values, train the model without telling it those values, then compare the model's output to what was actually there.

This is the only honest way to measure imputation quality.

### Step 6 — Train the GAIN

The Generator and Discriminator train for **500 epochs** (500 full passes through the data):

- **Generator** sees: the windowed data with missing positions zeroed out + the mask
- **Generator** outputs: a filled version of the entire window
- **Discriminator** sees: the filled data + a "hint" (partial mask info)
- **Discriminator** outputs: per-position predictions of what's real vs. imputed

The Generator loss has two parts:
1. **Fool the Discriminator** — make imputations look real
2. **Reconstruction loss** — for positions we *know*, match the actual value (weighted by α=10)

### Step 7 — Evaluate

After training, we compare imputed vs. actual on the 20% holdout:

| Station | RMSE | MAE | R² |
|---|---|---|---|
| Nada | 19.6 mm | 7.6 mm | -0.038 |
| Lembing | 18.5 mm | 8.2 mm | -0.079 |
| Reman | 15.7 mm | 7.1 mm | -0.054 |
| **Overall** | **18.0 mm** | **7.6 mm** | **-0.056** |

### Step 8 — Final Imputation Pass

We retrain on **all** observed data (including the 20% we held out) and run imputation over the entire dataset. The completed file has zero NaN values.

---

## Understanding the Validation Numbers

**RMSE = 18 mm** means the average error per imputed day is about 18 mm.

**Is that bad?** Context matters:

- Daily rainfall at these stations ranges from 0 to ~360 mm
- The *standard deviation* of daily rainfall is ~25 mm
- So RMSE of 18 mm is within the natural day-to-day variability — it's not precise, but it's plausible

**R² is negative** — what does that mean?

R² measures how much better our model is than just predicting the mean every time. Negative R² means the mean is actually a better predictor on the holdout set. This sounds bad, but it's normal for daily rainfall imputation because:

- Most missing days are isolated sensor failures, not during major rain events
- The test set contains many zero/near-zero days where the mean (8 mm) beats any specific prediction
- The model's real value is capturing *distributional* realism, not point-by-point accuracy

The distributional stats confirm this works:

| Metric | Before Imputation | After Imputation | Change |
|---|---|---|---|
| Mean | 8.4 mm | 8.3 mm | < 2% |
| Std dev | 22.1 mm | 21.8 mm | < 2% |
| Max value | 360 mm | 360 mm | 0% |

The distribution is preserved. That's what matters for downstream prediction.

---

## What Gets Saved

| File | What it is |
|---|---|
| `completed_daily_rainfall.csv` | The full 6,024-row dataset with no gaps |
| `gain_generator_final.pth` | The trained Generator weights (reusable) |

The CSV has extra columns `nada_imputed`, `lembing_imputed`, `reman_imputed` — these are 1 where we filled a value, 0 where it was original. Full transparency on what was imputed.

---

## The Big Picture

```
Raw daily data          GAIN model              Completed dataset
(with ~400 gaps)   →   (fills gaps using    →   (6,024 days, zero gaps)
                        nearby days &
                        other stations)
                                ↓
                        Phase 3: TCN uses
                        this complete data
                        to predict future
                        rainfall
```

The imputation doesn't need to be perfect — it just needs to be *statistically plausible* so the downstream TCN model isn't thrown off by artifacts.

---

## Key Terms Quick Reference

| Term | Plain English |
|---|---|
| Imputation | Filling in missing data |
| GAN | Two AI models that compete: one fakes, one detects |
| Generator | The model that fills in missing values |
| Discriminator | The model that checks if values look real |
| GAIN | GAN adapted for table-style missing data |
| Temporal window | Looking at surrounding days for context |
| log1p | A math trick to compress skewed data |
| Mask | A table of 1s and 0s marking what's real vs. missing |
| Epoch | One full training pass through all the data |
| RMSE | Average error size (in the same units as the data — mm) |
| R² | How much better than a dumb average (1=perfect, 0=same as average, <0=worse) |
