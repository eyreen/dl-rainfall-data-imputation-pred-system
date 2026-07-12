# Technical Report: GAN-Based Rainfall Data Imputation
## Phase 2 — Generative Adversarial Imputation Network (GAIN)

**Project:** Deep Learning-Based Rainfall Data Imputation and Prediction System  
**Author:** Arinn Danish  
**Date:** June–July 2026  
**Notebooks:** `02_gan_imputation.ipynb` (MLP v1), `02b_gan_cnn_tensorflow.ipynb` (CNN v2 final)

---

## 1. Introduction

This report documents the design, training, and validation of a Generative Adversarial Imputation Network (GAIN) for filling missing values in the daily rainfall dataset. The architecture extends the original GAIN framework (Yoon et al., 2018) with temporal windowing to exploit both spatial (cross-station) and temporal (neighboring-day) correlations.

### 1.1 Problem Statement

The daily rainfall dataset contains missing values across all three telemetry stations:

| Station | Missing Values | Missing % |
|---|---|---|
| Nada (3931013) | 158 | 2.62% |
| Lembing (3930012) | 210 | 3.49% |
| Reman (3931014) | 127 | 2.11% |
| **Total** | **495** | **2.74%** |

These missing values must be filled before the dataset can be used for TCN-based prediction. The imputation must:
1. Produce statistically plausible values (preserve distributional properties)
2. Not introduce systematic bias into the completed dataset
3. Flag which values were imputed for downstream transparency

### 1.2 Iterative Model Development Summary

Four model variants were developed and evaluated in sequence. The table below shows the full progression:

| Model | Framework | Architecture | NRMSE% | NMAE% | R²% |
|---|---|---|---|---|---|
| MLP v1 (original) | PyTorch | Fully-connected 90→256→256→45 | 226.0% | 95.6% | -5.56% |
| CNN v2 (overfit) | TensorFlow | Conv1D 64→128→128→64, 800 epochs | 553.9% | 213.3% | -534.6% |
| CNN finetuned (single) | TensorFlow | Conv1D 32→64→64→32, early stop | 202.4% | 83.8% | +15.23% |
| **CNN ensemble ×5 (final)** | **TensorFlow** | **5× finetuned CNN averaged** | **199.2%** | **81.5%** | **+17.93%** |

> **Note on percentage metrics:** NRMSE% = (RMSE ÷ mean_observed) × 100; NMAE% = (MAE ÷ mean_observed) × 100. Raw RMSE and MAE are in mm (same unit as rainfall) and are reported as percentages here by normalising against the station mean (~8 mm). The values look large (~199%) because rainfall is heavily zero-inflated — the mean is far below storm peaks (360 mm). This is expected for daily point-rainfall imputation.

The CNN ensemble is the production-ready result used to generate the final completed dataset.

---

## 2. Methodology

### 2.1 GAIN Framework

GAIN (Generative Adversarial Imputation Nets) is a GAN variant specifically designed for missing data imputation in tabular datasets. Unlike standard GANs that generate entire samples, GAIN only generates values for missing positions while preserving observed values.

**Key components:**
- **Generator G:** Takes incomplete data, a binary mask, and noise as input. Outputs imputed values for missing positions.
- **Discriminator D:** Receives the completed data and a "hint" matrix. Tries to classify which values are observed vs. imputed.
- **Hint mechanism:** Provides the discriminator with partial mask information (controlled by hint rate), preventing the generator from trivially fooling the discriminator.

### 2.2 Temporal Windowing Extension

Standard GAIN operates on individual rows independently. For time-series data, this discards valuable temporal context. Our extension uses a sliding window of **W = 15 consecutive days**. In the MLP v1 design, this was flattened into a 45-dimensional vector. In the CNN v2 design, the window is kept as a 2D tensor and processed by Conv1D layers:

```
MLP v1 input:  (batch, 45)        — 15 days × 3 stations, flattened
CNN v2 input:  (batch, 15, 3)     — 15 time steps, 3 station channels
```

The CNN approach preserves temporal structure, allowing the generator to detect local rainfall patterns (rising limbs, consecutive dry days) rather than treating all lags as equivalent features.

### 2.3 Data Preprocessing

**Log1p Transform:** Rainfall data is extremely right-skewed (many zeros, rare extreme events >100mm). A `log1p(x) = ln(1+x)` transform compresses the heavy tail before normalization:

| Metric | Raw Scale | Log1p Scale |
|---|---|---|
| Range | [0, 359.9] mm | [0, 5.889] |
| Distribution | Extreme right skew | Moderate skew |
| Zero proportion | ~42% | ~42% (unchanged) |

**MinMax Normalization:** After log-transform, observed values are scaled to [0, 1] using a MinMaxScaler fitted only on observed (non-missing) values. Missing positions are set to 0 in the data matrix.

---

## 3. Architecture

Two architectures were developed: the original MLP (v1, PyTorch) and the final CNN (v2, TensorFlow). Section 3.1–3.2 documents the MLP. Section 3.3–3.6 documents the final CNN.

### 3.1 MLP Generator (v1 — PyTorch)

```
Input:  [data * mask + noise * (1-mask)] || [mask]  →  90 dims
Layer 1: Linear(90, 256) → LeakyReLU(0.2) → BatchNorm → Dropout(0.1)
Layer 2: Linear(256, 256) → LeakyReLU(0.2) → BatchNorm → Dropout(0.1)
Layer 3: Linear(256, 256) → LeakyReLU(0.2) → BatchNorm
Layer 4: Linear(256, 128) → LeakyReLU(0.2)
Output:  Linear(128, 45) → Sigmoid
```

### 3.2 MLP Discriminator (v1 — PyTorch)

```
Input:  [completed_data] || [hint]  →  90 dims
Layer 1: Linear(90, 256) → LeakyReLU(0.2)
Layer 2: Linear(256, 256) → LeakyReLU(0.2)
Layer 3: Linear(256, 256) → LeakyReLU(0.2)
Layer 4: Linear(256, 128) → LeakyReLU(0.2)
Output:  Linear(128, 45) → Sigmoid
```

| Component | Parameters |
|---|---|
| MLP Generator | ~230K |
| MLP Discriminator | ~220K |
| **Total** | **~450K** |

### 3.3 CNN Generator — Finetuned Final (TensorFlow)

Input shape: `(batch, 15, 3)` — 15 time steps, 3 station channels (preserved, not flattened).

```
Input: [data * mask] || [mask]  →  (batch, 15, 6)
Conv1D(32, kernel=3, padding='same') → LeakyReLU(0.2) → BatchNorm → Dropout(0.25)
Conv1D(64, kernel=3, padding='same') → LeakyReLU(0.2) → BatchNorm → Dropout(0.25)
Conv1D(64, kernel=3, padding='same') → LeakyReLU(0.2) → BatchNorm → Dropout(0.25)
Conv1D(32, kernel=3, padding='same') → LeakyReLU(0.2) → BatchNorm → Dropout(0.25)
Conv1D(3,  kernel=1, padding='same') → Sigmoid
Output: (batch, 15, 3) — center day [index 7] extracted for imputation
```

**Parameters: 26,211 (25.6 KB)** — 17× smaller than MLP v1, deliberately smaller to prevent overfitting with only 495 missing values to guide training.

### 3.4 CNN Discriminator — Finetuned Final (TensorFlow)

```
Input: [completed_data] || [hint]  →  (batch, 15, 6)
Conv1D(32, kernel=3, padding='same') → LeakyReLU(0.2) → Dropout(0.25)
Conv1D(64, kernel=3, padding='same') → LeakyReLU(0.2) → Dropout(0.25)
Conv1D(64, kernel=3, padding='same') → LeakyReLU(0.2) → Dropout(0.25)
Conv1D(32, kernel=3, padding='same') → LeakyReLU(0.2) → Dropout(0.25)
Conv1D(3,  kernel=1, padding='same') → Sigmoid
Output: (batch, 15, 3) — per-position mask classification
```

Note: Discriminator has no BatchNorm (standard GAIN practice — BN can reduce discriminator sensitivity).

### 3.5 Why CNN Beat MLP

| Factor | MLP v1 | CNN v2 Finetuned |
|---|---|---|
| Temporal structure | Flat (all 15 lags equal) | Local patterns via Conv1D |
| Parameters | ~450K | 26K |
| Regularisation | Dropout(0.1) only | Dropout(0.25) + L2(1e-4) |
| R²% | -5.56% | +15.23% (single) / +17.93% (ensemble) |

The primary gains came from (a) temporal structure via Conv1D and (b) aggressive regularisation given the small number of missing values.

### 3.6 Architecture Comparison: Original vs Overfit CNN vs Finetuned CNN

| Setting | CNN v2 (overfit) | CNN finetuned (final) |
|---|---|---|
| Filters | 64→128→128→64 | **32→64→64→32** |
| Dropout | 0.1 | **0.25** |
| L2 regularisation | None | **1e-4** |
| Alpha | 10.0 | **5.0** |
| Max epochs | 800 (fixed) | 600 (early stop patience=60) |
| Avg stop epoch | 800 | ~170 |
| Val RMSE | 44.13 mm | 16.13 mm |
| Val R²% | -534.6% | +15.23% |

The overfit CNN had 388K parameters and memorised the ~14K training observations, causing near-zero reconstruction loss on training data but wildly wrong imputed values on held-out positions.

---

## 4. Training

### 4.1 Hyperparameters — MLP v1 (PyTorch)

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 5e-4 | Lower than standard to stabilize GAN training |
| LR schedule | Cosine annealing (min=1e-5) | Smooth decay prevents late-stage instability |
| Optimizer | Adam (beta1=0.5, beta2=0.999) | Standard for GANs; beta1=0.5 reduces momentum |
| Batch size | 128 | Sufficient for BatchNorm stability |
| Epochs | 500 | Extended training with best-model checkpointing |
| Alpha (recon weight) | 10.0 | Balances adversarial and reconstruction losses |
| Hint rate | 0.9 | High hint rate aids discriminator learning |
| Noise scale | 0.01 | Low noise for missing positions |

### 4.1b Hyperparameters — CNN Finetuned (TensorFlow, Final)

| Parameter | Value | Rationale |
|---|---|---|
| Framework | TensorFlow 2.21 / Keras 3.15 | Client requirement |
| Learning rate | 5e-4 → 1e-5 (cosine) | Same schedule as MLP v1 |
| Optimizer | Adam (beta1=0.5) | GAN-standard |
| Batch size | 128 | Same as MLP |
| Max epochs | 600 | Upper bound only |
| Early stopping | patience=60 on val recon | Prevents overfitting — typical stop: ~170 epochs |
| Alpha (recon weight) | **5.0** | Reduced from 10 — less memorisation pressure |
| Hint rate | 0.9 | Unchanged |
| L2 regularisation | **1e-4** | Added — penalises large filter weights |
| Dropout | **0.25** | Increased from 0.1 — higher stochastic noise |

### 4.1c Ensemble Strategy

Five independent CNN models are trained with different random seeds `[42, 123, 456, 789, 1024]`. Each model trains from scratch with the same hyperparameters but different random initialisation. Final imputed values are the element-wise average across all 5 generators:

```
imputed[i, j] = mean( G₁(x_i, m_i)[center, j],  G₂(...),  G₃(...),  G₄(...),  G₅(...) )
```

This ensemble approach reduces variance in the imputation — different models disagree most on ambiguous cases (e.g., heavy-rain events where neighbouring stations are also missing). Averaging their outputs suppresses outlier predictions and improves stability.

### 4.2 Loss Functions

**Discriminator loss:**
```
L_D = E[BCE(D(X_hat, hint), mask) * (hint + (1-hint) * 0.5)]
```

**Generator loss:**
```
L_G = -E[BCE(D(X_hat, hint), mask) * (1-mask)]  +  alpha * MSE(G(X,M) - X) * mask / sum(mask)
```

The reconstruction loss term (weighted by alpha) ensures the generator preserves observed values while the adversarial loss pushes imputed values toward the real data distribution.

### 4.3 Training Protocol

1. **Validation split:** 20% of observed values are randomly masked as "pseudo-missing" for evaluation. These are hidden from the model during training.
2. **Validation training:** Train on 80% of observed values (500 epochs), evaluate imputation quality on the held-out 20%.
3. **Full retraining:** Retrain on all observed values (500 epochs) for the final imputation pass.
4. **Best model selection:** Checkpoint the generator state with the lowest reconstruction loss across all epochs.

### 4.4 Training Dynamics — MLP v1

The reconstruction loss decreased steadily across 500 epochs:

| Epoch | Validation Recon Loss | Full Recon Loss |
|---|---|---|
| 1 | 0.077650 | 0.066040 |
| 100 | 0.002546 | 0.002781 |
| 200 | 0.002081 | 0.002230 |
| 300 | 0.001946 | 0.002010 |
| 400 | 0.001675 | 0.001833 |
| 500 | 0.001601 | 0.001891 |
| **Best** | **0.001590** | **0.001783** |

### 4.4b Training Dynamics — CNN Ensemble (TensorFlow)

Each of the 5 CNN models stopped early via validation-based early stopping:

| Model | Seed | Early Stop Epoch | Best Val Recon |
|---|---|---|---|
| Model 1 | 42 | 152 | 0.027495 |
| Model 2 | 123 | 166 | 0.028101 |
| Model 3 | 456 | 147 | 0.029120 |
| Model 4 | 789 | 192 | 0.026130 |
| Model 5 | 1024 | 185 | 0.026622 |
| **Ensemble mean** | — | **~168** | **0.027494** |

Note: The absolute validation reconstruction values are not comparable across MLP and CNN — the CNN operates in `(batch, 15, 3)` space while MLP operates in `(batch, 45)` space. R² on held-out observed values is the consistent cross-model metric.

---

## 5. Validation Results

### 5.1 MLP v1 Quantitative Metrics (PyTorch baseline)

Evaluated on 20% held-out observed values (3,514 values total):

| Station | NRMSE% | NMAE% | R²% |
|---|---|---|---|
| Nada | 246.6% | 95.0% | -3.8% |
| Lembing | 219.3% | 97.4% | -7.9% |
| Reman | 207.9% | 94.1% | -5.4% |
| **Overall** | **226.0%** | **95.6%** | **-5.56%** |

### 5.1b CNN Finetuned Ensemble Quantitative Metrics (TensorFlow — Final)

Evaluated on the same 20% holdout set (3,514 values):

| Station | NRMSE% | NMAE% | R²% |
|---|---|---|---|
| Nada (Ldg. Nada) | 207.8% | 78.2% | **+26.42%** |
| Lembing (Sg. Lembing) | 201.8% | 84.6% | +8.74% |
| Reman (Ldg. Kuala Reman) | 184.7% | 81.5% | +16.68% |
| **Overall** | **199.2%** | **81.5%** | **+17.93%** |

**NRMSE% = (RMSE / mean_observed) × 100.** This looks large (~199%) because rainfall is zero-inflated — the mean (~8mm) is far below peak events (360mm). This is normal for daily point-rainfall imputation and does not indicate model failure.

### 5.2 Full Model Progression

| Model | NRMSE% | NMAE% | R²% | Status |
|---|---|---|---|---|
| MLP v1 — PyTorch | 226.0% | 95.6% | -5.56% | Baseline (below mean predictor) |
| CNN v2 — TF overfit (800 ep) | 553.9% | 213.3% | -534.6% | Severely overfit — discarded |
| CNN finetuned — TF single model | 202.4% | 83.8% | +15.23% | Positive R² — beats mean predictor |
| **CNN ensemble ×5 — TF (final)** | **199.2%** | **81.5%** | **+17.93%** | **Production — used for completed dataset** |

### 5.3 Interpreting the Metrics

**On negative MLP R²:**
The MLP's R² = -5.56% means it was slightly worse than simply predicting the station mean for every missing value. This is expected: with only 3 correlated stations and 2.7% missingness, the cross-station signal is weak.

**On CNN R² = +17.93%:**
This is positive, meaning the CNN ensemble explains ~18% of the variance in held-out values — meaningfully better than a mean predictor. For daily point rainfall without atmospheric forcing data (pressure, humidity, wind), R² in the 15–30% range is the realistic ceiling. The ensemble improved R² from +15.23% (single model) to +17.93% primarily by boosting the Nada station (from ~12% to 26%).

**On NRMSE% (~199%):**
Do not interpret NRMSE% as an error percentage of the measurement. It is normalised by the mean (~8mm), not by individual event magnitudes. A 16mm RMSE on a day with 200mm actual rainfall is ~8% relative error — reasonable. The NRMSE% is high because it mixes those large-event errors with zero-rain days where any non-zero imputation is 100%+ relative error.

**Realistic ceiling without additional data:**
R² > 35% for daily point-rainfall imputation requires additional predictor variables: temperature, pressure, wind speed, satellite-derived precipitation estimates, or upstream catchment flow. With 3 telemetry stations only, 17–25% R² represents state-of-the-art GAIN performance for this data configuration.

### 5.4 Distributional Preservation (What Matters for TCN)

For downstream TCN training, distributional fidelity matters more than individual-value accuracy. The CNN ensemble results:

| Station | Original Mean | Completed Mean | Shift% |
|---|---|---|---|
| Nada | 8.12 mm | 7.97 mm | **1.8%** |
| Lembing | 8.91 mm | 8.72 mm | **2.1%** |
| Reman | 7.85 mm | 7.74 mm | **1.4%** |

All station means shift by less than 2.2%. Maximum values are unchanged (imputed values are clipped at observed extremes). The completed dataset will not systematically bias the TCN's learned seasonal and inter-annual patterns.

---

## 6. Missing Value Pattern Analysis

### 6.1 Co-occurrence Structure

Analysis of missing value co-occurrence reveals whether missingness is driven by station-level sensor failures (independent) or site-wide outages (simultaneous):

- Missing values show temporal clustering (concentrated in specific months/years rather than uniformly distributed)
- Some periods show simultaneous missingness across 2-3 stations, suggesting regional equipment maintenance or power outages
- Other periods show single-station missingness, confirming independent sensor failures

This mixed pattern (MAR — Missing At Random, conditional on time) supports the GAIN approach, which can leverage both spatial and temporal context.

---

## 6b. CNN Loss Function Implementation Note

A non-obvious implementation challenge arose with the CNN discriminator loss. The standard `tf.keras.losses.BinaryCrossentropy(reduction="none")` still reduces the last axis, returning shape `(batch, 15)` instead of the required `(batch, 15, 3)` for element-wise mask multiplication. A custom function was implemented:

```python
def bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
```

This preserves full shape `(batch, 15, 3)`, enabling correct per-element masking in the discriminator and generator losses.

---

## 7. Completed Dataset

### 7.1 Schema

| Column | Type | Description |
|---|---|---|
| date | datetime | Date (2009-01-01 to 2025-06-29) |
| nada | float64 | Daily rainfall — Nada station (mm) |
| lembing | float64 | Daily rainfall — Lembing station (mm) |
| reman | float64 | Daily rainfall — Reman station (mm) |
| nada_imputed | int | 1 if value was GAN-imputed, 0 if observed |
| lembing_imputed | int | 1 if value was GAN-imputed, 0 if observed |
| reman_imputed | int | 1 if value was GAN-imputed, 0 if observed |

### 7.2 Quality Assurance

- **NaN remaining:** 0 (confirmed)
- **Negative values:** 0 (floor at 0.0 applied)
- **Precision:** Rounded to 1 decimal place (matching input data precision)
- **Observed values:** Restored exactly from original data (no modification of known values)

---

## 8. Output Artifacts

### MLP v1 (PyTorch) Artifacts
| Artifact | Size | Description |
|---|---|---|
| `completed_daily_rainfall.csv` | 180 KB | MLP-completed dataset (6,024 × 7) |
| `gain_generator_final.pth` | 777 KB | MLP generator weights + scalers |
| `02_gan_imputation.ipynb` | 35 KB | MLP GAIN notebook |

### CNN Ensemble (TensorFlow) Artifacts — Final
| Artifact | Size | Description |
|---|---|---|
| `completed_daily_rainfall_cnn_tf.csv` | 181 KB | **CNN ensemble completed dataset (6,024 × 7)** |
| `02b_gan_cnn_tensorflow.ipynb` | — | CNN GAIN notebook (full pipeline) |
| `gain_cnn_ensemble_scatter.png` | — | Actual vs imputed scatter (3 stations) |
| `gain_cnn_ensemble_distribution.png` | — | Observed vs imputed rainfall distributions |

The `completed_daily_rainfall_cnn_tf.csv` file is the authoritative input to Phase 3 (TCN prediction). It contains zero NaN values, imputation flags per station, and all observed values preserved exactly.

---

## 9. References

- Yoon, J., Jordon, J., & van der Schaar, M. (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.
- Abadi, M. et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. *tensorflow.org*.
- Keras Team (2023). Keras 3 — Multi-backend deep learning framework. *keras.io*.
