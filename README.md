# dl-rainfall-data-imputation-pred-system

Deep Learning-Based Rainfall Data Imputation and Prediction System for Pahang, Malaysia.

## Overview

End-to-end deep learning pipeline covering data engineering, GAN-based gap imputation, and TCN-based multi-scale rainfall prediction across three telemetry stations (2009–2025):

- **Ldg. Nada** (3931013)
- **Sg. Lembing** (3930012)
- **Ldg. Kuala Reman** (3931014)

## Pipeline

| Phase | Notebook | Method | Key Result |
|---|---|---|---|
| 1 — Data Engineering | `01_e2e_data_engineering.ipynb` | EDA, gap detection, feature engineering | Clean dataset, gap map |
| 2 — GAN Imputation | `02b_gan_cnn_tensorflow.ipynb` | GAIN (GAN) with CNN generator, 5-seed ensemble | R²=+0.179, mean bias <2% |
| 3 — TCN Prediction | `03b_tcn_multihead_tensorflow.ipynb` | Multi-head TCN, 3-seed ensemble | Monthly R²=+0.238 |

## Architecture

### GAN Imputation (Phase 2)
- CNN generator: Conv1D 32→64→64→32, dropout=0.25, L2=1e-4
- 5-model ensemble (seeds 42, 123, 456, 789, 1024)
- Temporal window: 15 days

### Multi-Head TCN Prediction (Phase 3)
- 5 TCN blocks, 96 filters, dilations [1, 2, 4, 8, 16], causal padding
- 13 input features: rainfall + 7/30-day rolling means + sin/cos seasonal encoding
- Three separate output heads: daily (next 1d), weekly (7d sum), monthly (30d sum)
- 3-seed ensemble [42, 123, 456]

## Results

### GAN Imputation
| Model | R² | NRMSE |
|---|---|---|
| MLP baseline | -0.056 | 226.0% |
| CNN finetuned | +0.152 | 202.4% |
| **CNN ensemble ×5** | **+0.179** | **199.2%** |

### TCN Prediction (test set)
| Scale | R² | NRMSE |
|---|---|---|
| Daily (next 1d) | -0.032 | 239.0% |
| Weekly (7d sum) | +0.027 | 115.3% |
| **Monthly (30d sum)** | **+0.238** | **67.1%** |

## Project Structure

```
├── notebooks/
│   ├── 01_e2e_data_engineering.ipynb
│   ├── 02b_gan_cnn_tensorflow.ipynb
│   ├── 03b_tcn_multihead_tensorflow.ipynb
│   └── archive/          # superseded notebook versions
├── data/
│   ├── raw/              # original telemetry exports
│   └── processed/        # engineered + GAN-completed datasets
├── models/               # saved model weights (.keras, .pth)
├── predictions/          # test set prediction CSVs
├── figures/
│   ├── gan/              # GAN training and validation plots
│   └── tcn/              # TCN scatter, horizon, and comparison plots
├── reports/              # Technical reports (Phases 1–3)
├── briefs/               # Plain-language explanations
└── docs/                 # Proposal and invoice
```

## Environment

| Environment | Python | Framework | Used for |
|---|---|---|---|
| `.venv` | 3.14 | PyTorch | Phase 1, baseline models |
| `C:\tf_env\` | 3.11 | TensorFlow 2.21 | Phase 2 GAN, Phase 3 TCN |

## Framework

TensorFlow 2.21.0 / Keras 3.15.0 (Phases 2–3)  
PyTorch (Phase 1 baseline)
