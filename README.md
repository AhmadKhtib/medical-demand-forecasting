# Medical-demand-forecasting

> Weekly medication demand forecasting across Gaza clinic specialties,
> built from raw clinical visit records using TF-IDF clustering and SARIMAX time-series models.

---

## Overview

This project ingests raw Excel data from multiple clinic locations, normalises free-text
medical records, groups medications into semantic clusters per specialty, and produces
4-week demand forecasts for each cluster — all in a clean, reproducible Python pipeline.

**Tech stack:** Python · Pandas · scikit-learn · statsmodels · Matplotlib

---

## Project structure

```
medical-demand-forecasting/
│
├── main.py               # Full pipeline entry point (preprocessing + modeling)
│
├── src/
│   ├── preprocess.py     # Data loading, column normalisation, per-clinic parsing, merge
│   ├── parsers.py        # Per-clinic MEDICAL field parsers (12 specialties)
│   ├── text_utils.py     # Shared text normalisation utilities
│   ├── model.py          # TF-IDF clustering, weekly aggregation, SARIMAX forecasting
│   └── visualize.py      # Exploratory and date-combo chart generation
│
├── data/                 # (gitignored) raw and processed Excel files
├── charts/               # (gitignored) generated PNG charts and PDFs
│
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the pipeline

### Full pipeline (preprocessing + forecasting)

```bash
python main.py \
  --south  data/Data.xlsx data/Deir_El-Balah.xlsm \
  --middle data/middel_data_assi.xlsx \
  --out_south    data/final_data.xlsx \
  --out_middle   data/middel_final_data.xlsx \
  --out_forecast data/forecasts.xlsx \
  --horizon 4 \
  --charts_dir charts
```

### Preprocessing only

```bash
python -m src.preprocess \
  --south data/Data.xlsx data/Deir_El-Balah.xlsm \
  --middle data/middel_data_assi.xlsx \
  --out_south  data/final_data.xlsx \
  --out_middle data/middel_final_data.xlsx
```

### Modeling only (using already-preprocessed files)

```bash
python -m src.model \
  --south  data/final_data.xlsx \
  --middle data/middel_final_data.xlsx \
  --out_forecast data/forecasts.xlsx \
  --horizon 4
```

---

## Pipeline stages

| Stage | Description |
|---|---|
| **Load** | Read south-area Excel/xlsm files + middle-area Excel file |
| **Normalise** | Standardise `Clinics`, `Gender`, `GOVENORATES` columns |
| **Parse** | Apply specialty-specific parser to free-text `MEDICAL` field → `diagnosis`, `medication`, `dose_schedule` |
| **Merge** | Concatenate all 12 clinic DataFrames into one unified table |
| **Cluster** | TF-IDF vectorise medication strings per clinic → MiniBatchKMeans categories |
| **Aggregate** | Count weekly demand per `(clinic, med_category)` |
| **Forecast** | Fit SARIMAX per series → 4-week demand forecast |

---

## Clinics covered

Dermatology · Orthopedic · Urology · Nutrition · Pediatrics · ENT ·
General Surgery · Gyn & Obstetrics · Physiotherapy · Internal Medicine ·
Psychiatric · Deworming Clinic

---

## Outputs

| File | Description |
|---|---|
| `data/final_data.xlsx` | Preprocessed south-area data with structured medication columns |
| `data/middel_final_data.xlsx` | Preprocessed middle-area data |
| `data/forecasts.xlsx` | 4-week demand forecast for every clinic × medication cluster |
| `charts/` | Exploratory PNG charts + combined PDFs |
| `charts/forecast/` | Per-series history + forecast plots |
