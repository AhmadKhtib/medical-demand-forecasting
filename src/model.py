"""
Modeling pipeline.

Loads preprocessed data, clusters medication text per clinic with TF-IDF + MiniBatchKMeans,
aggregates weekly demand, fills gaps, fits SARIMAX per series, and produces a 4-week forecast
table — exactly matching the modeling notebook.

Usage:
    python -m src.model --south data/final_data.xlsx
                        --middle data/middel_final_data.xlsx
                        --out_forecast data/forecasts.xlsx
                        --horizon 4
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


# ---------------------------------------------------------------------------
# Step 1 — Load and combine
# ---------------------------------------------------------------------------

def load_data(south_path: str, middle_path: str | None = None) -> pd.DataFrame:
    """Load south (+ optionally middle) final Excel files and concatenate."""
    df_south = pd.read_excel(south_path)
    print(f"[load_data] south shape: {df_south.shape}")
    print(f"[load_data] south cols: {list(df_south.columns)}")

    if middle_path:
        df_middle = pd.read_excel(middle_path)
        print(f"[load_data] middle shape: {df_middle.shape}")
        print(f"[load_data] middle cols: {list(df_middle.columns)}")

        # Report columns only in south
        only_in_south = set(df_south.columns) - set(df_middle.columns)
        if only_in_south:
            print(f"[load_data] columns only in south: {sorted(only_in_south)}")

        data = pd.concat([df_south, df_middle], ignore_index=True)
    else:
        data = df_south

    print(f"[load_data] combined shape: {data.shape}")
    return data


# ---------------------------------------------------------------------------
# Step 2 — Clean Date + Clinics
# ---------------------------------------------------------------------------

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Coerce Date, normalize Clinics, drop rows missing either."""
    before = len(data)
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Clinics"] = data["Clinics"].astype(str).str.strip().str.lower()
    data = data.dropna(subset=["Date", "Clinics"])
    after = len(data)

    print(f"[clean_data] rows before={before} after={after}")
    print(f"[clean_data] date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"[clean_data] unique clinics: {data['Clinics'].nunique()}")
    print(data["Clinics"].value_counts().head(10).to_string())
    return data


# ---------------------------------------------------------------------------
# Step 3 — Build medication long table
# ---------------------------------------------------------------------------

def build_med_long(data: pd.DataFrame) -> pd.DataFrame:
    """
    Melt medication_1 … medication_N into a single 'med_text' column.
    Matches the notebook's melt approach.
    """
    med_cols = [c for c in data.columns if re.fullmatch(r"medication_\d+", c)]
    print(f"[build_med_long] medication columns: {med_cols}")

    med_long = (
        data[["Date", "Clinics"] + med_cols]
        .melt(
            id_vars=["Date", "Clinics"],
            value_vars=med_cols,
            var_name="slot",
            value_name="med_text",
        )
        .dropna(subset=["med_text"])
    )
    med_long = med_long[med_long["med_text"].astype(str).str.strip().ne("")]

    print(f"[build_med_long] shape: {med_long.shape}")
    print(f"[build_med_long] slot distribution:\n{med_long['slot'].value_counts().to_string()}")
    print(f"[build_med_long] unique raw med strings: {med_long['med_text'].nunique()}")
    print(f"[build_med_long] top 10:\n{med_long['med_text'].value_counts().head(10).to_string()}")
    return med_long


# ---------------------------------------------------------------------------
# Step 4 — Text normalization
# ---------------------------------------------------------------------------

FORM_STOPWORDS = {
    "tab","tabs","tablet","tablets","cap","caps","syrup","susp","spray","drops",
    "cream","ointment","oint","gel","tube","sachet","inj","injection","amp","vial",
    "solution","lotion","disp","ml","mg","iu",
}


def clean_med_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_med_text(med_long: pd.DataFrame) -> pd.DataFrame:
    """Apply text normalization to the med_text column."""
    med_long = med_long.copy()
    med_long["text_clean"] = med_long["med_text"].apply(clean_med_text)
    print(f"[normalize_med_text] unique uncleaned: {med_long['med_text'].nunique()}")
    print(f"[normalize_med_text] unique cleaned:   {med_long['text_clean'].nunique()}")
    return med_long


# ---------------------------------------------------------------------------
# Step 5 — TF-IDF + MiniBatchKMeans clustering per clinic
# ---------------------------------------------------------------------------

def choose_k(n_unique: int) -> int:
    if n_unique < 50:   return 3
    if n_unique < 200:  return 5
    if n_unique < 500:  return 8
    if n_unique < 1000: return 12
    return 15


def cluster_medications(med_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each clinic, TF-IDF vectorise text_clean, cluster with MiniBatchKMeans,
    assign category labels like 'dermatology__cat0'.

    Returns:
        med_clustered  – original rows with 'med_category' column added
        cluster_report – summary DataFrame with top_terms per (clinic, category)
    """
    all_rows = []
    report_rows = []

    for clinic, grp in med_long.groupby("Clinics"):
        texts = grp["text_clean"].astype(str).tolist()
        n_unique = len(set(texts))
        k = choose_k(n_unique)

        try:
            vec = TfidfVectorizer(
                min_df=1,
                max_features=500,
                stop_words=list(FORM_STOPWORDS),
                ngram_range=(1, 2),
            )
            X = vec.fit_transform(texts)

            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
            labels = km.fit_predict(X)

            # Top terms per cluster
            feature_names = vec.get_feature_names_out()
            centers = km.cluster_centers_
            for cluster_id in range(k):
                top_idx = centers[cluster_id].argsort()[::-1][:10]
                top_terms = ", ".join(feature_names[i] for i in top_idx)
                cat_name = f"{clinic}__cat{cluster_id}"
                report_rows.append({
                    "Clinics": clinic,
                    "category": cat_name,
                    "rows": int((labels == cluster_id).sum()),
                    "top_terms": top_terms,
                })

            grp = grp.copy()
            grp["med_category"] = [f"{clinic}__cat{lbl}" for lbl in labels]

        except Exception as e:
            print(f"  [cluster] WARNING {clinic}: {e} — assigning single category")
            grp = grp.copy()
            grp["med_category"] = f"{clinic}__cat0"
            report_rows.append({"Clinics": clinic, "category": f"{clinic}__cat0",
                                 "rows": len(grp), "top_terms": "(fallback)"})

        all_rows.append(grp)
        print(f"  [cluster] {clinic}: n_unique={n_unique} k={k}")

    med_clustered = pd.concat(all_rows, ignore_index=True)
    cluster_report = pd.DataFrame(report_rows)

    print(f"\n[cluster_medications] med_clustered shape: {med_clustered.shape}")
    print(cluster_report.sort_values(["Clinics", "rows"], ascending=[True, False]).head(25).to_string())
    return med_clustered, cluster_report


# ---------------------------------------------------------------------------
# Step 6 — Weekly aggregation
# ---------------------------------------------------------------------------

def build_weekly(med_clustered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate demand per (week_start, Clinics, med_category)."""
    med_clustered = med_clustered.copy()
    med_clustered["week_start"] = med_clustered["Date"].dt.to_period("W-SUN").dt.start_time

    weekly = (
        med_clustered
        .groupby(["week_start", "Clinics", "med_category"], as_index=False)
        .size()
        .rename(columns={"size": "demand"})
        .sort_values(["Clinics", "med_category", "week_start"])
        .reset_index(drop=True)
    )

    print(f"[build_weekly] shape: {weekly.shape}")
    print(f"[build_weekly] weekly demand stats:\n{weekly['demand'].describe().to_string()}")
    return weekly


# ---------------------------------------------------------------------------
# Step 7 — Fill missing weeks with zeros
# ---------------------------------------------------------------------------

def complete_weekly(g: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(g["week_start"].min(), g["week_start"].max(), freq="W-MON")
    out = (
        g.set_index("week_start")
        .reindex(idx)
        .rename_axis("week_start")
        .reset_index()
    )
    out["demand"] = out["demand"].fillna(0).astype(int)
    for col in ["Clinics", "med_category"]:
        out[col] = out[col].ffill().bfill()
    return out


def fill_weekly_gaps(weekly: pd.DataFrame) -> pd.DataFrame:
    """Ensure each (Clinics, med_category) series has continuous weekly dates."""
    weekly_full = (
        weekly
        .groupby(["Clinics", "med_category"], group_keys=False)
        .apply(complete_weekly)
        .reset_index(drop=True)
    )

    print(f"[fill_weekly_gaps] shape: {weekly_full.shape}")
    print(f"[fill_weekly_gaps] max demand: {weekly_full['demand'].max()}")
    print(f"[fill_weekly_gaps] non-zero: {(weekly_full['demand'] > 0).sum()}")
    lens = weekly_full.groupby(["Clinics", "med_category"]).size().head(10)
    print(f"[fill_weekly_gaps] series lengths (first 10):\n{lens.to_string()}")
    return weekly_full


# ---------------------------------------------------------------------------
# Step 8 — SARIMAX forecast
# ---------------------------------------------------------------------------

def forecast_count_sarimax(ts: pd.Series, H: int = 4) -> tuple:
    """
    Fit SARIMAX on weekly series ts and forecast H steps ahead.

    Falls back to simple mean baseline for sparse series.
    Returns (forecast Series, config dict or None, AIC or None).
    """
    # Sparse fallback
    if len(ts) < 12 or (ts == 0).mean() > 0.75 or ts.sum() < 30:
        base = int(round(ts.tail(4).mean()))
        idx = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
        fc = pd.Series([base] * H, index=idx)
        return fc, None, None

    best_fc, best_cfg, best_aic = None, None, np.inf
    candidates = [
        dict(order=(1,1,1), seasonal_order=(1,1,1,52)),
        dict(order=(1,1,0), seasonal_order=(1,0,0,52)),
        dict(order=(0,1,1), seasonal_order=(0,1,1,52)),
        dict(order=(1,0,0), seasonal_order=(0,0,0,1)),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cfg in candidates:
            try:
                model = SARIMAX(
                    ts,
                    order=cfg["order"],
                    seasonal_order=cfg["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=200)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_cfg = cfg
                    fc_obj = res.get_forecast(steps=H)
                    best_fc = fc_obj.predicted_mean.clip(lower=0).round().astype(int)
            except Exception:
                continue

    if best_fc is None:
        base = int(round(ts.tail(4).mean()))
        idx = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
        best_fc = pd.Series([base] * H, index=idx)

    return best_fc, best_cfg, best_aic if best_aic != np.inf else None


# ---------------------------------------------------------------------------
# Step 9 — Plot top series + forecast
# ---------------------------------------------------------------------------

def plot_top_series(weekly_full: pd.DataFrame, cluster_report: pd.DataFrame,
                    top_n: int = 5, H: int = 4, out_dir: str = "charts"):
    """Plot history + forecast for the top-N demand series."""
    os.makedirs(out_dir, exist_ok=True)

    top = (
        weekly_full.groupby(["Clinics", "med_category"])["demand"].sum()
        .sort_values(ascending=False).head(top_n).reset_index()
    )

    for _, r in top.iterrows():
        clinic_i = r["Clinics"]
        cat_i = r["med_category"]

        ts = (
            weekly_full[(weekly_full["Clinics"] == clinic_i) & (weekly_full["med_category"] == cat_i)]
            .sort_values("week_start")
            .set_index("week_start")["demand"]
            .asfreq("W-MON", fill_value=0)
        )

        fc, cfg, aic = forecast_count_sarimax(ts, H=H)

        # Describe what's in this category
        rep = cluster_report[
            (cluster_report["Clinics"] == clinic_i) & (cluster_report["category"] == cat_i)
        ]
        top_terms = rep["top_terms"].iloc[0] if len(rep) else "(no terms)"

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts.index, ts.values, label="History")
        ax.plot(fc.index, fc.values, label=f"Forecast (next {H}w)", linestyle="--", marker="o")

        title = f"{clinic_i} | {cat_i}"
        if cfg is None:
            title += "\nBaseline (sparse series)"
        else:
            title += f"\nSARIMAX {cfg['order']}×{cfg['seasonal_order']} | AIC={aic:.1f}"
        title += f"\nTop terms: {top_terms[:80]}"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Week")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()

        fname = f"{clinic_i}__{cat_i}.png".replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] {clinic_i} | {cat_i}")


# ---------------------------------------------------------------------------
# Step 10 — Build full forecast table
# ---------------------------------------------------------------------------

def build_forecast_table(weekly_full: pd.DataFrame, H: int = 4) -> pd.DataFrame:
    """
    Run SARIMAX for every (Clinics, med_category) series.
    Returns a long DataFrame with columns:
        Clinics, med_category, week_start, forecast_demand, sarimax_config
    """
    forecasts = []

    for (clinic_i, cat_i), g in weekly_full.groupby(["Clinics", "med_category"]):
        ts = (
            g.sort_values("week_start")
            .set_index("week_start")["demand"]
            .asfreq("W-MON", fill_value=0)
        )
        fc, cfg, aic = forecast_count_sarimax(ts, H=H)
        cfg_str = str(cfg) if cfg else "baseline"

        for week, val in fc.items():
            forecasts.append({
                "Clinics": clinic_i,
                "med_category": cat_i,
                "week_start": week,
                "forecast_demand": int(val),
                "sarimax_config": cfg_str,
                "aic": aic,
            })

    result = pd.DataFrame(forecasts)
    print(f"[build_forecast_table] shape: {result.shape}")
    print(result.head(10).to_string())
    return result


# ---------------------------------------------------------------------------
# Full modeling pipeline
# ---------------------------------------------------------------------------

def run_modeling(
    south_path: str,
    middle_path: str | None = None,
    out_forecast: str = "data/forecasts.xlsx",
    horizon: int = 4,
    charts_dir: str = "charts/forecast",
) -> pd.DataFrame:
    """
    End-to-end modeling pipeline.
    Returns the forecast DataFrame.
    """

    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    data = load_data(south_path, middle_path)

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning Date + Clinics")
    print("=" * 60)
    data = clean_data(data)

    print("\n" + "=" * 60)
    print("STEP 3: Building medication long table")
    print("=" * 60)
    med_long = build_med_long(data)

    print("\n" + "=" * 60)
    print("STEP 4: Text normalization")
    print("=" * 60)
    med_long = normalize_med_text(med_long)

    print("\n" + "=" * 60)
    print("STEP 5: Clustering medications per clinic")
    print("=" * 60)
    med_clustered, cluster_report = cluster_medications(med_long)

    print("\n" + "=" * 60)
    print("STEP 6: Weekly aggregation")
    print("=" * 60)
    weekly = build_weekly(med_clustered)

    print("\n" + "=" * 60)
    print("STEP 7: Filling missing weeks")
    print("=" * 60)
    weekly_full = fill_weekly_gaps(weekly)

    print("\n" + "=" * 60)
    print("STEP 8: Plotting top-5 series")
    print("=" * 60)
    plot_top_series(weekly_full, cluster_report, top_n=5, H=horizon, out_dir=charts_dir)

    print("\n" + "=" * 60)
    print("STEP 9: Building full forecast table")
    print("=" * 60)
    forecast_df = build_forecast_table(weekly_full, H=horizon)

    print(f"\nSaving forecast -> {out_forecast}")
    os.makedirs(os.path.dirname(out_forecast) or ".", exist_ok=True)
    forecast_df.to_excel(out_forecast, index=False)
    print(f"Saved. Shape: {forecast_df.shape}")

    return forecast_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(description="Run medication demand forecasting pipeline")
    parser.add_argument("--south", required=True, help="South final Excel file")
    parser.add_argument("--middle", default=None, help="Middle final Excel file (optional)")
    parser.add_argument("--out_forecast", default="data/forecasts.xlsx",
                        help="Output forecast Excel path")
    parser.add_argument("--horizon", type=int, default=4, help="Forecast horizon in weeks")
    parser.add_argument("--charts_dir", default="charts/forecast", help="Directory for forecast charts")

    args = parser.parse_args()

    run_modeling(
        south_path=args.south,
        middle_path=args.middle,
        out_forecast=args.out_forecast,
        horizon=args.horizon,
        charts_dir=args.charts_dir,
    )


if __name__ == "__main__":
    _main()
