"""
Chart generation functions matching the notebook outputs.
Produces PNGs and a combined PDF per chart group.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def clean_cat(s: pd.Series, upper=True) -> pd.Series:
        s = s.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        s = s.str.replace(r"[_\-]+", " ", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True)
        return s.str.upper() if upper else s

    for c in ["Gender", "Clinics", "GOVENORATES"]:
        if c in df.columns:
            df[c] = clean_cat(df[c], upper=True)

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    if "Date" in df.columns:
        d = df["Date"].astype("string").str.strip()
        df["Date"] = pd.to_datetime(d, errors="coerce")

    return df


def _save_fig(fig, out_path: str, dpi: int = 160):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _to_period(dt: pd.Series, freq: str) -> pd.Series:
    return dt.dt.to_period(freq)


def _topk_other(series: pd.Series, top_k: int) -> pd.Series:
    vc = series.value_counts(dropna=True)
    keep = set(vc.head(top_k).index.tolist())
    return series.where(series.isin(keep), other="OTHER")


def _pivot_counts(df, period_col, cat_col):
    p = pd.crosstab(df[period_col], df[cat_col]).sort_index()
    p.index = p.index.astype(str)
    return p


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def _plot_total_over_time(total, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total.index, total.values, marker="o", linewidth=1)
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def _plot_multiline(pivot, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col].values, marker="o", linewidth=1, label=str(col))
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="best", ncols=2)
    return fig


def _plot_stacked_area(pivot, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(pivot.index))
    ax.stackplot(x, pivot.values.T, labels=[str(c) for c in pivot.columns])
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Count")
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend(loc="best", ncols=2)
    return fig


def _plot_stacked_bar(pivot, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def _plot_heatmap(pivot, title):
    mat = pivot.values.T
    fig_w = max(10, 0.35 * mat.shape[1])
    fig_h = max(4, 0.35 * mat.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Category")
    ax.set_xticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[1]))
    ax.set_yticklabels([str(c) for c in pivot.columns])
    fig.colorbar(im, ax=ax, label="Count")
    if mat.shape[0] <= 18 and mat.shape[1] <= 12:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center")
    return fig


def _plot_share_over_time(pivot, title):
    denom = pivot.sum(axis=1).replace(0, np.nan)
    share = pivot.div(denom, axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(share.index))
    ax.stackplot(x, share.values.T, labels=[str(c) for c in share.columns])
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Share")
    ax.set_xticks(x); ax.set_xticklabels(share.index, rotation=45, ha="right")
    ax.legend(loc="best", ncols=2)
    return fig


def _plot_age_trend(df, period_str, title, age_max=100):
    sub = df.dropna(subset=["Age", period_str]).copy()
    sub = sub[(sub["Age"] >= 0) & (sub["Age"] <= age_max)]
    if sub.empty:
        return None
    g = sub.groupby(period_str)["Age"].agg(["count", "mean", "median"]).sort_index()
    g.index = g.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g.index, g["mean"].values, marker="o", linewidth=1, label="Mean age")
    ax.plot(g.index, g["median"].values, marker="o", linewidth=1, label="Median age")
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Age")
    ax.set_ylim(0, age_max)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="best")
    return fig


def _plot_age_box_by_period(df, period_str, title, age_max=100, max_periods=18):
    sub = df.dropna(subset=["Age", period_str]).copy()
    sub = sub[(sub["Age"] >= 0) & (sub["Age"] <= age_max)]
    if sub.empty:
        return None
    periods = sub[period_str].value_counts().sort_index().index.astype(str).tolist()
    if len(periods) > max_periods:
        periods = periods[-max_periods:]
    groups = [sub.loc[sub[period_str].astype(str) == p, "Age"].values for p in periods]
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(periods)), 4))
    ax.boxplot(groups, labels=periods, showfliers=False)
    ax.set_title(title); ax.set_xlabel("Period"); ax.set_ylabel("Age")
    ax.set_ylim(0, age_max)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


# ---------------------------------------------------------------------------
# Public: make_basic_charts
# ---------------------------------------------------------------------------

def make_basic_charts(
    df: pd.DataFrame,
    out_dir: str = "charts",
    top_n: int = 12,
    age_max: int = 100,
    top_gov_n: int = 8,
    make_pdf: bool = True,
) -> dict:
    """
    Produce a standard set of exploratory charts and optionally a combined PDF.
    Returns a dict of {name: file_path}.
    """
    df = _prep_df(df)
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    pdf = PdfPages(os.path.join(out_dir, "all_charts.pdf")) if make_pdf else None

    def add(fig):
        if pdf and fig:
            fig.tight_layout(); pdf.savefig(fig)

    # 1) Date: daily + monthly
    if "Date" in df.columns and df["Date"].notna().any():
        daily = df.dropna(subset=["Date"]).groupby(df["Date"].dt.date).size().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily.index, daily.values, marker="o", linewidth=1)
        ax.set_title("Visits per day"); ax.set_xlabel("Date"); ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        add(fig); path = os.path.join(out_dir, "date_daily_counts.png")
        _save_fig(fig, path); outputs["date_daily_counts"] = path

        monthly = df.dropna(subset=["Date"]).groupby(df["Date"].dt.to_period("M")).size().sort_index()
        monthly.index = monthly.index.astype(str)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(monthly.index, monthly.values)
        ax.set_title("Visits per month"); ax.set_xlabel("Month"); ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        add(fig); path = os.path.join(out_dir, "date_monthly_counts.png")
        _save_fig(fig, path); outputs["date_monthly_counts"] = path

    # 2) Gender
    if "Gender" in df.columns and df["Gender"].notna().any():
        g = df["Gender"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(g.index.astype(str), g.values)
        ax.set_title("Gender distribution"); ax.set_xlabel("Gender"); ax.set_ylabel("Count")
        add(fig); path = os.path.join(out_dir, "gender_counts.png")
        _save_fig(fig, path); outputs["gender_counts"] = path

    # 3) Clinics top N
    if "Clinics" in df.columns and df["Clinics"].notna().any():
        c = df["Clinics"].dropna().value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(c.index.astype(str), c.values)
        ax.set_title(f"Top {top_n} Clinics by count"); ax.set_xlabel("Clinic"); ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        add(fig); path = os.path.join(out_dir, "clinics_top_counts.png")
        _save_fig(fig, path); outputs["clinics_top_counts"] = path

    # 4) Governorates
    if "GOVENORATES" in df.columns and df["GOVENORATES"].notna().any():
        gov = df["GOVENORATES"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(gov.index.astype(str), gov.values)
        ax.set_title("Governorates distribution"); ax.set_xlabel("Governorate"); ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        add(fig); path = os.path.join(out_dir, "governorates_counts.png")
        _save_fig(fig, path); outputs["governorates_counts"] = path

    # 5) Age histogram
    if "Age" in df.columns and df["Age"].notna().any():
        age = df["Age"].dropna()
        age = age[(age >= 0) & (age <= age_max)]
        bins = np.arange(0, age_max + 5, 5)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(age.values, bins=bins)
        ax.set_title(f"Age distribution (0–{age_max})"); ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
        ax.set_xlim(0, age_max)
        add(fig); path = os.path.join(out_dir, "age_hist_0_100.png")
        _save_fig(fig, path); outputs["age_hist_0_100"] = path

    # 6) Age by Gender boxplot
    if all(c in df.columns for c in ["Age", "Gender"]):
        sub = df.dropna(subset=["Age", "Gender"])
        sub = sub[(sub["Age"] >= 0) & (sub["Age"] <= age_max)]
        if not sub.empty:
            order = sub["Gender"].value_counts().index.tolist()
            groups = [sub.loc[sub["Gender"] == k, "Age"].values for k in order]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(groups, labels=[str(k) for k in order], showfliers=False)
            ax.set_title(f"Age by Gender (0–{age_max})"); ax.set_xlabel("Gender"); ax.set_ylabel("Age")
            ax.set_ylim(0, age_max)
            add(fig); path = os.path.join(out_dir, "age_by_gender_box.png")
            _save_fig(fig, path); outputs["age_by_gender_box"] = path

    # 7) Clinic x Governorate heatmap
    if all(c in df.columns for c in ["Clinics", "GOVENORATES"]):
        sub = df.dropna(subset=["Clinics", "GOVENORATES"])
        if not sub.empty:
            top_clinics = sub["Clinics"].value_counts().head(top_n).index
            top_govs = sub["GOVENORATES"].value_counts().head(top_gov_n).index
            sub = sub[sub["Clinics"].isin(top_clinics) & sub["GOVENORATES"].isin(top_govs)]
            ct = pd.crosstab(sub["Clinics"], sub["GOVENORATES"])
            ct = ct.loc[top_clinics.intersection(ct.index), top_govs.intersection(ct.columns)]
            fig_w = max(9, 1.2 * ct.shape[1])
            fig_h = max(5, 0.6 * ct.shape[0])
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(ct.values, aspect="auto")
            ax.set_title(f"Counts: Clinic × Governorate\nTop {top_n} Clinics, Top {top_gov_n} Govs")
            ax.set_xlabel("Governorate"); ax.set_ylabel("Clinic")
            ax.set_xticks(np.arange(ct.shape[1]))
            ax.set_xticklabels([str(x) for x in ct.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(ct.shape[0]))
            ax.set_yticklabels([str(x) for x in ct.index])
            fig.colorbar(im, ax=ax, label="Count")
            if ct.shape[0] <= 15 and ct.shape[1] <= 10:
                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        ax.text(j, i, str(int(ct.values[i, j])), ha="center", va="center")
            add(fig); path = os.path.join(out_dir, "clinic_by_governorate_heatmap_clear.png")
            _save_fig(fig, path); outputs["clinic_by_governorate_heatmap_clear"] = path

    if pdf:
        pdf.close()
        outputs["all_charts_pdf"] = os.path.join(out_dir, "all_charts.pdf")

    return outputs


# ---------------------------------------------------------------------------
# Public: make_date_combo_charts
# ---------------------------------------------------------------------------

def make_date_combo_charts(
    df: pd.DataFrame,
    out_dir: str = "charts_date_combos",
    freq: str = "M",
    top_k: int = 8,
    age_max: int = 100,
    make_pdf: bool = True,
) -> dict:
    """
    Date × {Gender, GOVENORATES, Clinics, Age} combo charts.
    Returns a dict of {name: file_path}.
    """
    df = _prep_df(df)
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}

    if "Date" not in df.columns or not df["Date"].notna().any():
        raise ValueError("Date column is missing or entirely NaT.")

    df = df.dropna(subset=["Date"]).copy()
    df["_PERIOD"] = _to_period(df["Date"], freq)

    pdf = PdfPages(os.path.join(out_dir, f"all_date_combo_charts_{freq}.pdf")) if make_pdf else None

    def dump(fig, name):
        if fig is None:
            return
        if pdf:
            fig.tight_layout(); pdf.savefig(fig)
        path = os.path.join(out_dir, f"{name}.png")
        _save_fig(fig, path)
        outputs[name] = path

    # Total volume
    total = df.groupby("_PERIOD").size().sort_index()
    total.index = total.index.astype(str)
    dump(_plot_total_over_time(total, f"Total records over time ({freq})"), f"date_total_{freq}")

    # Categorical combos
    cat_cols = [c for c in ["Gender", "GOVENORATES", "Clinics"] if c in df.columns and df[c].notna().any()]
    for col in cat_cols:
        tmp = df.dropna(subset=[col, "_PERIOD"]).copy()
        tmp[col] = _topk_other(tmp[col], top_k=top_k)
        pivot = _pivot_counts(tmp, "_PERIOD", col)

        dump(_plot_multiline(pivot, f"{col} vs Date: counts ({freq}) (Top {top_k}+OTHER)"),
             f"date_{col}_multiline_{freq}")
        dump(_plot_stacked_area(pivot, f"{col} vs Date: stacked area ({freq})"),
             f"date_{col}_stacked_area_{freq}")
        dump(_plot_stacked_bar(pivot, f"{col} vs Date: stacked bar ({freq})"),
             f"date_{col}_stacked_bar_{freq}")
        dump(_plot_heatmap(pivot, f"{col} vs Date: heatmap ({freq})"),
             f"date_{col}_heatmap_{freq}")
        dump(_plot_share_over_time(pivot, f"{col} share over time ({freq})"),
             f"date_{col}_share_{freq}")

    # Age combos
    if "Age" in df.columns and df["Age"].notna().any():
        dump(_plot_age_trend(df, "_PERIOD", f"Age trend over time ({freq})", age_max=age_max),
             f"date_age_trend_{freq}")
        dump(_plot_age_box_by_period(df, "_PERIOD", f"Age distribution by period ({freq})", age_max=age_max),
             f"date_age_box_by_period_{freq}")

    if pdf:
        pdf.close()
        outputs["all_charts_pdf"] = os.path.join(out_dir, f"all_date_combo_charts_{freq}.pdf")

    return outputs
