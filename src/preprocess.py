"""
Preprocessing pipeline.

Loads raw Excel/xlsm sources, normalizes columns, applies per-clinic MEDICAL parsers,
merges all clinics, splits medication/dose_schedule into wide columns, and saves the
final Excel files — reproducing exactly what the notebook produces.

Usage:
    python -m src.preprocess --south data/Data.xlsx data/Deir_El-Balah.xlsm
                             --middle data/middel_data_assi.xlsx
                             --out_south data/final_data.xlsx
                             --out_middle data/middel_final_data.xlsx
                             --charts_dir charts
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

from .parsers import CLINIC_PARSERS, parse_medical_cell
from .visualize import make_basic_charts, make_date_combo_charts


# ---------------------------------------------------------------------------
# Step 1 — Load raw data
# ---------------------------------------------------------------------------

def load_south(south_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate south-area Excel files."""
    frames = [pd.read_excel(p) for p in south_paths]
    df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[load_south] shape={df.shape}")
    return df


def load_middle(middle_path: str) -> pd.DataFrame:
    """Load middle-area Excel file."""
    df = pd.read_excel(middle_path)
    print(f"[load_middle] shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Normalize columns
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Clinics, Gender, GOVENORATES columns.
    Exact logic from the preprocessing notebook.
    """
    df = df.copy()

    # Clinics
    df["Clinics"] = df["Clinics"].str.strip().str.lower()

    # Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.strip().str.lower()

    # GOVENORATES
    if "GOVENORATES" in df.columns:
        df["GOVENORATES"] = df["GOVENORATES"].str.lower()
        df["GOVENORATES"] = df["GOVENORATES"].str.replace("-", " ", regex=False)
        df["GOVENORATES"] = df["GOVENORATES"].str.replace("/", " ", regex=False)
        df["GOVENORATES"] = df["GOVENORATES"].str.strip()
        df["GOVENORATES"] = df["GOVENORATES"].str.replace(r"\s+", " ", regex=True)

    print(
        f"[normalize_columns] clinics={df['Clinics'].nunique()} "
        f"rows={len(df)}"
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 — Parse MEDICAL column per clinic
# ---------------------------------------------------------------------------

def parse_all_clinics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each known clinic, apply its dedicated MEDICAL parser and collect
    the four output columns: diagnosis, plan_text, medication, dose_schedule.
    Clinics not in CLINIC_PARSERS are passed through with NaN in those columns.
    """
    result_frames = []

    for clinic_name, parser in CLINIC_PARSERS.items():
        mask = df["Clinics"] == clinic_name
        if not mask.any():
            continue

        subset = df[mask].copy()
        cols = ["diagnosis", "plan_text", "medication", "dose_schedule"]
        subset = subset.drop(columns=cols, errors="ignore")
        subset[cols] = subset["MEDICAL"].apply(parser)
        result_frames.append(subset)
        print(f"  [{clinic_name}] rows={len(subset)}")

    # Clinics not covered by any parser
    known = set(CLINIC_PARSERS.keys())
    other_mask = ~df["Clinics"].isin(known)
    if other_mask.any():
        other = df[other_mask].copy()
        for col in ["diagnosis", "plan_text", "medication", "dose_schedule"]:
            other[col] = np.nan
        result_frames.append(other)
        print(f"  [other clinics] rows={len(other)}")

    parsed = pd.concat(result_frames, ignore_index=True, sort=False)
    print(f"[parse_all_clinics] total rows={len(parsed)}")
    return parsed


# ---------------------------------------------------------------------------
# Step 4 — Merge all clinic DataFrames
# ---------------------------------------------------------------------------

def merge_clinics(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """
    De-duplicate columns, unify schema, cast key dtypes, sort by ID.
    Matches the notebook merge logic exactly.
    """
    # Drop duplicated columns (common after concat)
    df = parsed_df.loc[:, ~parsed_df.columns.duplicated()].copy()

    # Cast key types
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype("string")

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Sort by numeric ID then original ID (stable, NaNs last)
    if "ID" in df.columns:
        df["_ID_num"] = pd.to_numeric(df["ID"], errors="coerce")
        df = df.sort_values(by=["_ID_num", "ID"], na_position="last", kind="mergesort")
        df = df.drop(columns=["_ID_num"]).reset_index(drop=True)

    print(f"[merge_clinics] rows={len(df)} cols={len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Split medication and dose_schedule into wide columns
# ---------------------------------------------------------------------------

def _split_pipe_column(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    Normalize separators in `col`, then split on ' | ' into
    {prefix}_1, {prefix}_2, ... and join back.
    """
    s = df[col].astype("string").str.strip().replace({"<NA>": pd.NA})

    # Normalize separators (newlines, tabs -> ' | ')
    s = s.str.replace(r"[\r\n\t]+", " | ", regex=True)
    s = s.str.replace(r"\s*\|\s*", " | ", regex=True)
    s = s.str.replace(r"\s{2,}", " ", regex=True).str.strip()
    s = s.where(s.notna() & (s != ""), pd.NA)

    max_parts = int(s.dropna().str.count(r"\|").max() + 1) if s.notna().any() else 1
    wide = s.str.split(r"\s*\|\s*", expand=True, n=max_parts - 1)
    wide = wide.apply(lambda c: c.astype("string").str.strip().replace("", pd.NA))
    wide.columns = [f"{prefix}_{i+1}" for i in range(wide.shape[1])]

    # Drop existing split columns to avoid duplicates, then join
    df = df.drop(columns=[c for c in df.columns if c.startswith(f"{prefix}_")], errors="ignore")
    return df.join(wide)


def split_wide_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split medication and dose_schedule into individual wide columns."""
    if "medication" in df.columns:
        df = _split_pipe_column(df, "medication", "medication")
    if "dose_schedule" in df.columns:
        df = _split_pipe_column(df, "dose_schedule", "dose_schedule")
    print(f"[split_wide_columns] cols={[c for c in df.columns if c.startswith('medication_') or c.startswith('dose_schedule_')]}")
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    south_paths: list[str],
    middle_path: str | None,
    out_south: str,
    out_middle: str | None = None,
    charts_dir: str = "charts",
    make_charts: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Full end-to-end preprocessing pipeline.

    Returns a dict with keys 'south' and (optionally) 'middle'.
    """

    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    south_df = load_south(south_paths)

    if middle_path:
        middle_df = load_middle(middle_path)

    print("\n" + "=" * 60)
    print("STEP 2: Normalizing columns")
    print("=" * 60)

    south_df = normalize_columns(south_df)

    if make_charts:
        print("\nGenerating exploratory charts...")
        make_basic_charts(south_df, out_dir=os.path.join(charts_dir, "charts02"), top_n=12, make_pdf=True)
        make_date_combo_charts(south_df, out_dir=os.path.join(charts_dir, "charts_date_combos"),
                               freq="M", top_k=8, age_max=100, make_pdf=True)
        print("Charts saved.")

    print("\n" + "=" * 60)
    print("STEP 3: Parsing MEDICAL column per clinic (south)")
    print("=" * 60)

    south_parsed = parse_all_clinics(south_df)

    print("\n" + "=" * 60)
    print("STEP 4: Merging & sorting")
    print("=" * 60)

    south_merged = merge_clinics(south_parsed)

    print("\n" + "=" * 60)
    print("STEP 5: Splitting wide medication/dose columns")
    print("=" * 60)

    south_final = split_wide_columns(south_merged)

    print(f"\nSaving south final data -> {out_south}")
    os.makedirs(os.path.dirname(out_south) or ".", exist_ok=True)
    south_final.to_excel(out_south, index=False)
    print(f"Saved. Shape: {south_final.shape}")

    result = {"south": south_final}

    if middle_path and out_middle:
        print("\n" + "=" * 60)
        print("Processing middle-area data")
        print("=" * 60)

        middle_df = normalize_columns(middle_df)
        middle_parsed = parse_all_clinics(middle_df)
        middle_merged = merge_clinics(middle_parsed)
        middle_final = split_wide_columns(middle_merged)

        print(f"\nSaving middle final data -> {out_middle}")
        os.makedirs(os.path.dirname(out_middle) or ".", exist_ok=True)
        middle_final.to_excel(out_middle, index=False)
        print(f"Saved. Shape: {middle_final.shape}")

        result["middle"] = middle_final

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(description="Run clinic data preprocessing pipeline")
    parser.add_argument("--south", nargs="+", required=True,
                        help="South-area Excel/xlsm file(s)")
    parser.add_argument("--middle", default=None,
                        help="Middle-area Excel file (optional)")
    parser.add_argument("--out_south", default="data/final_data.xlsx",
                        help="Output path for south final Excel")
    parser.add_argument("--out_middle", default=None,
                        help="Output path for middle final Excel")
    parser.add_argument("--charts_dir", default="charts",
                        help="Directory to save charts")
    parser.add_argument("--no_charts", action="store_true",
                        help="Skip chart generation")

    args = parser.parse_args()

    run_preprocessing(
        south_paths=args.south,
        middle_path=args.middle,
        out_south=args.out_south,
        out_middle=args.out_middle,
        charts_dir=args.charts_dir,
        make_charts=not args.no_charts,
    )


if __name__ == "__main__":
    _main()
