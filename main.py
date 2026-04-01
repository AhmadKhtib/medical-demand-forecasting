"""
clinic-rx-demand-forecast — main entry point.

Runs preprocessing then modeling in one shot.

Usage:
    python main.py --south data/Data.xlsx data/Deir_El-Balah.xlsm
                   --middle data/middel_data_assi.xlsx
                   --out_south data/final_data.xlsx
                   --out_middle data/middel_final_data.xlsx
                   --out_forecast data/forecasts.xlsx
                   --horizon 4
                   --charts_dir charts
                   --no_charts          # skip chart generation
"""

import argparse
import os
import sys

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import run_preprocessing
from src.model import run_modeling


def main():
    parser = argparse.ArgumentParser(
        description="Full clinic medication demand preprocessing + forecasting pipeline"
    )

    # --- Preprocessing args ---
    parser.add_argument(
        "--south", nargs="+", required=True,
        help="South-area raw Excel/xlsm file(s)",
    )
    parser.add_argument(
        "--middle", default=None,
        help="Middle-area raw Excel file (optional)",
    )
    parser.add_argument(
        "--out_south", default="data/final_data.xlsx",
        help="Output path for south preprocessed Excel",
    )
    parser.add_argument(
        "--out_middle", default=None,
        help="Output path for middle preprocessed Excel",
    )

    # --- Modeling args ---
    parser.add_argument(
        "--out_forecast", default="data/forecasts.xlsx",
        help="Output path for forecast Excel",
    )
    parser.add_argument(
        "--horizon", type=int, default=4,
        help="Forecast horizon in weeks (default: 4)",
    )

    # --- Shared ---
    parser.add_argument(
        "--charts_dir", default="charts",
        help="Directory to write all charts",
    )
    parser.add_argument(
        "--no_charts", action="store_true",
        help="Skip chart generation",
    )

    args = parser.parse_args()

    # --- PREPROCESSING ---
    print("\n" + "=" * 70)
    print("PHASE 1 — PREPROCESSING")
    print("=" * 70)

    run_preprocessing(
        south_paths=args.south,
        middle_path=args.middle,
        out_south=args.out_south,
        out_middle=args.out_middle,
        charts_dir=args.charts_dir,
        make_charts=not args.no_charts,
    )

    # --- MODELING ---
    print("\n" + "=" * 70)
    print("PHASE 2 — MODELING & FORECASTING")
    print("=" * 70)

    run_modeling(
        south_path=args.out_south,
        middle_path=args.out_middle,
        out_forecast=args.out_forecast,
        horizon=args.horizon,
        charts_dir=os.path.join(args.charts_dir, "forecast"),
    )

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"  Preprocessed south  -> {args.out_south}")
    if args.out_middle:
        print(f"  Preprocessed middle -> {args.out_middle}")
    print(f"  Forecast table      -> {args.out_forecast}")
    print(f"  Charts              -> {args.charts_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
