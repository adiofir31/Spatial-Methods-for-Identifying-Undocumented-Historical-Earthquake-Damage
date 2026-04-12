"""
main.py — Orchestrator for the earthquake spatial-analysis pipeline.

Usage examples
--------------
    # Full pipeline (Phase 1 + all Phase 2 models) for one earthquake:
    python main.py --earthquake M_7.2_Nippes_2021

    # Phase 2 only (skip preprocessing — requires an existing near table):
    python main.py --earthquake M_6.4_Dead_Sea_1927 --skip_preprocessing

    # Run a single model:
    python main.py --earthquake M_6.4_Dead_Sea_1927 --skip_preprocessing --model linear
    python main.py --earthquake M_6.4_Dead_Sea_1927 --skip_preprocessing --model knn
    python main.py --earthquake M_6.9_Kamariotissa_2014 --skip_preprocessing --model kriging
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from config import DATA_DIR, EARTHQUAKE_PARAMS, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Earthquake intensity spatial-analysis pipeline."
    )
    parser.add_argument(
        "--earthquake", required=True,
        help=f"Earthquake key. Available: {list(EARTHQUAKE_PARAMS.keys())}",
    )
    parser.add_argument(
        "--skip_preprocessing", action="store_true",
        help="Skip Phase 1 (arcpy). Assumes near table already exists in data/.",
    )
    parser.add_argument(
        "--model", default="all",
        choices=["all", "linear", "knn", "kriging"],
        help="Which Phase-2 model to run (default: all).",
    )
    parser.add_argument(
        "--near_table", default=None,
        help="Override: path to a pre-existing near table CSV.",
    )
    args = parser.parse_args()

    eq = args.earthquake
    if eq not in EARTHQUAKE_PARAMS:
        print(f"Unknown earthquake: {eq}")
        print(f"Available: {list(EARTHQUAKE_PARAMS.keys())}")
        sys.exit(1)

    params = EARTHQUAKE_PARAMS[eq]

    # ── Phase 1 ──────────────────────────────────────────────────────────
    if not args.skip_preprocessing:
        from preprocessing import run_preprocessing

        print(f"\n{'='*60}")
        print(f" Phase 1 — Preprocessing: {eq}")
        print(f"{'='*60}\n")
        near_table = run_preprocessing(eq)
    else:
        near_table = args.near_table or os.path.join(DATA_DIR, params["near_table_csv"])
        needs_near_table = args.model in ("all", "linear", "knn")
        if needs_near_table and not os.path.isfile(near_table):
            print(f"\n[ERROR] Near table not found: {near_table}")
            print(f"        Place the file in the data/ directory, or specify a path with --near_table")
            sys.exit(1)
        print(f"\n>> Skipping preprocessing. Using near table: {near_table}")

    # ── Phase 2 ──────────────────────────────────────────────────────────
    run_linear = args.model in ("all", "linear")
    run_knn = args.model in ("all", "knn")
    run_kriging = args.model in ("all", "kriging")

    # Per-earthquake distance filters (historical vs instrumental ranges)
    filters = params.get("filters", {})

    if run_linear and os.path.isfile(near_table):
        from models.linear_regression import run_linear_regression

        print(f"\n{'='*60}")
        print(f" Phase 2A — Linear Regression: {eq}")
        print(f"{'='*60}\n")
        df_lr = run_linear_regression(
            near_table, earthquake_name=eq,
            total_dist_list=filters.get("total_dist_list"),
            nei_dist_list=filters.get("nei_dist_list"),
        )
        out = os.path.join(RESULTS_DIR, f"{eq}_linear_results.csv")
        df_lr.to_csv(out, index=False)
        print(f"  [OK] Saved {len(df_lr)} rows → {out}")

    if run_knn and os.path.isfile(near_table):
        from models.knn import run_knn_filtered, run_knn_unfiltered

        print(f"\n{'='*60}")
        print(f" Phase 2B — KNN (filtered + unfiltered): {eq}")
        print(f"{'='*60}\n")

        df_filt = run_knn_filtered(
            near_table, earthquake_name=eq,
            total_dist_list=filters.get("total_dist_list"),
            nei_dist_list=filters.get("nei_dist_list"),
        )
        df_unfilt = run_knn_unfiltered(
            near_table, earthquake_name=eq,
            nei_dist_list=filters.get("nei_dist_list"),
        )
        df_knn = pd.concat([df_filt, df_unfilt], ignore_index=True)

        out = os.path.join(RESULTS_DIR, f"{eq}_knn_results.csv")
        df_knn.to_csv(out, index=False)
        print(f"  [OK] Saved {len(df_knn)} rows → {out}")

    if run_kriging:
        # Kriging requires projected XY coordinates, which are created by Phase 1.
        # It reads from the Damage_locations table (not the near table).
        kriging_path = os.path.join(RESULTS_DIR, f"{eq}_Damage_locations.csv")

        if os.path.isfile(kriging_path):
            from models.kriging import run_kriging as _run_kriging

            angle = params.get("anisotropy_angle", 0)
            print(f"\n{'='*60}")
            print(f" Phase 2C — Kriging: {eq}  (angle={angle}°)")
            print(f"{'='*60}\n")
            df_k = _run_kriging(kriging_path, earthquake_name=eq, anisotropy_angle=angle)
            out = os.path.join(RESULTS_DIR, f"{eq}_kriging_results.csv")
            df_k.to_csv(out, index=False)
            print(f"  [OK] Saved {len(df_k)} rows → {out}")
        else:
            print(f"\n[WARNING] Kriging input not found: {kriging_path}")
            print(f"          Kriging requires projected XY coordinates (columns: X, Y, int)")
            print(f"          generated by Phase 1. Run preprocessing first, or provide the")
            print(f"          file manually via:  python -m models.kriging --csv <path>")

    print(f"\n{'='*60}")
    print(" Pipeline finished.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
