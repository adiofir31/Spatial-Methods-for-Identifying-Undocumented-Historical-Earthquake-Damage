"""
models/linear_regression.py — Phase 2A: Linear Regression model.

The model learns a linear relationship between inter-point distance and the
mean absolute intensity difference, then uses it to predict the intensity of
test sites from their trained neighbours.

Can be run standalone:
    python -m models.linear_regression --near_table data/D1927_near_new.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

# Allow running as ``python -m models.linear_regression`` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    compute_error_metrics,
    is_angle_within_range,
    load_near_table,
)
from config import (
    ANGLE_RANGES,
    TOTAL_DIST_LIST,
    NEI_DIST_LIST,
    PRED_NEIGHBORS_LIST,
    MIN_NEIGHBORS,
    TEST_SIZE,
    RANDOM_STATE,
    RESULTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: bin-level statistics for the regression training step
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_bin_stats(
    data: np.ndarray, jumps: range, nei_dist: int
) -> pd.DataFrame:
    """
    For each *jump* size, partition [0, nei_dist) into bins of width *jump*
    and compute the mean absolute intensity difference per bin (averaged
    across unique IN_FID sites).
    """
    rows = []
    for jump in jumps:
        for dist_start in range(0, nei_dist, jump):
            dist_end = dist_start + jump
            site_means = []
            for fid in np.unique(data["IN_FID"]):
                mask = (
                    (data["IN_FID"] == fid)
                    & (data["NEAR_DIST"] >= dist_start)
                    & (data["NEAR_DIST"] < dist_end)
                )
                if np.any(mask):
                    m = np.mean(data["abs_int_diff"][mask])
                    if not np.isnan(m):
                        site_means.append(m)
            rows.append({
                "distance_end": dist_end,
                "jump": jump,
                "mean_int": np.mean(site_means) if site_means else -1,
                "count": len(site_means),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_regression(
    near_table_path: str,
    earthquake_name: str = "earthquake",
    *,
    angle_ranges: list[int] | None = None,
    total_dist_list: list[int] | None = None,
    nei_dist_list: list[int] | None = None,
    pred_neighbors_list: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run the filtered linear-regression + KNN_k models over a parameter grid.

    Parameters
    ----------
    near_table_path : str
        Path to the enriched near-table CSV produced by Phase 1.
    earthquake_name : str
        Label used in the output table.

    Returns
    -------
    pd.DataFrame  with one row per (angle, dist, nei_dist, K, model) combo.
    """
    angle_ranges = angle_ranges or ANGLE_RANGES
    total_dist_list = total_dist_list or TOTAL_DIST_LIST
    nei_dist_list = nei_dist_list or NEI_DIST_LIST
    pred_neighbors_list = pred_neighbors_list or PRED_NEIGHBORS_LIST

    df = load_near_table(near_table_path)
    results: list[dict] = []

    total_iters = (
        len(angle_ranges) * len(total_dist_list) * len(nei_dist_list) * len(pred_neighbors_list)
    )
    counter = 0

    for angle_range in angle_ranges:
        for total_dist in total_dist_list:
            for nei_dist in nei_dist_list:

                # ── Spatial filtering ────────────────────────────────────
                mask = (df["epic_dist"] < total_dist) & (df["NEAR_DIST"] < nei_dist)
                df_filt = df[mask]

                # Vectorised azimuth filtering (safe when df_filt is empty)
                if len(df_filt) > 0:
                    ea = df_filt["epic_angle"]
                    nea = df_filt["near_epic_angle"]

                    # Direct azimuth comparison
                    diff = np.abs(ea - nea) % 360
                    direct = (diff <= angle_range) | (diff >= 360 - angle_range)

                    # Anti-podal azimuth comparison (+180 degrees)
                    anti_ea = (ea + 180) % 360
                    diff2 = np.abs(anti_ea - nea) % 360
                    anti = (diff2 <= angle_range) | (diff2 >= 360 - angle_range)

                    # Apply combined filter
                    df_filt = df_filt[direct | anti]

                base = {
                    "earthquake": earthquake_name,
                    "data_size": len(df_filt),
                    "angle_range": angle_range,
                    "total_dist": total_dist,
                    "nei_dist": nei_dist,
                }

                if len(df_filt) == 0:
                    for pn in pred_neighbors_list:
                        counter += 1
                    continue

                # ── Train / test split (site-level) ──────────────────────
                unique_ids = np.unique(
                    np.concatenate((df_filt["IN_FID"], df_filt["NEAR_FID"]))
                )
                train_ids, test_ids = train_test_split(
                    unique_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
                )
                train_df = df_filt[
                    np.isin(df_filt["IN_FID"], train_ids)
                    & np.isin(df_filt["NEAR_FID"], train_ids)
                ]

                # ── Fit linear regression on distance bins ───────────────
                jumps = range(1, 51)
                bin_stats = _calculate_bin_stats(train_df, jumps, nei_dist)

                best = {"r_squared": -1, "slope": 0, "intercept": 0, "p_value": 1}
                filtered_bins = bin_stats[bin_stats["count"] > MIN_NEIGHBORS]
                for jump_val in filtered_bins["jump"].unique():
                    grp = filtered_bins[filtered_bins["jump"] == jump_val]
                    if len(grp) > 1:
                        slope, intercept, r_val, p_val, _ = linregress(
                            grp["distance_end"], grp["mean_int"]
                        )
                        r2 = r_val ** 2
                        if r2 > best["r_squared"] and p_val <= 0.10:
                            best.update(slope=slope, intercept=intercept,
                                        r_squared=r2, p_value=p_val)

                # ── Predict on test set ──────────────────────────────────
                test_df = df_filt[
                    np.isin(df_filt["IN_FID"], test_ids)
                    & np.isin(df_filt["NEAR_FID"], train_ids)
                ]
                if len(test_df) == 0:
                    for pn in pred_neighbors_list:
                        counter += 1
                    continue

                test_df = np.copy(test_df)
                # Append predicted-intensity field
                from models.utils import compute_error_metrics as _cem  # noqa: already imported

                abs_pred = best["slope"] * test_df["NEAR_DIST"] + best["intercept"]
                cond = test_df["near_epic_dist"] > test_df["epic_dist"]
                int_pred = np.where(
                    cond,
                    test_df["near_int"] + abs_pred,
                    test_df["near_int"] - abs_pred,
                )

                unique_test_ids = np.unique(test_df["IN_FID"])

                for pred_neighbors in pred_neighbors_list:
                    counter += 1
                    if counter % 500 == 0:
                        print(
                            f"  [{counter}/{total_iters}]  angle={angle_range}  "
                            f"dist={total_dist}  nei={nei_dist}  K={pred_neighbors}"
                        )

                    row = dict(base)
                    row["pred_neighbors"] = pred_neighbors

                    # — Linear regression aggregation —
                    lr_preds, lr_true = [], []
                    for fid in unique_test_ids:
                        idx = test_df["IN_FID"] == fid
                        subset = test_df[idx]
                        order = np.argsort(subset["NEAR_DIST"])
                        topk = subset[order][:pred_neighbors]
                        if len(topk) == 0:
                            continue
                        pred_vals = int_pred[np.where(idx)[0][order[:pred_neighbors]]]
                        lr_preds.append(float(np.mean(pred_vals)))
                        lr_true.append(float(topk[0]["int"]))

                    if lr_preds:
                        metrics = compute_error_metrics(
                            np.array(lr_true), np.array(lr_preds)
                        )
                        results.append({**row, "model": "linear", **metrics})



    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Run linear-regression model.")
    parser.add_argument("--near_table", required=True, help="Path to near-table CSV.")
    parser.add_argument("--name", default="earthquake", help="Earthquake label.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Phase 2A — Linear Regression: {args.name}")
    print(f"{'='*60}\n")

    df = run_linear_regression(args.near_table, earthquake_name=args.name)
    out = args.out or os.path.join(RESULTS_DIR, f"{args.name}_linear_results.csv")
    df.to_csv(out, index=False)
    print(f"\n[DONE] Results saved: {out}  ({len(df)} rows)")


if __name__ == "__main__":
    _cli()
