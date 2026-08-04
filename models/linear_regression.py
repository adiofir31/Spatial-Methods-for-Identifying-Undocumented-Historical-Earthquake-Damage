"""
models/linear_regression.py — Phase 2A: Linear Regression model.

The model learns a linear relationship between inter-point distance and the
mean absolute intensity difference, then uses it to predict the intensity of
test sites from their trained neighbours.

Can be run standalone:
    python -m models.linear_regression \\
        --near_table data/Dead_Sea_1927_near_table.csv \\
        --sites_csv data/Dead_sea_1927.csv --name Dead_Sea_1927
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Allow running as ``python -m models.linear_regression`` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import (
    compute_error_metrics,
    is_angle_within_range,
    load_near_table,
)
from models.cross_validation import (
    build_site_folds,
    n_folds_of,
    split_ids_for_fold,
)
from config import (
    ANGLE_RANGES,
    TOTAL_DIST_LIST,
    NEI_DIST_LIST,
    PRED_NEIGHBORS_LIST,
    MIN_NEIGHBORS,
    CV_SCHEMES,
    K_FOLDS,
    RANDOM_STATE,
    FOLD_COORD_SPACE_NEAR_TABLE,
    SITE_ID_COL,
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
    and compute the mean absolute intensity difference per bin, averaged
    across unique IN_FID sites.

    Vectorised rewrite of the original triple loop
    (jump -> distance bin -> IN_FID). One ``np.digitize`` plus one stable
    ``np.lexsort`` per jump, i.e. O(N log N) per jump instead of
    O(N * n_FIDs * n_bins) boolean masks. This is what makes the largest
    datasets (e.g. Ridgecrest, ~1,500 sites) tractable.

    The result is identical to the original loop, value for value:

    * bin membership uses ``np.digitize`` against exact integer edges,
      reproducing the original ``(NEAR_DIST >= start) & (NEAR_DIST < end)``
      comparisons exactly, including the exclusion of NaN and
      out-of-range values;
    * ``np.lexsort`` is stable and ordered bin-major / IN_FID-minor,
      matching the original iteration order (bins ascending, ``np.unique``
      FIDs ascending) and preserving the original row order within each
      group;
    * per-group means and the per-bin mean-of-means are still computed
      with ``np.mean`` on the float32 data.

    ``pandas.groupby().mean()`` is deliberately **not** used here: it
    accumulates in float64, which changes low-order bits that then
    propagate through ``linregress`` into the fitted slope.
    """
    rows = []

    fid_all = data["IN_FID"]
    dist_all = data["NEAR_DIST"]
    vals_all = data["abs_int_diff"]

    for jump in jumps:
        starts = range(0, nei_dist, jump)
        n_bins = len(starts)

        # Integer bin edges 0, jump, 2*jump, ... (exact in float64)
        edges = np.arange(0, (n_bins + 1) * jump, jump, dtype=np.float64)

        # edges[i] <= x < edges[i+1]  ->  bin i
        bin_idx = np.digitize(dist_all, edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)

        b = bin_idx[valid]
        f = fid_all[valid]
        v = vals_all[valid]

        bin_mean_lists: list[list] = [[] for _ in range(n_bins)]

        if b.size:
            # stable sort; primary key = bin, secondary key = site ID
            order = np.lexsort((f, b))
            b_s, f_s, v_s = b[order], f[order], v[order]

            new_group = np.empty(b_s.size, dtype=bool)
            new_group[0] = True
            new_group[1:] = (b_s[1:] != b_s[:-1]) | (f_s[1:] != f_s[:-1])
            g_starts = np.flatnonzero(new_group)
            g_ends = np.append(g_starts[1:], b_s.size)

            for gs, ge in zip(g_starts, g_ends):
                m = np.mean(v_s[gs:ge])
                if not np.isnan(m):
                    bin_mean_lists[b_s[gs]].append(m)

        for i, dist_start in enumerate(starts):
            means = bin_mean_lists[i]
            rows.append({
                "distance_end": dist_start + jump,
                "jump": jump,
                "mean_int": np.mean(means) if means else -1,
                "count": len(means),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_regression(
    near_table_path: str,
    sites_csv_path: str,
    earthquake_name: str = "earthquake",
    *,
    angle_ranges: list[int] | None = None,
    total_dist_list: list[int] | None = None,
    nei_dist_list: list[int] | None = None,
    pred_neighbors_list: list[int] | None = None,
    schemes: list[str] | None = None,
    k_folds: int = K_FOLDS,
    lon_col: str = "POINT_X",
    lat_col: str = "POINT_Y",
    id_col: str = SITE_ID_COL,
) -> pd.DataFrame:
    """
    Run the filtered linear-regression model over a parameter grid, under
    k-fold cross-validation with both partitioning schemes.

    For every (scheme, fold, angle_range, epicentral distance, neighbour
    distance, K) combination the model is fitted on the training sites of
    the fold and evaluated on the withheld sites, so that each site is
    tested exactly once per scheme.

    Parameters
    ----------
    near_table_path : str
        Path to the enriched near-table CSV produced by Phase 1.
    sites_csv_path : str
        Path to the site table used to build the fold map (one row per
        reporting location, with coordinate and identifier columns).
    earthquake_name : str
        Label written to the output table.
    schemes : list of str, optional
        Cross-validation schemes to run. Defaults to ``CV_SCHEMES``.

    Returns
    -------
    pd.DataFrame
        One row per (scheme, fold, angle, dist, nei_dist, K) combination,
        with the error metrics defined in ``models.utils``.
    """
    angle_ranges = angle_ranges or ANGLE_RANGES
    total_dist_list = total_dist_list or TOTAL_DIST_LIST
    nei_dist_list = nei_dist_list or NEI_DIST_LIST
    pred_neighbors_list = pred_neighbors_list or PRED_NEIGHBORS_LIST
    schemes = schemes or CV_SCHEMES

    df = load_near_table(near_table_path)
    results: list[dict] = []

    for scheme in schemes:
        site_to_fold = build_site_folds(
            sites_csv_path,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
            scheme=scheme,
            k=k_folds,
            random_state=RANDOM_STATE,
            coord_space=FOLD_COORD_SPACE_NEAR_TABLE,
        )
        n_folds = n_folds_of(site_to_fold)
        print(f"  scheme={scheme}: {n_folds} populated fold(s)")

        for angle_range in angle_ranges:
            for total_dist in total_dist_list:
                for nei_dist in nei_dist_list:

                    # ── Spatial filtering ────────────────────────────────
                    mask = (df["epic_dist"] < total_dist) & (
                        df["NEAR_DIST"] < nei_dist
                    )
                    df_filt = df[mask]

                    if len(df_filt) > 0:
                        ea = df_filt["epic_angle"]
                        nea = df_filt["near_epic_angle"]

                        # Direct azimuth comparison
                        diff = np.abs(ea - nea) % 360
                        direct = (diff <= angle_range) | (
                            diff >= 360 - angle_range
                        )

                        # Anti-podal azimuth comparison (+180 degrees)
                        anti_ea = (ea + 180) % 360
                        diff2 = np.abs(anti_ea - nea) % 360
                        anti = (diff2 <= angle_range) | (
                            diff2 >= 360 - angle_range
                        )

                        df_filt = df_filt[direct | anti]

                    if len(df_filt) == 0:
                        continue

                    base = {
                        "earthquake": earthquake_name,
                        "scheme": scheme,
                        "data_size": len(df_filt),
                        "angle_range": angle_range,
                        "total_dist": total_dist,
                        "nei_dist": nei_dist,
                    }

                    unique_ids = np.unique(
                        np.concatenate(
                            (df_filt["IN_FID"], df_filt["NEAR_FID"])
                        )
                    )

                    # ── Cross-validation folds ───────────────────────────
                    for fold in range(n_folds):
                        train_ids, test_ids = split_ids_for_fold(
                            unique_ids, site_to_fold, fold
                        )
                        if len(train_ids) < 5 or len(test_ids) == 0:
                            continue

                        train_df = df_filt[
                            np.isin(df_filt["IN_FID"], train_ids)
                            & np.isin(df_filt["NEAR_FID"], train_ids)
                        ]
                        if len(train_df) == 0:
                            continue

                        # ── Fit linear model on distance bins ────────────
                        jumps = range(1, 51)
                        bin_stats = _calculate_bin_stats(
                            train_df, jumps, nei_dist
                        )

                        best = {
                            "r_squared": -1,
                            "slope": 0,
                            "intercept": 0,
                            "p_value": 1,
                        }
                        filtered_bins = bin_stats[
                            bin_stats["count"] > MIN_NEIGHBORS
                        ]
                        for jump_val in filtered_bins["jump"].unique():
                            grp = filtered_bins[
                                filtered_bins["jump"] == jump_val
                            ]
                            if len(grp) > 1:
                                slope, intercept, r_val, p_val, _ = linregress(
                                    grp["distance_end"], grp["mean_int"]
                                )
                                r2 = r_val ** 2
                                if r2 > best["r_squared"] and p_val <= 0.10:
                                    best.update(
                                        slope=slope,
                                        intercept=intercept,
                                        r_squared=r2,
                                        p_value=p_val,
                                    )

                        # ── Predict on the withheld sites ────────────────
                        test_df = df_filt[
                            np.isin(df_filt["IN_FID"], test_ids)
                            & np.isin(df_filt["NEAR_FID"], train_ids)
                        ]
                        if len(test_df) == 0:
                            continue

                        test_df = np.copy(test_df)
                        abs_pred = (
                            best["slope"] * test_df["NEAR_DIST"]
                            + best["intercept"]
                        )
                        cond = test_df["near_epic_dist"] > test_df["epic_dist"]
                        int_pred = np.where(
                            cond,
                            test_df["near_int"] + abs_pred,
                            test_df["near_int"] - abs_pred,
                        )

                        unique_test_ids = np.unique(test_df["IN_FID"])

                        for pred_neighbors in pred_neighbors_list:
                            row = dict(base)
                            row["fold"] = fold
                            row["pred_neighbors"] = pred_neighbors

                            lr_preds, lr_true = [], []
                            for fid in unique_test_ids:
                                idx = test_df["IN_FID"] == fid
                                subset = test_df[idx]
                                order = np.argsort(subset["NEAR_DIST"])
                                topk = subset[order][:pred_neighbors]
                                if len(topk) == 0:
                                    continue
                                pred_vals = int_pred[
                                    np.where(idx)[0][order[:pred_neighbors]]
                                ]
                                lr_preds.append(float(np.mean(pred_vals)))
                                lr_true.append(float(topk[0]["int"]))

                            if lr_preds:
                                metrics = compute_error_metrics(
                                    np.array(lr_true), np.array(lr_preds)
                                )
                                results.append(
                                    {**row, "model": "linear", **metrics}
                                )

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Run linear-regression model.")
    parser.add_argument("--near_table", required=True, help="Path to near-table CSV.")
    parser.add_argument("--sites_csv", required=True,
                        help="Path to the site table used to build the CV folds.")
    parser.add_argument("--name", default="earthquake", help="Earthquake label.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Phase 2A — Linear Regression: {args.name}")
    print(f"{'='*60}\n")

    df = run_linear_regression(
        args.near_table, args.sites_csv, earthquake_name=args.name
    )
    out = args.out or os.path.join(RESULTS_DIR, f"{args.name}_linear_results.csv")
    df.to_csv(out, index=False)
    print(f"\n[DONE] Results saved: {out}  ({len(df)} rows)")


if __name__ == "__main__":
    _cli()
