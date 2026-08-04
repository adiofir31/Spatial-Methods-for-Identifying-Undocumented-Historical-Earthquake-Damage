"""
models/knn.py — Phase 2B: K-Nearest Neighbours models (filtered & unfiltered).

Standalone usage:
    python -m models.knn --near_table data/Dead_Sea_1927_near_table.csv \\
        --sites_csv data/Dead_sea_1927.csv --name Dead_Sea_1927
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

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
    CV_SCHEMES,
    K_FOLDS,
    RANDOM_STATE,
    FOLD_COORD_SPACE_NEAR_TABLE,
    SITE_ID_COL,
    RESULTS_DIR,
)

MIN_TEST_PRED = 5  # minimum predictions to report a result


# ─────────────────────────────────────────────────────────────────────────────
# Filtered KNN (KNN_d + KNN_k with angle + epicentral filtering)
# ─────────────────────────────────────────────────────────────────────────────

def run_knn_filtered(
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
    KNN_d and KNN_k with spatial filtering (epicentral-distance cap +
    azimuth window), evaluated under k-fold cross-validation.

    - **KNN_d** (radius-based): for each test site, average the intensities
      of all training neighbours within the *nei_dist* radius.
    - **KNN_k** (fixed-k): for each test site, average the intensities of
      the *K* nearest training neighbours (sorted by NEAR_DIST).

    Both variants are run for every fold of every scheme, so that each site
    is tested exactly once per scheme.
    """
    angle_ranges = angle_ranges or ANGLE_RANGES
    total_dist_list = total_dist_list or TOTAL_DIST_LIST
    nei_dist_list = nei_dist_list or NEI_DIST_LIST
    pred_neighbors_list = pred_neighbors_list or PRED_NEIGHBORS_LIST
    schemes = schemes or CV_SCHEMES

    df_pd = pd.read_csv(near_table_path)
    required = ["IN_FID", "NEAR_FID", "int", "near_int",
                "epic_dist", "NEAR_DIST", "epic_angle", "near_epic_angle"]
    if not all(c in df_pd.columns for c in required):
        raise ValueError(f"Missing columns: {set(required) - set(df_pd.columns)}")

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
                    mask = (df_pd["epic_dist"] < total_dist) & (
                        df_pd["NEAR_DIST"] < nei_dist
                    )
                    df_f = df_pd[mask].copy()
                    keep = [
                        is_angle_within_range(
                            r["epic_angle"], r["near_epic_angle"], angle_range
                        )
                        or is_angle_within_range(
                            (r["epic_angle"] + 180) % 360,
                            r["near_epic_angle"],
                            angle_range,
                        )
                        for _, r in df_f.iterrows()
                    ]
                    df_f = df_f[keep]
                    if len(df_f) == 0:
                        continue

                    unique_ids = np.unique(
                        np.concatenate(
                            (df_f["IN_FID"].values, df_f["NEAR_FID"].values)
                        )
                    )

                    base = {
                        "earthquake": earthquake_name,
                        "scheme": scheme,
                        "data_size": len(df_f),
                        "angle_range": angle_range,
                        "total_dist": total_dist,
                        "nei_dist": nei_dist,
                    }

                    for fold in range(n_folds):
                        train_ids, test_ids = split_ids_for_fold(
                            unique_ids, site_to_fold, fold
                        )
                        if len(train_ids) < 5 or len(test_ids) == 0:
                            continue

                        test_df = df_f[
                            df_f["IN_FID"].isin(test_ids)
                            & df_f["NEAR_FID"].isin(train_ids)
                        ]
                        if test_df.empty:
                            continue

                        base_fold = dict(base)
                        base_fold["fold"] = fold

                        # ── KNN_d (radius-based) ─────────────────────────
                        knn_d_preds, knn_d_true = [], []
                        for fid in test_df["IN_FID"].unique():
                            nb = test_df[test_df["IN_FID"] == fid]
                            nb_in = nb[nb["NEAR_DIST"] < nei_dist]
                            if len(nb_in) == 0:
                                continue
                            knn_d_preds.append(float(nb_in["near_int"].mean()))
                            knn_d_true.append(float(nb_in["int"].iloc[0]))

                        if len(knn_d_preds) >= MIN_TEST_PRED:
                            metrics = compute_error_metrics(
                                np.array(knn_d_true), np.array(knn_d_preds)
                            )
                            results.append({
                                **base_fold, "model": "KNN_d",
                                "count": len(knn_d_preds), **metrics,
                            })

                        # ── KNN_k (fixed-k) ──────────────────────────────
                        for K in pred_neighbors_list:
                            knn_k_preds, knn_k_true = [], []
                            for fid in test_df["IN_FID"].unique():
                                nb = test_df[test_df["IN_FID"] == fid]
                                if nb.empty:
                                    continue
                                nearest = nb.sort_values("NEAR_DIST").head(K)
                                knn_k_preds.append(
                                    float(nearest["near_int"].mean())
                                )
                                knn_k_true.append(
                                    float(nearest["int"].iloc[0])
                                )

                            if len(knn_k_preds) >= MIN_TEST_PRED:
                                metrics = compute_error_metrics(
                                    np.array(knn_k_true),
                                    np.array(knn_k_preds),
                                )
                                results.append({
                                    **base_fold, "model": "KNN_k",
                                    "pred_neighbors": K,
                                    "count": len(knn_k_preds),
                                    **metrics,
                                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Unfiltered KNN (no angle / epicentral-distance filtering)
# ─────────────────────────────────────────────────────────────────────────────

def run_knn_unfiltered(
    near_table_path: str,
    sites_csv_path: str,
    earthquake_name: str = "earthquake",
    *,
    nei_dist_list: list[int] | None = None,
    pred_neighbors_list: list[int] | None = None,
    schemes: list[str] | None = None,
    k_folds: int = K_FOLDS,
    lon_col: str = "POINT_X",
    lat_col: str = "POINT_Y",
    id_col: str = SITE_ID_COL,
) -> pd.DataFrame:
    """
    KNN without spatial filtering — KNN_d_Unfiltered (by radius) and
    KNN_k_Unfiltered (by K nearest), evaluated under the same k-fold
    cross-validation schemes as the filtered variants.

    Because the angular and epicentral-distance filters are what the
    parameter grid varies, removing them leaves only the neighbour radius
    and K to iterate over. The unfiltered variants therefore produce far
    fewer rows than the filtered ones. They are reported in the
    Supplementary Materials of the accompanying paper, not in the main
    comparison.
    """
    nei_dist_list = nei_dist_list or [100, 200, 300, 400, 500]
    pred_neighbors_list = pred_neighbors_list or PRED_NEIGHBORS_LIST
    schemes = schemes or CV_SCHEMES

    df = pd.read_csv(near_table_path)
    required = ["IN_FID", "NEAR_FID", "int", "near_int", "NEAR_DIST"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

    unique_ids = np.unique(
        np.concatenate((df["IN_FID"].values, df["NEAR_FID"].values))
    )

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

        for fold in range(n_folds):
            train_ids, test_ids = split_ids_for_fold(
                unique_ids, site_to_fold, fold
            )
            if len(train_ids) < 5 or len(test_ids) == 0:
                continue

            test_df = df[
                df["IN_FID"].isin(test_ids) & df["NEAR_FID"].isin(train_ids)
            ]
            if test_df.empty:
                continue

            # ── KNN_d_Unfiltered (by radius) ─────────────────────────────
            for nei_dist in nei_dist_list:
                preds, trues = [], []
                for fid in test_df["IN_FID"].unique():
                    nb = test_df[test_df["IN_FID"] == fid]
                    nb_in = nb[nb["NEAR_DIST"] < nei_dist]
                    if len(nb_in) == 0:
                        continue
                    preds.append(float(nb_in["near_int"].mean()))
                    trues.append(float(nb_in["int"].iloc[0]))

                if len(preds) >= MIN_TEST_PRED:
                    metrics = compute_error_metrics(
                        np.array(trues), np.array(preds)
                    )
                    results.append({
                        "earthquake": earthquake_name,
                        "scheme": scheme,
                        "fold": fold,
                        "model": "KNN_d_Unfiltered",
                        "nei_dist": nei_dist,
                        "K": None,
                        "count": len(preds),
                        **metrics,
                    })

            # ── KNN_k_Unfiltered (by K neighbours) ───────────────────────
            for K in pred_neighbors_list:
                preds, trues = [], []
                for fid in test_df["IN_FID"].unique():
                    nb = test_df[test_df["IN_FID"] == fid]
                    if len(nb) == 0:
                        continue
                    topk = nb.sort_values("NEAR_DIST").head(K)
                    preds.append(float(topk["near_int"].mean()))
                    trues.append(float(topk["int"].iloc[0]))

                if len(preds) >= MIN_TEST_PRED:
                    metrics = compute_error_metrics(
                        np.array(trues), np.array(preds)
                    )
                    results.append({
                        "earthquake": earthquake_name,
                        "scheme": scheme,
                        "fold": fold,
                        "model": "KNN_k_Unfiltered",
                        "nei_dist": None,
                        "K": K,
                        "count": len(preds),
                        **metrics,
                    })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Run KNN models.")
    parser.add_argument("--near_table", required=True)
    parser.add_argument("--name", default="earthquake")
    parser.add_argument("--mode", choices=["filtered", "unfiltered", "both"], default="both")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Phase 2B — KNN: {args.name}  (mode={args.mode})")
    print(f"{'='*60}\n")

    frames = []
    if args.mode in ("filtered", "both"):
        frames.append(run_knn_filtered(args.near_table, args.name))
    if args.mode in ("unfiltered", "both"):
        frames.append(run_knn_unfiltered(args.near_table, args.name))

    df = pd.concat(frames, ignore_index=True)
    out = args.out or os.path.join(RESULTS_DIR, f"{args.name}_knn_results.csv")
    df.to_csv(out, index=False)
    print(f"\n[DONE] KNN results saved: {out}  ({len(df)} rows)")


if __name__ == "__main__":
    _cli()
