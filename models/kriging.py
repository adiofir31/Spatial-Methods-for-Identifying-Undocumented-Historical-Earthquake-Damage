"""
models/kriging.py — Phase 2C: Ordinary & Universal Kriging.

NOTE: This model does NOT use the near table. It operates directly on a
dedicated point dataset with columns: X, Y, int (intensity).

Standalone usage:
    python -m models.kriging --csv data/M_6_9_Kamariótissa_2014.csv \\
           --name M_6.9_Kamariótissa_2014 --anisotropy_angle 250
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import product as iterproduct

import numpy as np
import pandas as pd
from pykrige import OrdinaryKriging, UniversalKriging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VARIOGRAM_MODELS,
    NLAGS_LIST,
    N_CLOSEST_POINTS_LIST,
    DRIFT_TERMS_LIST,
    TEST_SIZE,
    RANDOM_STATE,
    RESULTS_DIR,
)


def _error_metrics_simple(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Lighter metric set used in the kriging comparison."""
    n = len(y_pred)
    if n == 0:
        return {}
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "error_no_rounded_+-0.5": float(np.sum(np.abs(y_true - y_pred) > 0.5)) / n,
        "error_no_rounded_+-1":   float(np.sum(np.abs(y_true - y_pred) > 1.0)) / n,
    }


def run_kriging(
    csv_path: str,
    earthquake_name: str = "earthquake",
    anisotropy_angle: float = 0,
    *,
    variogram_models: list[str] | None = None,
    nlags_list: list[int] | None = None,
    n_closest_list: list[int | None] | None = None,
    drift_terms_list: list[str] | None = None,
) -> pd.DataFrame:
    """
    Grid-search over Ordinary and Universal Kriging hyper-parameters.

    Parameters
    ----------
    csv_path : str
        CSV with columns X, Y, int (projected coordinates).
    anisotropy_angle : float
        Strike azimuth for universal kriging (degrees).

    Returns
    -------
    pd.DataFrame with one row per (method × variogram × nlags × n_closest × drift).
    """
    variogram_models = variogram_models or VARIOGRAM_MODELS
    nlags_list = nlags_list or NLAGS_LIST
    n_closest_list = n_closest_list or N_CLOSEST_POINTS_LIST
    drift_terms_list = drift_terms_list or DRIFT_TERMS_LIST

    df = pd.read_csv(csv_path)
    for col in ("X", "Y", "int"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results: list[dict] = []
    total = (
        len(variogram_models) * len(nlags_list) * len(n_closest_list)
        * (1 + len(drift_terms_list))
    )
    counter = 0

    # Suppress ill-conditioned matrix warnings from PyKrige grid search.
    # Many hyperparameter combinations intentionally produce poor fits;
    # these are filtered by MSE in the results, not by solver warnings.
    warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

    for var_model, nlags, n_closest in iterproduct(
        variogram_models, nlags_list, n_closest_list
    ):
        # ── Ordinary Kriging ─────────────────────────────────────────────
        counter += 1
        if counter % 20 == 0:
            print(f"  [{counter}/{total}]  OK  {var_model}  nlags={nlags}  n={n_closest}")
        try:
            ok = OrdinaryKriging(
                train_df["X"], train_df["Y"], train_df["int"],
                variogram_model=var_model, nlags=nlags,
            )
            z_list, ss_list = [], []
            for x, y in zip(test_df["X"], test_df["Y"]):
                kw = {"backend": "loop"}
                if n_closest is not None:
                    kw["n_closest_points"] = n_closest
                z, ss = ok.execute("points", x, y, **kw)
                z_list.append(z[0])
                ss_list.append(ss[0] if ss[0] >= 0 else np.nan)

            y_true = test_df["int"].values
            y_pred = np.array(z_list)
            row = {
                "earthquake": earthquake_name,
                "method": "ordinary",
                "variogram_model": var_model,
                "nlags": nlags,
                "n_closest": n_closest,
                "mean_ss": np.nanmean(ss_list),
                "model": "kriging",
            }
            params = ok.variogram_model_parameters
            if params is not None:
                row["var_range"], row["var_sill"], row["var_nugget"] = params
            row.update(_error_metrics_simple(y_true, y_pred))
            results.append(row)

        except Exception as e:
            print(f"  [WARNING] OrdinaryKriging error ({var_model}, nlags={nlags}): {e}")

        # ── Universal Kriging ────────────────────────────────────────────
        for drift in drift_terms_list:
            counter += 1
            try:
                uk = UniversalKriging(
                    train_df["X"], train_df["Y"], train_df["int"],
                    variogram_model=var_model,
                    nlags=nlags,
                    anisotropy_angle=anisotropy_angle % 180,
                    drift_terms=drift,
                )
                z_list, ss_list = [], []
                for x, y in zip(test_df["X"], test_df["Y"]):
                    z, ss = uk.execute("points", x, y, backend="loop")
                    z_list.append(z[0])
                    ss_list.append(ss[0] if ss[0] >= 0 else np.nan)

                y_true = test_df["int"].values
                y_pred = np.array(z_list)
                row = {
                    "earthquake": earthquake_name,
                    "method": "universal",
                    "variogram_model": var_model,
                    "nlags": nlags,
                    "n_closest": n_closest,
                    "anisotropy_angle": anisotropy_angle,
                    "drift_term": drift,
                    "mean_ss": np.nanmean(ss_list),
                    "model": "kriging",
                }
                params = uk.variogram_model_parameters
                if params is not None:
                    row["var_range"], row["var_sill"], row["var_nugget"] = params
                row.update(_error_metrics_simple(y_true, y_pred))
                results.append(row)

            except Exception as e:
                print(f"  [WARNING] UniversalKriging error ({var_model}, {drift}): {e}")

    # Restore default warning behavior
    warnings.resetwarnings()
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Run Kriging models.")
    parser.add_argument("--csv", required=True, help="CSV with X, Y, int columns.")
    parser.add_argument("--name", default="earthquake")
    parser.add_argument("--anisotropy_angle", type=float, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Phase 2C — Kriging: {args.name}")
    print(f"{'='*60}\n")

    df = run_kriging(args.csv, args.name, args.anisotropy_angle)
    out = args.out or os.path.join(RESULTS_DIR, f"{args.name}_kriging_results.csv")
    df.to_csv(out, index=False)
    print(f"\n[DONE] Kriging results saved: {out}  ({len(df)} rows)")


if __name__ == "__main__":
    _cli()
