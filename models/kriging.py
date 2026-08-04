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
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans

from config import (
    VARIOGRAM_MODELS,
    NLAGS_LIST,
    N_CLOSEST_POINTS_LIST,
    DRIFT_TERMS_LIST,
    CV_SCHEMES,
    K_FOLDS,
    RANDOM_STATE,
    FOLD_COORD_SPACE_KRIGING,
    RESULTS_DIR,
)


def _assign_kriging_folds(
    df: pd.DataFrame, scheme: str, k: int, random_state: int
) -> np.ndarray:
    """
    Assign fold identifiers to the rows of the kriging point table.

    Unlike the near-table models, kriging works directly on the projected
    point table (columns X, Y, int) rather than on a separate site table,
    so the folds are built here from those coordinates. With
    ``FOLD_COORD_SPACE_KRIGING = "degrees"`` the k-means blocks are formed
    in the raw coordinate space of the table; see the README section
    "Reproducing the published results".
    """
    if scheme == "random":
        return np.random.default_rng(random_state).integers(0, k, len(df))

    coords = df[["X", "Y"]].values
    if FOLD_COORD_SPACE_KRIGING == "projected":
        x = coords[:, 0] - coords[:, 0].mean()
        y = coords[:, 1] - coords[:, 1].mean()
        coords = np.c_[x, y]

    k_eff = max(2, min(k, len(df) // 8))
    return KMeans(
        n_clusters=k_eff, n_init=10, random_state=random_state
    ).fit_predict(coords)


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
    schemes: list[str] | None = None,
    k_folds: int = K_FOLDS,
) -> pd.DataFrame:
    """
    Grid-search over Ordinary and Universal Kriging hyper-parameters, under
    k-fold cross-validation with both partitioning schemes.

    The variogram parameters (range, sill, nugget) are fitted automatically
    to the training data of each fold by PyKrige's least-squares procedure;
    only the variogram family, the number of lags, the neighbourhood size
    and the drift term are varied across the grid. The fitted parameters
    are written to the output for inspection.

    Parameters
    ----------
    csv_path : str
        CSV with columns X, Y, int (projected coordinates).
    anisotropy_angle : float
        Strike azimuth for universal kriging (degrees).
    schemes : list of str, optional
        Cross-validation schemes to run. Defaults to ``CV_SCHEMES``.

    Returns
    -------
    pd.DataFrame
        One row per (scheme, fold, method, variogram, nlags, n_closest,
        drift) combination.

    Notes
    -----
    Ordinary Kriging is evaluated once per neighbourhood size
    (``n_closest``); Universal Kriging does not take a neighbourhood size
    and is therefore evaluated once per drift term, outside that loop.
    Occasional configurations extrapolate outside the envelope of the
    training data and return implausible values, producing very large
    squared errors. These rows are retained rather than discarded; the
    accompanying paper reports the median MSE across runs for this reason.
    """
    variogram_models = variogram_models or VARIOGRAM_MODELS
    nlags_list = nlags_list or NLAGS_LIST
    n_closest_list = n_closest_list or N_CLOSEST_POINTS_LIST
    drift_terms_list = drift_terms_list or DRIFT_TERMS_LIST
    schemes = schemes or CV_SCHEMES

    df = pd.read_csv(csv_path)
    for col in ("X", "Y", "int"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    results: list[dict] = []

    # Suppress ill-conditioned matrix warnings from PyKrige grid search.
    # Many hyperparameter combinations intentionally produce poor fits;
    # these are filtered by MSE in the results, not by solver warnings.
    warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

    for scheme in schemes:
        folds = _assign_kriging_folds(df, scheme, k_folds, RANDOM_STATE)
        n_folds = len(set(folds))
        print(f"  scheme={scheme}: {n_folds} populated fold(s)")

        for fold in sorted(set(folds)):
            train_df = df[folds != fold]
            test_df = df[folds == fold]
            if len(train_df) < 5 or len(test_df) == 0:
                continue

            xt = test_df["X"].values
            yt = test_df["Y"].values
            y_true = test_df["int"].values

            for var_model, nlags in iterproduct(variogram_models, nlags_list):

                # ── Ordinary Kriging ─────────────────────────────────────
                for n_closest in n_closest_list:
                    try:
                        ok = OrdinaryKriging(
                            train_df["X"], train_df["Y"], train_df["int"],
                            variogram_model=var_model, nlags=nlags,
                        )
                        if n_closest is not None:
                            z, ss = ok.execute(
                                "points", xt, yt,
                                backend="loop", n_closest_points=n_closest,
                            )
                        else:
                            z, ss = ok.execute(
                                "points", xt, yt, backend="vectorized"
                            )
                        z = np.asarray(z)
                        ss = np.asarray(ss)

                        row = {
                            "earthquake": earthquake_name,
                            "scheme": scheme,
                            "fold": fold,
                            "method": "ordinary",
                            "variogram_model": var_model,
                            "nlags": nlags,
                            "n_closest": n_closest,
                            "mean_ss": float(
                                np.nanmean(np.where(ss >= 0, ss, np.nan))
                            ),
                            "model": "kriging",
                        }
                        params = ok.variogram_model_parameters
                        if params is not None:
                            (row["var_range"], row["var_sill"],
                             row["var_nugget"]) = params
                        row.update(_error_metrics_simple(y_true, z))
                        results.append(row)

                    except Exception as e:
                        print(
                            f"  [WARNING] OrdinaryKriging error "
                            f"({var_model}, nlags={nlags}): {e}"
                        )

                # ── Universal Kriging ────────────────────────────────────
                # UK has no neighbourhood-size parameter, so it is fitted
                # once per drift term rather than once per n_closest value.
                for drift in drift_terms_list:
                    try:
                        uk = UniversalKriging(
                            train_df["X"], train_df["Y"], train_df["int"],
                            variogram_model=var_model,
                            nlags=nlags,
                            anisotropy_angle=anisotropy_angle % 180,
                            drift_terms=drift,
                        )
                        z, ss = uk.execute(
                            "points", xt, yt, backend="vectorized"
                        )
                        z = np.asarray(z)
                        ss = np.asarray(ss)

                        row = {
                            "earthquake": earthquake_name,
                            "scheme": scheme,
                            "fold": fold,
                            "method": "universal",
                            "variogram_model": var_model,
                            "nlags": nlags,
                            "anisotropy_angle": anisotropy_angle,
                            "drift_term": drift,
                            "mean_ss": float(
                                np.nanmean(np.where(ss >= 0, ss, np.nan))
                            ),
                            "model": "kriging",
                        }
                        params = uk.variogram_model_parameters
                        if params is not None:
                            (row["var_range"], row["var_sill"],
                             row["var_nugget"]) = params
                        row.update(_error_metrics_simple(y_true, z))
                        results.append(row)

                    except Exception as e:
                        print(
                            f"  [WARNING] UniversalKriging error "
                            f"({var_model}, {drift}): {e}"
                        )

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
