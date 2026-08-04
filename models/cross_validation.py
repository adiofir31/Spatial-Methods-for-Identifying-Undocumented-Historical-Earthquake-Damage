"""
models/cross_validation.py — Fold construction for k-fold cross-validation.

Two partitioning schemes are used throughout the study:

  * ``random``  — sites are assigned to folds at random. Each withheld site
                  remains embedded within the reported field, so the task is
                  prediction at a location surrounded by reported neighbours.

  * ``spatial`` — folds are contiguous spatial blocks obtained by k-means
                  clustering of the site coordinates, so that entire regions
                  are withheld together and the withheld sites must be
                  predicted from reports lying outside the block.

Folds are built once per (event, scheme) at site level, never at pair level,
so that a site never appears in both the training and the test set of the
same fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def build_site_folds(
    sites_csv_path: str,
    lon_col: str,
    lat_col: str,
    id_col: str | None = None,
    *,
    scheme: str = "spatial",
    k: int = 5,
    random_state: int = 42,
    coord_space: str = "projected",
) -> dict:
    """
    Assign every reporting site to a fold.

    Parameters
    ----------
    sites_csv_path : str
        Path to the site table (one row per reporting location) containing
        the longitude, latitude and identifier columns.
    lon_col, lat_col : str
        Column names holding the coordinates.
    id_col : str or None
        Column holding the persistent site identifier used in the near
        table (``OID_``). If None, a 1-based row index is used instead.
    scheme : {"random", "spatial"}
        Partitioning scheme (see module docstring).
    k : int
        Requested number of folds. For the spatial scheme the effective
        number is reduced when the event has too few sites to form ``k``
        populated blocks (see Notes).
    random_state : int
        Seed for both the random assignment and the k-means initialisation.
    coord_space : {"projected", "degrees"}
        Coordinate space in which the spatial blocks are formed.
        ``projected`` converts longitude/latitude to a local
        equidistant frame in kilometres before clustering, so that blocks
        are geometrically compact on the ground. ``degrees`` clusters the
        raw coordinates directly.

        These two options do not produce identical blocks: at mid
        latitudes a degree of longitude is shorter on the ground than a
        degree of latitude, so clustering in degree space stretches the
        north–south axis. The published results use ``projected`` for the
        linear and KNN models and ``degrees`` for kriging; see the README
        section "Reproducing the published results".

    Returns
    -------
    dict
        Mapping ``{site_id: fold_id}``.

    Notes
    -----
    For the spatial scheme the effective number of folds is
    ``max(2, min(k, n_sites // 8))``. Sparsely reported events therefore
    yield fewer than ``k`` blocks, and their spatial-block results are
    based on a smaller number of tested sites than the corresponding
    random k-fold results. This behaviour is intentional and is reported
    as a limitation in the accompanying paper.
    """
    if scheme not in ("random", "spatial"):
        raise ValueError(
            f"Unknown scheme {scheme!r}; expected 'random' or 'spatial'."
        )
    if coord_space not in ("projected", "degrees"):
        raise ValueError(
            f"Unknown coord_space {coord_space!r}; "
            "expected 'projected' or 'degrees'."
        )

    d = pd.read_csv(sites_csv_path, encoding="utf-8-sig")
    d["sid"] = d[id_col] if id_col else np.arange(1, len(d) + 1)
    d = d.rename(columns={lon_col: "lon", lat_col: "lat"}).dropna(
        subset=["lon", "lat"]
    )

    if scheme == "random":
        rng = np.random.default_rng(random_state)
        folds = rng.integers(0, k, size=len(d))
        return dict(zip(d["sid"].values, folds))

    # ── spatial: contiguous blocks via k-means ───────────────────────────
    if coord_space == "projected":
        lon0, lat0 = d["lon"].mean(), d["lat"].mean()
        x = (d["lon"] - lon0) * 111.32 * np.cos(np.radians(lat0))
        y = (d["lat"] - lat0) * 110.57
        coords = np.c_[x, y]
    else:
        coords = d[["lon", "lat"]].values

    k_eff = max(2, min(k, len(d) // 8))
    folds = KMeans(
        n_clusters=k_eff, n_init=10, random_state=random_state
    ).fit_predict(coords)

    return dict(zip(d["sid"].values, folds))


def split_ids_for_fold(
    unique_ids: np.ndarray, site_to_fold: dict, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the site identifiers present in a (filtered) near table into the
    training and test sets of a given fold.

    Returns ``(train_ids, test_ids)``. Sites absent from ``site_to_fold``
    are treated as training sites.
    """
    test_ids = np.array(
        [s for s in unique_ids if site_to_fold.get(s) == fold], dtype=int
    )
    train_ids = np.array(
        [s for s in unique_ids if site_to_fold.get(s) != fold], dtype=int
    )
    return train_ids, test_ids


def n_folds_of(site_to_fold: dict) -> int:
    """Number of populated folds in a fold map."""
    return len(set(site_to_fold.values()))
