"""
models/utils.py — Shared helpers for all spatial models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def custom_round(values: np.ndarray) -> np.ndarray:
    """Round to nearest 0.5 increment."""
    return np.round(values * 2) / 2


def is_angle_within_range(
    epic_angle: float, near_epic_angle: float, angle_range: float
) -> bool:
    """
    Check whether two epicentral azimuths lie within *angle_range* of each
    other (accounting for the 360° wrap-around).
    """
    diff = abs(epic_angle - near_epic_angle) % 360
    return diff <= angle_range or diff >= (360 - angle_range)


def angle_filter_mask(df: pd.DataFrame, angle_range: float) -> pd.Series:
    """
    Vectorised boolean mask: True for rows whose IN / NEAR epicentral
    azimuths are within *angle_range* (considering both the direct and
    anti-podal directions, as in the original methodology).
    """
    ea = df["epic_angle"].values
    nea = df["near_epic_angle"].values
    diff = np.abs(ea - nea) % 360
    direct = (diff <= angle_range) | (diff >= 360 - angle_range)
    anti_ea = (ea + 180) % 360
    diff2 = np.abs(anti_ea - nea) % 360
    anti = (diff2 <= angle_range) | (diff2 >= 360 - angle_range)
    return pd.Series(direct | anti, index=df.index)


def compute_error_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Return the full set of error metrics used in the paper.
    """
    n = len(y_pred)
    if n == 0:
        return {}

    cr = custom_round(y_pred)
    rr = np.round(y_pred)

    return {
        "mse": mean_squared_error(y_true, y_pred),
        "error_custom_rounded_+-0.5": float(np.sum(np.abs(y_true - cr) > 0.5)) / n,
        "error_custom_rounded_+-1":   float(np.sum(np.abs(y_true - cr) > 1.0)) / n,
        "error_reg_rounded_+-0.5":    float(np.sum(np.abs(y_true - rr) > 0.5)) / n,
        "error_reg_rounded_+-1":      float(np.sum(np.abs(y_true - rr) > 1.0)) / n,
        "error_no_rounded_+-0.5":     float(np.sum(np.abs(y_true - y_pred) > 0.5)) / n,
        "error_no_rounded_+-1":       float(np.sum(np.abs(y_true - y_pred) > 1.0)) / n,
    }


def site_level_train_test_split(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split unique site IDs (union of IN_FID and NEAR_FID) so that the same
    location never appears in both train and test — prevents data leakage.

    Returns (train_ids, test_ids).
    """
    from sklearn.model_selection import train_test_split

    unique_ids = np.unique(
        np.concatenate((df["IN_FID"].values, df["NEAR_FID"].values))
    )
    return train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )


def load_near_table(path: str) -> np.ndarray:
    """
    Load a near-table CSV into a NumPy structured array with the canonical
    dtype used by the modelling scripts.
    """
    dtype = np.dtype([
        ("IN_FID", "i4"), ("name", "U50"), ("int", "f4"),
        ("epic_dist", "f8"), ("epic_angle", "f4"), ("NEAR_FID", "i4"),
        ("near_name", "U50"), ("near_int", "f4"), ("near_epic_dist", "f8"),
        ("near_epic_angle", "f4"), ("NEAR_DIST", "f8"), ("NEAR_RANK", "i4"),
        ("NEAR_ANGLE", "f4"), ("intensity_diff", "i4"), ("abs_int_diff", "f4"),
    ])
    return np.genfromtxt(
        path, delimiter=",", names=True, dtype=dtype, encoding="utf-8-sig"
    )
