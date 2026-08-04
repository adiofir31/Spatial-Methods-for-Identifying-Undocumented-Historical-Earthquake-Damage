"""
config.py — Central configuration for the earthquake spatial-analysis pipeline.

Edit the EARTHQUAKE_PARAMS dictionary to define new earthquake events.
The pipeline supports two common input formats:

  - Instrumental (DYFI):  City / MMI (Roman) / Latitude / Longitude
  - Historical:           SITE_NAME / Damage (numeric) / POINT_X / POINT_Y

Each event entry includes a 'fields' sub-dictionary that maps the generic
column roles (lon, lat, name, intensity) to the actual column names in the
CSV, along with the intensity format ("roman" or "numeric").
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Default field mappings for the two common input formats
# ─────────────────────────────────────────────────────────────────────────────

DYFI_FIELDS = {
    "lon": "Longitude",
    "lat": "Latitude",
    "name": "City",
    "intensity": "MMI",
    "intensity_format": "roman",       # Roman numerals — converted automatically
}

HISTORICAL_FIELDS = {
    "lon": "POINT_X",
    "lat": "POINT_Y",
    "name": "SITE_NAME",
    "intensity": "Damage",
    "intensity_format": "numeric",     # already numeric — used as-is
}

# Template for datasets with non-standard column names.
# Copy, rename, and edit the values to match your CSV headers.
# Example:
#
#   MY_SURVEY_FIELDS = {
#       "lon": "x_coord",
#       "lat": "y_coord",
#       "name": "locality",
#       "intensity": "EMS98",
#       "intensity_format": "numeric",   # "roman" or "numeric"
#   }
#
# Then reference it in your earthquake entry:  "fields": MY_SURVEY_FIELDS

# ─────────────────────────────────────────────────────────────────────────────
# Spatial filter presets (distance ranges differ by data density)
# ─────────────────────────────────────────────────────────────────────────────

INSTRUMENTAL_FILTERS = {
    "total_dist_list": list(range(100, 501, 100)),   # 100, 200, ... 500 km
    "nei_dist_list":   list(range(100, 501, 100)),   # 100, 200, ... 500 km
}

HISTORICAL_FILTERS = {
    "total_dist_list": list(range(50, 301, 50)),     # 50, 100, ... 300 km
    "nei_dist_list":   list(range(50, 301, 50)),     # 50, 100, ... 300 km
}

# ─────────────────────────────────────────────────────────────────────────────
# Earthquake event definitions
# ─────────────────────────────────────────────────────────────────────────────
# Required keys per event:
#   epicenter_lon, epicenter_lat : WGS 84 coordinates of the epicenter
#   input_csv       : filename inside DATA_DIR for the raw intensity table
#   near_table_csv  : output filename for the enriched near table
#   anisotropy_angle: fault-strike azimuth for universal kriging (degrees)
#   fields          : column-name mapping (use DYFI_FIELDS or HISTORICAL_FIELDS)
#   filters         : distance filter ranges (use INSTRUMENTAL_FILTERS or HISTORICAL_FILTERS)

EARTHQUAKE_PARAMS = {

    # ── Historical earthquakes ───────────────────────────────────────────

    "Dead_Sea_1927": {
        "epicenter_lon": 35.5668,
        "epicenter_lat": 31.9027,
        "input_csv": "Dead_sea_1927.csv",
        "near_table_csv": "Dead_Sea_1927_near_table.csv",
        "anisotropy_angle": 2,
        "fields": HISTORICAL_FIELDS,
        "filters": HISTORICAL_FILTERS,
    },
    "South_Lebanon_1837": {
        "epicenter_lon": 35.5394,
        "epicenter_lat": 33.3449,
        "input_csv": "south_lebanon_1837.csv",
        "near_table_csv": "South_Lebanon_1837_near_table.csv",
        "anisotropy_angle": 340,
        "fields": HISTORICAL_FIELDS,
        "filters": HISTORICAL_FILTERS,
    },

    # ── Instrumental (DYFI) earthquakes ──────────────────────────────────

    "Nippes_2021": {
        "epicenter_lon": -73.482,
        "epicenter_lat": 18.434,
        "input_csv": "Nippes_2021.csv",
        "near_table_csv": "Nippes_2021_near_table.csv",
        "anisotropy_angle": 264,
        "fields": DYFI_FIELDS,
        "filters": INSTRUMENTAL_FILTERS,
    },
    "Lucea_2020": {
        "epicenter_lon": -78.756,
        "epicenter_lat": 19.419,
        "input_csv": "lucea_2020.csv",
        "near_table_csv": "Lucea_2020_near_table.csv",
        "anisotropy_angle": 262,
        "fields": DYFI_FIELDS,
        "filters": INSTRUMENTAL_FILTERS,
    },
    "Ridgecrest_2019": {
        "epicenter_lon": -117.599,
        "epicenter_lat": 35.770,
        "input_csv": "ridgecrest_2019.csv",
        "near_table_csv": "Ridgecrest_2019_near_table.csv",
        "anisotropy_angle": 325,
        "fields": DYFI_FIELDS,
        "filters": INSTRUMENTAL_FILTERS,
    },
    "South_Napa_2014": {
        "epicenter_lon": -122.312,
        "epicenter_lat": 38.215,
        "input_csv": "south_napa_2014.csv",
        "near_table_csv": "South_Napa_2014_near_table.csv",
        "anisotropy_angle": 333,
        "fields": DYFI_FIELDS,
        "filters": INSTRUMENTAL_FILTERS,
    },
    "Kamariotissa_2014": {
        "epicenter_lon": 25.389,
        "epicenter_lat": 40.289,
        "input_csv": "Kamariotissa_2014.csv",
        "near_table_csv": "Kamariotissa_2014_near_table.csv",
        "anisotropy_angle": 250,
        "fields": DYFI_FIELDS,
        "filters": INSTRUMENTAL_FILTERS,
    },
    "Duzce_2022": {
        "epicenter_lon": 30.983,
        "epicenter_lat": 40.836,
        "input_csv": "duzce_2022.csv",
        "near_table_csv": "Duzce_2022_near_table.csv",
        "anisotropy_angle": 265,
        "fields": DYFI_FIELDS,
        "filters": INSTRUMENTAL_FILTERS,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Roman-numeral look-up table (MMI values in DYFI reports)
# ─────────────────────────────────────────────────────────────────────────────
ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12,
}

# ─────────────────────────────────────────────────────────────────────────────
# Modeling hyper-parameter grids
# ─────────────────────────────────────────────────────────────────────────────
ANGLE_RANGES = list(range(15, 91, 15))           # 15, 30, ... 90 (90 = no filter)
TOTAL_DIST_LIST = list(range(100, 501, 100))     # epicentral distance caps (km)
NEI_DIST_LIST = list(range(100, 501, 100))       # neighbour-distance caps (km)
PRED_NEIGHBORS_LIST = list(range(5, 51, 5))      # k values for KNN_k
MIN_NEIGHBORS = 5                                # minimum bin count for regression

# Kriging grid-search parameters
# The Gaussian variogram was tested during development but excluded from the
# reported analysis: it produced unstable fits, with extremely large MSE values
# and large standard deviations across all parameter combinations.
VARIOGRAM_MODELS = ["exponential", "spherical"]
NLAGS_LIST = list(range(5, 50, 5))
N_CLOSEST_POINTS_LIST = [5, 15, 25, 50, None]   # None = use all training points
DRIFT_TERMS_LIST = ["regional_linear", "linear", "quadratic"]

# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────
# Every model is evaluated under k-fold cross-validation, in which each site is
# held out for testing exactly once and the reported metrics are averaged over
# folds and over the parameter grid. Two partitioning schemes are used:
#
#   random   — sites assigned to folds at random; each withheld site stays
#              embedded within the reported field.
#   spatial  — folds are contiguous spatial blocks (k-means on site
#              coordinates); entire regions are withheld together.
#
# See models/cross_validation.py and the README for details.
CV_SCHEMES = ["random", "spatial"]
K_FOLDS = 5
RANDOM_STATE = 42

# Coordinate space used to form the spatial blocks, per model family.
# These values reproduce the results reported in the paper — see the README
# section "Reproducing the published results" before changing them.
FOLD_COORD_SPACE_NEAR_TABLE = "projected"   # linear regression and KNN
FOLD_COORD_SPACE_KRIGING = "degrees"        # kriging

# Site identifier column in the site tables, used to build the fold map.
SITE_ID_COL = "OID_"
