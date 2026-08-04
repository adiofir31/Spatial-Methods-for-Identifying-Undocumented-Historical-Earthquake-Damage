# Spatial Methods for Identifying Undocumented Historical Earthquake Damage

Reproducible code accompanying the academic paper.

This pipeline processes earthquake damage report data (instrumental or historical sources) into spatial near tables, then runs three modelling approaches — **Linear Regression**, **K-Nearest Neighbours (KNN)**, and **Kriging** — to estimate seismic intensity at unreported sites.

All models are evaluated under **5-fold cross-validation** with two partitioning schemes, *random* and *spatial-block*.

---

## Requirements

| Component      | Version | Notes                                      |
| -------------- | ------- | ------------------------------------------ |
| **Python**     | 3.9 +   |                                            |
| **ArcGIS Pro** | 2.9 +   | Provides the `arcpy` module (Phase 1 only) |
| NumPy          | ≥ 1.22  |                                            |
| pandas         | ≥ 1.4   |                                            |
| scikit-learn   | ≥ 1.0   |                                            |
| SciPy          | ≥ 1.7   |                                            |
| PyKrige        | ≥ 1.7   | Phase 2C (Kriging) only                    |

> **Phase 1 (preprocessing)** requires a licensed ArcGIS Pro `arcpy` environment.
> **Phase 2 (modelling)** runs in any standard Python environment — no ArcGIS needed — as long as you supply a computed near table.

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
earthquake_project
├── config.py                   # Central configuration (event params, grids, CV settings)
├── preprocessing.py            # Phase 1 — arcpy-based spatial preprocessing
├── main.py                     # Orchestrator entry point
├── models/
│   ├── __init__.py
│   ├── utils.py                # Shared helpers (metrics, angle filtering, I/O)
│   ├── cross_validation.py     # Fold construction (random / spatial-block)
│   ├── linear_regression.py    # Phase 2A — Linear Regression
│   ├── knn.py                  # Phase 2B — KNN (filtered + unfiltered)
│   └── kriging.py              # Phase 2C — Ordinary & Universal Kriging
├── data/                       # Place input CSVs here
├── results/                    # Output CSVs are written here
├── requirements.txt
└── README.md
```

---

## Cross-Validation

Every model is evaluated under k-fold cross-validation (k = 5). Each site is held out for testing exactly once, and the reported metrics are averaged over folds **and** over the parameter grid — no single "best" configuration is selected.

Two partitioning schemes are used:

| Scheme | How folds are formed | What it tests |
| ------ | -------------------- | ------------- |
| `random` | Sites assigned to folds at random | Prediction at a site that remains surrounded by reported neighbours |
| `spatial` | Contiguous blocks via k-means clustering of site coordinates | Prediction when an entire region is unreported and neighbours lie outside the block |

Folds are always built **at site level**, never at pair level, so a site never appears in both the training and test set of the same fold. In the near-table models this means the test set is restricted to pairs whose target site is in the fold *and* whose neighbour is in the training set.

**Sparse events.** For the spatial scheme the effective number of folds is `max(2, min(k, n_sites // 8))`. Events with few reporting sites therefore yield fewer than five blocks, and their spatial-block results are based on a smaller number of tested sites than the corresponding random k-fold results. This is reported as a limitation in the paper.

### Reproducing the published results

The spatial blocks are formed by k-means on site coordinates, but the two model families do this in different coordinate spaces, and the published results reflect that:

- **Linear regression and KNN** cluster in a **local equidistant projection** (degrees converted to kilometres), set by `FOLD_COORD_SPACE_NEAR_TABLE = "projected"`.
- **Kriging** clusters the **raw coordinates** of its projected point table, set by `FOLD_COORD_SPACE_KRIGING = "degrees"`.

Both are valid k-means spatial blocking, but they do not produce identical block boundaries. Both constants are exposed in `config.py`; changing them will produce internally consistent folds across model families, but the resulting numbers will no longer match the published tables.

---

## Input Data Formats

The pipeline accepts any tabular dataset containing site coordinates and seismic intensity values. Column names are configured per earthquake in `config.py` via the `fields` mapping. Two built-in presets are provided:

**USGS DYFI reports (instrumental)** — standard USGS "Did You Feel It?" file:

```
City, State/Region, Country, Zip Code, MMI, Responses, Distance, Latitude, Longitude
```

**Historical** — manually compiled macroseismic catalogues:

```
SITE_NAME, POINT_X, POINT_Y, Damage
```

For datasets with different column names, define a custom `fields` mapping in `config.py` (see the commented template).

---

## Quick Start

### 1. Full pipeline (Phase 1 + Phase 2)

Requires ArcGIS Pro. Add your earthquake to `config.py` (or use an existing one), then:

```bash
python main.py --earthquake Nippes_2021
```

### 2. Phase 2 only (skip preprocessing)

If you already have a near table, place it in `data/` and run:

```bash
# All models
python main.py --earthquake Dead_Sea_1927 --skip_preprocessing

# Single model
python main.py --earthquake Dead_Sea_1927 --skip_preprocessing --model linear
python main.py --earthquake Dead_Sea_1927 --skip_preprocessing --model knn

# Single cross-validation scheme
python main.py --earthquake Dead_Sea_1927 --skip_preprocessing --scheme spatial
```

> **Note on Kriging:** Kriging requires projected XY coordinates (columns `X`, `Y`, `int`) generated by Phase 1 and saved to `results/`. It cannot run with `--skip_preprocessing` unless `<event>_Damage_locations.csv` already exists in `results/` with those columns.

### 3. Run individual models directly

Each model module has its own CLI. The near-table models additionally need the **site table**, which is used to build the fold map:

```bash
# Linear Regression
python -m models.linear_regression \
    --near_table data/Dead_Sea_1927_near_table.csv \
    --sites_csv  data/Dead_sea_1927.csv \
    --name Dead_Sea_1927

# KNN (filtered + unfiltered)
python -m models.knn \
    --near_table data/Dead_Sea_1927_near_table.csv \
    --sites_csv  data/Dead_sea_1927.csv \
    --name Dead_Sea_1927

# Kriging (needs projected XY — see note above)
python -m models.kriging \
    --csv results/Dead_Sea_1927_Damage_locations.csv \
    --name Dead_Sea_1927 --anisotropy_angle 2
```

---

## Adding a New Earthquake

1. Add an entry to `EARTHQUAKE_PARAMS` in `config.py` with the epicenter coordinates, anisotropy angle (for universal kriging), and a `fields` mapping.
2. Set the `fields` key to one of the built-in mappings, or define your own:
   - `DYFI_FIELDS` — instrumental DYFI data (`City`, `MMI`, `Latitude`, `Longitude`)
   - `HISTORICAL_FIELDS` — historical data (`SITE_NAME`, `Damage`, `POINT_X`, `POINT_Y`)
   - **Custom** — copy the template in `config.py` and set the column names to match your CSV
3. Set the `filters` key to control the spatial distance ranges used by Linear Regression and KNN:
   - `INSTRUMENTAL_FILTERS` — epicentral and neighbour distances 100–500 km (100 km steps)
   - `HISTORICAL_FILTERS` — epicentral and neighbour distances 50–300 km (50 km steps)
4. Place the input CSV in `data/`.
5. Run: `python main.py --earthquake <your_key>`.

---

## Methodology Summary

### Phase 1 — Preprocessing (`preprocessing.py`)

1. Load input CSV → ArcGIS point feature class (WGS 84), using the field mappings in `config.py`.
2. Create a persistent site-ID field (`MY_FID`) for later use.
3. Create the epicenter point feature class (WGS 84).
4. `arcpy.analysis.Near` — distance and azimuth from each site to the epicenter.
5. `arcpy.analysis.GenerateNearTable` — inter-site distance, angle, and rank.
6. Project to UTM and add XY coordinates (used by the kriging model).
7. Export the damage-location table; convert intensity to numeric (Roman → integer for DYFI data, pass-through for historical data).
8. Merge epicentral distance/angle attributes into the near table; compute `intensity_diff` and `abs_int_diff`.

### Phase 2A — Linear Regression (`models/linear_regression.py`)

For each cross-validation fold and each combination of spatial filters (azimuth window × epicentral distance × neighbour distance):

- Bin inter-site distances and compute the mean intensity difference per bin, using training sites only.
- Fit a linear model (distance → intensity difference) on the training bins.
- Predict test-site intensity by applying the fitted slope to each neighbour's intensity, then averaging over the K nearest neighbours.

The distance-bin statistics are computed with a vectorised routine
(`np.digitize` + stable `np.lexsort`), which is O(N log N) per bin width
instead of the O(N x n_sites x n_bins) of a naive triple loop. This is what
makes the largest datasets tractable. The routine returns values identical to
the naive implementation; see the docstring of `_calculate_bin_stats` for why
`pandas.groupby().mean()` is deliberately avoided.

> **Distance filter ranges** are configured per earthquake in `config.py`. Historical events use shorter ranges (50–300 km) due to sparser, geographically concentrated data; instrumental events use wider ranges (100–500 km) to match the broader spatial distribution of reports.

### Phase 2B — KNN (`models/knn.py`)

- **Filtered Fixed-k KNN**: within the spatial filter envelope (azimuth + epicentral distance), average the intensities of the K nearest training neighbours.
- **Filtered Radius-based KNN**: within the same envelope, average the intensities of all training neighbours within a distance radius.
- **Unfiltered variants**: the same two models without any angular or epicentral filtering.

The filters are what the parameter grid varies, so the unfiltered variants iterate only over the neighbour radius and K, and therefore produce far fewer rows. They are reported in the Supplementary Materials of the paper, not in the main comparison.

### Phase 2C — Kriging (`models/kriging.py`)

Operates on the processed damage-locations table (columns `X`, `Y`, `int`) rather than the near table.

For each fold, the variogram parameters (range, sill, nugget) are **fitted automatically** to the training data by PyKrige's least-squares procedure. The grid varies only the variogram family, the number of lags, the neighbourhood size (Ordinary Kriging) and the drift term (Universal Kriging). Fitted parameters are written to the output for inspection.

The **Gaussian** variogram was tested during development and excluded from the reported analysis: it produced unstable fits with extremely large MSE values across all parameter combinations. It can be re-enabled via `VARIOGRAM_MODELS` in `config.py`.

Universal Kriging has no neighbourhood-size parameter and is therefore fitted once per drift term, outside the `n_closest` loop.

---

## Output Format

All result CSVs contain at minimum:

| Column | Meaning |
| ------ | ------- |
| `earthquake` | Event label |
| `scheme` | `random` or `spatial` |
| `fold` | Fold index within the scheme |
| `model` | `linear`, `KNN_k`, `KNN_d`, `KNN_k_Unfiltered`, `KNN_d_Unfiltered`, or `kriging` |
| `mse` | Mean squared error for that run |
| `error_no_rounded_+-0.5` | Proportion of predictions further than 0.5 intensity units from the observed value |
| `error_no_rounded_+-1` | Proportion further than 1.0 intensity units |

Additional columns vary by model (`angle_range`, `total_dist`, `nei_dist`, `pred_neighbors`; `variogram_model`, `nlags`, `n_closest`, `drift_term`, `var_range`, `var_sill`, `var_nugget`).

**Success rate**, the metric reported in the paper, is `1 − error_no_rounded_+-X`.

**MSE.** Kriging occasionally extrapolates outside the envelope of the training data and returns implausible values, producing very large squared errors. These runs are retained rather than discarded, since removing them would bias the comparison in favour of kriging. The paper therefore reports the **median** MSE across runs rather than the mean; success rates are bounded and are reported as means.

---

## Citation

If you use this code, please cite the accompanying paper. Version-specific archives are available via Zenodo.

## License

MIT License — see `LICENSE`.
