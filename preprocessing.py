"""
preprocessing.py — Phase 1: Data Preprocessing (requires arcpy / ArcGIS Pro).

Supports both DYFI (instrumental) and historical intensity data formats.
Column names are read from the 'fields' mapping in each earthquake's config
entry, so the same pipeline handles any tabular intensity dataset with
location and intensity columns.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

def get_utm_epsg(lon: float, lat: float) -> int:
    """Calculates the UTM EPSG code based on longitude and latitude."""
    zone = int((lon + 180) / 6) + 1
    # EPSG: 32601-32660 for Northern Hemisphere, 32701-32760 for Southern
    base = 32600 if lat >= 0 else 32700
    return base + zone

def roman_to_arabic(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping)

def normalize_angle(angle: float) -> float:
    return angle + 360 if angle < 0 else angle

def run_preprocessing(earthquake_key: str) -> str:
    import arcpy
    from config import DATA_DIR, EARTHQUAKE_PARAMS, RESULTS_DIR, ROMAN_TO_INT

    params = EARTHQUAKE_PARAMS[earthquake_key]

    # ── Paths ────────────────────────────────────────────────────────────
    input_csv = os.path.join(DATA_DIR, params["input_csv"])
    # ArcGIS feature class names cannot contain dots or special characters
    safe_name = earthquake_key.replace(".", "_")
    workspace = os.path.join(RESULTS_DIR, f"{safe_name}.gdb")
    fields = params["fields"]

    arcpy.env.overwriteOutput = True

    if not arcpy.Exists(workspace):
        arcpy.management.CreateFileGDB(RESULTS_DIR, f"{safe_name}.gdb")

    arcpy.env.workspace = workspace
    input_crs = arcpy.SpatialReference(4326) # WGS 84

    # ── 1. CSV → Point Feature Class (WGS 84) ───────────────────────────
    xy_fc = os.path.join(workspace, f"{safe_name}_XYToPoint")
    arcpy.management.XYTableToPoint(
        in_table=input_csv,
        out_feature_class=xy_fc,
        x_field=fields["lon"],
        y_field=fields["lat"],
        coordinate_system=input_crs,
    )

    # Create a persistent ID field (OBJECTID is dropped on CSV export)
    arcpy.management.CalculateField(
        in_table=xy_fc,
        field="MY_FID",
        expression="!OBJECTID!",
        expression_type="PYTHON3",
        field_type="LONG"
    )
    print(f"  [OK] Created point FC with explicit MY_FID: {xy_fc}")

    # ── 2. Create Epicenter (WGS 84) ────────────────────────────────────
    epic_fc = os.path.join(workspace, f"{safe_name}_Epicenter")
    arcpy.management.CreateFeatureclass(
        out_path=workspace,
        out_name=f"{safe_name}_Epicenter",
        geometry_type="POINT",
        spatial_reference=input_crs,
    )
    with arcpy.da.InsertCursor(epic_fc, ["SHAPE@"]) as cur:
        pt = arcpy.Point(params["epicenter_lon"], params["epicenter_lat"])
        cur.insertRow([pt])
    print(f"  [OK] Epicenter created: {epic_fc}")

    # ── 3. Near analysis — each damage point → epicenter (GEODESIC) ──────
    arcpy.analysis.Near(
        in_features=xy_fc,
        near_features=epic_fc,
        search_radius=None,
        location="NO_LOCATION",
        angle="ANGLE",
        method="GEODESIC",  # accurate on WGS 84 without projection
        field_names="NEAR_FID NEAR_FID;NEAR_DIST NEAR_DIST;NEAR_ANGLE NEAR_ANGLE",
        distance_unit="Kilometers",
    )
    print("  [OK] Near (to epicenter) completed [Geodesic]")

    # ── 4. Generate all-pairs near table (GEODESIC) ──────────────────────
    raw_near_csv = os.path.join(RESULTS_DIR, f"{earthquake_key}_Near_Table_raw.csv")

    arcpy.analysis.GenerateNearTable(
        in_features=xy_fc,
        near_features=xy_fc,
        out_table=raw_near_csv,
        search_radius=None,
        location="NO_LOCATION",
        angle="ANGLE",
        closest="ALL",
        closest_count=0,
        method="GEODESIC",
        distance_unit="Kilometers",
    )
    print(f"  [OK] All-pairs near table: {raw_near_csv} [Geodesic]")

    # ── 5. Project to UTM and add XY coordinates (for Kriging input) ─────
    # Automatically determine the appropriate UTM zone from the epicenter
    utm_epsg = get_utm_epsg(params["epicenter_lon"], params["epicenter_lat"])
    utm_crs = arcpy.SpatialReference(utm_epsg)

    utm_fc = os.path.join(workspace, f"{safe_name}_UTM")
    arcpy.management.Project(xy_fc, utm_fc, utm_crs)

    # Add projected XY coordinates as fields (POINT_X, POINT_Y)
    arcpy.management.AddXY(utm_fc)

    arcpy.management.AlterField(utm_fc, "POINT_X", "X", "X")
    arcpy.management.AlterField(utm_fc, "POINT_Y", "Y", "Y")

    print(f"  [OK] Projected to UTM Zone {utm_epsg} and added XY coordinates.")

    # ── 6. Export damage-location table ──────────────────────────────────
    damage_csv = os.path.join(RESULTS_DIR, f"{earthquake_key}_Damage_locations.csv")
    arcpy.conversion.ExportTable(
        in_table=utm_fc,
        out_table=damage_csv,
        use_field_alias_as_name="NOT_USE_ALIAS",
    )
    print(f"  [OK] Exported damage locations: {damage_csv}")

    # ── 7. Enrich: merge epicenter fields & compute abs_int_diff ─────────
    location_df = pd.read_csv(damage_csv, encoding='utf-8-sig')
    near_df = pd.read_csv(raw_near_csv, encoding='utf-8-sig')

    # Convert seismic intensity to a unified numeric "int" column
    int_col = fields["intensity"]
    if int_col in location_df.columns:
        if fields["intensity_format"] == "roman":
            location_df["int"] = roman_to_arabic(location_df[int_col], ROMAN_TO_INT)
        else:
            location_df["int"] = location_df[int_col]

    location_df.to_csv(damage_csv, index=False, encoding='utf-8-sig')

    location_df["epic_angle"] = location_df["NEAR_ANGLE"].apply(normalize_angle)
    location_df["epic_dist"] = location_df["NEAR_DIST"]

    # Unify site-name column
    name_col = fields["name"]
    if name_col in location_df.columns and name_col != "name":
        location_df.rename(columns={name_col: "name"}, inplace=True)
    elif "name" not in location_df.columns:
        location_df["name"] = "Site_" + location_df["MY_FID"].astype(str)

    # --- First merge: attach IN_FID site attributes via MY_FID ---
    enriched = near_df.merge(
        location_df[["MY_FID", "name", "int", "epic_dist", "epic_angle"]],
        left_on="IN_FID",
        right_on="MY_FID",
        suffixes=('', '_IN')
    )

    enriched.rename(columns={"name": "IN_name", "int": "IN_int", "epic_dist": "IN_epic_dist", "epic_angle": "IN_epic_angle"}, inplace=True)

    # --- Second merge: attach NEAR_FID site attributes via MY_FID ---
    enriched = enriched.merge(
        location_df[["MY_FID", "name", "int", "epic_dist", "epic_angle"]],
        left_on="NEAR_FID",
        right_on="MY_FID",
        suffixes=('', '_near')
    )

    enriched.rename(columns={"name": "near_name", "int": "near_int", "epic_dist": "near_epic_dist", "epic_angle": "near_epic_angle"}, inplace=True)
    enriched.rename(columns={"IN_name": "name", "IN_int": "int", "IN_epic_dist": "epic_dist", "IN_epic_angle": "epic_angle"}, inplace=True)

    # Drop auxiliary join keys
    enriched.drop(columns=["MY_FID", "MY_FID_near"], inplace=True, errors="ignore")

    # Compute intensity differences between each pair
    enriched["intensity_diff"] = enriched["int"] - enriched["near_int"]
    enriched["abs_int_diff"] = enriched["intensity_diff"].abs()

    # Reorder columns and save final near table
    final_cols = [
        "IN_FID", "name", "int", "epic_dist", "epic_angle",
        "NEAR_FID", "near_name", "near_int", "near_epic_dist", "near_epic_angle",
        "NEAR_DIST", "NEAR_RANK", "NEAR_ANGLE", "intensity_diff", "abs_int_diff"
    ]
    final_cols = [c for c in final_cols if c in enriched.columns]
    enriched = enriched[final_cols]

    enriched["NEAR_DIST"] = enriched["NEAR_DIST"] / 1000
    enriched["near_epic_dist"] = enriched["near_epic_dist"] / 1000
    enriched["epic_dist"] = enriched["epic_dist"] / 1000

    out_path = os.path.join(RESULTS_DIR, params["near_table_csv"])
    enriched.to_csv(out_path, index=False)
    print(f"  [OK] Enriched near table saved: {out_path}")
    return out_path

if __name__ == "__main__":
    from config import EARTHQUAKE_PARAMS
    eq_key = sys.argv[1] if len(sys.argv) > 1 else list(EARTHQUAKE_PARAMS.keys())[0]
    run_preprocessing(eq_key)