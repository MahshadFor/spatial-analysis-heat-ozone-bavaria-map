"""
temp_o3_landuse_10km_pair.py
============================
Generates landuse buffer maps for paired temperature (TEMP) and ozone (O₃)
monitoring stations within a 10 km radius.

For each station pair, two maps are saved:
  - TEMP-centered: buffer around the temperature station
  - O₃-centered:  buffer around the ozone station

Both maps show landuse polygons by macro class, both station markers,
a dashed pair-connection line with a distance label, and a basemap.

Author : Mahshadfor – Global Change Ecology thesis
Date   : 12.2025
## All scripts have been created with JupyterLab

Inputs
------
- temp_o3_close_pairs_10km.csv   : CSV of matched TEMP–O₃ station pairs
- landnutzung.gpkg               : Multi-layer GeoPackage of landuse polygons

Output
------
- PNG maps saved to OUT_DIR (one pair → two PNGs)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import re
from pathlib import Path

import contextily as cx
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration – edit these paths before running
# ---------------------------------------------------------------------------
PAIRS_CSV = Path(
    r"C:\Users\mahsh\OneDrive\Desktop\GCE\Thesis"
    r"\Data_preparation\10 - Analysis\EEA_o3_match"
    r"\temp_o3_close_pairs_10km.csv"
)
LANDUSE_PATH = Path(
    r"C:\Users\mahsh\OneDrive\Desktop\GCE\Thesis\GIS\landnutzung.gpkg"
)
OUT_DIR = Path(
    r"C:\Users\mahsh\OneDrive\Desktop\GCE\Thesis"
    r"\Data_preparation\10 - Analysis\EEA_o3_match"
    r"\Figures\PAIR_maps_bothstations_10km"
)

# Buffer radius applied to both map types
BUFFER_M = 10_000  # 10 km

# Column names in the pairs CSV
TEMP_ID_FIELD = "temp_station_id"
O3_NAME_FIELD = "airquality_o3_station"

# Station marker colours
TEMP_COLOR = "#1f78b4"  # blue
O3_COLOR   = "#984ea3"  # purple

# ---------------------------------------------------------------------------
# Landuse macro-class definitions
# ---------------------------------------------------------------------------
MACRO_CLASSES = {
    "Residential & Social": {
        "ln_wohnnutzung",
        "ln_oeffentlicheeinrichtungen",
        "ln_kulturundunterhaltung",
        "ln_sportanlage",
        "ln_freizeitanlage",
        "ln_bestattung",
    },
    "Economic / Industrial / Commercial": {
        "ln_industrieundverarbeitendesgewerbe",
        "ln_gewerblichedienstleistungen",
        "ln_lagerung",
        "ln_abbau",
        "ln_versorgungundentsorgung",
    },
    "Transportation": {
        "ln_strassenundwegeverkehr",
        "ln_bahnverkehr",
        "ln_schiffsverkehr",
        "ln_flugverkehr",
    },
    "Agriculture": {
        "ln_landwirtschaft",
        "ln_aquakulturundfischereiwirtschaft",
    },
    "Forest": {
        "ln_forstwirtschaft",
    },
    "Green & Open": {
        "ln_freiluftundnaherholung",
        "ln_ohnenutzung",
        "ln_wasserwirtschaft",
    },
}

MACRO_COLORS = {
    "Residential & Social":               "#d73027",
    "Economic / Industrial / Commercial": "#fc8d59",
    "Transportation":                     "#4d4d4d",
    "Agriculture":                        "#a6d96a",
    "Forest":                             "#1a9850",
    "Green & Open":                       "#66c2a5",
    "Other":                              "#bdbdbd",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_filename(text: str) -> str:
    """Return a Windows-safe filename string (no special characters)."""
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(text))
    text = re.sub(r"\s+", "_", text).strip("_")
    return text


def classify_landuse(layer_name: str) -> str:
    """Map a raw landuse layer name to its macro class label."""
    for label, members in MACRO_CLASSES.items():
        if layer_name in members:
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(csv_path: Path) -> pd.DataFrame:
    """Load the TEMP–O₃ station pairs CSV."""
    return pd.read_csv(csv_path)


def build_geodataframes(df: pd.DataFrame):
    """
    Build two GeoDataFrames (TEMP stations, O₃ stations) from the pairs table.
    Both use EPSG:4326.
    """
    gdf_temp = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon_temp"], df["lat_temp"]),
        crs="EPSG:4326",
    )
    gdf_o3 = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon_o3"], df["lat_o3"]),
        crs="EPSG:4326",
    )
    return gdf_temp, gdf_o3


def load_landuse(gpkg_path: Path) -> gpd.GeoDataFrame:
    """
    Load all layers from the landuse GeoPackage, concatenate them, and add a
    'landuse' column (layer name) and 'macro_class' column.
    """
    layers = fiona.listlayers(str(gpkg_path))
    parts = []
    for lyr in layers:
        g = gpd.read_file(gpkg_path, layer=lyr)
        g["landuse"] = lyr
        parts.append(g)

    gdf = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=parts[0].crs,
    )
    gdf["macro_class"] = gdf["landuse"].apply(classify_landuse)
    return gdf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pair_map(
    center_geom,
    buffer_m: int,
    title: str,
    save_path: Path,
    temp_point,
    o3_point,
    gdf_landuse: gpd.GeoDataFrame,
    distance_km: float | None = None,
) -> None:
    """
    Render and save a single pair map.

    Parameters
    ----------
    center_geom  : Shapely geometry – centre of the buffer
    buffer_m     : Buffer radius in metres
    title        : Map title string
    save_path    : Output PNG path
    temp_point   : Shapely point for the TEMP station
    o3_point     : Shapely point for the O₃ station
    gdf_landuse  : Full landuse GeoDataFrame (projected CRS)
    distance_km  : Pair distance to display as a label (optional)
    """
    # --- Buffer ---------------------------------------------------------------
    buf_geom = center_geom.buffer(buffer_m)
    buf_gdf  = gpd.GeoDataFrame(geometry=[buf_geom], crs=gdf_landuse.crs)

    # --- Clip landuse to buffer -----------------------------------------------
    candidates = gpd.sjoin(
        gdf_landuse, buf_gdf, predicate="intersects", how="inner"
    ).drop(columns=["index_right"])

    if candidates.empty:
        print(f"  [skip] No landuse candidates for: {save_path.name}")
        return

    lu_clip = gpd.clip(candidates, buf_geom)
    if lu_clip.empty:
        print(f"  [skip] Clip empty for: {save_path.name}")
        return

    lu_clip = lu_clip.copy()
    lu_clip["plot_color"] = (
        lu_clip["macro_class"].map(MACRO_COLORS).fillna(MACRO_COLORS["Other"])
    )

    # --- Figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    # Landuse polygons
    lu_clip.plot(ax=ax, color=lu_clip["plot_color"], linewidth=0.15, edgecolor="white")

    # Buffer boundary ring
    buf_gdf.boundary.plot(ax=ax, linewidth=2, edgecolor="black")

    # Dashed connection line between the two stations
    ax.plot(
        [temp_point.x, o3_point.x],
        [temp_point.y, o3_point.y],
        linestyle="--", linewidth=1.2, color="black", alpha=0.8,
    )

    # Station markers
    gpd.GeoSeries([temp_point], crs=gdf_landuse.crs).plot(
        ax=ax, markersize=100, marker="^", color=TEMP_COLOR, edgecolor="black"
    )
    gpd.GeoSeries([o3_point], crs=gdf_landuse.crs).plot(
        ax=ax, markersize=100, marker="o", color=O3_COLOR, edgecolor="black"
    )

    # Distance label at midpoint
    if distance_km is not None:
        mid_x = (temp_point.x + o3_point.x) / 2
        mid_y = (temp_point.y + o3_point.y) / 2
        ax.text(
            mid_x, mid_y,
            f"{distance_km:.2f} km",
            fontsize=10, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.25"),
        )

    # Map extent (buffer + small padding)
    minx, miny, maxx, maxy = buf_gdf.total_bounds
    pad = 800
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # Basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=gdf_landuse.crs)
    except Exception as exc:
        print(f"  [warn] Basemap failed: {exc}")

    # --- Legends --------------------------------------------------------------
    # Landuse legend (only classes present in this clip)
    present_classes = set(lu_clip["macro_class"])
    land_handles = [
        Line2D(
            [0], [0], marker="s", linestyle="",
            markerfacecolor=MACRO_COLORS[c], markeredgecolor="black",
            markersize=9, label=c,
        )
        for c in MACRO_COLORS
        if c in present_classes
    ]
    leg_land = ax.legend(
        handles=land_handles, title="Landuse (macro)", loc="lower left", frameon=True
    )
    ax.add_artist(leg_land)

    # Station legend
    station_handles = [
        Line2D(
            [0], [0], marker="^", linestyle="",
            markerfacecolor=TEMP_COLOR, markeredgecolor="black",
            markersize=10, label="TEMP station",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=O3_COLOR, markeredgecolor="black",
            markersize=10, label="O₃ station",
        ),
        Line2D([0], [0], linestyle="--", color="black", label="Pair connection"),
    ]
    ax.legend(handles=station_handles, title="Stations", loc="lower right", frameon=True)

    # --- Finalise -------------------------------------------------------------
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading pairs CSV …")
    df = load_pairs(PAIRS_CSV)

    print("Building station GeoDataFrames …")
    gdf_temp, gdf_o3 = build_geodataframes(df)

    print("Loading landuse GeoPackage …")
    gdf_landuse = load_landuse(LANDUSE_PATH)

    # Reproject stations to landuse CRS for spatial operations
    gdf_temp_m = gdf_temp.to_crs(gdf_landuse.crs)
    gdf_o3_m   = gdf_o3.to_crs(gdf_landuse.crs)

    buffer_km = int(BUFFER_M / 1000)
    n = len(df)
    print(f"Generating maps for {n} pairs (buffer = {buffer_km} km) …")

    for i in range(n):
        temp_id = str(df.loc[i, TEMP_ID_FIELD])
        o3_name = str(df.loc[i, O3_NAME_FIELD])
        dist_km = float(df.loc[i, "distance_km"])

        temp_pt = gdf_temp_m.loc[i, "geometry"]
        o3_pt   = gdf_o3_m.loc[i, "geometry"]

        base = f"{safe_filename(temp_id)}__{safe_filename(o3_name)}"

        # TEMP-centered map
        plot_pair_map(
            center_geom=temp_pt,
            buffer_m=BUFFER_M,
            title=f"TEMP-centered ({buffer_km} km): {temp_id}  |  closest O₃: {o3_name}",
            save_path=OUT_DIR / f"{base}_TEMPcenter_{buffer_km}km.png",
            temp_point=temp_pt,
            o3_point=o3_pt,
            gdf_landuse=gdf_landuse,
            distance_km=dist_km,
        )

        # O₃-centered map
        plot_pair_map(
            center_geom=o3_pt,
            buffer_m=BUFFER_M,
            title=f"O₃-centered ({buffer_km} km): {o3_name}  |  closest TEMP: {temp_id}",
            save_path=OUT_DIR / f"{base}_O3center_{buffer_km}km.png",
            temp_point=temp_pt,
            o3_point=o3_pt,
            gdf_landuse=gdf_landuse,
            distance_km=dist_km,
        )

        print(f"  [{i + 1}/{n}] {temp_id} ↔ {o3_name}")

    print("\nDone: pair maps (TEMP-centered + O₃-centered), both stations shown, distance labeled.")


if __name__ == "__main__":
    main()