"""Shared constants and utilities."""
from __future__ import annotations
from pathlib import Path
import os, subprocess
import geopandas as gpd
import rasterio

ALBERS = "EPSG:5070"
THRESH_COLS = [f"ah{p:02d}" for p in (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)]
KM2_TO_CELLS_AT_10M = 10_000   # exact in 5070 (equal-area, 10 m grid) — used for threshold tables, not the MERIT DEM

# Use $SLURM_TMPDIR for node-local fast scratch; fall back to /scratch for shared
SCRATCH = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("SCRATCH", "/scratch")))
WORK    = Path(os.environ["MISS_WORK"])    # set in sbatch (shared scratch root)
OUT     = Path(os.environ["MISS_OUT"])     # final outputs (project storage)

def warp_tiles_for_aoi(aoi_albers_geom, tile_idx_4269, out_dem,
                        target_crs, buffer_m=5000, nodata=-9999.0):
    """Reproject + mosaic tiles into the target CRS (UTM) for the given AOI."""
    aoi_target = gpd.GeoSeries([aoi_albers_geom], crs=ALBERS)
    aoi_buf    = aoi_target.buffer(buffer_m).iloc[0]
    aoi_src    = gpd.GeoSeries([aoi_buf], crs=target_crs).iloc[0]

    paths = tile_idx_4269.loc[tile_idx_4269.intersects(aoi_src), "path"].tolist()
    if not paths:
        raise RuntimeError("No tiles intersect AOI")

    xmin, ymin, xmax, ymax = aoi_buf.bounds
    subprocess.run([
        "gdalwarp",
        "-t_srs", target_crs,
        "-r", "near",
        "-ot", "Float32",
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-srcnodata", str(nodata), "-dstnodata", str(nodata),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2",
        "-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
        "-multi", "-wo", f"NUM_THREADS={os.environ.get('SLURM_CPUS_PER_TASK', 4)}",
        "-overwrite",
        *paths, str(out_dem),
    ], check=True)

    return out_dem

def load_huc4(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path).to_crs(ALBERS).sort_values("huc4").reset_index(drop=True)

def load_thresh_catchments(path: str | Path) -> gpd.GeoDataFrame:
    """Catchments (HUC12s) with ah05..ah95 columns in km²."""
    g = gpd.read_file(path).to_crs(ALBERS)
    missing = [c for c in THRESH_COLS if c not in g.columns]
    if missing:
        raise KeyError(f"Threshold columns missing: {missing}")
    return g

def utm_crs_for_geom(geom_albers):
    """Return the UTM CRS (as EPSG string) appropriate for a geometry's centroid.
    Input geometry must be in EPSG:5070 or any projected CRS — reproject centroid
    to geographic first to get lon/lat for zone calculation."""
    centroid = gpd.GeoSeries([geom_albers], crs=ALBERS).to_crs("EPSG:4326").iloc[0].centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

# def load_thresh_catchments(path: str | Path) -> gpd.GeoDataFrame:
#     return gpd.read_file(path).to_crs(ALBERS).sort_values("huc12").reset_index(drop=True)