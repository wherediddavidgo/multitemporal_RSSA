import geopandas as gpd, rasterio, subprocess
from pathlib import Path
from rasterio.features import bounds as feat_bounds
from tqdm import tqdm
import os

ALBERS = "EPSG:5070"
PNTR_TILES_DIR = Path("/scratch/dego/miss/corrected_nhd_fdr")
UPAREA_TILES_DIR = Path("/scratch/dego/miss/corrected_nhd_fac")
WORK = Path("/scratch/dego/miss"); WORK.mkdir(parents=True, exist_ok=True)

def build_tile_index(tiles_dir: Path, out_path: Path):
    """One-time: index all tiles by their bounds in source CRS (5070)."""
    rows = []
    for p in tqdm(tiles_dir.glob("*c.tif")):
        with rasterio.open(p) as s:
            rows.append({"path": str(p), "geometry": gpd.GeoSeries.from_wkt(
                [f"POLYGON(({s.bounds.left} {s.bounds.bottom},"
                 f"{s.bounds.right} {s.bounds.bottom},"
                 f"{s.bounds.right} {s.bounds.top},"
                 f"{s.bounds.left} {s.bounds.top},"
                 f"{s.bounds.left} {s.bounds.bottom}))"]).iloc[0]})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:5070")
    gdf.to_file(out_path, driver="GPKG")
    return gdf

huc4   = gpd.read_file("/home/dego/headwater_network_extraction/catchment_geodata/huc4_mississippi.gpkg").to_crs(ALBERS)
huc12  = gpd.read_file("/home/dego/headwater_network_extraction/catchment_geodata/sheds_w_nhd_merit_atts_ahthresh_4269.gpkg").to_crs(ALBERS)
build_tile_index(PNTR_TILES_DIR, WORK / "nhd_fdr_tile_idx.gpkg")
build_tile_index(UPAREA_TILES_DIR, WORK / "nhd_fac_tile_idx.gpkg")