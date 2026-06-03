"""Phase 2: extract streams at 10 thresholds for every HUC12 within one HUC4."""
import argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from pyproj import Transformer
from shapely.ops import transform as shp_transform
import whitebox
import subprocess

from common import (WORK, ALBERS, THRESH_COLS, KM2_TO_CELLS_AT_10M,
                    load_huc4, load_thresh_catchments)
THRESH_COLS.reverse()

HUC12_BUFFER_M = 5000

def clip_to_huc12(src_path, geom, out_path, geom_crs, buffer_m=HUC12_BUFFER_M):
    with rasterio.open(src_path) as src:
        raster_crs = src.crs
        if rasterio.crs.CRS.from_user_input(geom_crs) != raster_crs:
            proj = Transformer.from_crs(geom_crs, raster_crs, always_xy=True).transform
            geom = shp_transform(proj, geom)
        out, t = rio_mask(src, [geom.buffer(buffer_m)], crop=True, all_touched=True)
        prof = src.profile
        prof.update(height=out.shape[1], width=out.shape[2],
                    transform=t, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out)
    return out_path

def extract_one_huc12(row, parent_pntr, parent_facc, out_root, geom_crs):
    hid = f'{int(row["huc12"]):012d}'
    h = out_root / hid; h.mkdir(parents=True, exist_ok=True)
    pntr = clip_to_huc12(parent_pntr, row.geometry, h / "pntr_clip.tif", geom_crs)
    facc = clip_to_huc12(parent_facc, row.geometry, h / "facc_clip.tif", geom_crs)

    with rasterio.open(parent_pntr) as src:
        utm = src.crs

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(True); wbt.set_compress_rasters(True)
    wbt.set_max_procs(1)  # one thread per worker; parallelism comes from Python workers

    written = {}
    for col in THRESH_COLS:
        t_cells = int(round(float(row[col]) * KM2_TO_CELLS_AT_10M))
        if t_cells < 1:
            written[col] = None; continue
        sr = h / f"streams_{col}.tif"
        sv = h / f"streams_{col}.shp"
        wbt.extract_streams(flow_accum=str(facc), output=str(sr), threshold=t_cells)
        wbt.raster_streams_to_vector(streams=str(sr), d8_pntr=str(pntr), output=str(sv))

        if not sv.exists():
            written[col] = None; continue
        gdf = gpd.read_file(sv)
        gdf = gdf.set_crs(utm)
        if len(gdf) == 0:
            written[col] = None; continue
        gdf = gpd.clip(gdf, row.geometry)
        gdf = gdf[gdf.geom_type == "LineString"].copy()
        gdf["huc12"] = hid
        gpkg = h / f"streams_{col}.gpkg"
        gdf.to_file(gpkg, driver="GPKG")
        written[col] = str(gpkg)

    pntr.unlink(missing_ok=True)
    facc.unlink(missing_ok=True)
    for f in h.glob("streams_*.tif"): f.unlink()
    for f in h.glob("streams_*.shp*"): f.unlink()
    for f in h.glob("streams_*.dbf*"): f.unlink()
    for f in h.glob("streams_*.shx*"): f.unlink()
    return hid, written

def main(huc4_idx: int, huc4_path: str, catchments_path: str, n_workers: int):
    huc4 = load_huc4(huc4_path).iloc[huc4_idx]
    hid_h4 = huc4["huc4"]
    parent_pntr = WORK / "huc4" / hid_h4 / "pntr.tif"
    parent_facc = WORK / "huc4" / hid_h4 / "facc.tif"

    utm = (WORK / "huc4" / hid_h4 / "utm_crs.txt").read_text().strip()

    if not parent_facc.exists():
        raise FileNotFoundError(f"Phase 1 output missing for {hid_h4}")

    catch = load_thresh_catchments(catchments_path)
    catch_in_huc4 = catch[catch.intersects(huc4.geometry)].copy()
    # belt-and-suspenders: keep only HUC12s whose centroid lies in this HUC4
    catch_in_huc4 = catch_in_huc4[catch_in_huc4.representative_point().within(huc4.geometry)]
    catch_in_huc4 = catch_in_huc4.to_crs(utm)
    print(f"[{hid_h4}] {len(catch_in_huc4)} HUC12s")

    out_root = WORK / "huc12"
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(extract_one_huc12, row, parent_pntr, parent_facc, out_root, utm)
                for _, row in catch_in_huc4.iterrows()]
        for f in as_completed(futs):
            try: f.result()
            except Exception as e: print(f"[fail] {e}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--huc4-idx", type=int, required=True)
    ap.add_argument("--huc4-path", required=True)
    ap.add_argument("--catchments-path", required=True)
    ap.add_argument("--n-workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    main(**vars(ap.parse_args()))