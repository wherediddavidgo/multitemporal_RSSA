"""Phase 1: warp+burn+breach+D8 pointer+flow accumulation for one HUC4."""
import argparse, os, sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
import whitebox
from common import WORK, ALBERS, warp_tiles_for_aoi, load_huc4, utm_crs_for_geom
import geopandas as gpd
import datetime


def _ensure_float64(path: Path) -> None:
    """Rewrite raster as float64 in-place if it isn't already."""
    with rasterio.open(path) as src:
        if src.dtypes[0] == "float64":
            return
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    profile.update(dtype="float64", bigtiff="YES")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    print(f"  [dtype] converted {path.name} to float64")


def burn_reference_network(dem_path, grwl_path, huc4_geom_albers, utm_crs,
                            out_path, burn_depth=10.0):
    """Lower DEM cells that fall on GRWL centerlines by burn_depth metres.

    Returns out_path on success, or dem_path unchanged if GRWL has no
    features in the AOI (so the caller can always use the return value).
    """
    grwl = gpd.read_file(grwl_path)

    # Buffer AOI in the equal-area projection before converting to GRWL CRS,
    # so we catch lines that cross the boundary and extend into the 5 km DEM buffer.
    aoi_in_grwl_crs = (gpd.GeoSeries([huc4_geom_albers], crs=ALBERS)
                       .buffer(10_000)
                       .to_crs(grwl.crs)
                       .iloc[0])
    grwl_clipped = grwl[grwl.intersects(aoi_in_grwl_crs)]
    if grwl_clipped.empty:
        print("  [burn] no GRWL features in AOI — skipping burn")
        return dem_path

    grwl_utm = grwl_clipped.to_crs(utm_crs)

    with rasterio.open(dem_path) as src:
        dem_arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    stream_mask = rio_rasterize(
        [(geom, 1) for geom in grwl_utm.geometry if geom is not None],
        out_shape=dem_arr.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    valid = stream_mask == 1
    del stream_mask
    if nodata is not None:
        valid &= dem_arr != nodata
    dem_arr[valid] -= burn_depth  # in-place: avoids a second full-array allocation
    print(f"  [burn] lowered {int(valid.sum()):,} cells by {burn_depth} m")
    del valid

    profile.update(dtype="float64", bigtiff="YES")
    dem_arr = dem_arr.astype(np.float64)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem_arr, 1)
    return out_path


def main(huc4_idx: int, huc4_path: str, pntr_tile_idx_path: str, uparea_tile_idx_path: str, grwl_path: str | None):
    huc4 = load_huc4(huc4_path).iloc[huc4_idx]
    pntr_tiles = gpd.read_file(pntr_tile_idx_path)
    uparea_tiles = gpd.read_file(uparea_tile_idx_path)
    hid = huc4["huc4"]

    # utm = utm_crs_for_geom(huc4.geometry)
    # print(f'[{hid}] using {utm}')

    h = WORK / "huc4" / hid
    h.mkdir(parents=True, exist_ok=True)

    # dem      = h / "dem_utm.tif"
    # filled   = h / "filled.tif"
    # burned   = h / "dem_burned.tif"
    # breached = h / "breached.tif"
    pntr     = h / "pntr.tif"
    facc     = h / "facc.tif"

    # print(facc)

    if pntr.exists() and facc.exists():
        # print(f"[{hid}] already done"); return
        os.remove(pntr)
        os.remove(facc)

    warp_tiles_for_aoi(huc4.geometry, pntr_tiles, pntr, target_crs=ALBERS, buffer_m=5000)
    warp_tiles_for_aoi(huc4.geometry, uparea_tiles, facc, target_crs=ALBERS, buffer_m=5000)
    # Save the CRS used so Phase 2 can read it without recomputing
    # (h / "utm_crs.txt").write_text(utm)

    # wbt = whitebox.WhiteboxTools()
    # wbt.exe_path = '/home/dego/headwater_network_extraction/whitebox_tools/WBT/'
    # wbt.set_verbose_mode(True)
    # wbt.set_compress_rasters(True)
    # wbt.set_max_procs(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))

    # if grwl_path:
    #     burned = burn_reference_network(
    #         dem, grwl_path, huc4.geometry, utm, burned, 10.0
    #     )
    # else:
    #     burned = dem


    # ret = wbt.breach_depressions_least_cost(
    #     dem=burned, output=str(breached), dist=20, fill=False
    # )
    # if ret != 0:
    #     raise RuntimeError(f"[{hid}] breach_depressions_least_cost failed (code {ret})")
    

    # ret = wbt.fill_depressions_wang_and_liu(
    #     dem=str(breached), output=str(filled)
    # )
    # if ret != 0:
    #     raise RuntimeError(f"[{hid}] fill_depressions_wang_and_liu failed (code {ret})")


    # ret = wbt.d8_pointer(dem=str(filled), output=str(pntr))
    # if ret != 0:
    #     raise RuntimeError(f"[{hid}] d8_pointer failed (code {ret})")
    # _ensure_float64(pntr)

    # ret = wbt.d8_flow_accumulation(
    #     i=str(pntr), output=str(facc), out_type="cells", pntr=True,
    # )
    # if ret != 0:
    #     raise RuntimeError(f"[{hid}] d8_flow_accumulation failed (code {ret})")
    # _ensure_float64(facc)
    # dem.unlink(missing_ok=True)
    # filled.unlink(missing_ok=True)
    # burned.unlink(missing_ok=True)
    # breached.unlink(missing_ok=True)
    print(f"[{hid}] done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--huc4-idx", type=int, required=True)   # SLURM_ARRAY_TASK_ID
    ap.add_argument("--huc4-path", required=True)
    ap.add_argument("--pntr-tile-idx-path", required=True)
    ap.add_argument("--uparea-tile-idx-path", required=True)
    ap.add_argument("--grwl-path", default=None,
                    help="GRWL centerlines shapefile; omit to skip stream burning")
    print(f'Starting at {datetime.datetime.now()}')
    main(**vars(ap.parse_args()))
    print(f'Finished at {datetime.datetime.now()}')
