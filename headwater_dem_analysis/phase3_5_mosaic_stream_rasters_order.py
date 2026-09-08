"""
Mosaic ~30,000 HUC12 stream-network rasters into a single raster and
assign stream order (Strahler and/or Shreve) using WhiteboxTools.

Input layout assumed:
    STREAMS_DIR/
        <HUC12_id_1>/<stream file>.tif
        <HUC12_id_2>/<stream file>.tif
        ...
i.e. one subdirectory per HUC12, each containing that catchment's stream
raster. A single flow-direction (D8 pointer) raster already covers the
whole study area and is used as-is — it does NOT need to be mosaicked.

--------------------------------------------------------------------------
IMPORTANT CAVEAT — read before running at full scale
--------------------------------------------------------------------------
Stream order (Strahler, Shreve, etc.) is a topological property computed
from flow connectivity. WhiteboxTools' order tools need BOTH:
    1. A streams raster
    2. A D8 flow-direction ("pointer") raster covering the same area

Because the D8 pointer here already covers the entire study area (derived
once from a continuous conditioned DEM), you avoid the worst seam problem
— mismatched flow direction at HUC12 boundaries. That's normally the
biggest source of ordering artifacts in this kind of workflow, and it's
already solved by your inputs.

One thing still matters, though: the stream mosaic and the study-area D8
raster MUST share the same grid — identical cell size, alignment, extent,
and CRS — or WhiteboxTools will error out or silently misalign the two
rasters. If your per-HUC12 stream rasters were originally clipped from
this same D8 raster's grid, this should already be true. If not, set
RESAMPLE_STREAMS_TO_D8_GRID = True below to snap the stream mosaic onto
the D8 raster's grid before running the order tools.

Stream order is still a network-wide property, so it must be computed
ONCE on the full mosaic, not per-HUC12 and then mosaicked — a stream near
a HUC12 edge may pick up additional order from tributaries that live in
a neighboring catchment's raster.

--------------------------------------------------------------------------
Why hierarchical (batched) mosaicking?
--------------------------------------------------------------------------
WhiteboxTools' `mosaic` tool is invoked as a CLI call under the hood with
a comma-separated list of input file paths. Passing 30,000 paths at once
risks hitting OS command-line length limits and spikes memory usage when
building one enormous VRT-like merge. Instead, this script mosaics inputs
in small batches, then repeatedly mosaics the batch outputs together
("pyramid reduction") until only one file remains.
"""

import os
import sys
import glob
import shutil
import logging
from datetime import datetime

import rasterio
from rasterio.crs import CRS
from whitebox import WhiteboxTools

# ------------------------------------------------------------------------
# CONFIG — edit these paths/patterns for your data
# ------------------------------------------------------------------------

# Parent directory containing one subdirectory per HUC12, each holding
# that catchment's stream raster, e.g.:
#   /data/huc12_streams/060102030401/streams.tif
#   /data/huc12_streams/060102030402/streams.tif
STREAMS_DIR = "/scratch/dego/miss/huc12"

# Glob pattern for the stream file WITHIN each HUC12 subdirectory.
# "*.tif" works if each subdirectory contains exactly one raster; narrow
# it (e.g. "streams.tif" or "*_streams.tif") if subdirectories contain
# other files too.
STREAM_FILE_GLOB = "streams_ah45.tif"

# Single flow-direction (D8 pointer) raster covering the entire study
# area. This is used directly — it is NOT mosaicked.
D8_RASTER = "/home/dego/headwater_network_extraction/NHD_fdr/nhd_fdr_mos_arc_corr.tif"

# Working / output locations
WORK_DIR = "/scratch/dego/miss"
TEMP_DIR = os.path.join(WORK_DIR, "tmp_mosaic_batches")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")

STREAMS_MOSAIC = os.path.join(OUTPUT_DIR, "streams_mosaic.tif")
STREAMS_MOSAIC_RESAMPLED = os.path.join(OUTPUT_DIR, "streams_mosaic_on_d8_grid.tif")
STRAHLER_OUT = os.path.join(OUTPUT_DIR, "strahler_order.tif")
SHREVE_OUT = os.path.join(OUTPUT_DIR, "shreve_order.tif")

# Set True only if the stream mosaic's grid does not already exactly
# match the D8 raster's grid (cell size / alignment / extent). This adds
# a resampling step so the order tools don't fail or silently misalign.
RESAMPLE_STREAMS_TO_D8_GRID = False

# How many rasters to mosaic per batch at each level of the pyramid.
# Lower this if you hit memory or command-line-length errors; raise it
# to reduce the number of intermediate files (fewer, larger batches).
BATCH_SIZE = 50

# Mosaic resampling method: "nn" (nearest neighbour) is correct for
# categorical data like stream masks and D8 pointer codes. Do NOT use
# bilinear/cubic here — it will corrupt integer pointer/stream codes.
MOSAIC_METHOD = "nn"

# Set True if your D8 pointer rasters use ESRI's pointer convention
# instead of Whitebox's native convention.
ESRI_PNTR = True

# ------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def init_wbt(working_dir):
    wbt = WhiteboxTools()
    wbt.set_working_dir(working_dir)
    wbt.set_verbose_mode(False)
    return wbt


def hierarchical_mosaic(wbt, file_list, batch_size, temp_dir, final_output,
                         method="nn", label="mosaic"):
    """
    Mosaic a large list of rasters in batches, then recursively mosaic
    the batch outputs, until a single output raster remains.
    """
    if len(file_list) == 0:
        raise ValueError(f"No input files provided for {label}.")

    if len(file_list) == 1:
        shutil.copy(file_list[0], final_output)
        log.info(f"[{label}] Only one input file — copied directly to {final_output}")
        return final_output

    os.makedirs(temp_dir, exist_ok=True)
    level = 0
    current_files = list(file_list)

    while len(current_files) > 1:
        level += 1
        next_files = []
        batches = [
            current_files[i:i + batch_size]
            for i in range(0, len(current_files), batch_size)
        ]
        log.info(f"[{label}] Level {level}: mosaicking {len(current_files)} "
                  f"files in {len(batches)} batch(es) of up to {batch_size}.")

        for b_idx, batch in enumerate(batches):
            if len(batch) == 1:
                # Nothing to mosaic, pass the single file through unchanged.
                next_files.append(batch[0])
                continue

            out_name = os.path.join(
                temp_dir, f"{label}_lvl{level}_batch{b_idx:05d}.tif"
            )
            inputs_str = ",".join(batch)

            ret = wbt.mosaic(output=out_name, inputs=inputs_str, method=method)
            if ret != 0:
                raise RuntimeError(
                    f"WhiteboxTools mosaic failed at {label} level {level}, "
                    f"batch {b_idx} (return code {ret})."
                )
            next_files.append(out_name)

        current_files = next_files

    # Move/rename the final remaining file to the requested output path
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    shutil.move(current_files[0], final_output)
    log.info(f"[{label}] Final mosaic written to {final_output}")
    return final_output


def gather_huc12_stream_rasters(streams_dir, file_glob):
    """
    Walk each HUC12 subdirectory under `streams_dir` and collect its
    stream raster. Returns (stream_files, huc_dirs): the list of stream
    raster paths to mosaic, and the full list of HUC12 subdirectories
    found (for logging/sanity-checking counts). Any subdirectory with
    zero or multiple matches for `file_glob` is logged as a warning
    (worth investigating before a 30,000-file run).
    """
    huc_dirs = sorted(
        d for d in glob.glob(os.path.join(streams_dir, "*"))
        if os.path.isdir(d)
    )

    stream_files = []
    problem_dirs = []

    for huc_dir in huc_dirs:
        matches = glob.glob(os.path.join(huc_dir, file_glob))
        if len(matches) == 1:
            stream_files.append(matches[0])
        elif len(matches) == 0:
            problem_dirs.append((os.path.basename(huc_dir), "no match"))
        else:
            problem_dirs.append((os.path.basename(huc_dir), f"{len(matches)} matches"))
            # Be permissive: include all matches rather than silently
            # dropping the HUC12 entirely.
            stream_files.extend(matches)

    if problem_dirs:
        log.warning(
            f"{len(problem_dirs)} HUC12 subdirectories had zero or multiple "
            f"files matching '{file_glob}' (first few: {problem_dirs[:5]})"
        )

    return stream_files, huc_dirs


def main():
    start_time = datetime.now()

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wbt = init_wbt(WORK_DIR)

    if not os.path.isfile(D8_RASTER):
        log.error(f"D8_RASTER not found at {D8_RASTER} — check the path.")
        sys.exit(1)

    log.info("Gathering stream rasters from HUC12 subdirectories...")
    streams_files, huc_dirs = gather_huc12_stream_rasters(STREAMS_DIR, STREAM_FILE_GLOB)
    log.info(f"Found {len(huc_dirs)} HUC12 subdirectories, "
              f"{len(streams_files)} stream rasters to mosaic.")

    if len(streams_files) == 0:
        log.error("No stream rasters found — check STREAMS_DIR / STREAM_FILE_GLOB.")
        sys.exit(1)

    # --- Step 1: mosaic the stream rasters ---
    log.info("Mosaicking stream rasters (this covers the full ~30,000-tile set)...")
    hierarchical_mosaic(
        wbt, streams_files, BATCH_SIZE,
        os.path.join(TEMP_DIR, "streams"),
        STREAMS_MOSAIC, method=MOSAIC_METHOD, label="streams"
    )

    with rasterio.open(STREAMS_MOSAIC, 'r+') as dst:
        dst.crs = CRS.from_epsg(5070)

    streams_for_order = STREAMS_MOSAIC

    # --- Step 2 (optional): snap the stream mosaic onto the D8 raster's
    # grid if they don't already share identical cell size/alignment/extent.
    if RESAMPLE_STREAMS_TO_D8_GRID:
        log.info("Resampling stream mosaic onto the D8 raster's grid...")
        ret = wbt.resample(
            inputs=STREAMS_MOSAIC,
            output=STREAMS_MOSAIC_RESAMPLED,
            cell_size=None,
            base=D8_RASTER,
            method="nn",
        )
        if ret != 0:
            raise RuntimeError(f"resample failed (return code {ret}).")
        streams_for_order = STREAMS_MOSAIC_RESAMPLED
        log.info(f"Resampled stream mosaic written to {STREAMS_MOSAIC_RESAMPLED}")

    # --- Step 3: compute stream order using the existing whole-area D8
    # raster directly (no D8 mosaicking needed) ---
    log.info("Computing Strahler stream order against the study-area D8 raster...")
    ret = wbt.strahler_stream_order(
        d8_pntr=D8_RASTER,
        streams=streams_for_order,
        output=STRAHLER_OUT,
        esri_pntr=ESRI_PNTR,
        zero_background=False,
    )
    if ret != 0:
        raise RuntimeError(
            f"strahler_stream_order failed (return code {ret}). If this is a "
            f"grid-mismatch error, set RESAMPLE_STREAMS_TO_D8_GRID = True."
        )
    log.info(f"Strahler order raster written to {STRAHLER_OUT}")

    # log.info("Computing Shreve stream order against the study-area D8 raster...")
    # ret = wbt.shreve_stream_order(
    #     d8_pntr=D8_RASTER,
    #     streams=streams_for_order,
    #     output=SHREVE_OUT,
    #     esri_pntr=ESRI_PNTR,
    #     zero_background=False,
    # )
    # if ret != 0:
    #     raise RuntimeError(
    #         f"shreve_stream_order failed (return code {ret}). If this is a "
    #         f"grid-mismatch error, set RESAMPLE_STREAMS_TO_D8_GRID = True."
    #     )
    # log.info(f"Shreve order raster written to {SHREVE_OUT}")

    # # --- Step 4: clean up temp batch files ---
    # if os.path.isdir(TEMP_DIR):
    #     shutil.rmtree(TEMP_DIR, ignore_errors=True)
    #     log.info("Cleaned up temporary batch mosaics.")

    elapsed = datetime.now() - start_time
    log.info(f"Done. Total elapsed time: {elapsed}")


if __name__ == "__main__":
    main()