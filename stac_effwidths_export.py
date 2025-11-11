# run_effwidths_win.py
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("AWS_NO_SIGN_REQUEST","YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD","NO")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif,.TIF,.vrt")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE","32768")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY","0.5")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd, geopandas as gpd
from tqdm import tqdm
import numpy as np

# import your top-level functions from a module:
from stac_utils import (ref_geoms_from_b3href, process_image_from_hrefs,
                        identify_river, count_pixels)

G_CI=G_CL=HREFS=None


def _init_worker(square_path, circle_path, centerline_path, pt_path, href_dict_in, valid_iids):
    global G_SQ, G_CI, G_CL, G_PT, HREFS
    squares = gpd.read_file(square_path).set_crs(4326)
    circles = gpd.read_file(circle_path).set_crs(4326)
    pts = gpd.read_file(pt_path).set_crs(4326)
    G_CL = gpd.read_file(centerline_path).set_crs(4326, allow_override=True)

    # duplicated_iidxes = circles.iindex[circles.iindex.duplicated()]
    # circles = circles.loc[~circles.iindex.isin(duplicated_iidxes)]

    # pairs = circles.sjoin(circles, how='inner', predicate='intersects')
    # pairs = pairs[pairs.iindex_left != pairs.iindex_right].drop_duplicates()

    # G = nx.from_pandas_edgelist(pairs, source='iindex_left', target='iindex_right')
    # G.add_nodes_from(circles['iindex'])

    # comp_map = {n:i for i, comp in enumerate(nx.connected_components(G)) for n in comp}
    # circles['cluster'] = circles.iindex.map(comp_map)

    # keep_idx = circles.groupby('cluster').sample(1, random_state=0).index
    G_CI = circles.loc[circles.iindex.isin(valid_iids)].set_index('iindex')
    G_PT = pts.loc[pts.iindex.isin(valid_iids)].set_index('iindex')
    G_SQ = squares.loc[squares.iindex.isin(valid_iids)].set_index('iindex')
    # print(len(G_CI))
    # .drop(columns='cluster')

    
    # squares = squares.loc[squares.iindex.isin(circles.iindex)]

    HREFS = href_dict_in


def _worker(rec):
    img_id, idx = rec
    b3, b8, scl = HREFS[(str(img_id), (idx))]
    # print(rec)
    square, circle, lines = ref_geoms_from_b3href(b3, (idx), G_SQ, G_CI, G_CL)
    ndwi, wmask, cloud, snow, valid, transform = process_image_from_hrefs(b3, b8, scl, square, circle)
    p = G_PT.loc[idx]
    xcoord = p.geometry.x
    ycoord = p.geometry.y

    if ndwi.size <= 1:
        return (img_id, idx, -999, -999, -999, -999, -999, -999, -999, -999, xcoord, ycoord)
    rmask = identify_river(wmask, lines, transform)
    
    return (img_id, idx, *count_pixels(rmask, cloud, snow, valid, transform, circle), xcoord, ycoord)


def build_href_dict(df_in):
    df = df_in.copy().reset_index()
    df["img_id"] = df['img_id'].astype(str).str.strip()
    df["iindex"] = np.uint64(df['iindex'])
    for c in ("b3_href","b8_href","scl_href"): df[c] = df[c].astype(str).str.strip()
    return {(row.img_id,row.iindex):(row.b3_href,row.b8_href,row.scl_href) for _,row in df.iterrows()}

if __name__ == "__main__":
    mp.freeze_support()  # harmless elsewhere; required in some Windows launches

    # # inputs (replace with your paths)
    # squares = r"C:\Users\dego\Documents\local_files\RSSA\Platte_centerlines_masks\squares_15x_20251010.shp"
    # pills   = r"C:\path\gage_pills_3L_3W_20250904.shp"

    outdir = r"C:\Users\dego\Documents\local_files\RSSA\effwidth_results\static"
    completed_files = os.listdir(outdir)

    for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        if f'effwidths_{year}.csv' not in (completed_files):
            break


    squares = r"C:\Users\dego\Documents\local_files\RSSA\Platte_centerlines_masks\squares_15x_20251010.shp"
    circles = r"C:\Users\dego\Documents\local_files\RSSA\Platte_centerlines_masks\circles_3x_20251010.shp"
    clines  = r"C:\Users\dego\Documents\local_files\RSSA\Platte_centerlines_masks\Vector_centerlines\s2_platte_centerlines.shp"
    href_csv=fr"C:\Users\dego\Documents\local_files\RSSA\stac_img_ids_{year}_20251012.csv"
    pts =     r"C:\Users\dego\Documents\local_files\RSSA\Platte_centerlines_masks\points_20251010.shp"

    href_df = pd.read_csv(href_csv).set_index(['img_id', 'iindex'])
    valid_iids = href_df.index.get_level_values(1).unique()
    HREFS_PARENT = build_href_dict(href_df)
    # work = pd.read_csv(work_csv)
    # work["img_id"] = work["img_id"].astype(str)
    # work["iindex"] = work["iindex"].astype(int)

    # records = list(zip(href_df["img_id"].tolist(), href_df["iindex"].tolist()))
    records = href_df.index.tolist()
    # print(records)

    ctx = mp.get_context("spawn")  # explicit on Windows
    with ProcessPoolExecutor(max_workers=min(6, mp.cpu_count()),
                             mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(squares, circles, clines, pts, HREFS_PARENT, valid_iids)) as ex:
        rows = list(tqdm(ex.map(_worker, records, chunksize=8), total=len(records)))
    out = pd.DataFrame(rows, columns=[
        "img_id","iindex","n_pixels","n_valid","n_river","n_cloud","n_snow","n_cloudriver","n_edge","n_edgeriver", "x", "y"
    ])
    os.makedirs(outdir, exist_ok=True)
    out.to_csv(os.path.join(outdir, f"effwidths_{year}.csv"), index=False)