"""Phase 3: merge HUC12 networks for one percentile, build digraph, assign Strahler."""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import geopandas as gpd
import igraph as ig
from common import WORK, OUT, THRESH_COLS, ALBERS

def collect_paths(percentile_col: str):
    return sorted((WORK / "huc12").glob(f"*/streams_{percentile_col}.gpkg"))
    # return sorted((WORK / "huc12").glob(f"0501*/streams_{percentile_col}.gpkg"))

def _read_one(p):
    return gpd.read_file(p).to_crs(ALBERS)

def merge(paths):
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        gdfs = list(pool.map(_read_one, paths))
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

def strahler_igraph(streams_gdf, snap_decimals=0):
    import igraph as ig
    nodes, edges, idxs, lens = {}, [], [], []
    for idx, geom in streams_gdf.geometry.items():
        if geom is None or geom.is_empty: continue
        c = list(geom.coords)
        # u = tuple(round(x, snap_decimals) for x in c[0])
        # v = tuple(round(x, snap_decimals) for x in c[-1])
        u = tuple(5 * round(x / 5) for x in c[0])
        v = tuple(5 * round(x / 5) for x in c[-1])
        if u == v: continue
        for k in (u, v): nodes.setdefault(k, len(nodes))
        edges.append((nodes[u], nodes[v]))
        idxs.append(idx); lens.append(geom.length)

    g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    g.es["idx"] = idxs; g.es["length"] = lens
    if not g.is_dag():
        # Remove cycle-causing edges (usually flat-area pointer artifacts)
        cyc = g.feedback_arc_set(method="ip")
        print(f"[warn] removing {len(cyc)} cycle-causing edges", flush=True)
        removed_idxs = {g.es[e]["idx"] for e in cyc}
        g.delete_edges(cyc)
        idxs = [i for i in idxs if i not in removed_idxs]
        lens = [l for i, l in zip(
            [g.es[e]["idx"] for e in range(g.ecount())] , lens) if i not in removed_idxs]
        # rebuild clean edge lists after deletion
        idxs = g.es["idx"]
        lens = g.es["length"]

    topo = g.topological_sorting(mode="out")
    edge_order = [0] * g.ecount()
    for v in topo:
        in_es = g.incident(v, mode="in")
        if not in_es:
            o = 1
        else:
            ords = [edge_order[e] for e in in_es]
            mx = max(ords); o = mx + 1 if ords.count(mx) >= 2 else mx
        for e in g.incident(v, mode="out"):
            edge_order[e] = o

    out = streams_gdf.copy()
    idx_to_o = dict(zip(idxs, edge_order))
    out["strahler"] = out.index.map(idx_to_o).astype("Int32")
    return out

def main(percentile_idx: int):
    col = THRESH_COLS[percentile_idx]
    paths = collect_paths(col)
    if not paths:
        raise RuntimeError(f"No HUC12 outputs for {col}")
    print(f"[{col}] merging {len(paths)} HUC12 files", flush=True)

    gdf = merge(paths)
    # gdf = strahler_igraph(gdf)

    OUT.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT / f"network_{col}_merged.gpkg", driver="GPKG")

    # summary = (gdf.assign(length_m=gdf.geometry.length)
    #               .groupby("strahler")["length_m"].sum().reset_index())
    # summary["percentile"] = col
    # summary.to_csv(OUT / f"lengths_by_order_{col}_test.csv", index=False)
    print(f"[{col}], "
          f"total {gdf.geometry.length.sum()/1e3:.0f} km", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile-idx", type=int, required=True)  # 0..9
    main(**vars(ap.parse_args()))
