"""Phase 2.5: merge DEM-derived headwater networks with MERIT Hydro mainstems.

DEM networks accurately map headwaters but lose connectivity in wide, flat rivers.
MERIT Hydro captures large-river routing but misses headwater extent. This script
combines the two by Strahler order threshold (MERIT column: order):

  order > threshold  — high-order MERIT reaches replace parallel DEM reaches.
  order <= threshold — low-order MERIT reaches are excluded; DEM reaches are kept.

Two modes
---------
Single-file (one-off / testing):
    python phase2_5_merge_networks.py \\
        --dem-network-path dem.gpkg \\
        --merit-network-path merit.shp \\
        --output-path merged.gpkg

Batch (all 10 threshold columns, parallel workers):
    python phase2_5_merge_networks.py \\
        --huc12-dir   /path/to/huc12 \\
        --merit-path  /path/to/merit.shp \\
        --out-dir     /path/to/output \\
        --n-workers   4
"""
import os
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import igraph as ig
import pandas as pd
from shapely import Point, LineString
from shapely.ops import substring
from tqdm import tqdm
# Source - https://stackoverflow.com/a/79163867
# Posted by patman
# Retrieved 2026-05-19, License - CC BY-SA 4.0

import pyproj
pyproj.network.set_network_enabled(False)

ALBERS = 'EPSG:5070'
THRESH_COLS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "p100"]

# Endpoint snapping tolerance (m). Controls:
#   - internal DEM reach connectivity in find_downstream_dangles
#   - minimum split-segment length in split_merit_at_junctions
#   - node grid resolution in strahler_igraph
GRAPH_SNAP = 10.0


# ---------------------------------------------------------------------------
# Core geometry functions
# ---------------------------------------------------------------------------

def filter_parallel_reaches(network, reference, max_distance_parallel,
                            require_overlap=True):
    """Remove reaches from *network* where both endpoints are within
    max_distance_parallel of any reach in *reference*.

    When require_overlap=True (default), a reach is only removed if a reference
    reach also intersects a buffer around the line body — ensuring MERIT
    actually covers the reach span, not just its endpoints.  Without this guard,
    a DEM mainstem that starts and ends near MERIT but traverses a wetland lobe
    MERIT does not cover would be erroneously removed.
    """
    geoms = network.geometry.tolist()
    n = len(geoms)

    ups = gpd.GeoDataFrame(
        {'_row': range(n), 'geometry': [Point(g.coords[0]) for g in geoms]},
        crs=network.crs,
    )
    dns = gpd.GeoDataFrame(
        {'_row': range(n), 'geometry': [Point(g.coords[-1]) for g in geoms]},
        crs=network.crs,
    )
    ref_geom = reference[['geometry']]

    d0 = (gpd.sjoin_nearest(ups, ref_geom, how='left', distance_col='_d')
          .groupby('_row')['_d'].min())
    d1 = (gpd.sjoin_nearest(dns, ref_geom, how='left', distance_col='_d')
          .groupby('_row')['_d'].min())

    endpoint_close = {
        i for i in range(n)
        if d0.get(i, float('inf')) <= max_distance_parallel
        and d1.get(i, float('inf')) <= max_distance_parallel
    }

    if not endpoint_close or not require_overlap:
        keep = [i for i in range(n) if i not in endpoint_close]
        return network.iloc[keep].copy().reset_index(drop=True)

    # Among endpoint-close rows, require a reference reach to also intersect
    # the line body (buffered at half the parallel threshold).
    cands = gpd.GeoDataFrame(
        {'_row': list(endpoint_close),
         'geometry': [geoms[i].buffer(max_distance_parallel / 2)
                      for i in endpoint_close]},
        crs=network.crs,
    )
    sj = gpd.sjoin(cands, ref_geom.reset_index(drop=True),
                   how='inner', predicate='intersects')
    body_covered = set(sj['_row'].unique())

    keep = [i for i in range(n) if i not in (endpoint_close & body_covered)]
    return network.iloc[keep].copy().reset_index(drop=True)


def orient_merit_reaches(merit, progress=True):
    """Orient MERIT reach geometries so coordinates run upstream → downstream.

    MERIT encodes flow topology via NextDownID (downstream COMID) and up1–up4
    (upstream COMIDs). After this function, coords[-1] of every reach is its
    downstream endpoint — the convention assumed by strahler_igraph.

    Algorithm
    ---------
    1. Build a directed adjacency map from NextDownID / up1–up4.
    2. Identify outlet reaches (NextDownID not present in this tile).
    3. Orient outlets: the upstream endpoint is the one nearest to any
       upstream neighbor (distance to either endpoint of the neighbor,
       so the check is independent of the neighbor's orientation).
    4. BFS upstream from each outlet: for each interior reach, the downstream
       endpoint is the one nearest to the already-confirmed upstream endpoint
       of its downstream neighbor — no distance threshold, exact topology match.
    5. Any reaches not reachable via BFS (disconnected tile fragments) fall
       back to the distance-to-neighbor heuristic.
    """
    comid_to_idx = {int(row['COMID']): idx for idx, row in merit.iterrows()}
    n = len(merit)

    # Build per-reach adjacency (integer row indices, not COMIDs)
    dn_of  = {}   # idx -> downstream idx
    ups_of = {}   # idx -> [upstream idx, ...]

    for idx, row in tqdm(merit.iterrows(), total=n,
                         desc="  building MERIT topology", disable=not progress):
        nxt = int(row['NextDownID'])
        if nxt and nxt in comid_to_idx:
            dn_of[idx] = comid_to_idx[nxt]
        for col in ('up1', 'up2', 'up3', 'up4'):
            uid = int(row[col])
            if uid and uid in comid_to_idx:
                ups_of.setdefault(idx, []).append(comid_to_idx[uid])

    # Pre-extract (x, y) endpoints — ignore Z so distance checks stay 2-D
    first_pt = [list(g.coords)[0][:2] for g in merit.geometry]
    last_pt  = [list(g.coords)[-1][:2] for g in merit.geometry]

    def sq_dist(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def min_sq_dist_to_reach(pt, ref_idx):
        """Minimum squared distance from pt to either endpoint of reach ref_idx."""
        return min(sq_dist(pt, first_pt[ref_idx]),
                   sq_dist(pt, last_pt[ref_idx]))

    # flipped[idx] = True  →  reverse this reach's coordinates
    flipped = [False] * n

    # --- Step 1: orient outlet reaches ---
    # An outlet has no downstream neighbor in the dataset.  Its upstream
    # endpoint is whichever end is nearest to an upstream neighbor.
    outlets = [idx for idx in range(n) if idx not in dn_of]
    for idx in outlets:
        up_idxs = ups_of.get(idx, [])
        if not up_idxs:
            continue  # true headwater island — no topology info, leave as-is
        d_first = min(min_sq_dist_to_reach(first_pt[idx], u) for u in up_idxs)
        d_last  = min(min_sq_dist_to_reach(last_pt[idx],  u) for u in up_idxs)
        # upstream end should be first; if last is closer to upstream → flip
        if d_last < d_first:
            flipped[idx] = True

    # --- Step 2: BFS upstream, orienting each reach from its downstream neighbor ---
    visited = set(outlets)
    queue = deque()
    for outlet_idx in outlets:
        for up_idx in ups_of.get(outlet_idx, []):
            if up_idx not in visited:
                queue.append((up_idx, outlet_idx))

    with tqdm(total=n, desc="  orienting MERIT reaches",
              disable=not progress) as pbar:
        pbar.update(len(outlets))
        while queue:
            idx, dn_idx = queue.popleft()
            if idx in visited:
                continue
            visited.add(idx)
            pbar.update(1)

            # The upstream endpoint of dn_idx (after its own orientation) is
            # the point that idx's downstream end connects to.
            #   flipped[dn_idx]=False → new first_pt = old first_pt (upstream end)
            #   flipped[dn_idx]=True  → new first_pt = old last_pt  (upstream end)
            dn_upstream_pt = (last_pt[dn_idx] if flipped[dn_idx]
                              else first_pt[dn_idx])

            d_first = sq_dist(first_pt[idx], dn_upstream_pt)
            d_last  = sq_dist(last_pt[idx],  dn_upstream_pt)
            # idx's downstream end should be close to dn_upstream_pt;
            # if first is closer → first is downstream → flip
            if d_first < d_last:
                flipped[idx] = True

            for up_idx in ups_of.get(idx, []):
                if up_idx not in visited:
                    queue.append((up_idx, idx))

    # --- Step 3: distance fallback for disconnected fragments ---
    for idx in range(n):
        if idx in visited:
            continue
        dn_idx = dn_of.get(idx)
        up_idxs = ups_of.get(idx, [])
        if dn_idx is not None:
            d_first = min_sq_dist_to_reach(first_pt[idx], dn_idx)
            d_last  = min_sq_dist_to_reach(last_pt[idx],  dn_idx)
            if d_first < d_last:
                flipped[idx] = True
        elif up_idxs:
            d_first = min(min_sq_dist_to_reach(first_pt[idx], u) for u in up_idxs)
            d_last  = min(min_sq_dist_to_reach(last_pt[idx],  u) for u in up_idxs)
            if d_last < d_first:
                flipped[idx] = True

    n_flipped = sum(flipped)
    print(f"  flipped {n_flipped}/{n} MERIT reaches", flush=True)

    if n_flipped == 0:
        return merit

    result = merit.copy()
    result['geometry'] = [
        LineString(list(g.coords)[::-1]) if flipped[i] else g
        for i, g in enumerate(merit.geometry)
    ]
    return result


def find_downstream_dangles(degsub_filtered, progress=True):
    """Return row indices whose downstream endpoint (last coord) does not connect
    to any other reach's upstream endpoint (first coord) within GRAPH_SNAP.
    These are the reaches that need to snap onto MERIT.
    """
    def snapped(c):
        return (GRAPH_SNAP * round(c[0] / GRAPH_SNAP),
                GRAPH_SNAP * round(c[1] / GRAPH_SNAP))

    n = len(degsub_filtered)
    upstream_set = set()
    for _, row in tqdm(degsub_filtered.iterrows(), total=n,
                       desc="  building upstream index", disable=not progress):
        g = row.geometry
        if g and not g.is_empty:
            upstream_set.add(snapped(g.coords[0]))

    return [
        idx for idx, row in tqdm(degsub_filtered.iterrows(), total=n,
                                 desc="  finding dangles", disable=not progress)
        if row.geometry and not row.geometry.is_empty
        and snapped(row.geometry.coords[-1]) not in upstream_set
    ]


def find_floating_terminals(dem_gdf, merit_high_gdf, progress=True):
    """Return indices of terminal DEM reaches that cannot route downstream to MERIT.

    Builds the directed DEM graph and propagates MERIT-reachability backward
    from MERIT endpoint nodes: a node is reachable if going downstream from it
    eventually reaches a MERIT endpoint.  Returns terminal reaches (no downstream
    DEM neighbor) whose downstream node is still not reachable — snapping them
    bridges entire floating components that per-reach dangle detection misses
    because the component's interior reaches look connected to each other.
    """
    def snapped(c):
        return (GRAPH_SNAP * round(c[0] / GRAPH_SNAP),
                GRAPH_SNAP * round(c[1] / GRAPH_SNAP))

    up_node = {}
    dn_node = {}
    for idx, row in tqdm(dem_gdf.iterrows(), total=len(dem_gdf),
                         desc="  DEM reachability graph", disable=not progress):
        g = row.geometry
        if g is None or g.is_empty:
            continue
        u = snapped(g.coords[0])
        v = snapped(g.coords[-1])
        if u != v:
            up_node[idx] = u
            dn_node[idx] = v

    # Seed reachable set with all MERIT endpoint nodes
    reachable = set()
    for _, row in merit_high_gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        reachable.add(snapped(g.coords[0]))
        reachable.add(snapped(g.coords[-1]))

    # Propagate backward: if a reach's downstream node is reachable, so is its upstream
    changed = True
    while changed:
        changed = False
        for idx in up_node:
            v, u = dn_node[idx], up_node[idx]
            if v in reachable and u not in reachable:
                reachable.add(u)
                changed = True

    # Terminal floating reaches: no downstream DEM neighbor and not MERIT-reachable
    all_up_nodes = set(up_node.values())
    return [
        idx for idx in dn_node
        if dn_node[idx] not in reachable and dn_node[idx] not in all_up_nodes
    ]


def snap_to_merit(degsub_filtered, merit_high, dangle_idxs, max_snap_distance,
                  progress=True):
    """Snap downstream endpoints of dangling DEM reaches onto high-order MERIT reaches.

    For each dangling reach:
      - If the DEM line geometrically crosses a MERIT reach: trim the DEM reach
        at the last crossing point (closest to its downstream end).
      - Otherwise: extend the downstream endpoint to the nearest point on MERIT.

    Junction points on MERIT are recorded so those reaches can be split there,
    exposing the junction as a graph node for the Strahler algorithm.

    Returns
    -------
    result         : GeoDataFrame — modified copy of degsub_filtered
    junction_points: dict         — {merit_index: [Point, ...]}
    """
    if not dangle_idxs:
        return degsub_filtered.copy(), {}

    merit_sj = merit_high[['geometry']].copy().reset_index()
    merit_sj = merit_sj.rename(columns={'index': '_merit_orig_idx'})

    dangle_geoms = [Point(degsub_filtered.loc[i, 'geometry'].coords[-1])
                    for i in dangle_idxs]
    dangle_gdf = gpd.GeoDataFrame(
        {'_dem_idx': dangle_idxs, 'geometry': dangle_geoms},
        crs=degsub_filtered.crs,
    )
    nearest = (gpd.sjoin_nearest(dangle_gdf, merit_sj, how='left', distance_col='_dist')
               .drop_duplicates(subset='_dem_idx'))

    result = degsub_filtered.copy()
    junction_points = {}

    for _, nr in tqdm(nearest.iterrows(), total=len(nearest),
                      desc="  snapping to MERIT", disable=not progress):
        dem_idx = int(nr['_dem_idx'])
        dist = float(nr['_dist'])
        merit_orig_idx = int(nr['_merit_orig_idx'])

        if dist > max_snap_distance:
            continue

        dem_line = result.loc[dem_idx, 'geometry']
        merit_line = merit_high.loc[merit_orig_idx, 'geometry']
        downstream_pt = Point(dem_line.coords[-1])

        if dist <= GRAPH_SNAP:
            snap_pt = merit_line.interpolate(merit_line.project(downstream_pt))
            junction_points.setdefault(merit_orig_idx, []).append(snap_pt)
            continue

        intersection = dem_line.intersection(merit_line)
        if not intersection.is_empty and intersection.geom_type in ('Point', 'MultiPoint'):
            pts = (list(intersection.geoms)
                   if intersection.geom_type == 'MultiPoint' else [intersection])
            snap_pt = max(pts, key=lambda p: dem_line.project(p))
            cut_dist = dem_line.project(snap_pt)
            if cut_dist > GRAPH_SNAP:
                result.at[dem_idx, 'geometry'] = substring(dem_line, 0, cut_dist)
        else:
            snap_pt = merit_line.interpolate(merit_line.project(downstream_pt))
            dem_coords = list(dem_line.coords)
            ndim = len(dem_coords[0])  # match DEM line's 2D or 3D dimensionality
            result.at[dem_idx, 'geometry'] = LineString(
                dem_coords + [snap_pt.coords[0][:ndim]]
            )

        junction_points.setdefault(merit_orig_idx, []).append(snap_pt)

    return result, junction_points


def split_merit_at_junctions(merit_high, junction_points, progress=True):
    """Split high-order MERIT reaches at tributary junction points.

    The Strahler graph only sees line endpoints as nodes, so each MERIT reach
    must be split wherever a DEM tributary meets it.

    Every reach in merit_high is guaranteed to appear in the output — either as
    one or more split segments or, if splitting would drop the reach entirely
    (all segments too short, or any unexpected error), as the original unsplit row.
    """
    rows = []
    represented = set()

    for idx, row in tqdm(merit_high.iterrows(), total=len(merit_high),
                         desc="  splitting MERIT", disable=not progress):
        pts = junction_points.get(idx)
        if not pts:
            rows.append(row)
            represented.add(idx)
            continue

        merit_line = row.geometry
        # Keep junction points anywhere along the reach except exactly at the
        # endpoints (d == 0 or d == length would produce a zero-length segment).
        distances = sorted({
            d for d in (merit_line.project(pt) for pt in pts)
            if 0 < d < merit_line.length
        })

        if not distances:
            rows.append(row)
            represented.add(idx)
            continue

        breakpoints = [0.0] + distances + [merit_line.length]
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            if end <= start:  # floating-point safety only
                continue
            seg_row = row.copy()
            seg_row['geometry'] = substring(merit_line, start, end)
            rows.append(seg_row)
        represented.add(idx)

    # Safety net: catch any reach that was skipped (e.g. unexpected geometry error).
    missing = set(merit_high.index) - represented
    if missing:
        print(f"  [warn] {len(missing)} MERIT reaches not represented; adding unsplit",
              flush=True)
        for i in missing:
            rows.append(merit_high.loc[i])

    # Explicit geometry='geometry' avoids silent null-geometry rows when
    # building a GeoDataFrame from a plain list of Series.
    return gpd.GeoDataFrame(
        pd.DataFrame(rows).reset_index(drop=True),
        geometry='geometry', crs=merit_high.crs,
    )


def strahler_igraph(streams_gdf):
    """Assign Strahler stream order to a directed network of LineStrings.

    Nodes are identified by snapping endpoint coordinates to the nearest 5 m
    grid (matching GRAPH_SNAP). Edges point upstream → downstream following
    the coordinate order of each geometry.

    A `strahler` column (Int32, NA for degenerate/cycle-removed edges) is
    added to a copy of *streams_gdf* and returned.
    """
    nodes, edges, idxs = {}, [], []
    for idx, geom in streams_gdf.geometry.items():
        if geom is None or geom.is_empty:
            continue
        c = list(geom.coords)
        u = tuple(GRAPH_SNAP * round(x / GRAPH_SNAP) for x in c[0])
        v = tuple(GRAPH_SNAP * round(x / GRAPH_SNAP) for x in c[-1])
        if u == v:
            continue
        for k in (u, v):
            nodes.setdefault(k, len(nodes))
        edges.append((nodes[u], nodes[v]))
        idxs.append(idx)

    g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    g.es["idx"] = idxs

    if not g.is_dag():
        cyc = g.feedback_arc_set(method="ip")
        print(f"  [warn] removing {len(cyc)} cycle-causing edges", flush=True)
        removed = {g.es[e]["idx"] for e in cyc}
        g.delete_edges(cyc)
        idxs = g.es["idx"]

    topo = g.topological_sorting(mode="out")
    edge_order = [0] * g.ecount()
    for v in topo:
        in_es = g.incident(v, mode="in")
        if not in_es:
            o = 1
        else:
            ords = [edge_order[e] for e in in_es]
            mx = max(ords)
            o = mx + 1 if ords.count(mx) >= 2 else mx
        for e in g.incident(v, mode="out"):
            edge_order[e] = o

    out = streams_gdf.copy()
    out["strahler"] = out.index.map(dict(zip(idxs, edge_order))).astype("Int32")
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _merge_one(dem_network_path, merit_network_path, output_path,
               max_distance_parallel, max_snap_distance, order_threshold=4,
               tag="", progress=True):
    label = f"[{tag}] " if tag else ""

    print(f"{label}loading...", flush=True)
    degsub = gpd.read_file(dem_network_path).to_crs(ALBERS).reset_index(drop=True)
    merit = (gpd.read_file(merit_network_path, layer='intersection',)
             .to_crs(ALBERS)
             .explode(index_parts=False)   # MultiLineString → LineString
             .reset_index(drop=True))
    print(f"{label}DEM: {len(degsub)} reaches | MERIT: {len(merit)} reaches "
          f"(order range {merit['order'].min()}–{merit['order'].max()})", flush=True)

    # Orient MERIT before splitting so NextDownID lookups span both halves
    print(f"{label}orienting MERIT reaches...", flush=True)
    merit = orient_merit_reaches(merit, progress=progress)

    # Split MERIT by order threshold:
    #   merit_high: replaces parallel DEM reaches; retained in output
    #   merit_low:  excluded entirely; DEM reaches are retained instead
    merit_high = merit[merit['order'] > order_threshold].copy().reset_index(drop=True)
    merit_low_n = len(merit) - len(merit_high)
    print(f"{label}MERIT split: {len(merit_high)} high-order (>{order_threshold}), "
          f"{merit_low_n} low-order (≤{order_threshold}, excluded)", flush=True)

    # Remove DEM reaches parallel to high-order MERIT; the overlap guard prevents
    # removing wetland mainstem reaches where MERIT covers only the endpoints.
    print(f"{label}filtering DEM reaches parallel to high-order MERIT...", flush=True)
    degsub_f = filter_parallel_reaches(degsub, merit_high, max_distance_parallel,
                                       require_overlap=True)
    print(f"{label}kept {len(degsub_f)}/{len(degsub)} DEM reaches", flush=True)

    # Iterative dangle snap: repeat until the dangle set stops changing.
    # Each pass can expose new dangles when snapping one reach changes the
    # topology visible to its upstream neighbors.
    all_jpts = {}
    prev_dangle_set = None
    for iteration in range(1, 6):
        dangle_idxs = find_downstream_dangles(degsub_f, progress=progress)
        dangle_set = set(dangle_idxs)
        print(f"{label}dangle iter {iteration}: {len(dangle_idxs)} endpoints", flush=True)
        if not dangle_idxs or dangle_set == prev_dangle_set:
            break
        prev_dangle_set = dangle_set
        degsub_f, new_jpts = snap_to_merit(degsub_f, merit_high, dangle_idxs,
                                           max_snap_distance, progress=progress)
        for k, pts in new_jpts.items():
            all_jpts.setdefault(k, []).extend(pts)
        n_jcts = sum(len(v) for v in new_jpts.values())
        print(f"{label}  snapped {n_jcts} junctions on {len(new_jpts)} MERIT reaches",
              flush=True)
        if not new_jpts:
            break

    # Reachability pass: find DEM reaches whose entire downstream chain never
    # reaches a MERIT node — floating components that dangle iteration misses
    # because interior reaches look connected to each other.  Snap their terminal
    # reaches with a larger tolerance to bridge wider wetland gaps.
    print(f"{label}checking for floating components...", flush=True)
    float_idxs = find_floating_terminals(degsub_f, merit_high, progress=progress)
    if float_idxs:
        print(f"{label}{len(float_idxs)} floating terminals; snapping "
              f"(up to {max_snap_distance * 2:.0f} m)...", flush=True)
        degsub_f, new_jpts = snap_to_merit(degsub_f, merit_high, float_idxs,
                                           max_snap_distance * 2, progress=progress)
        for k, pts in new_jpts.items():
            all_jpts.setdefault(k, []).extend(pts)
        n_jcts = sum(len(v) for v in new_jpts.values())
        print(f"{label}  snapped {n_jcts} floating junctions on {len(new_jpts)} MERIT reaches",
              flush=True)
    else:
        print(f"{label}no floating components found", flush=True)

    # Split MERIT reaches at all accumulated junction points
    merit_split = split_merit_at_junctions(merit_high, all_jpts, progress=progress)
    print(f"{label}MERIT: {len(merit_high)} → {len(merit_split)} reaches after splitting",
          flush=True)

    degsub_f = degsub_f.copy()
    degsub_f['from_dem'] = True
    merit_split = merit_split.copy()
    merit_split['from_dem'] = False

    combined = gpd.GeoDataFrame(
        pd.concat([degsub_f, merit_split], ignore_index=True),
        crs=degsub.crs,
    )

    print(f"{label}assigning Strahler orders...", flush=True)
    combined = strahler_igraph(combined)
    max_order = combined["strahler"].max()
    print(f"{label}orders 1–{max_order}, "
          f"{combined['strahler'].isna().sum()} reaches unordered", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_path, driver="GPKG")
    print(f"{label}written: {len(combined)} total reaches → {output_path}", flush=True)

    csv_path = Path(output_path).with_suffix('.csv')
    (combined.assign(length_m=combined.geometry.length)
     .groupby('strahler', dropna=True)['length_m']
     .agg(reach_count='count', total_length_m='sum')
     .reset_index()
     .to_csv(csv_path, index=False))
    print(f"{label}written: order summary → {csv_path}", flush=True)

    return len(combined)


def _process_one_column(args):
    col, huc12_dir, merit_path, out_dir, max_dist_par, max_snap, order_threshold = args
    paths = sorted(Path(huc12_dir).glob(f"*/streams_{col}.gpkg"))
    if not paths:
        print(f"[{col}] no HUC12 files found, skipping", flush=True)
        return col, 0

    print(f"[{col}] merging {len(paths)} HUC12 files...", flush=True)
    gdfs = [gpd.read_file(p).to_crs(ALBERS) for p in tqdm(paths, desc=f"[{col}] reading HUC12 files")]
    dem_tmp = Path(out_dir) / f"_dem_tmp_{col}.gpkg"
    gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=ALBERS).to_file(
        dem_tmp, driver="GPKG"
    )

    out_path = Path(out_dir) / f"merged_{col}.gpkg"
    n = _merge_one(str(dem_tmp), merit_path, str(out_path),
                   max_dist_par, max_snap, order_threshold,
                   tag=col, progress=False)
    dem_tmp.unlink(missing_ok=True)
    return col, n


def main_single(dem_network_path, merit_network_path, output_path,
                max_distance_parallel, max_snap_distance, order_threshold):
    _merge_one(dem_network_path, merit_network_path, output_path,
               max_distance_parallel, max_snap_distance, order_threshold)


def main_batch(huc12_dir, merit_path, out_dir, n_workers,
               max_distance_parallel, max_snap_distance, order_threshold):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    job_args = [
        (col, huc12_dir, merit_path, out_dir,
         max_distance_parallel, max_snap_distance, order_threshold)
        for col in THRESH_COLS
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_one_column, a): a[0] for a in job_args}
        with tqdm(total=len(THRESH_COLS), desc="columns complete") as pbar:
            for f in as_completed(futs):
                col = futs[f]
                try:
                    _, n = f.result()
                    pbar.set_postfix_str(f"last: {col} ({n} reaches)")
                except Exception as exc:
                    pbar.set_postfix_str(f"{col} FAILED: {exc}")
                pbar.update(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- single-file mode ---
    sg = ap.add_argument_group("single-file mode")
    sg.add_argument("--dem-network-path", type=str)
    sg.add_argument("--merit-network-path", type=str)
    sg.add_argument("--output-path", type=str)

    # --- batch mode ---
    bg = ap.add_argument_group("batch mode (parallel columns)")
    bg.add_argument("--huc12-dir", type=str, default=None,
                    help="Directory containing {id}/streams_{col}.gpkg files")
    bg.add_argument("--merit-path", type=str, help="Path to MERIT Hydro GeoPackage")
    bg.add_argument("--out-dir", type=str, default=None,
                    help="Output directory for merged_{col}.gpkg files")
    bg.add_argument("--col", type=str, default=None, choices=THRESH_COLS,
                    help="Process a single threshold column (for SLURM array jobs). "
                         "Omit to process all columns with --n-workers.")
    bg.add_argument("--n-workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)),
                    help="Number of parallel workers when --col is not set (default: all local CPUs)")

    # --- shared ---
    ap.add_argument("--max-distance-parallel", type=float, default=400.0,
                    help="Distance threshold for parallel-reach filtering (m)")
    ap.add_argument("--max-snap-distance", type=float, default=500.0,
                    help="Max distance to snap a DEM endpoint to MERIT (m)")
    ap.add_argument("--order-threshold", type=int, default=4,
                    help="MERIT reaches with order > this replace parallel DEM reaches; "
                         "reaches at or below are excluded (default: 4)")

    args = ap.parse_args()

    if args.dem_network_path:
        if not args.merit_network_path or not args.output_path:
            ap.error("single-file mode requires --dem-network-path, "
                     "--merit-network-path, --output-path")
        main_single(
            dem_network_path=args.dem_network_path,
            merit_network_path=args.merit_network_path,
            output_path=args.output_path,
            max_distance_parallel=args.max_distance_parallel,
            max_snap_distance=args.max_snap_distance,
            order_threshold=args.order_threshold,
        )
    else:
        if not args.merit_path or not args.huc12_dir or not args.out_dir:
            ap.error("batch mode requires --merit-path, --huc12-dir, and --out-dir")
        if args.col:
            _process_one_column((
                args.col, args.huc12_dir, args.merit_path, args.out_dir,
                args.max_distance_parallel, args.max_snap_distance, args.order_threshold,
            ))
        else:
            main_batch(
                huc12_dir=args.huc12_dir,
                merit_path=args.merit_path,
                out_dir=args.out_dir,
                n_workers=args.n_workers,
                max_distance_parallel=args.max_distance_parallel,
                max_snap_distance=args.max_snap_distance,
                order_threshold=args.order_threshold,
            )
