import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pyproj
pyproj.network.set_network_enabled(False)

from tqdm import tqdm
from collections import deque
from shapely import LineString, force_2d
import igraph as ig
GRAPH_SNAP = 10.0

CHUNK_SIZE = 100_000

print('Starting')
nhd = gpd.read_file("/home/dego/headwater_network_extraction/catchment_geodata/NHDPlus_H_National_Release_2_GDB.gdb/NHDPlus_H_National_Release_2_GDB.gdb", 
                    layer='NetworkNHDFlowline', 
                    columns=['streamorde', 'fromnode', 'tonode', 'hydroseq', 'lengthkm', 'geometry', 'divdasqkm', 'fcode', 'ftype'])
nhd = nhd.loc[(np.isfinite(nhd['hydroseq'])) & (nhd['hydroseq'] > 1)]
nhd['geometry'] = nhd['geometry'].apply(force_2d)
nhd = nhd.loc[(nhd['hydroseq'] > 1) & (np.isfinite(nhd['hydroseq']))]
nhd = nhd.to_crs(5070)
print("NHD read complete")

catchments = gpd.read_file("/home/dego/headwater_network_extraction/catchment_geodata/sheds_w_nhd_merit_atts_ahthresh.gpkg")
catchments = catchments.to_crs(5070)
print('Catchment read complete')

nhd_j = gpd.sjoin(nhd, catchments, how='left', predicate='covered_by')
print('Join complete')



def strahler_igraph(streams_gdf):
    """Assign Strahler stream order to a directed network of LineStrings.
    Nodes are identified by snapping endpoint coordinates to the nearest 5 m
    grid (matching GRAPH_SNAP). Edges point upstream → downstream following
    the coordinate order of each geometry.
    A `strahler` column (Int32, NA for degenerate/cycle-removed edges) is
    added to a copy of *streams_gdf* and returned.
    """
    nodes, edges, idxs = {}, [], []

    def _norm(g):
        if g.geom_type != 'MultiLineString':
            return g
        from shapely.ops import linemerge
        m = linemerge(g)
        return m if m.geom_type == 'LineString' else list(m.geoms)[0]

    geoms = [_norm(g) for g in streams_gdf.geometry]
    streams_gdf = streams_gdf.copy()
    streams_gdf['geometry'] = geoms

    for idx, geom in streams_gdf['geometry'].items():
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
        g.delete_edges(cyc)
        idxs = g.es["idx"]

    # ------------------------------------------------------------------
    # Label edges with their "braid group": edges that descend from the
    # same split node (out-degree >= 2) share a group ID. At a merge
    # node, all incoming edges of the same group count as ONE tributary.
    # ------------------------------------------------------------------
    braid_group = {}
    topo = g.topological_sorting(mode="out")

    for v in topo:
        out_es = g.incident(v, mode="out")
        in_es  = g.incident(v, mode="in")

        arriving = frozenset()
        for e in in_es:
            arriving |= braid_group.get(e, frozenset())

        if len(out_es) >= 2:
            split_tag = frozenset(arriving | {v})
            for e in out_es:
                braid_group[e] = split_tag
        else:
            for e in out_es:
                braid_group[e] = arriving

    # ------------------------------------------------------------------
    # Strahler assignment (topological, upstream -> downstream)
    # At each node, group incoming edges by braid group and count only
    # one representative per group (use the max order within the group).
    # ------------------------------------------------------------------
    edge_order = [0] * g.ecount()

    for v in topo:
        in_es = g.incident(v, mode="in")

        if not in_es:
            o = 1
        else:
            groups = {}
            independent = []

            for e in in_es:
                grp = braid_group.get(e, frozenset())
                if grp:
                    groups.setdefault(grp, []).append(edge_order[e])
                else:
                    independent.append(edge_order[e])

            representative_orders = [max(v_list) for v_list in groups.values()]
            representative_orders += independent

            mx = max(representative_orders)
            count_mx = representative_orders.count(mx)
            o = mx + 1 if count_mx >= 2 else mx

        for e in g.incident(v, mode="out"):
            edge_order[e] = o

    out = streams_gdf.copy()
    out["strahler"] = out.index.map(
        dict(zip(idxs, edge_order))
    ).astype("Int32")
    return out


outcols = ['lengthkm', 'ftype', 'fcode', 'streamorde', 'strahler', 'hydroseq', 'divdasqkm', 'geometry', 'huc12']

order_column = 'strahler' # strahler for dynamic reordering, streamorde for static NHD parameter

lendf = []
thresholds = ['ah95', 'ah85', 'ah75', 'ah65', 'ah55', 'ah45', 'ah35', 'ah25', 'ah15', 'ah05']

for t in tqdm(range(len(thresholds))):
    threshold = thresholds[t]
    dynamic_nhd = nhd_j.copy()
    dynamic_nhd = dynamic_nhd.loc[(dynamic_nhd['divdasqkm'] >= dynamic_nhd[threshold])]
    dynamic_nhd = strahler_igraph(dynamic_nhd)

    print(dynamic_nhd[outcols].head())

    if threshold == 'ah95':
        dynamic_nhd[outcols].to_file('/scratch/dego/miss/nhd_trimmed_ah95.gpkg')
    if threshold == 'ah45':
        dynamic_nhd[outcols].to_file('/scratch/dego/miss/nhd_trimmed_ah45.gpkg')
    if threshold == 'ah05':
        dynamic_nhd[outcols].to_file('/scratch/dego/miss/nhd_trimmed_ah05.gpkg')

    len_per_order = dynamic_nhd.groupby(order_column) \
    .agg(
        lpo = ('lengthkm', lambda x: x.sum())
    )
    len_per_order['Q_decile'] = t
    lendf.append(len_per_order)

lendf = pd.concat(lendf)
lendf.to_csv('/home/dego/headwater_network_extraction/nhd_trim_df.csv')