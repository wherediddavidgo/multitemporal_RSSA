import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
import pygris
from glob import glob
import pandas as pd
from tqdm import tqdm
from os.path import join
from shapely import Polygon, union_all, snap
from shapely.ops import linemerge
from shapely.strtree import STRtree

# cb_centerlines = gpd.read_file('C:/Users/dego/Documents/local_files/RSSA/Platte_centerlines_masks/vector_centerlines/s2_platte_centerlines.shp')\
#     .set_crs(4326, allow_override=True)\
#     .to_crs(3857)


# geoms = cb_centerlines.geometry.values
# tree = STRtree(geoms)
# tol = 1
# snapped = []
# for geom in tqdm(geoms):
#     env = geom.envelope.buffer(tol)
#     nearby_idxs = tree.query(env)
#     local_union = union_all(geoms[nearby_idxs])
#     snapped.append(snap(geom, local_union, tol))


# cb_centerlines['geometry'] = np.array(snapped, dtype=object)

# merged = (
#     cb_centerlines.groupby("NewSO")
#         .agg({"geometry": lambda g: linemerge(union_all(g.values))})
#         .reset_index()
# )
# merged = merged.set_geometry('geometry', crs=3857)
# merged.to_file('C:/Users/dego/Desktop/cb_centerlines_vis.gpkg', overwrite=True)


# merit = gpd.read_file('C:/Users/dego/Documents/local_files/RSSA/merit_noplatte.gpkg').to_crs(3857)
# glow_pts = gpd.read_file('C:/Users/dego/Documents/local_files/RSSA/glow_pts_on_grwl.gpkg').to_crs(3857)
# COMID_list = glow_pts.COMID.unique().tolist()
# merit_filt = merit.loc[merit.COMID.isin(COMID_list)]
# merit_filt['order'] = merit_filt['order'] + 6

# geoms = merit_filt.geometry.values
# tree = STRtree(geoms)
# tol = 1
# snapped = []
# for geom in tqdm(geoms):
#     env = geom.envelope.buffer(tol)
#     nearby_idxs = tree.query(env)
#     local_union = union_all(geoms[nearby_idxs])
#     snapped.append(snap(geom, local_union, tol))


# merit_filt['geometry'] = np.array(snapped, dtype=object)

# merged = (
#     merit_filt.groupby("order")
#         .agg({"geometry": lambda g: linemerge(union_all(g.values))})
#         .reset_index()
# )
# merged = merged.set_geometry('geometry', crs=3857)
# merged.to_file('C:/Users/dego/Desktop/merit_filt_vis.gpkg', overwrite=True)



cb_cl_vis = gpd.read_file('C:/Users/dego/Desktop/cb_centerlines_vis.gpkg').to_crs(5070)
cb_cl_vis = cb_cl_vis.loc[cb_cl_vis.NewSO <= 9]
platte_wbd = gpd.read_file(r'c:\Users\dego\Documents\local_files\RSSA\watershed_boundaries\platte\watershed_boundary.shp').to_crs(5070).dissolve()
merit_filt = gpd.read_file(r'C:\Users\dego\Documents\local_files\RSSA\merit_on_grwl_filt.gpkg').to_crs(5070)

merit_filt['order'] = merit_filt['order'] + 6
ms_gages = gpd.read_file('C:/Users/dego/Documents/local_files/ms_gages_grwl_filtered.gpkg').to_crs(5070)
platte_gages = gpd.read_file('C:/Users/dego/Documents/local_files/RSSA/platte_gages.gpkg').to_crs(5070)


states = pygris.states().to_crs(5070)

# combine WBDs comprising MS
ws = []
for fp in tqdm(glob(join(r'c:\Users\dego\Documents\local_files\RSSA\watershed_boundaries', '*huc*'))):
    fn = fr'{fp}\Shape\WBDHU2.shp'
    ws.append(gpd.read_file(fn))

ms = pd.concat(ws)\
    .to_crs(5070)\
    .dissolve()

ms['geometry'] = Polygon(ms.geometry.exterior[0])
l, b, r, t = ms.total_bounds
sizes = np.linspace(0.5, 3, 10)
order_size_map = {5: sizes[0],
                  6: sizes[1],
                  7: sizes[2],
                  8: sizes[3],
                  9: sizes[4],
                  10:sizes[5],
                  11:sizes[6],
                  12:sizes[7],
                  13:sizes[8],
                  14:sizes[9]}

cb_cl_vis['linewidth'] = cb_cl_vis['NewSO'].map(order_size_map)
merit_filt['linewidth'] = merit_filt['order'].map(order_size_map)


fig, ax = plt.subplots()
ms.plot(ax=ax, facecolor='none', zorder=2, capstyle='round')
merit_filt.plot(ax=ax, zorder=4, linewidth=merit_filt['linewidth'], capstyle='round', label='Observed by Landsat')


states.plot(ax=ax, facecolor='none', edgecolor='tab:gray', linewidth=0.4, zorder=1)
cb_cl_vis.plot(ax=ax, color='tab:orange', zorder=5, linewidth=cb_cl_vis['linewidth'])
cb_cl_vis.loc[cb_cl_vis.NewSO == 9].plot(ax=ax, color='tab:orange', zorder=5, linewidth=cb_cl_vis['linewidth'], label='Observed by Sentinel-2')
platte_wbd.plot(ax=ax, facecolor='none', zorder=3)
ms_gages.plot(ax=ax, color='black', zorder=6, markersize=3)
platte_gages.plot(ax=ax, color='black', zorder=7, label='Gage', markersize=3)
ax.legend()

ax.set_xlim((l - (r - l) / 8), (r + (r - l) / 8))
ax.set_ylim((b - (t - b) / 8), (t + (t - b) / 8))

ax.set_xticks([])
ax.set_yticks([])


plt.show()

print('Complete')