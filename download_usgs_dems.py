from bmi_topography import Topography
import shapely
import glob
import argparse
import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Script to bulk download 10 m USGS DEM for MS basin extent")
parser.add_argument("--api_key", help='Opentopography API key')
args = parser.parse_args()
api_key = str(args.api_key)

ms_watershed = []
for file in glob.glob('/home/dego/headwater_network_extraction/catchment_geodata/ms_watershed_boundaries/*/Shape/WBDHU2.shp'):
    ms_watershed.append(gpd.read_file(file))

ms_watershed = pd.concat(ms_watershed)

west, south, east, north = ms_watershed.total_bounds

lonser = np.linspace(np.floor(west), np.ceil(east), int(np.ceil(east) - np.floor(west) + 1))
latser = np.linspace(np.floor(south), np.ceil(north), int(np.ceil(north) - np.floor(south) + 1))





for x in tqdm.tqdm(lonser):
    west = x
    east = x + 1
    for y in latser:
        south = y
        north = y + 1

        box = gpd.GeoDataFrame({'geometry': [shapely.box(west, south, east, north)]}, crs=4269)

        if len(gpd.overlay(ms_watershed, box, how='intersection')) > 0:
#             print(west, south, east, north)
#             print(gpd.overlay(ms_watershed, box, how='intersection'))

#             break
# fig, ax = plt.subplots()
# ms_watershed.plot(ax=ax, facecolor='none')
# box.plot(ax=ax, facecolor='tab:orange')
# plt.show()
            params = {'dem_type': 'USGS10m',
                      'west': x,
                      'east': x + 1,
                      'south': y,
                      'north': y + 1,
                      'api_key': api_key,
                      'cache_dir': '/home/dego/headwater_network_extraction/dems'}
            # url = Topography(**params).url
            # print(url)
            dem = Topography(**params).fetch()