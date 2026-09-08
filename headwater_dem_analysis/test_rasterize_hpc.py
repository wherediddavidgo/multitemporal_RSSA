import geopandas as gpd
import rioxarray
from geocube.api.core import make_geocube
import xarray as xr
import numpy as np
import datetime

print(f'Beginning at {datetime.datetime.now()}')

vector_data = gpd.read_file('/home/dego/headwater_network_extraction/catchment_geodata/sheds_w_ahthresh_corr.gpkg').head(500)
print(f'Vector read at {datetime.datetime.now()}')
dir_raster = rioxarray.open_rasterio('/home/dego/headwater_network_extraction/NHD_fdr/nhd_fdr_mos_arc_corr.tif', chunks=True)
acc_raster = rioxarray.open_rasterio('/home/dego/headwater_network_extraction/NHD_fac/nhd_fac_burn.tif', chunks=True)
print(f'Raster read at {datetime.datetime.now()}')

dir_clipped = dir_raster.rio.clip(vector_data.geometry.values, vector_data.crs, drop=True)
acc_clipped = acc_raster.rio.clip(vector_data.geometry.values, vector_data.crs, drop=True)

out_grid = make_geocube(vector_data=vector_data,
                        like=dir_clipped,
                        measurements=['ah05', 'ah15', 'ah25', 'ah35', 'ah45', 'ah55', 'ah65', 'ah75', 'ah85', 'ah95'])
print(f'Grid made at {datetime.datetime.now()}')

out_grid.rio.to_raster('/home/dego/headwater_network_extraction/test_rasterize_catchments.tif')
print(f'Rasterize complete at {datetime.datetime.now()}')


mask = rioxarray.open_rasterio('/home/dego/headwater_network_extraction/test_rasterize_catchments.tif')
print(mask)

masked = xr.where(acc_clipped >= mask.sel(1), np.nan, 1)

masked.rio.to_raster('/home/dego/headwater_network_extraction/test_extract.tif')