import geopandas as gpd
from matplotlib import pyplot as plt

grwl = gpd.read_file(r"C:\Users\dego\Desktop\ms_grwl.gpkg")\
    .head(1)\
    .to_crs(3857)

print(grwl.crs)

for seg in grwl:
    midpoint = gpd.GeoSeries[seg].centroid
    

fig, ax = plt.subplots()
seg.plot(ax=ax)
midpoint.plot(ax=ax, color='tab:orange')
plt.show()