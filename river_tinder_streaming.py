import pandas as pd
import geopandas as gpd
import numpy as np
# import shapely
# import dask
# from dask.distributed import Client
import xarray as xr
import numpy as np

from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

import rasterio
from rasterio.windows import from_bounds
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.enums import Resampling

from shapely import bounds
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# from rio_cogeo import cogeo
# from google.colab import drive
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap


import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import rasterio
import numpy as np
import glob
import os
import itertools
from datetime import datetime
import sys

sys.path.append(r'C:\Users\dego\AppData\Local\Programs\Python\Python313\Lib\site-packages')
sys.path.append(r'C:\Users\dego\AppData\Local\Programs\Python\Python313\Scripts')
import gdown
from pystac_client import Client as stac_client


def search_stac_by_id(img_id):
    search = stac.search(collections=['sentinel-2-l2a'],
                         ids = [img_id],
                         fields={'include': ['id', 'assets.B03', 'assets.B08', 'assets.SCL', 'bbox', 'properties.datetime']})

    return list(search.items())[0]



def ref_geoms(img, poly_idx, view_geoms, otsu_geoms, effwidth_geoms, centerline_geoms):
    b3_href = img.assets["green"].href
    with rasterio.open(b3_href) as src:
        img_crs = src.crs

    view_geom  = view_geoms.loc[poly_idx].geometry
    otsu_geom  = otsu_geoms.loc[poly_idx].geometry
    eff_geom   = effwidth_geoms.loc[poly_idx].geometry

    t_view  = Transformer.from_crs(view_geoms.crs, img_crs,  always_xy=True).transform
    t_otsu  = Transformer.from_crs(otsu_geoms.crs, img_crs,  always_xy=True).transform
    t_eff   = Transformer.from_crs(effwidth_geoms.crs, img_crs, always_xy=True).transform

    view_src = shp_transform(t_view, view_geom)
    otsu_src = shp_transform(t_otsu, otsu_geom)
    eff_src  = shp_transform(t_eff,  eff_geom)

    # Filter lines by intersection in image CRS, then reproject only those (cheap)
    if centerline_geoms.crs != img_crs:
        cl_img = centerline_geoms.to_crs(img_crs)
    else:
        cl_img = centerline_geoms
    hits = list(cl_img.sindex.query(otsu_src, predicate="intersects"))
    lines_in_bound = cl_img.iloc[hits].copy()

    return view_src, otsu_src, eff_src, lines_in_bound

def dn_to_reflectance(band):
    return np.float32(band) * 1e-4


def normalized_difference(b1, b2):
    denominator = b1 + b2
    numerator = b1 - b2
    return numerator / np.where(denominator != 0, denominator, np.nan)



def process_image(img, view_geom, otsu_geom):
    b3_href = img.assets['green'].href
    b8_href = img.assets['nir'].href
    scl_href = img.assets['scl'].href

    l, b, r, t = view_geom.bounds


    with rasterio.open(b3_href) as b3_src:
        wwindow = rasterio.windows.from_bounds(l, b, r, t, b3_src.transform).round_offsets().round_lengths()
        # cwindow = rasterio.windows.from_bounds(l, b, r, t, scl_src.transform)

        wwindow_transform = b3_src.window_transform(wwindow)    
        # cwindow_transform = scl_src.window_transform(cwindow)

        b3v = b3_src.read(1, window=wwindow, masked=True)
        h, w = wwindow.height, wwindow.width

    with rasterio.open(b8_href) as b8_src:
        b8v = b8_src.read(1, window=wwindow, masked=True)

    with rasterio.open(scl_href) as scl_src:
        sclv = scl_src.read(1, window=rasterio.windows.from_bounds(l, b, r, t, transform=scl_src.transform).round_offsets().round_lengths(), out_shape=(1, h, w), resampling=Resampling.nearest)

    sclv = np.ma.MaskedArray(sclv, mask=np.zeros_like(sclv, dtype=bool))
    b3v = dn_to_reflectance(b3v)
    b8v = dn_to_reflectance(b8v)
    
    ndwi_v = normalized_difference(b3v, b8v)
    # print(ndwi_v.min())
    # print(ndwi_v.max())

    otsu_geom_mask = geometry_mask([otsu_geom], out_shape=(h, w), transform=wwindow_transform, invert=True)

    ndwi_o = np.ma.array(ndwi_v, mask=~otsu_geom_mask).compressed()
    b8o = np.ma.array(b8v, mask=~otsu_geom_mask).compressed()

    if ndwi_o.size >= 10:
        ndwi_threshold = threshold_otsu(ndwi_o)
        nir_threshold = threshold_otsu(b8o)
    else:
        ndwi_threshold = 1
        nir_threshold = 1

    # print(ndwi_threshold)
    # print(nir_threshold)


    wmask = (ndwi_v >= ndwi_threshold) & (b8v <= nir_threshold)

    cloudmask = np.isin(sclv, [7, 8, 9]).astype('uint8')
    snowmask = (sclv == 11).astype('uint8')

    return ndwi_v, wmask, cloudmask, snowmask, wwindow_transform


def identify_river(wmask, lines, window_trans):
    # vcl_reproj = vector_centerlines.to_crs(img_crs)
    # line_nos_in_bounds = list(vcl_reproj.sindex.query(otsu_geom, predicate='intersects'))
    # lines_in_bound = vcl_reproj.iloc[line_nos_in_bounds]
    h, w = wmask.shape
    wbool = wmask.filled(0) > 0
    rbuffs = lines.copy().buffer(5)
    shapes = ((geom, 1) for geom in rbuffs)
    river_seed = rasterize(
        shapes=shapes,
        out_shape= (h, w),
        transform=window_trans,
        fill=0,
        dtype='uint8',
        all_touched=True
    )
    structure = np.ones((3, 3), dtype=bool)
    river_mask = ndi.binary_propagation(input=river_seed * wbool, mask=wbool, structure=structure)
    return river_mask


def GENERATE_MASKS(img_id, poly_idx, squares, pills, circles, vector_centerlines):
    image = search_stac_by_id(img_id)

    square, pill, circle_geom, lines = ref_geoms(image, poly_idx, squares, pills, circles, vector_centerlines)

    ndwi_v, wmask, cloudmask, snowmask, wwindow_transform = process_image(image, square, pill)

    rmask = identify_river(wmask, lines, wwindow_transform)

    # circle_mask, n_river, n_cloud, n_snow = count_pixels(rmask, cloudmask, snowmask, wwindow_transform, circle_geom)


    return ndwi_v, rmask, cloudmask, snowmask, wwindow_transform, circle_geom
    

    

def count_pixels(rmask, cloudmask, snowmask, transform, circle):
    circle_mask = rasterize([circle], out_shape = rmask.shape, transform=transform, dtype='uint8', all_touched=True) == 1

    n_river = (circle_mask & rmask).sum()
    n_cloud = (circle_mask & cloudmask).sum()
    n_snow = (circle_mask & snowmask).sum()

    return circle_mask, n_river, n_cloud, n_snow
    

class TifViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TIF Viewer")

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # # Button to load new image
        # self.btn_load = tk.Button(root, text="Load Images", command=self.load_image)
        # self.btn_load.pack()

        # test button
        self.btn_like = tk.Button(root, text='Like', command=lambda: [self.like(), self.show_plot()])
        self.btn_like.pack()

        self.btn_dislike = tk.Button(root, text='Dislike', command=lambda: [self.dislike(), self.show_plot()])
        self.btn_dislike.pack()

        self.btn_undo = tk.Button(root, text='Undo', command=lambda: [self.undo(), self.show_plot()])
        self.btn_undo.pack()

        self.btn_save = tk.Button(root, text='Save', command=self.save)
        self.btn_save.pack()

        self.root.bind("<Right>", lambda event: [self.like(), self.show_plot()])

        self.root.bind("<Left>", lambda event: [self.dislike(), self.show_plot()])
        self.root.bind("<Control-x>", lambda event: [self.undo(), self.show_plot()])
        self.root.bind("<Control-s>", lambda event: self.save())

        # test label
        self.lbl_iid = tk.Label(root, text=None)
        self.lbl_iid.pack()

        self.ndwi_artist = None  # will hold the AxesImage object
        self.wm_artist = None
        self.cm_artist = None


        rm_cdict = {
            'red':   [[0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0]],
            'green': [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]],
            'blue':  [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]],
            'alpha': [[0.0, 0.0, 0.0], 
                      [1.0, 0.35, 1.0]]
        }
        self.rm_cmap = LinearSegmentedColormap('custom_cmap', segmentdata=rm_cdict)

        cm_cdict = {
            'red':   [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]],
            'green': [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]],
            'blue':  [[0.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0]],
            'alpha': [[0.0, 0.0, 0.0], 
                      [1.0, 0.2, 1.0]]
        }
        self.cm_cmap = LinearSegmentedColormap('custom_cmap', segmentdata=cm_cdict)

        self.i = -1
        self.img_id = None
        self.poly_id = None
        self.liked_ids = []
        self.disliked_ids = []
 
        # global img_ids
        # global iidxes
    

    def advance(self):
        self.i += 1
        if self.i >= len(img_ids):
            self.i = len(img_ids) - 1
            return
        self.img_id = img_ids.iloc[self.i]
        self.poly_id = iidxes.iloc[self.i]
        self.img_name = f'{self.img_id}_{self.poly_id}'

        print(self.liked_ids)
        print(self.disliked_ids)



    def like(self):
        # global liked_ids

        if self.i >= 0 and self.img_name:
            self.liked_ids.append(self.img_name)
        self.advance()

        # print('likes: ', self.liked_ids)
        # print('dislikes: ', self.disliked_ids)
        # print(' ')

    def dislike(self):

        if self.i >= 0 and self.img_name:
            self.disliked_ids.append(self.img_name)
        self.advance()
        
        # print('likes: ', self.liked_ids)
        # print('dislikes: ', self.disliked_ids)
        # print(' ')


    def undo(self):
        # if self.ndwi_artist is not None or self.wm_artist is not None:
        # global img_id
        # global liked_ids
        # global disliked_ids
        if self.i <= 0:
            return
        
        self.i -= 1
        self.img_id = img_ids.iloc[self.i]
        self.poly_id = iidxes.iloc[self.i]
        self.img_name = f'{self.img_id}_{self.poly_id}'

        if self.img_id in self.liked_ids:
            self.liked_ids.remove(self.img_name)
        elif self.img_name in self.disliked_ids:
            self.disliked_ids.remove(self.img_name)

        print(self.liked_ids)
        print(self.disliked_ids)

        # print('likes: ', self.liked_ids)
        # print('dislikes: ', self.disliked_ids)


    def save(self):

        current_datetime = datetime.now()
        dt_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

        # with open(r"C:\Users\dego\Documents\local_files\RSSA\river_mask_tinder\checked\dislikes_" + dt_string + ".txt", 'w') as dislike_file:
        with open(r"C:\Users\dego\Documents\river_tinder_assets\gage_sites\likes_dislikes\dislikes_" + dt_string + ".txt", 'w') as dislike_file:
            for id in self.disliked_ids:
                dislike_file.write(id + "\n")
        
        # with open(r"C:\Users\dego\Documents\local_files\RSSA\river_mask_tinder\checked\likes_" + dt_string + ".txt", 'w') as like_file:
        with open(r"C:\Users\dego\Documents\river_tinder_assets\gage_sites\likes_dislikes\likes_" + dt_string + ".txt", 'w') as like_file:
            for id in self.liked_ids:
                like_file.write(id + "\n")

        self.disliked_ids.clear()
        self.liked_ids.clear()

    def show_plot(self):

        if self.i < 0:
            self.advance()
        if self.i < 0 or self.i >= len(img_ids):
            return

        ndwi, rmask, cloudmask, snowmask, wwindow_transform, circle = GENERATE_MASKS(self.img_id, int(self.poly_id), squares, pills, circles, vector_centerlines)
        # cloudmask = GENERATE_MASKS(current_img_id, current_poly_iidx, squares, pil, circles, vector_centerlines)
        # print(circlemask)
        # print('---------------------------------------------------')
        ndwi = np.uint8(np.clip((ndwi + 1) * 255 / 2, 0, 255))
        if self.wm_artist is None:

            self.ax.clear()
            self.ndwi_artist = show(ndwi, vmin=0, vmax=255, cmap='Greys_r', ax=self.ax, transform=wwindow_transform)
            self.rm_artist = show(rmask, cmap=self.rm_cmap, vmin=0, vmax=1, ax=self.ax, transform=wwindow_transform)
            self.cm_artist = show(cloudmask, cmap = self.cm_cmap, vmin=0, vmax=1, ax=self.ax, transform=wwindow_transform)
            self.circle_artist = gpd.GeoSeries([circle]).plot(ax=self.ax, facecolor='none')
            # self.mask_artist = show(circle_mask & rmask, cmap=self.rm_cmap, ax=self.ax, transform=wwindow_transform)
            self.title_artist = self.ax.set_title(self.img_name)


        else:

            self.ndwi_artist.set_data(ndwi)
            self.rm_artist.set_data(rmask)
            self.cm_artist.set_data(cloudmask)
            # self.mask_artist.set_data(n_river)
            self.circle_artist.set_data(gpd.GeoSeries([circle]))
            self.ax.relim()
            self.ax.autoscale_view()
            self.title_artist.set_title(self.img_name)
            # print(ndwi[10])

        self.canvas.draw()
        # self.lbl_iid['text'] = self.img_name

if __name__ == "__main__":
    ### load assets

    # identify user profile

    userdir = glob.glob((r"C:\Users\*"))
    dsts = []
    for dir in userdir:
        dsts.append(dir.split('\\')[-1])

    user_pids = ['dego', 'demko22b', 'dtasha']
    for user in user_pids:
        if user in dsts:
            dst = user
    
    
    
    # download assets from google drive using gdown package
    asset_loc = f'C:/Users/{dst}/Documents/river_tinder_assets'

    if (glob.glob(f'{asset_loc}/gage_sites/*')) == []:
        gage_sites_url = 'https://drive.google.com/drive/folders/1g8rcfbHuIaF1Zad91O_FATsQyncmm0vY?usp=sharing'
        centerline_url = 'https://drive.google.com/drive/folders/1HFiBh-X1xtvoXDqFq5KYup7vc1hq3Xn-?usp=sharing'
        all_sites_url = 'https://drive.google.com/drive/folders/1f19N2qtnHmfCidoTp00tHcG1JBC9fPcv?usp=sharing'

        gdown.download_folder(url=gage_sites_url, output=f'{asset_loc}/gage_sites', quiet=False, use_cookies=True)
        gdown.download_folder(url=centerline_url, output=f'{asset_loc}/centerline', quiet=False, use_cookies=True)
        gdown.download_folder(url=all_sites_url, output=f'{asset_loc}/all_sites', quiet=False, use_cookies=True)


    # load assets for processing
    squares = gpd.read_file(f'{asset_loc}/gage_sites/gage_squares_15x_20250909.shp')\
        .set_index('iindex', drop=False)\
        .to_crs('EPSG:4326')

    pills = gpd.read_file(f'{asset_loc}/gage_sites/gage_pills_3L_3W_20250909.shp')\
        .set_index('iindex', drop=False)\
        .to_crs('EPSG:4326')


    circles = gpd.read_file(f'{asset_loc}/gage_sites/gage_circles_3x_20250909.shp')\
        .set_index('iindex', drop=False)\
        .to_crs('EPSG:4326')
    
    checked_img_path = 'C:/Users/dego/Documents/river_tinder_assets/gage_sites/likes_dislikes'
    lols = []
    for fn in (glob.glob(os.path.join(checked_img_path, '*.txt'))):
        with open(fn, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        lols.append(lines)

    checked_id_list = list(itertools.chain.from_iterable(lols))

    checked_img_ids = []
    checked_iidxes = []

    for id in checked_id_list:
        img_id = str(id[0:24])
        iidx = int(id[25:])
        checked_img_ids.append(img_id)
        checked_iidxes.append(iidx)

    print(len(checked_id_list))
    print(len(set(checked_id_list)))

    
    checked_df = pd.DataFrame({'img_id': checked_img_ids, 'iindex': checked_iidxes}, index=checked_id_list)
    # print(checked_df)
    # print('n checked ids: ', len(checked_df))
    vector_centerlines = gpd.read_file(f'{asset_loc}/centerline/s2_platte_centerlines_4326.shp')
    

    imgs_w_ids = pd.read_csv(f'{asset_loc}/gage_sites/gage_stac_ids_iidx_clouds_lt20.csv')
    imgs_w_ids['img_id'] = imgs_w_ids.apply(lambda x: str(x['0'].split('=')[1][:24]), axis=1)
    imgs_w_ids = imgs_w_ids.rename(columns={'1': 'iindex'})[['img_id', 'iindex']]
    imgs_w_ids = imgs_w_ids.set_index(pd.Series([f'{a}_{b}' for a, b in zip(imgs_w_ids['img_id'], imgs_w_ids['iindex'])]))
    # print('total ids: ', len(imgs_w_ids))
    unchecked_ids = imgs_w_ids[~imgs_w_ids[['img_id', 'iindex']].isin(checked_df[['img_id', 'iindex']]).all(axis=1)]

    # print(unchecked_ids.head())
    # print('unchecked ids: ', len(unchecked_ids))
    

    iidxes = unchecked_ids['iindex']
    img_ids = unchecked_ids['img_id']
    # n_imgs = len(img_series)
    # img_ids = np.repeat('000000000000000000000000', n_imgs)

    # for n in range(n_imgs):
    #     id = img_series[n]
    #     id = id.split('=')[1][0:24]
    #     img_ids[n] = id

    stac = stac_client.open('https://earth-search.aws.element84.com/v1')



    ### gui stuff
    root = tk.Tk()
    app = TifViewerApp(root)
    root.mainloop()
