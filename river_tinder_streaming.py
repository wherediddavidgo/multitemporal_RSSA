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
from rasterio.vrt import WarpedVRT

from shapely import bounds
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# from rio_cogeo import cogeo
# from google.colab import drive
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
from cv2 import dilate


import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import rasterio
import numpy as np
import glob
import os
import tqdm
from datetime import datetime
import sys

sys.path.append(r'C:\Users\dego\AppData\Local\Programs\Python\Python313\Lib\site-packages')
sys.path.append(r'C:\Users\dego\AppData\Local\Programs\Python\Python313\Scripts')
# import gdown
from pystac_client import Client as stac_client





def ref_geoms_from_b3href(b3_href, poly_idx, view_geoms, effwidth_geoms, centerline_geoms):
    with rasterio.open(b3_href) as src:
        img_crs = src.crs
    # print(poly_idx)
    view_geom = view_geoms.set_index('iindex').loc[poly_idx].geometry
    eff_geom = effwidth_geoms.set_index('iindex').loc[poly_idx].geometry

    t_view = Transformer.from_crs(view_geoms.crs, img_crs,  always_xy=True).transform
    t_eff = Transformer.from_crs(effwidth_geoms.crs, img_crs, always_xy=True).transform

    view_src = shp_transform(t_view, view_geom)
    eff_src  = shp_transform(t_eff,  eff_geom)

    # Filter lines by intersection in image CRS, then reproject only those (cheap)
    if centerline_geoms.crs != img_crs:
        cl_img = centerline_geoms.to_crs(img_crs)
    else:
        cl_img = centerline_geoms
    hits = list(cl_img.sindex.query(eff_src, predicate="intersects"))
    lines_in_bound = cl_img.iloc[hits].copy()

    return view_src, eff_src, lines_in_bound


def dn_to_reflectance(band):
    return np.float32(band) * 1e-4


def normalized_difference(b1, b2):
    denominator = b1 + b2
    numerator = b1 - b2
    return numerator / np.where(denominator != 0, denominator, np.nan)




def process_image_from_hrefs(b3_href, b8_href, scl_href, view_geom, otsu_geom):
    l, b, r, t = map(float, view_geom.bounds)

    with rasterio.open(b3_href) as b3_src:
        window = rasterio.windows.from_bounds(l, b, r, t, b3_src.transform).round_offsets().round_lengths()
        transform = b3_src.window_transform(window)

        h, w = window.height, window.width

        if h == 0 or w == 0:
            return np.array([1]), None, None, None, None, None

        b3v = b3_src.read(1, window=window, masked=True)
        b3_crs = b3_src.crs

    with rasterio.open(b8_href) as b8_src:
        b8v = b8_src.read(1, window=window, masked=True)

    with rasterio.open(scl_href) as scl_src, WarpedVRT(
        scl_src, crs=b3_crs, transform=transform, width=w, height=h, resampling=Resampling.nearest
    ) as vrt:
        sclv = vrt.read(1)

    b3v = dn_to_reflectance(b3v)
    b8v = dn_to_reflectance(b8v)
    ndwi_v = normalized_difference(b3v, b8v)
    otsu_geom_mask = geometry_mask([otsu_geom], out_shape=(h, w), transform=transform, invert=True)
    # return otsu_geom_mask, ndwi_v
    if otsu_geom_mask.size <= ndwi_v.size:
        ndwi_o = np.ma.array(ndwi_v, mask=~otsu_geom_mask).compressed()
        b8o = np.ma.array(b8v, mask=~otsu_geom_mask).compressed()

        if ndwi_o.size >= 10:
            # ndwi_threshold = threshold_otsu(ndwi_o)
            # nir_threshold = threshold_otsu(b8o)
            nir_threshold = 0.20
            ndwi_threshold = -0.15
        else:
            ndwi_threshold = 1
            nir_threshold = 1

        wmask = (ndwi_v >= ndwi_threshold) & (b8v <= nir_threshold)
        cloudmask = np.isin(sclv, [7, 8, 9]).astype('uint8')
        snowmask = (sclv == 11).astype('uint8')
        ndmask = (sclv != 0).astype('uint8')

        return ndwi_v, wmask, cloudmask, snowmask, ndmask, transform, ndwi_threshold, nir_threshold
    else:
        return np.array([1]), None, None, None, None, None, None, None


def identify_river(wmask, lines, window_trans):
    h, w = wmask.shape
    wbool = wmask.filled(0) > 0
    # rbuffs = lines.copy().buffer(5)
    shapes = ((geom, 1) for geom in lines.geometry)
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


def count_pixels(rmask, cloudmask, snowmask, ndmask, transform, circle, ndwi_v):
    if rmask is not None:
        circle_mask = rasterize([circle], out_shape = rmask.shape, transform=transform, dtype='uint8', all_touched=True) == 1

        kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]).astype('uint8')

        ring_mask = dilate(circle_mask.astype('uint8'), kernel, iterations=1) & ~circle_mask

        r = rmask.astype(bool)
        c = cloudmask.astype(bool)
        s = snowmask.astype(bool)
        v = ndmask.astype(bool)

        n_pixels = np.count_nonzero(circle_mask)
        n_valid = np.count_nonzero(circle_mask & v)
        n_river = np.count_nonzero(circle_mask & r)
        n_cloud = np.count_nonzero(circle_mask & c)
        n_snow = np.count_nonzero(circle_mask & s)
        n_cloudriver = np.count_nonzero(circle_mask & r & c)

        n_edge = np.count_nonzero(ring_mask)
        n_edgeriver = np.count_nonzero(ring_mask & r)

        return n_pixels, n_valid, n_river, n_cloud, n_snow, n_cloudriver, n_edge, n_edgeriver, np.nanmean(np.where(circle_mask, ndwi_v, np.nan))

    else:
        return -999, -999, -999, -999, -999, -999, -999, -999
    

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
        self.rm_cmap = LinearSegmentedColormap('custom_cmap1', segmentdata=rm_cdict)

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
        self.cm_cmap = LinearSegmentedColormap('custom_cmap2', segmentdata=cm_cdict)

        sm_cdict = {
            'red':   [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]],
            'green': [[0.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0]],
            'blue':  [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]],
            'alpha': [[0.0, 0.0, 0.0], 
                      [1.0, 0.2, 1.0]]
        }
        self.sm_cmap = LinearSegmentedColormap('custom_cmap3', segmentdata=sm_cdict)

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
        print('like')
        if self.i >= 0 and self.img_name:
            self.liked_ids.append(self.img_name)
        self.advance()

        # print('likes: ', self.liked_ids)
        # print('dislikes: ', self.disliked_ids)
        # print(' ')

    def dislike(self):
        print('dislike')
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
        
        # if (self.img_id in (Qdf.img_id.unique())) and (self.poly_id in (Qdf.iindex.unique())):
        #     Q_cms = imgs_w_ids.loc[(imgs_w_ids.img_id == self.img_id) & (imgs_w_ids.iindex == self.poly_id)].reset_index().loc[0, 'Q_cms']
        #     Qperc = imgs_w_ids.loc[(imgs_w_ids.img_id == self.img_id) & (imgs_w_ids.iindex == self.poly_id)].reset_index().loc[0, 'Q_percentile']
        # else: 
        #     Qperc = -999
        #     Q_cms = -999

        b3 = imgs_w_ids.loc[(imgs_w_ids.img_id == self.img_id) & (imgs_w_ids.iindex == self.poly_id)].reset_index().loc[0, 'b3_href']
        b8 = imgs_w_ids.loc[(imgs_w_ids.img_id == self.img_id) & (imgs_w_ids.iindex == self.poly_id)].reset_index().loc[0, 'b8_href']
        scl = imgs_w_ids.loc[(imgs_w_ids.img_id == self.img_id) & (imgs_w_ids.iindex == self.poly_id)].reset_index().loc[0, 'scl_href']

        square, circle, lines = ref_geoms_from_b3href(b3, int(self.poly_id), squares, circles, vector_centerlines)

        ndwi, wmask, cloud, snow, valid, transform, ndwi_threshold, nir_threshold = process_image_from_hrefs(b3, b8, scl, square, circle)
        
        if (valid) is not None:
            ndwi = (ndwi + 1) * 127.5
            rmask = identify_river(wmask, lines, transform)

            # print(np.unique(rmask.data))
            if self.wm_artist is None:

                self.ax.clear()
                self.ndwi_artist = show(ndwi.data, vmin=0, vmax=255, cmap='Greys_r', ax=self.ax, transform=transform)
                self.rm_artist = show(rmask, cmap=self.rm_cmap, vmin=0, vmax=1, ax=self.ax, transform=transform)
                self.cm_artist = show(cloud, cmap =self.cm_cmap, vmin=0, vmax=1, ax=self.ax, transform=transform)
                self.sm_artist = show(snow, cmap=self.sm_cmap, vmin=0, vmax=1, ax=self.ax, transform=transform)
                self.circle_artist = gpd.GeoSeries([circle]).plot(ax=self.ax, facecolor='none', edgecolor='tab:blue')
                self.line_artist = lines.plot(ax=self.ax, color='tab:blue')
                self.title_artist = self.ax.set_title(self.img_name)
                # self.desc_artist = self.ax.set_xlabel(f'Discharge = {Q_cms:.02f} cms\nDischarge percentile = {Qperc:.02f}')


                # n_pixels, n_valid, n_river, n_cloud, n_snow, n_cloudriver, n_edge, n_edgeriver, mean_ndwi = count_pixels(rmask, cloud, snow, valid, transform, circle, ndwi)

                # print(n_pixels)
                # print(n_valid)
                # print(n_river)
                # print(' ')
            else:

                self.ndwi_artist.set_data(ndwi)
                self.rm_artist.set_data(rmask)
                self.cm_artist.set_data(cloud)
                self.sm_artist.set_data(snow)
                self.circle_artist.set_data(gpd.GeoSeries([circle]))
                self.line_artist.set_data(lines)
                self.ax.relim()
                self.ax.autoscale_view()
                self.title_artist.set_title(self.img_name)
                # self.desc_artist.set_xlabel('0')
                # print(ndwi[10])

            self.canvas.draw()

        else: 
            self.title_artist = self.ax.set_title(self.img_name)
            self.ndwi_artist = show(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), ax=self.ax)




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
    
    
    
    # # download assets from google drive using gdown package
    # asset_loc = f'C:/Users/{dst}/Documents/river_tinder_assets'

    # if (glob.glob(f'{asset_loc}/gage_sites/*')) == []:
    #     gage_sites_url = 'https://drive.google.com/drive/folders/1g8rcfbHuIaF1Zad91O_FATsQyncmm0vY?usp=sharing'
    #     centerline_url = 'https://drive.google.com/drive/folders/1HFiBh-X1xtvoXDqFq5KYup7vc1hq3Xn-?usp=sharing'
    #     all_sites_url = 'https://drive.google.com/drive/folders/1f19N2qtnHmfCidoTp00tHcG1JBC9fPcv?usp=sharing'

    #     gdown.download_folder(url=gage_sites_url, output=f'{asset_loc}/gage_sites', quiet=False, use_cookies=True)
    #     gdown.download_folder(url=centerline_url, output=f'{asset_loc}/centerline', quiet=False, use_cookies=True)
    #     gdown.download_folder(url=all_sites_url, output=f'{asset_loc}/all_sites', quiet=False, use_cookies=True)


    # load assets for processing
    squares = gpd.read_file(f'C:/Users/dego/Documents/local_files/RSSA/Platte_centerlines_masks/squares_15x_20251010.shp')\
        .set_index('iindex', drop=False)
        # .to_crs('EPSG:4326')

    # pills = gpd.read_file(f'{asset_loc}/gage_sites/gage_pills_3L_3W_20250909.shp')\
    #     .set_index('iindex', drop=False)\
    #     .to_crs('EPSG:4326')


    circles = gpd.read_file(f'C:/Users/dego/Documents/local_files/RSSA/Platte_centerlines_masks/circles_3x_20251010.shp')\
        .set_index('iindex', drop=False)
        # .to_crs('EPSG:4326')
    
    # checked_img_path = 'C:/Users/dego/Documents/river_tinder_assets/gage_sites/likes_dislikes'
    # lols = []
    # for fn in (glob.glob(os.path.join(checked_img_path, '*.txt'))):
    #     with open(fn, 'r') as f:
    #         lines = [line.strip() for line in f.readlines()]
    #     lols.append(lines)

    # checked_name_list = list(itertools.chain.from_iterable(lols))

    # Qdf = pd.read_csv('C:/Users/dego/Documents/local_files/RSSA/gage_iid_Q.csv')
    # Qdf['iindex'] = np.uint64(Qdf['iindex'])

    # checked_img_ids = []
    # checked_iidxes = []
    # checked_names = []

    # for name in set(checked_name_list):
    #     img_id = str(name[0:24])
    #     iidx = int(name[25:])
    #     checked_img_ids.append(img_id)
    #     checked_iidxes.append(iidx)
    #     checked_names.append(name)

    # print(len(checked_img_ids))
    # print(len(checked_iidxes))
    # print(len(checked_names))
    
    # checked_df = pd.DataFrame({'img_id': checked_img_ids, 'iindex': checked_iidxes, 'name': checked_names}).set_index('name')
    # checked_df.to_csv('C:/Users/dego/Desktop/checked_df2.csv')
    # print(checked_df)
    # print('n checked ids: ', len(checked_df))
    vector_centerlines = gpd.read_file(f'C:/Users/dego/Documents/river_tinder_assets/centerline/s2_platte_centerlines_4326.shp')
    # normal mode
    {
    # imgs_w_ids = pd.read_csv(f'{asset_loc}/gage_sites/gage_stac_ids_iidx_clouds_lt20.csv')
    # imgs_w_ids['img_id'] = imgs_w_ids.apply(lambda x: str(x['0'].split('=')[1][:24]), axis=1)
    # imgs_w_ids = imgs_w_ids.rename(columns={'1': 'iindex'})[['img_id', 'iindex']]
    # imgs_w_ids = imgs_w_ids.set_index(pd.Series([f'{a}_{b}' for a, b in zip(imgs_w_ids['img_id'], imgs_w_ids['iindex'])]))
    # unchecked_ids = imgs_w_ids[~imgs_w_ids[['img_id', 'iindex']].isin(checked_df[['img_id', 'iindex']]).all(axis=1)]
    }
    
    
    # spot check mode, use to see same day gage and satellite width measurements
    imgs_w_ids = pd.read_csv(r"C:\Users\dego\Documents\local_files\RSSA\stac_img_ids_20251012.csv").sample(frac=1, random_state=1)
    imgs_w_ids['iindex'] = np.uint64(imgs_w_ids['iindex'])
    imgs_w_ids['date'] = pd.to_datetime(imgs_w_ids['date'])
    # imgs_w_ids = pd.merge(imgs_w_ids, Qdf, on=['img_id', 'iindex'], how='outer')
    



    # imgs_w_ids = imgs_w_ids.loc[imgs_w_ids.Q_cms >= 0]

    # imgs_w_ids = imgs_w_ids.loc[(imgs_w_ids.iindex == 245238)]
    # 225563
    # 68053
    # 82571

    # imgs_w_ids = imgs_w_ids.sort_values('Q_percentile')

    # imgs_w_ids = imgs_w_ids.drop_duplicates(['img_id', 'iindex'])
    # imgs_w_ids['w_percentile'] = imgs_w_ids.groupby('iindex')['sat_width_m'].rank(pct=True)

    imgs_w_ids = imgs_w_ids.set_index(pd.Series([f'{a}_{b}' for a, b in zip(imgs_w_ids['img_id'], imgs_w_ids['iindex'])]))
    
    unchecked_ids = imgs_w_ids.copy()
    # [~imgs_w_ids[['img_id', 'iindex']].isin(checked_df[['img_id', 'iindex']]).all(axis=1)]
    
    

    iidxes = unchecked_ids['iindex']
    img_ids = unchecked_ids['img_id']
    # n_imgs = len(img_series)
    # img_ids = np.repeat('000000000000000000000000', n_imgs)

    # for n in range(n_imgs):
    #     id = img_series[n]
    #     id = id.split('=')[1][0:24]
    #     img_ids[n] = id

    stac = stac_client.open('https://earth-search.aws.element84.com/v1')




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
    rm_cmap = LinearSegmentedColormap('custom_cmap1', segmentdata=rm_cdict)

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
    cm_cmap = LinearSegmentedColormap('custom_cmap2', segmentdata=cm_cdict)

    sm_cdict = {
        'red':   [[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]],
        'green': [[0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0]],
        'blue':  [[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]],
        'alpha': [[0.0, 0.0, 0.0], 
                    [1.0, 0.2, 1.0]]
    }
    sm_cmap = LinearSegmentedColormap('custom_cmap3', segmentdata=sm_cdict)

    # fig, ax = plt.subplots()
    # transform = rasterio.Affine(10.0, 0.0, 199980.0,
    #                             0.0, -10.0, 4600020.0)
    # ndwi_im = show(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), vmin=0, vmax=255, cmap='Greys_r', ax=ax, transform=transform)
    # rm_im = show(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), cmap=rm_cmap, vmin=0, vmax=1, ax=ax, transform=transform)
    # cm_im = show(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), cmap=cm_cmap, vmin=0, vmax=1, ax=ax, transform=transform)
    # sm_im = show(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), cmap=sm_cmap, vmin=0, vmax=1, ax=ax, transform=transform)

    # img_id_list = []
    # poly_ids = []
    # Qpercs = []
    # Qs = []
    # wpercs = []
    # ws = []


    # plot_out = True
    
    # for i, row in tqdm.tqdm(imgs_w_ids.iterrows()):

    #     img_id = row['img_id']
    #     poly_id = row['iindex']

    #     if (img_id in (Qdf.img_id.unique())) and (poly_id in (Qdf.iindex.unique())):
    #         Q_cms = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'Q_cms']
    #         Qperc = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'Q_percentile']
    #         w_m = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'sat_width_m']
    #         wperc = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'w_percentile']
    #     else: 
    #         Qperc = -999
    #         Q_cms = -999

    #     img_id_list.append(img_id)
    #     poly_ids.append(poly_id)
    #     Qpercs.append(Qperc)
    #     Qs.append(Q_cms)
    #     wpercs.append(wperc)
    #     ws.append(w_m)

    #     if plot_out:
    #         b3 = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'b3_href']
    #         b8 = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'b8_href']
    #         scl = imgs_w_ids.loc[(imgs_w_ids.img_id == img_id) & (imgs_w_ids.iindex == poly_id)].reset_index().loc[0, 'scl_href']

    #         square, circle, lines = ref_geoms_from_b3href(b3, int(poly_id), squares, circles, vector_centerlines)

    #         ndwi, wmask, cloud, snow, valid, transform, ndwi_threshold, nir_threshold = process_image_from_hrefs(b3, b8, scl, square, circle)

    #         if (valid) is not None:

    #             fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    #             ndwi = (ndwi + 1) * 127.5
    #             rmask = identify_river(wmask, lines, transform)

    #             show(ndwi, vmin=0, vmax=255, cmap='Greys_r', ax=ax, transform=transform)
    #             show(rmask, vmin=0, vmax=1, cmap=rm_cmap, ax=ax, transform=transform)
    #             show(cloud, vmin=0, vmax=1, cmap=cm_cmap, ax=ax, transform=transform)
    #             show(snow, vmin=0, vmax=1, cmap=sm_cmap, ax=ax, transform=transform)
    #             gpd.GeoSeries([circle]).plot(ax=ax, facecolor='none', edgecolor='tab:blue')
    #             lines.plot(ax=ax, color='tab:blue')
    #             ax.set_xlabel(f'Discharge = {Q_cms:.02f} cms\nDischarge percentile = {Qperc:.02f}\nWidth = {w_m:.02f} m\nWidth percentile = {wperc:.02f}')
    #             ax.set_title(f'{img_id}_{poly_id}')

    #             fig.savefig(f'C:/Users/dego/Documents/local_files/RSSA/RT_exports/{img_id}_{poly_id}.png', dpi=100)
    #             plt.close(fig)


    # percentile_comp_df = pd.DataFrame({'img_id': img_id_list,
    #                                    'poly_id': poly_ids,
    #                                    'Q_percentile': Qpercs,
    #                                    'Q_cms': Qs,
    #                                    'w_percentile': wpercs,
    #                                    'width_m': ws})
    # percentile_comp_df.to_csv(f'C:/Users/dego/Desktop/percentile_comparison_{poly_id}.csv')

    ### gui stuff
    root = tk.Tk()
    app = TifViewerApp(root)
    root.mainloop()
