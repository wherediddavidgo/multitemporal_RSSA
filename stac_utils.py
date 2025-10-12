import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
from cv2 import dilate
from shapely.ops import transform as shp_transform
from pyproj import Transformer





def ref_geoms_from_b3href(b3_href, poly_idx, effwidth_geoms, centerline_geoms):
    with rasterio.open(b3_href) as src:
        img_crs = src.crs
    print(poly_idx)
    # view_geom = view_geoms.loc[poly_idx].geometry
    eff_geom = effwidth_geoms.loc[poly_idx].geometry

    # t_view = Transformer.from_crs(view_geoms.crs, img_crs,  always_xy=True).transform
    t_eff = Transformer.from_crs(effwidth_geoms.crs, img_crs, always_xy=True).transform

    # view_src = shp_transform(t_view, view_geom)
    eff_src  = shp_transform(t_eff,  eff_geom)

    # Filter lines by intersection in image CRS, then reproject only those (cheap)
    if centerline_geoms.crs != img_crs:
        cl_img = centerline_geoms.to_crs(img_crs)
    else:
        cl_img = centerline_geoms
    hits = list(cl_img.sindex.query(eff_src, predicate="intersects"))
    lines_in_bound = cl_img.iloc[hits].copy()

    return eff_src, lines_in_bound


def dn_to_reflectance(band):
    return np.float32(band) * 1e-4


def normalized_difference(b1, b2):
    denominator = b1 + b2
    numerator = b1 - b2
    return numerator / np.where(denominator != 0, denominator, np.nan)




def process_image_from_hrefs(b3_href, b8_href, scl_href, otsu_geom):
    l, b, r, t = map(float, otsu_geom.bounds)

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
            ndwi_threshold = threshold_otsu(ndwi_o)
            nir_threshold = threshold_otsu(b8o)
        else:
            ndwi_threshold = 1
            nir_threshold = 1

        wmask = (ndwi_v >= ndwi_threshold) & (b8v <= nir_threshold)

        cloudmask = np.isin(sclv, [7, 8, 9]).astype('uint8')
        snowmask = (sclv == 11).astype('uint8')
        ndmask = (sclv != 0).astype('uint8')

        return ndwi_v, wmask, cloudmask, snowmask, ndmask, transform
    else:
        return np.array([1]), None, None, None, None, None





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


def GENERATE_MASKS(img_id, poly_idx, circles, vector_centerlines, HREFS):
    b3_href, b8_href, scl_href = HREFS[(img_id, poly_idx)]

    circle_geom, lines = ref_geoms_from_b3href(b3_href, poly_idx, circles, vector_centerlines)

    ndwi_v, wmask, cloudmask, snowmask, ndmask, wwindow_transform = process_image_from_hrefs(b3_href, b8_href, scl_href, circle_geom)

    if ndwi_v.size > 1:

        rmask = identify_river(wmask, lines, wwindow_transform)

        return ndwi_v, rmask, cloudmask, snowmask, ndmask, wwindow_transform, circle_geom
    else:
        return np.array([1]), None, None, None, None, None, None


def count_pixels(rmask, cloudmask, snowmask, ndmask, transform, circle):
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

        return n_pixels, n_valid, n_river, n_cloud, n_snow, n_cloudriver, n_edge, n_edgeriver

    else:
        return -999, -999, -999, -999, -999, -999, -999, -999