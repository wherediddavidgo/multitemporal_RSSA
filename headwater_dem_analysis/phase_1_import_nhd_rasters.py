import glob
import geopandas as gpd
import argparse
import pandas as pd
import zipfile
import shutil
import urllib.request
import os


def main(catchments_path: str, raster_list_path: str, fdr_dir: str, fac_dir: str, tmp_dir: str):
    print('poopy', flush=True)
    rasters = pd.read_csv(raster_list_path)
    catchments = gpd.read_file(catchments_path)

    hucs = catchments['huc4']

    for h in hucs:
        print(h, flush=True)
        if (os.path.join(fac_dir, f'fac_{h}.tif') not in glob.glob(os.path.join(fac_dir, 'fac*.tif'))) and (os.path.join(fdr_dir, f'fdr_{h}.tif') not in glob.glob(os.path.join(fdr_dir, 'fdr*.tif'))):
            print(h, flush=True)
            r = rasters.loc[(rasters['file'].str.contains(f'_{h}_')) & (rasters['file'].str.contains('zip')) & (rasters['file'].str.contains('HU4')), 'file'].to_numpy()[0]
            url = f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/VPU/Current/Raster/{r}'
            print(url, flush=True)
            zip_path = os.path.join(tmp_dir, r)
            print(f"zip_path: {zip_path}", flush=True)
            # Download
            print(f"Downloading {h}", flush=True)
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded {os.path.getsize(zip_path) / (1024*1024):.1f} MB", flush=True)

            # Unzip
            print(f"Extracting {h}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                base = names[0]
                print(base)
                if os.path.join(base, 'fdr.tif') in names:
                    zf.extract(os.path.join(base, 'fdr.tif'), fdr_dir)
                    shutil.move(os.path.join(fdr_dir, base, 'fdr.tif'), os.path.join(fdr_dir, f'fdr_{h}.tif'))
                else:
                    print('fdr.tif does not exist.', flush=True)

                if os.path.join(base, 'fac.tif') in names:
                    zf.extract(os.path.join(base, 'fac.tif'), fac_dir)
                    shutil.move(os.path.join(fac_dir, base, 'fac.tif'), os.path.join(fac_dir, f'fac_{h}.tif'))
                else:
                    print('fac.tif does not exist.', flush=True)

            print(f"Extracted {h}", flush=True)
            os.remove(zip_path)
            
            # os.remove(os.path.join(fdr_dir, base))
            # os.remove(os.path.join(fac_dir, base))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--catchments-path', required=True)
    ap.add_argument('--raster-list-path', required=True)
    ap.add_argument('--fdr-dir', required=True)
    ap.add_argument('--fac-dir', required=True)
    ap.add_argument('--tmp-dir', required=True)
    main(**vars(ap.parse_args()))


