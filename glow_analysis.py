import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm, linregress
from tqdm import tqdm
import datetime as dt
import polars as pl

pl.Config.set_streaming_chunk_size(5000)
pl.Config.set_engine_affinity('streaming')
CHUNK_SIZE = 1000

print('Starting....')

# Read in GLOW points
ms_pts = gpd.read_file(r'C:\Users\dego\Documents\local_files\RSSA\MS_GLOW_GRADESDL_pts.gpkg')\
    .drop('geometry', axis=1)

# Convert to lazyframe
ms_pts_lz = pl.from_pandas(ms_pts).lazy()

# List of relevant COMIDs
ID_arr = ms_pts['ID'].unique()
COMID_arr = ms_pts['COMID'].unique()

# Load GLOW widths
glow = pl.scan_parquet(r'c:\Users\dego\Documents\local_files\sandbox\GLOW\width\width\GLOW_width_region_7.parquet')
glow = glow.filter(pl.col('ID').is_in(ID_arr))

# Load in GRADES DL discharge
fps = [rf'C:\Users\dego\Documents\local_files\RSSA\grades_Q\GRADES_DL_thru_1991.parquet',
       rf'C:\Users\dego\Documents\local_files\RSSA\grades_Q\GRADES_DL_thru_2001.parquet',
       rf'C:\Users\dego\Documents\local_files\RSSA\grades_Q\GRADES_DL_thru_2011.parquet',
       rf'C:\Users\dego\Documents\local_files\RSSA\grades_Q\GRADES_DL_thru_2021.parquet']
GDL_Q = pl.concat([pl.scan_parquet(p) for p in fps])
GDL_Q = GDL_Q.filter(pl.col('COMID').is_in(COMID_arr))



n_batches = int(np.ceil(len(ID_arr) / CHUNK_SIZE))
print(f'Total COMIDs: {len(ID_arr)}')
print(f'N batches: {n_batches}')

chunk_files = []

# for i in tqdm(range(n_batches)):
for i in tqdm(range(5)):
    batch_comids = COMID_arr[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]

    glow_chunk = glow.filter(pl.col('COMID').is_in(batch_comids))    

    GDL_chunk = GDL_Q.filter(pl.col('COMID').is_in(batch_comids)).collect()
    
    GDL_chunk = GDL_chunk.group_by('COMID').map_groups(lambda df: df.with_columns([
        pl.Series('Q_rank', df['Qout'].rank().cast(pl.Float32)),
    ])).lazy()
    
    GDL_chunk = GDL_chunk.with_columns([
        (pl.col('Q_rank')
            .over('COMID') / 
            (pl.len().over('COMID') - 1))
            .alias('Q_percentile')
            ])
    
    GDL_chunk = GDL_chunk.with_columns([
        (pl.col('Q_percentile') * 10)
            .floor()
            .clip(0,9)
            .cast(pl.Int8)
            .alias('Q_decile')
            ])
    
    
    chunk_lz = glow_chunk\
        .join(ms_pts_lz, on='COMID', how='inner')\
        .join(GDL_chunk, on=['COMID', 'date'], how='inner')

    print(glow_chunk.explain())
    print(GDL_chunk.explain())
    df_batch = chunk_lz.collect()
    path = rf'C:\Users\dego\Documents\local_files\RSSA\GLOW_analysis_chunks\chunk_{i}.parquet'
    df_batch.write_parquet(path)

    chunk_files.append(path)


print('Scanning results')
result = pl.concat([pl.scan_parquet(f) for f in tqdm(chunk_files)])

print('Grouping')

grouped = result\
    .group_by(['order', 'Q_decile'])\
    .agg([pl.col("width").drop_nulls().alias("width_list")])\
    .collect()

print('Fitting distributions')
fits = []
for row in tqdm(grouped.iter_rows(named=True)):
    o = row['order']
    qd = row['Q_decile']
    w = np.asarray(row['width_list'], dtype=float)

    w = w[w > 0]

    shape, loc, scale = lognorm.fit(w)

    fits.append({'order': o,
                 'Q_decile': qd,
                 'shape': shape,
                 'scale': scale,
                 'loc': loc})
    

fit_df = pd.DataFrame(fits)

fit_df.to_csv('C:/Users/dego/Desktop/testfits.csv')

print('Complete')