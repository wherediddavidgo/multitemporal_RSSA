import dask.multiprocessing
dask.config.set(scheduler='processes', num_workers=5)
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm

print('Starting imports')
hydrosheds_ah = gpd.read_file(r"C:\Users\dego\Documents\local_files\big_datasets\hydrosheds_prancevic_ah.gpkg")

catchments = gpd.read_file(r'C:\Users\dego\Documents\local_files\big_datasets\Prancevic_et_al_2025\sheds_w_nhd_merit_atts_ahthresh.gpkg')
catchments = catchments.loc[catchments['beta'] > 0.0001]
catchments = catchments.loc[catchments['beta'] < 1]


meanahs = [catchments['ah05'].median(),
           catchments['ah15'].median(),
           catchments['ah25'].median(),
           catchments['ah35'].median(),
           catchments['ah45'].median(),
           catchments['ah55'].median(),
           catchments['ah65'].median(),
           catchments['ah75'].median(),
           catchments['ah85'].median(),
           catchments['ah95'].median()]


hydrosheds_ah['ah05'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[0], hydrosheds_ah['ah05']) * 100
hydrosheds_ah['ah15'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[1], hydrosheds_ah['ah15']) * 100
hydrosheds_ah['ah25'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[2], hydrosheds_ah['ah25']) * 100
hydrosheds_ah['ah35'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[3], hydrosheds_ah['ah35']) * 100
hydrosheds_ah['ah45'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[4], hydrosheds_ah['ah45']) * 100
hydrosheds_ah['ah55'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[5], hydrosheds_ah['ah55']) * 100
hydrosheds_ah['ah65'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[6], hydrosheds_ah['ah65']) * 100
hydrosheds_ah['ah75'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[7], hydrosheds_ah['ah75']) * 100
hydrosheds_ah['ah85'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[8], hydrosheds_ah['ah85']) * 100
hydrosheds_ah['ah95'] = np.where((hydrosheds_ah['beta'] > 1) | (hydrosheds_ah['beta'] < 0.0001) | np.isnan(hydrosheds_ah['beta']), meanahs[9], hydrosheds_ah['ah95']) * 100


hydrosheds_ah['STRM_UP_0'] = hydrosheds_ah['STRM_UP'].apply(lambda x: int(x.split('_')[0]) if x != '' else '')
hydrosheds_ah['STRM_UP_1'] = hydrosheds_ah['STRM_UP'].apply(lambda x: int(x.split('_')[1]) if x != '' else '')

print('Imports complete')

def order_streams_dynamic(unfiltered_stream_network, Qd):
    streams = unfiltered_stream_network.copy()
    if Qd != None:
        column = f'ah{Qd}5'
        streams = streams.loc[streams['UPLAND_SKM'] >= streams[column]]
    streams['dynamic_order'] = 0

    streams.loc[(~streams['STRM_UP_0'].isin(streams['STRM_ID'])) &
                (~streams['STRM_UP_1'].isin(streams['STRM_ID'])), 'dynamic_order'] = 1
    
    streams_unordered = streams.loc[streams['dynamic_order'] == 0]


    x = 0
    while len(streams_unordered) > 0:
        streams_unordered = streams.loc[streams['dynamic_order'] == 0]
        next_level_down = streams_unordered.loc[(streams_unordered['STRM_ID'].isin(streams['STRM_DN'])) & (~streams_unordered['STRM_ID'].isin(streams_unordered['STRM_DN']))]

        for i, row in next_level_down.iterrows():
            up_id_0 = row['STRM_UP_0']
            up_id_1 = row['STRM_UP_1']

            stream_ids = streams['STRM_ID'].to_numpy()
            unordered_ids = streams_unordered['STRM_ID'].to_numpy()

            if (up_id_0 not in unordered_ids) & (up_id_1 not in unordered_ids) & (up_id_0 in stream_ids) & (up_id_1 in stream_ids):
                do0 = streams.loc[streams['STRM_ID'] == up_id_0, 'dynamic_order'].to_numpy()[0]
                do1 = streams.loc[streams['STRM_ID'] == up_id_1, 'dynamic_order'].to_numpy()[0]

                if do0 != do1:
                    streams.loc[i, 'dynamic_order'] = np.max([do0, do1])
                elif do0 == do1:
                    streams.loc[i, 'dynamic_order'] = do0 + 1
            

            elif (up_id_0 not in unordered_ids) & (up_id_0 not in stream_ids) & (up_id_1 not in unordered_ids) & (up_id_1 in stream_ids):
                do1 = streams.loc[streams['STRM_ID'] == up_id_1, 'dynamic_order'].to_numpy()[0]
                streams.loc[i, 'dynamic_order'] = do1

            elif (up_id_0 not in unordered_ids) & (up_id_0 in stream_ids) & (up_id_1 not in unordered_ids) & (up_id_1 not in stream_ids):
                do0 = streams.loc[streams['STRM_ID'] == up_id_0, 'dynamic_order'].to_numpy()[0]
                streams.loc[i, 'dynamic_order'] = do0
        if x % 25 == 0:
            print(f'{len(streams_unordered)} remaining')

    lengths = []
    orders = []
    for o in np.arange(1, streams['dynamic_order'].max() + 1, 1):
        dynamic_network_filt = streams.loc[streams['dynamic_order'] == o]
        lengths.append(np.sum(dynamic_network_filt['LENGTH_KM']))
        orders.append(o)
    if Qd != None:
        Qds = [Qd] * len(lengths)
    else:
        Qds = [-1] * len(lengths)
    return pd.DataFrame({'order': orders, 'length_km': lengths, 'Q_decile': Qds})


futures = []
results = []

for Qd in tqdm(range(10)):
    results.append(order_streams_dynamic(hydrosheds_ah, Qd))

for i in range(len(results)):
    df = results[i]
    df.to_csv(f'C:/Users/dego/Desktop/200260820_hydrosheds_lengths_{i}_ah100.csv')

print('Complete')