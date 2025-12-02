import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import lognorm
from tqdm import tqdm
import os

# ms_pts = (
#     gpd.read_file(r"C:\Users\dego\Documents\local_files\RSSA\MS_GLOW_GRADESDL_pts.gpkg")
#     .drop(columns="geometry")
# )

# COMID_arr = ms_pts["COMID"].unique().tolist()

con = duckdb.connect()
# con.register("ms_pts", ms_pts)

# con = duckdb.connect()
# con.register("ms_pts", ms_pts)

# con.execute("""
# CREATE VIEW glow AS 
#   SELECT ID, COMID, date, width
#   FROM parquet_scan('c:/Users/dego/Documents/local_files/sandbox/GLOW/width/width/GLOW_width_region_7.parquet')
# """);

# con.execute(f"""
# CREATE VIEW GDL AS 
#   SELECT COMID, date, Qout
#   FROM parquet_scan('C:/Users/dego/Documents/local_files/RSSA/grades_Q/*.parquet')
#   WHERE COMID IN ({','.join(map(str, COMID_arr))})
# """);

# CHUNK_SIZE = 1000
# n_batches = (len(COMID_arr) + CHUNK_SIZE - 1) // CHUNK_SIZE
# print("Total COMIDs:", len(COMID_arr))
# print("Batches:", n_batches)

out_dir = r"C:\Users\dego\Documents\local_files\RSSA\GLOW_analysis_chunks"
# os.makedirs(out_dir, exist_ok=True)

# chunk_files = []

# for i in tqdm(range(n_batches)):
#     batch_comids = COMID_arr[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
#     batch_str = ",".join(map(str, batch_comids))

#     df_batch = con.execute(f"""
#         WITH glow_chunk AS (
#             SELECT
#                 g.ID,
#                 g.COMID,
#                 g.date,
#                 g.width,
#                 ROW_NUMBER() OVER (PARTITION BY g.ID ORDER BY g.width) AS w_rank,
#                 COUNT(*)     OVER (PARTITION BY g.ID)                 AS w_count
#             FROM glow g
#             WHERE g.COMID IN ({batch_str})
#         ),
#         glow_ranked AS (
#             SELECT *,
#                 (w_rank - 1)::DOUBLE / NULLIF((w_count - 1), 0) AS w_percentile,
#                 FLOOR((w_rank - 1)::DOUBLE / NULLIF((w_count - 1), 0) * 10) AS w_decile
#             FROM glow_chunk
#         ),
#         Q_chunk AS (
#             SELECT
#                 q.COMID,
#                 q.date,
#                 q.Qout,
#                 ROW_NUMBER() OVER (PARTITION BY q.COMID ORDER BY q.Qout) AS Q_rank,
#                 COUNT(*)     OVER (PARTITION BY q.COMID)                 AS Q_count
#             FROM GDL q
#             WHERE q.COMID IN ({batch_str})
#         ),
#         Q_ranked AS (
#             SELECT *,
#                 (Q_rank - 1)::DOUBLE / NULLIF((Q_count - 1), 0) AS Q_percentile,
#                 FLOOR((Q_rank - 1)::DOUBLE / NULLIF((Q_count - 1), 0) * 10) AS Q_decile
#             FROM Q_chunk
#         )
#         SELECT
#             m.order,
#             Q_ranked.Q_decile,
#             glow_ranked.width
#         FROM glow_ranked
#         JOIN ms_pts m
#         ON glow_ranked.COMID = m.COMID
#         JOIN Q_ranked
#         ON glow_ranked.COMID = Q_ranked.COMID
#            AND glow_ranked.date = Q_ranked.date
#         WHERE glow_ranked.width > 0
#     """).df()

#     out_path = f"{out_dir}/chunk_{i}.parquet"
#     con.execute(f"""
#                 COPY (SELECT * FROM df_batch)
#                 TO '{out_path}'
#                 (FORMAT PARQUET);
#                 """)
#     chunk_files.append(out_path)

result = duckdb.query(f"""
SELECT * FROM parquet_scan('C:/Users/dego/Documents/local_files/RSSA/GLOW_analysis_chunks/chunk*.parquet')
""")

# ================================================================
# LOGNORMAL MLE SUMMARY STATS (NO LISTS!)
# ================================================================
grouped = con.execute("""
    SELECT
        "order",
        Q_decile,
        COUNT(*) AS n,
        AVG(LOG(width))      AS mean_logw,
        VAR_POP(LOG(width))  AS var_logw
    FROM parquet_scan('C:/Users/dego/Documents/local_files/RSSA/GLOW_analysis_chunks/*.parquet')
    GROUP BY 1,2
""").df()

print(grouped.head())

# ================================================================
# CONVERT MLE STATS → LOGNORMAL PARAMETERS
# ================================================================
fits = []

for _, row in tqdm(grouped.iterrows()):
    mu = row["mean_logw"]
    var = row["var_logw"]
    sigma = np.sqrt(var)

    # Lognormal MLE parameters:
    shape = sigma      # shape parameter (sigma)
    scale = np.exp(mu) # scale parameter (exp(mu))
    loc = 0            # loc is always 0 for MLE

    fits.append({
        "order": row["order"],
        "Q_decile": row["Q_decile"],
        "shape": shape,
        "scale": scale,
        "loc": loc,
        "n": row["n"]
    })

fit_df = pd.DataFrame(fits)
fit_df.to_csv(r"C:\Users\dego\Desktop\lognormal_fits_duckdb.csv", index=False)

print("Complete.")