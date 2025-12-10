import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import lognorm
from tqdm import tqdm
import os

# ================================================================
# LOAD MS_PTS LOOKUP TABLE
# ================================================================
ms_pts = (
    gpd.read_file(r"C:\Users\dego\Documents\local_files\RSSA\glow_pts_on_grwl_nolake.gpkg")
    .drop(columns="geometry")
)

ID_arr = ms_pts["ID"].unique().tolist()

# Connect and register ms_pts
con = duckdb.connect()
con.register("ms_pts", ms_pts)

# ================================================================
# CREATE VIEWS FOR GLOW + GDL
# ================================================================
con.execute("""
    CREATE OR REPLACE VIEW glow AS 
        SELECT ID, COMID, date, width
        FROM parquet_scan(
            'c:/Users/dego/Documents/local_files/sandbox/GLOW/width/width/GLOW_width_region_7.parquet'
        );
""")

con.execute("""
    CREATE OR REPLACE VIEW GDL AS 
        SELECT COMID, date, Qout
        FROM parquet_scan(
            'C:/Users/dego/Documents/local_files/RSSA/grades_Q/*.parquet'
        );
""")

# Directory for output
out_dir = r"C:\Users\dego\Documents\local_files\RSSA\GLOW_analysis_chunks"
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# BATCH EXECUTION
# ================================================================
CHUNK_SIZE = 1000
n_batches = (len(ID_arr) + CHUNK_SIZE - 1) // CHUNK_SIZE

print("Total IDs:", len(ID_arr))
print("Batches:", n_batches)

chunk_files = []

# ---------------------------------------------------------------
# PREPARED QUERY (parameterized, no manual string building)
# ---------------------------------------------------------------
QUERY = r"""
    WITH glow_chunk AS (
        SELECT
            g.ID,
            g.COMID,
            g.date,
            g.width,
            ROW_NUMBER() OVER (PARTITION BY g.ID ORDER BY g.width) AS w_rank,
            COUNT(*)     OVER (PARTITION BY g.ID)                  AS w_count
        FROM glow g
        WHERE g.ID = ANY($1)
    ),
    glow_ranked AS (
        SELECT *,
            CASE 
                WHEN (w_rank - 1)::DOUBLE / NULLIF(w_count - 1, 0) = 1.0 THEN 9
                ELSE FLOOR((w_rank - 1)::DOUBLE / NULLIF(w_count - 1, 0) * 10)
            END AS w_decile
        FROM glow_chunk
    ),
    Q_chunk AS (
        SELECT
            q.COMID,
            q.date,
            q.Qout,
            ROW_NUMBER() OVER (PARTITION BY q.COMID ORDER BY q.Qout) AS Q_rank,
            COUNT(*)     OVER (PARTITION BY q.COMID)                  AS Q_count
        FROM GDL q
        WHERE q.COMID = ANY($2)
    ),
    Q_ranked AS (
        SELECT *,
            CASE 
                WHEN (Q_rank - 1)::DOUBLE / NULLIF(Q_count - 1, 0) = 1.0 THEN 9
                ELSE FLOOR((Q_rank - 1)::DOUBLE / NULLIF(Q_count - 1, 0) * 10)
            END AS Q_decile
        FROM Q_chunk
    )

    SELECT
        m.NHD_order,
        Q_ranked.Q_decile,
        glow_ranked.width
    FROM glow_ranked
    JOIN ms_pts AS m USING (COMID)
    JOIN Q_ranked USING (COMID, date)
    WHERE glow_ranked.width > 0
    """

# ================================================================
# PROCESS BATCHES
# ================================================================
for i in tqdm(range(n_batches)):
    
    batch_ids = ID_arr[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]

    # -----------------------------------------------------------
    # GET COMIDs belonging to these IDs (correct + safe)
    # -----------------------------------------------------------
    batch_comids = (
        con.execute("SELECT DISTINCT COMID FROM glow WHERE ID = ANY($1)", [batch_ids])
          .fetchdf()
          .COMID
          .tolist()
    )

    # Edge case: If no COMIDs exist (rare, but safe to check)
    if len(batch_comids) == 0:
        print(f"WARNING: No COMIDs found for batch {i}. Skipping.")
        continue

    # -----------------------------------------------------------
    # EXECUTE THE MAIN QUERY (parameterized)
    # -----------------------------------------------------------
    df_batch = con.execute(QUERY, [batch_ids, batch_comids]).df()

    # -----------------------------------------------------------
    # WRITE CHUNK TO PARQUET
    # -----------------------------------------------------------
    out_path = f"{out_dir}/chunk_{i}.parquet"
    con.execute(f"""
        COPY (SELECT * FROM df_batch)
        TO '{out_path}' (FORMAT PARQUET);
    """)
    chunk_files.append(out_path)

print("Complete.")
