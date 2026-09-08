#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=import_rasters
#SBATCH --time=6:00:00
#SBATCH --output=/home/dego/headwater_network_extraction/logs/import_nhd_rasters_%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

python /home/dego/headwater_network_extraction/scripts/claude/phase_1_import_nhd_rasters.py \
    --catchments-path /home/dego/headwater_network_extraction/catchment_geodata/huc4_mississippi.gpkg \
    --raster-list-path /home/dego/headwater_network_extraction/catchment_geodata/nhd_raster_files.csv \
    --fdr-dir /home/dego/headwater_network_extraction/NHD_fdr \
    --fac-dir /home/dego/headwater_network_extraction/NHD_fac \
    --tmp-dir /scratch/dego/miss