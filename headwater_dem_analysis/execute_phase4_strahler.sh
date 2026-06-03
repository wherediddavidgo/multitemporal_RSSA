#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=merge_p95
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --output=/home/dego/headwater_network_extraction/logs/order_%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

python /home/dego/headwater_network_extraction/scripts/claude/phase4_strahler.py \
    --dem-network-path /home/dego/headwater_network_extraction/output/network_ah95_merged.gpkg \
    --merit-network-path /home/dego/headwater_network_extraction/catchment_geodata/merit_usms_filt.gpkg \
    --output-path /home/dego/headwater_network_extraction/output/ordered_ah95.gpkg