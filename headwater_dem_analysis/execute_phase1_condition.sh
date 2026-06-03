#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=miss_p1
#SBATCH --array=39              # 30 HUC4s in the Mississippi
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                 # headroom for largest HUC4s
#SBATCH --time=18:00:00
#SBATCH --output=/home/dego/headwater_network_extraction/logs/condition_%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

export MISS_WORK=/scratch/$USER/miss
export MISS_OUT=/scratch/$USER/out
mkdir -p "$MISS_WORK" "$MISS_OUT" logs

python -u /home/dego/headwater_network_extraction/scripts/claude/phase1_condition.py \
    --huc4-idx        ${SLURM_ARRAY_TASK_ID} \
    --huc4-path       /home/$USER/headwater_network_extraction/catchment_geodata/huc4_mississippi.gpkg \
    --tile-idx-path   /scratch/$USER/miss/tile_idx_4269.gpkg \
    --grwl-path       /home/$USER/headwater_network_extraction/catchment_geodata/ms_grwl.shp