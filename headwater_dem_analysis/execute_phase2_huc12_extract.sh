#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=miss_p1
#SBATCH --array=0-83                # 30 HUC4s in the Mississippi
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/dego/headwater_network_extraction/logs/extract_%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

export MISS_WORK=/scratch/$USER/miss/
export MISS_OUT=/scratch/$USER/out
mkdir -p "$MISS_WORK" "$MISS_OUT" logs

python phase2_extract.py \
    --huc4-idx          ${SLURM_ARRAY_TASK_ID} \
    --huc4-path         /home/$USER/headwater_network_extraction/catchment_geodata/huc4_mississippi.gpkg \
    --catchments-path   /home/$USER/headwater_network_extraction/catchment_geodata/sheds_w_nhd_merit_atts_ahthresh_4269.gpkg