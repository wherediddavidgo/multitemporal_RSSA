#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=mosaic_fdr_fac_rasters
#SBATCH --time=2:00:00
#SBATCH --mem=512G
#SBATCH --output=/home/dego/headwater_network_extraction/logs/mosaic_nhd_inputs_%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

python /home/dego/headwater_network_extraction/scripts/claude/phase0a_mosaic_nhd_inputs.py \
    --fdr-in-dir /scratch/dego/miss/corrected_nhd_fdr \
    --fac-in-dir /scratch/dego/miss/corrected_nhd_fac \
    --fdr-out-dir /scratch/dego/miss/corrected_nhd_fdr \
    --fac-out-dir /scratch/dego/miss/corrected_nhd_fac \