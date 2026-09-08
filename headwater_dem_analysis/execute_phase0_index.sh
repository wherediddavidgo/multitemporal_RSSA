#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=miss_p1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                      # headroom for largest HUC4s
#SBATCH --time=2:00:00
#SBATCH --output=/home/dego/headwater_network_extraction/logs/create_tile_index_%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

python phase0_index.py