#!/bin/bash
#SBATCH --account=rivermap
#SBATCH --job-name=miss_p1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --mem=400G
#SBATCH --qos=tc_normal_short
#SBATCH --time=4:00:00
#SBATCH --output=/home/dego/headwater_network_extraction/logs/merge_nostrahler%A_%a.out

module load Miniforge3
source activate /home/dego/.conda/envs/headwater_env_tc

export MISS_WORK=/scratch/$USER/miss/
export MISS_OUT=/home/$USER/headwater_network_extraction/output
mkdir -p "$MISS_WORK" "$MISS_OUT" logs

python phase3_merge_networks.py \
    --percentile-idx    ${SLURM_ARRAY_TASK_ID}
