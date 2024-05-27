#!/bin/bash
#SBATCH --mail-user=ck696@cornell.edu    # Email for status updates
#SBATCH --mail-type=END                  # Request status by email
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 80:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition
#SBATCH --constraint="gpu-high|gpu-mid"  # GPU constraint
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=60G
#SBATCH --cpus-per-task=16                # Number of CPU cores per task
#SBATCH -N 1                             # Number of nodes
#SBATCH--output=watch_folder/%x-%j.log   # Output file name
#SBATCH --requeue                        # Requeue job if it fails

# Setup python path and env
# source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
source /share/hariharan/ck696/env_bh/bin/activate
conda activate /share/hariharan/ck696/env_bh/anaconda/envs/allclear 
cd /share/hariharan/ck696/allclear_0529_lllll/baselines/DiffCR3

python diffcr_0526.py --train_dataset "allclear" --workers 8
# python diffcr_0526.py --train_dataset "sen2mtc"