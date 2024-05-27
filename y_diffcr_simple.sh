#!/bin/bash
# Setup python path and env
# source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
# source /share/hariharan/ck696/env_bh/bin/activate
conda activate /share/hariharan/ck696/env_bh/anaconda/envs/allclear 
cd /share/hariharan/ck696/allclear_0529_lllll/baselines/DiffCR3

/share/hariharan/ck696/env_bh/anaconda/envs/allclear/bin/python diffcr_0526.py --train_dataset "allclear" --workers 8
/share/hariharan/ck696/env_bh/anaconda/envs/allclear/bin/python diffcr_0526.py --train_dataset "allclear" --workers 16
/share/hariharan/ck696/env_bh/anaconda/envs/allclear/bin/python diffcr_0526.py --train_dataset "sen2mtc"