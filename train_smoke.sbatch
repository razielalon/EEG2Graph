#!/bin/bash
### EEG2Graph smoke-test training job for the BGU CIS Slurm cluster.
### Submit from the cluster's manager node with `conda` deactivated:
###     sbatch train_smoke.sbatch

#SBATCH --account=eliyanac                        ### Aliya's accounting group (per IT)
#SBATCH --qos=eliyanac
#SBATCH --time=0-02:00:00
#SBATCH --job-name=eeg2graph-smoke
#SBATCH --output=logs/eeg2graph-smoke-%J.out
#SBATCH --mail-user=razielal@post.bgu.ac.il    ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1                         ### 24GB card — enough for BART-large at batch 16
#SBATCH --mem=48G                                 ### override default 24G; HF model load + dataloaders

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph                         ### replace with your conda env name

cd "$HOME/EEG2Graph/model"                        ### adjust if the repo lives elsewhere on the cluster

python train.py \
    --processed_dir ../processed_zuco \
    --triplets_path ../processed_zuco/sentence_triplets.json \
    --output_dir ../checkpoints_smoke \
    --epochs 3 \
    --batch_size 16 \
    --limit_train 200 \
    --limit_val 50 \
    --limit_test 50 \
    --num_workers 4 \
    --save_every 100

echo "END=$(date)"
