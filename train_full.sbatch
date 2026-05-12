#!/bin/bash
### EEG2Graph full training job (ZuCo 2, 80 epochs) for the BGU CIS Slurm cluster.
### Submit from the cluster's manager node with `conda` deactivated:
###     sbatch train_full.sbatch
### Job will queue (PD) until an L40S slot frees in your group's quota — Slurm will email when it runs.

#SBATCH --qos=eliyanac                            ### group's prioritized qos
#SBATCH --partition=l40s                          ### 48GB L40S — fits BART-large at batch 16 comfortably
#SBATCH --time=0-06:00:00                         ### ~4h expected for 80 epochs + beam-search test eval; 2h safety margin
#SBATCH --job-name=eeg2graph-full
#SBATCH --output=logs/eeg2graph-full-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il          ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=48G

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph

cd "$HOME/EEG2Graph/model"

python train.py \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --output_dir ../checkpoints \
    --epochs 80 \
    --batch_size 16 \
    --num_workers 4 \
    --save_every 10 \
    --warmup_epochs 10 \
    --bridge_layers 2 \
    --patience 20

echo "END=$(date)"
