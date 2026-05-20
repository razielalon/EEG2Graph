#!/bin/bash
### EEG2Graph tiny-overfit diagnostic for the BGU CIS Slurm cluster.
### Submit from the cluster's manager node with `conda` deactivated:
###     sbatch train_overfit.sbatch
###
### Purpose: can the FULL model (bridge + unfrozen BART, no dropout, no label
### smoothing) memorize 32 training samples? Watch the `train_F1=` column.
###   - train_F1 climbs toward ~1.0  -> pipeline is wired correctly and capable;
###     the real failure is generalization / data scale.
###   - train_F1 stuck near 0 while train_loss -> 0 -> generation/conditioning
###     bug to hunt (the model cannot turn a memorized signal into triplets).

#SBATCH --qos=debug                               ### debug qos for a short diagnostic
#SBATCH --partition=debug
#SBATCH --time=0-00:30:00                         ### ~15 min expected for 40 epochs on 32 samples
#SBATCH --job-name=eeg2graph-overfit
#SBATCH --output=logs/eeg2graph-overfit-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il          ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=48G

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph                         ### replace with your conda env name

cd "$HOME/EEG2Graph/model"                        ### adjust if the repo lives elsewhere on the cluster

python train.py \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --output_dir ../checkpoints_overfit \
    --epochs 40 \
    --batch_size 16 \
    --limit_train 32 \
    --limit_val 8 \
    --limit_test 8 \
    --num_workers 4 \
    --save_every 100 \
    --eval_train \
    `### Maximum capacity to fit: BART unfrozen from epoch 1, no regularization` \
    --warmup_epochs 0 \
    --patience 0 \
    --bridge_layers 2 \
    --bridge_lr 3e-4 \
    --bart_lr 5e-5 \
    --dropout 0.0 \
    --bart_dropout 0.0 \
    --bart_attention_dropout 0.0 \
    --label_smoothing 0.0 \
    --max_src_len 64 \
    --max_tgt_len 96

echo "END=$(date)"
