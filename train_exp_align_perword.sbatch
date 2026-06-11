#!/bin/bash
### EEG2Graph EXPERIMENT: ALIGN-PERWORD for the BGU CIS Slurm cluster.
### Branch: exp/align-perword
### Delta vs baseline: replaces pooled InfoNCE alignment with per-position
### InfoNCE — every EEG word position and every text token position is its
### own anchor, with positives drawn from the same sentence across modalities.
### Targets the ~2.7 align-loss plateau all four earlier runs hit, which the
### pooled (B*B) loss could not push past.
###
### Also pushes the BART unfreeze from epoch 11 to epoch 31 (longer Phase 1)
### to give the denser alignment loss more time to actually shape the bridge
### before the decoder shortcut becomes available.
###     sbatch train_exp_align_perword.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-06:00:00
#SBATCH --job-name=eeg2graph-align-perword
#SBATCH --output=logs/eeg2graph-align-perword-%J.out
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
    --output_dir ../checkpoints_align_perword \
    --epochs 80 \
    --batch_size 16 \
    --num_workers 4 \
    --save_every 10 \
    --bridge_layers 2 \
    --patience 20 \
    --bridge_lr 3e-4 \
    --bart_lr 3e-5 \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --align_weight 1.0 \
    --align_temp 0.07 \
    --max_text_len 96 \
    `### EXPERIMENT KNOB 1: per-position InfoNCE instead of pooled` \
    --align_mode perword \
    `### EXPERIMENT KNOB 2: longer Phase 1 (BART frozen 30 epochs vs baseline 10)` \
    --warmup_epochs 30 \
    --bart_warmup_epochs 3 \
    --llrd_gamma 0.8 \
    --bart_dropout 0.2 \
    --bart_attention_dropout 0.2 \
    --bart_weight_decay 0.05

echo "END=$(date)"
