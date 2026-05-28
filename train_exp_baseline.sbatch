#!/bin/bash
### EEG2Graph EXPERIMENT: BASELINE (seeded reference run) for the BGU CIS Slurm cluster.
### Branch: exp/baseline
### Purpose: the controlled reference for the experiment matrix. Identical to
### train_full.sbatch but with a fixed --seed so the experiment branches
### (recon-head / subject-cond / temporal-bridge) are compared against a
### deterministic baseline rather than an un-seeded historical run.
###     sbatch train_exp_baseline.sbatch

#SBATCH --qos=eliyanac                            ### group's prioritized qos
#SBATCH --partition=l40s                          ### 48GB L40S — fits BART-large at batch 16 comfortably
#SBATCH --time=0-06:00:00                         ### ~4h expected for 80 epochs + beam-search test eval; 2h safety margin
#SBATCH --job-name=eeg2graph-baseline
#SBATCH --output=logs/eeg2graph-baseline-%J.out
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

cd "$HOME/exp_baseline/EEG2Graph/model"

python train.py \
    --processed_dir $HOME/exp_baseline/EEG2Graph/processed_zuco2 \
    --triplets_path $HOME/exp_baseline/EEG2Graph/processed_zuco2/sentence_triplets.json \
    --output_dir ../checkpoints_baseline \
    --seed 42 \
    --epochs 80 \
    --batch_size 16 \
    --num_workers 4 \
    --save_every 10 \
    --warmup_epochs 10 \
    --bridge_layers 2 \
    --patience 20 \
    --bridge_lr 3e-4 \
    --bart_lr 3e-5 \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    `### Text-encoder alignment: direct gradient into the bridge, bypasses the decoder shortcut` \
    --align_weight 1.0 \
    --align_temp 0.07 \
    --max_text_len 96 \
    `### Phase 2 (post-unfreeze) regularization — see TUTORIAL.md §10` \
    --bart_warmup_epochs 3 \
    --llrd_gamma 0.8 \
    --bart_dropout 0.2 \
    --bart_attention_dropout 0.2 \
    --bart_weight_decay 0.05

echo "END=$(date)"
