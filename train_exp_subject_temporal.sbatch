#!/bin/bash
### EEG2Graph EXPERIMENT: SUBJECT-TEMPORAL for the BGU CIS Slurm cluster.
### Branch: exp/subject-temporal
### Delta vs baseline: stacks the two changes that helped most in the first
### round of experiments —
###   1) per-subject embedding (the only run with a nonzero test F1)
###   2) temporal Transformer pre-encoder in the bridge (best Phase-1 fit)
### AND keeps BART FROZEN for the entire run (--freeze_bart) so the Phase 2
### unfreezing collapse — the dominant failure mode in all 4 earlier runs —
### never happens. Bridge-only training with the strongest representation
### learner we've tried.
###     sbatch train_exp_subject_temporal.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-04:00:00
#SBATCH --job-name=eeg2graph-subject-temporal
#SBATCH --output=logs/eeg2graph-subject-temporal-%J.out
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
    --output_dir ../checkpoints_subject_temporal \
    --seed 42 \
    --epochs 60 \
    --batch_size 16 \
    --num_workers 4 \
    --save_every 10 \
    --bridge_layers 2 \
    --patience 25 \
    --bridge_lr 3e-4 \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --align_weight 1.0 \
    --align_temp 0.07 \
    --max_text_len 96 \
    `### EXPERIMENT KNOB 1: per-subject embedding (18 subjects in ZuCo 2.0, 64 buckets w/ headroom)` \
    --n_subject_buckets 64 \
    `### EXPERIMENT KNOB 2: temporal Transformer mixing words in EEG space before REBEL's encoder` \
    --bridge_transformer_layers 2 \
    --bridge_nhead 8 \
    `### EXPERIMENT KNOB 3: BART frozen for the entire run (no Phase 2 collapse)` \
    --freeze_bart

echo "END=$(date)"
