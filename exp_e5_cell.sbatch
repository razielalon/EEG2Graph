#!/bin/bash
### EEG2Graph E5 — frozen-BART component ablation grid (RQ4), ONE grid cell.
### Branch: exp/tier2-grid
###
### Parametrized over the 3 grid axes via --export env vars:
###   SUBJ   per-subject embedding buckets   (0 = off, 64 = on)
###   TEMP   bridge Transformer layers        (0 = off, 2 = on)
###   ALIGN  alignment granularity            (pooled | perword)
###   TAG    short cell id, used in output_dir + log name
###
### All other knobs are FIXED across the 8 cells so any test-F1 difference is
### attributable to the 3 axes alone (frozen BART, bridge_layers 2,
### align_weight 1.0, align_temp 0.07, seed 42, 60 epochs / patience 25).
###
### Submit the 7 new cells (cell 4 = on/on/pooled re-run for a clean table):
###   sbatch --export=ALL,SUBJ=0,TEMP=0,ALIGN=pooled,TAG=s0t0aPL exp_e5_cell.sbatch  # cell 1
###   sbatch --export=ALL,SUBJ=64,TEMP=0,ALIGN=pooled,TAG=s1t0aPL exp_e5_cell.sbatch  # cell 2
###   sbatch --export=ALL,SUBJ=0,TEMP=2,ALIGN=pooled,TAG=s0t1aPL exp_e5_cell.sbatch  # cell 3
###   sbatch --export=ALL,SUBJ=64,TEMP=2,ALIGN=pooled,TAG=s1t1aPL exp_e5_cell.sbatch  # cell 4 (re-run)
###   sbatch --export=ALL,SUBJ=0,TEMP=0,ALIGN=perword,TAG=s0t0aPW exp_e5_cell.sbatch  # cell 5
###   sbatch --export=ALL,SUBJ=64,TEMP=0,ALIGN=perword,TAG=s1t0aPW exp_e5_cell.sbatch  # cell 6
###   sbatch --export=ALL,SUBJ=0,TEMP=2,ALIGN=perword,TAG=s0t1aPW exp_e5_cell.sbatch  # cell 7
###   sbatch --export=ALL,SUBJ=64,TEMP=2,ALIGN=perword,TAG=s1t1aPW exp_e5_cell.sbatch  # cell 8

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-04:00:00
#SBATCH --job-name=eeg2graph-e5
#SBATCH --output=logs/eeg2graph-e5-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il          ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=48G

set -euo pipefail

### Required axis env vars (fail loudly if a submit forgot --export=ALL,...).
: "${SUBJ:?set SUBJ (0 or 64) via --export}"
: "${TEMP:?set TEMP (0 or 2) via --export}"
: "${ALIGN:?set ALIGN (pooled or perword) via --export}"
: "${TAG:?set TAG (short cell id) via --export}"

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "E5 CELL: TAG=$TAG  SUBJ=$SUBJ  TEMP=$TEMP  ALIGN=$ALIGN"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph

cd "$HOME/EEG2Graph/model"

python train.py \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --output_dir "../checkpoints_e5_${TAG}" \
    --seed 42 \
    --epochs 60 \
    --batch_size 16 \
    --num_workers 4 \
    --save_every 10 \
    --bridge_layers 2 \
    --patience 25 \
    --warmup_epochs 10 \
    --bridge_lr 3e-4 \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --align_weight 1.0 \
    --align_temp 0.07 \
    --max_text_len 96 \
    `### AXIS 1: per-subject embedding (0 = off, 64 = on)` \
    --n_subject_buckets "$SUBJ" \
    `### AXIS 2: temporal Transformer pre-encoder (0 = off, 2 layers = on)` \
    --bridge_transformer_layers "$TEMP" \
    --bridge_nhead 8 \
    `### AXIS 3: alignment granularity (pooled | perword)` \
    --align_mode "$ALIGN" \
    `### FIXED: BART frozen for the entire run (no Phase 2 collapse)` \
    --freeze_bart

echo "END=$(date)"
