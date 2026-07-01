#!/bin/bash
### EEG2Graph E6 — data-scaling curve (RQ4 follow-up), ONE fraction point.
### Branch: exp/tier2-grid
###
### E5 closed the architecture question: under frozen BART no architectural axis
### lifts the ~0 ceiling (subject / temporal / per-word alignment all flat at the
### floor). The remaining lever is DATA SCALE. E6 holds the best E5 cell fixed
### (#2: subject=64, temporal=0, align=pooled, frozen BART, seed 42) and sweeps
### the fraction of TRAIN *sentences* seen during training.
###
### Read-off: if test F1 climbs with more sentences, the frozen ceiling is a
### sample-size artifact (more data would help). If the curve is flat at ~0, the
### E1 fidelity cliff is the binding constraint at this data scale.
###
### Parametrized via --export env vars:
###   FRAC   fraction of TRAIN unique sentences, in (0,1]   (0.1 | 0.25 | 0.5 | 1.0)
###   TAG    short id used in output_dir + log name         (f010|f025|f050|f100)
###
### Everything else is byte-identical to E5 cell 2 so the only moving part is FRAC.
### frac=1.0 (TAG=f100) reproduces E5 cell 2 (checkpoints_e5_s1t0aPL, test F1
### 0.0083) under identical code/seed — re-run it fresh for a self-contained E6
### table (cheap insurance), or symlink checkpoints_e6_f100 -> checkpoints_e5_s1t0aPL.
###
### Submit the 4 fraction points (all queue in parallel):
###   sbatch --export=ALL,FRAC=0.1,TAG=f010  exp_e6_scale.sbatch
###   sbatch --export=ALL,FRAC=0.25,TAG=f025 exp_e6_scale.sbatch
###   sbatch --export=ALL,FRAC=0.5,TAG=f050  exp_e6_scale.sbatch
###   sbatch --export=ALL,FRAC=1.0,TAG=f100  exp_e6_scale.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-04:00:00
#SBATCH --job-name=eeg2graph-e6
#SBATCH --output=logs/eeg2graph-e6-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il          ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=48G

set -euo pipefail

### Required axis env vars (fail loudly if a submit forgot --export=ALL,...).
: "${FRAC:?set FRAC (0.1|0.25|0.5|1.0) via --export}"
: "${TAG:?set TAG (short id, e.g. f050) via --export}"

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "E6 POINT: TAG=$TAG  FRAC=$FRAC"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph

cd "$HOME/EEG2Graph/model"

python train.py \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --output_dir "../checkpoints_e6_${TAG}" \
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
    `### FIXED = best E5 cell (#2): subject ON (64), temporal OFF, pooled align` \
    --n_subject_buckets 64 \
    --bridge_transformer_layers 0 \
    --bridge_nhead 8 \
    --align_mode pooled \
    `### THE E6 AXIS: fraction of TRAIN unique sentences (val/test stay full)` \
    --train_sentence_frac "$FRAC" \
    `### FIXED: BART frozen for the entire run (no Phase 2 collapse)` \
    --freeze_bart

echo "END=$(date)"
