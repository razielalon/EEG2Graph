#!/bin/bash
### EEG2Graph EXPERIMENT: TEMPORAL-BRIDGE for the BGU CIS Slurm cluster.
### Branch: exp/temporal-bridge
### Delta vs baseline: inserts a small Transformer encoder (--bridge_transformer_layers)
### after the bridge projection, before REBEL's encoder, so the bridge can mix
### EEG information ACROSS word positions instead of projecting each word
### independently. Tests whether the pointwise bridge is the capacity limit.
### Everything else identical to train_exp_baseline.sbatch.
###     sbatch train_exp_temporal.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-06:00:00
#SBATCH --job-name=eeg2graph-temporal
#SBATCH --output=logs/eeg2graph-temporal-%J.out
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
    --output_dir ../checkpoints_temporal \
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
    --align_weight 1.0 \
    --align_temp 0.07 \
    --max_text_len 96 \
    `### EXPERIMENT KNOB: temporal Transformer pre-encoder over word positions` \
    --bridge_transformer_layers 3 \
    --bridge_nhead 8 \
    --bart_warmup_epochs 3 \
    --llrd_gamma 0.8 \
    --bart_dropout 0.2 \
    --bart_attention_dropout 0.2 \
    --bart_weight_decay 0.05

echo "END=$(date)"
