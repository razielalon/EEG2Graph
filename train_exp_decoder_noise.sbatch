#!/bin/bash
### EEG2Graph EXPERIMENT: DECODER-NOISE for the BGU CIS Slurm cluster.
### Branch: exp/decoder-noise
### Delta vs baseline: random token corruption on the decoder INPUT during
### training. Each non-structural input token has p=0.25 chance of being
### replaced with <mask>; labels stay unchanged. The decoder can no longer
### rely purely on perfect teacher-forced previous tokens — at corrupted
### steps it MUST use cross-attention (and therefore the EEG-side encoder
### representation) to predict the next token.
###
### Targets the dominant failure across all 4 prior runs: identical predictions
### for every batch entry, train_loss → ~4.5, val/test F1 ≈ 0 (decoder solves
### the task as a language model and ignores the EEG).
###     sbatch train_exp_decoder_noise.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-06:00:00
#SBATCH --job-name=eeg2graph-decoder-noise
#SBATCH --output=logs/eeg2graph-decoder-noise-%J.out
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
    --output_dir ../checkpoints_decoder_noise \
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
    `### EXPERIMENT KNOB: token noise on decoder input (mirrors BART's denoising pretraining)` \
    --decoder_noise_prob 0.25 \
    --bart_warmup_epochs 3 \
    --llrd_gamma 0.8 \
    --bart_dropout 0.2 \
    --bart_attention_dropout 0.2 \
    --bart_weight_decay 0.05

echo "END=$(date)"
