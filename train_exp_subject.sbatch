#!/bin/bash
### EEG2Graph EXPERIMENT: SUBJECT-COND for the BGU CIS Slurm cluster.
### Branch: exp/subject-cond
### Delta vs baseline: adds a learned per-subject embedding (subject_id hashed
### into --n_subject_buckets buckets) to every bridge output. Tests whether
### explicitly factoring out inter-subject EEG variability (Probe 2 = 0.246
### subject decodability) frees REBEL to focus on content and lifts F1.
### Everything else identical to train_exp_baseline.sbatch.
###     sbatch train_exp_subject.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-06:00:00
#SBATCH --job-name=eeg2graph-subject
#SBATCH --output=logs/eeg2graph-subject-%J.out
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
    --output_dir ../checkpoints_subject \
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
    `### EXPERIMENT KNOB: per-subject embedding (18 subjects in ZuCo 2.0, 64 buckets w/ headroom)` \
    --n_subject_buckets 64 \
    --bart_warmup_epochs 3 \
    --llrd_gamma 0.8 \
    --bart_dropout 0.2 \
    --bart_attention_dropout 0.2 \
    --bart_weight_decay 0.05

echo "END=$(date)"
