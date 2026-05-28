#!/bin/bash
### EEG2Graph EXPERIMENT: RECON-HEAD for the BGU CIS Slurm cluster.
### Branch: exp/recon-head
### Delta vs baseline: adds a sentence-reconstruction auxiliary loss
###   loss = CE(triplet) + align_weight*InfoNCE + recon_weight*CE(gold text)
### sharing REBEL's decoder. Tests whether dense, position-aware lexical
### supervision (the EEG->text mapping the Ridge probe proved is learnable)
### improves downstream triplet extraction. Everything else identical to
### train_exp_baseline.sbatch (same seed, LRs, schedule, regularization).
###     sbatch train_exp_recon.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-06:00:00
#SBATCH --job-name=eeg2graph-recon
#SBATCH --output=logs/eeg2graph-recon-%J.out
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
    --output_dir ../checkpoints_recon \
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
    `### EXPERIMENT KNOB: sentence-reconstruction auxiliary head` \
    --recon_weight 0.5 \
    --bart_warmup_epochs 3 \
    --llrd_gamma 0.8 \
    --bart_dropout 0.2 \
    --bart_attention_dropout 0.2 \
    --bart_weight_decay 0.05

echo "END=$(date)"
