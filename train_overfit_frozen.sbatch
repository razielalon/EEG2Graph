#!/bin/bash
### EEG2Graph FROZEN-BART overfit diagnostic for the BGU CIS Slurm cluster.
### Submit from the cluster's manager node with `conda` deactivated:
###     sbatch train_overfit_frozen.sbatch
###
### Companion to train_overfit.sbatch. That run let BART unfreeze from epoch 1;
### the 406M-param decoder memorized the target text as a language model and
### ignored the EEG entirely -- every sample in the batch generated the SAME
### triplet and train_F1 stayed ~0 even as train_loss fell to 0.16. This run
### FREEZES BART for the whole run, so bridge -> frozen REBEL is the ONLY path
### from EEG to loss; the decoder cannot fall back on its language-model prior.
###
### Purpose: is there ANY decodable EEG->triplet signal? Watch `train_F1=`.
###   - train_F1 climbs (toward >0.5) -> the bridge CAN carry EEG->triplet
###     signal; the design is salvageable -> keep BART frozen far longer / use
###     a much longer Phase 1 before any unfreeze.
###   - train_F1 stuck near 0 -> even with the decoder forced to depend on the
###     encoder the bridge cannot memorize 32 mappings -> the EEG signal (or
###     the bridge) is too weak; a different conditioning approach is needed.

#SBATCH --qos=debug                               ### debug qos for a short diagnostic
#SBATCH --partition=debug
#SBATCH --time=0-00:30:00                         ### bridge-only epochs are fast
#SBATCH --job-name=eeg2graph-ovf-frozen
#SBATCH --output=logs/eeg2graph-ovf-frozen-%J.out
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
    --output_dir ../checkpoints_overfit_frozen \
    --epochs 150 \
    --batch_size 16 \
    --limit_train 32 \
    --limit_val 8 \
    --limit_test 8 \
    --num_workers 4 \
    --save_every 200 \
    --eval_train \
    `### BART frozen for the ENTIRE run: bridge -> frozen REBEL is the only path` \
    --freeze_bart \
    --warmup_epochs 0 \
    --patience 0 \
    `### Maximum bridge capacity to fit: deeper bridge, high LR, no regularization` \
    --bridge_layers 3 \
    --bridge_lr 1e-3 \
    --dropout 0.0 \
    --bart_dropout 0.0 \
    --bart_attention_dropout 0.0 \
    --label_smoothing 0.0 \
    --max_src_len 64 \
    --max_tgt_len 96 \
    --beam_size 1

echo "END=$(date)"
