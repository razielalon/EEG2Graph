#!/bin/bash
### EEG2Graph E8 — cross-subject: subject-shared vs subject-specific codes (RQ5), ONE point.
### Branch: exp/tier2-grid
###
### E5 (architecture) and E6 (data scale) both left the frozen-BART ~0 ceiling
### untouched. E8 asks the last RQ5 question: is the decodable EEG->triplet code
### SHARED across people, or SUBJECT-SPECIFIC? A model trained without subject S's
### data is scored on S at test time (val/test always keep all 18 subjects).
###
### PAIRED design to strip out the sample-count confound (removing subjects also
### removes training samples — already measured by E6):
###   excl arm  (EXCLUDE set): drop N whole subjects from TRAIN.
###   ctrl arm  (FRAC set):    drop a MATCHED sample count but keep all 18 subjects
###                            (sentence-fraction subsample, deterministic by seed).
### Signal = F1(ctrl) - F1(excl) at matched data size:
###   excl << ctrl  -> codes are subject-specific (can't decode an unseen person).
###   excl ~= ctrl  -> codes are subject-shared (only sample count matters).
###
### Subject embedding is OFF (--n_subject_buckets 0) on purpose: a per-subject
### vector is by construction subject-specific, so leaving it on would trivially
### confound content-transfer with "untrained embedding for an unseen subject."
### => the k0 baseline reproduces E5 cell s0t0aPL (subject OFF, temporal OFF,
### pooled) as a built-in cross-check. Everything else matches E5/E6 (seed 42,
### frozen BART, temporal OFF, pooled align) so the only moving part is the
### train-set composition.
###
### Parametrized via --export env vars:
###   TAG      output-dir + log id: k0 | x03 | x06 | x09 | c03 | c06 | c09
###   EXCLUDE  held-out subject ids, '-' delimited (converted to CSV), '' = none
###   FRAC     --train_sentence_frac for the ctrl arm, '' = full train
###
### After training, seen-vs-unseen subject F1 is scored (cross_subject_eval.py ->
### <ckpt>/test_by_subject.json); aggregate test F1 still lands in test_metrics.json.
###
### Submit the 7 points (all queue in parallel; k0 is the shared baseline):
###   sbatch --export=ALL,TAG=k0,EXCLUDE=,FRAC=                                            exp_e8_cross_subject.sbatch
###   # excl arm — drop whole subjects (nested tail of the sorted 18):
###   sbatch --export=ALL,TAG=x03,EXCLUDE=YSD-YSL-YTL,FRAC=                                exp_e8_cross_subject.sbatch
###   sbatch --export=ALL,TAG=x06,EXCLUDE=YRH-YRK-YRP-YSD-YSL-YTL,FRAC=                    exp_e8_cross_subject.sbatch
###   sbatch --export=ALL,TAG=x09,EXCLUDE=YLS-YMD-YMS-YRH-YRK-YRP-YSD-YSL-YTL,FRAC=        exp_e8_cross_subject.sbatch
###   # ctrl arm — keep all subjects, drop a MATCHED #samples (frac = kept/10216):
###   sbatch --export=ALL,TAG=c03,EXCLUDE=,FRAC=0.827                                      exp_e8_cross_subject.sbatch
###   sbatch --export=ALL,TAG=c06,EXCLUDE=,FRAC=0.664                                      exp_e8_cross_subject.sbatch
###   sbatch --export=ALL,TAG=c09,EXCLUDE=,FRAC=0.491                                      exp_e8_cross_subject.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-04:00:00
#SBATCH --job-name=eeg2graph-e8
#SBATCH --output=logs/eeg2graph-e8-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il          ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=48G

set -euo pipefail

### TAG is required; EXCLUDE/FRAC may be empty (=all subjects / full train).
: "${TAG:?set TAG (k0|x03|x06|x09|c03|c06|c09) via --export}"
EXCLUDE="${EXCLUDE:-}"
FRAC="${FRAC:-}"
### '-'-delimited EXCLUDE -> comma-separated CSV that train.py expects (avoids the
### sbatch --export comma-splitting trap).
EXCLUDE_CSV="${EXCLUDE//-/,}"

OUT="../checkpoints_e8_${TAG}"

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "E8 POINT: TAG=$TAG  EXCLUDE=[$EXCLUDE_CSV]  FRAC=[$FRAC]"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph

cd "$HOME/EEG2Graph/model"

### Build optional flags only when set, so defaults (full train / no exclusion) hold.
FRAC_ARG=(); [ -n "$FRAC" ] && FRAC_ARG=(--train_sentence_frac "$FRAC")

python train.py \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --output_dir "$OUT" \
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
    `### FIXED axes (differ from E6: subject embedding OFF — see header)` \
    --n_subject_buckets 0 \
    --bridge_transformer_layers 0 \
    --bridge_nhead 8 \
    --align_mode pooled \
    `### THE E8 AXIS: train-set composition (excl arm) or matched size (ctrl arm)` \
    --exclude_subjects "$EXCLUDE_CSV" \
    "${FRAC_ARG[@]}" \
    `### FIXED: BART frozen for the entire run (no Phase 2 collapse)` \
    --freeze_bart

### Seen-vs-unseen subject breakdown (aggregate test_metrics.json dilutes the
### held-out minority). Non-fatal: aggregate F1 is already saved by train.py.
python cross_subject_eval.py \
    --checkpoint "$OUT/best_model.pt" \
    --tokenizer_dir "$OUT/tokenizer" \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --exclude_subjects "$EXCLUDE_CSV" \
    --output "$OUT/test_by_subject.json" \
    || echo "WARN: cross_subject_eval.py failed; aggregate test_metrics.json still valid"

echo "END=$(date)"
