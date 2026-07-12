#!/bin/bash
### EEG2Graph E8 — RE-SCORE ONLY: fix the seen/unseen beam-width mismatch.
### Branch: exp/tier2-grid
###
### BUG being fixed: the original exp_e8_cross_subject.sbatch called
### cross_subject_eval.py WITHOUT --beam_size, so it fell back to its old default of
### 1 (greedy) — while train.py's test eval uses its own default of 4 (beam). Every
### test_f1 in e8_cross_subject.json is therefore beam-4, and every seen_f1/unseen_f1
### is greedy: two different decoders on the same checkpoint. The tell was k0, which
### excludes nobody (so its "seen" partition IS the whole test set) yet reported
### seen_f1=0.0000 against test_f1=0.0040.
###
### unseen F1 is the direct generalize-to-a-new-person number RQ5 turns on, so it has
### to be on the same decoding path as everything else in the thesis.
###
### This re-runs ONLY the eval step (inference on the test split) against the 7
### checkpoints that already exist — no retraining, nothing in the training path is
### touched, so E5/E6/E8 training reproducibility is unaffected. It overwrites each
### checkpoints_e8_<TAG>/test_by_subject.json in place.
###
### Run:
###   sbatch exp_e8_rescore.sbatch
### then re-aggregate:
###   cd model && python aggregate_e8.py

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-02:00:00
#SBATCH --job-name=eeg2graph-e8-rescore
#SBATCH --output=logs/eeg2graph-e8-rescore-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il          ### replace with your BGU email if different
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=48G

set -euo pipefail

echo "JOBID=$SLURM_JOBID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "START=$(date)"
nvidia-smi

module load anaconda
source activate eeg2graph

cd "$HOME/EEG2Graph/model"

### TAG -> held-out subject CSV. Must match exactly what each run was TRAINED with,
### or the seen/unseen partition is wrong. ctrl arm (c*) and k0 excluded nobody.
declare -A EXCL=(
    [k0]=""
    [x03]="YSD,YSL,YTL"
    [x06]="YRH,YRK,YRP,YSD,YSL,YTL"
    [x09]="YLS,YMD,YMS,YRH,YRK,YRP,YSD,YSL,YTL"
    [c03]=""
    [c06]=""
    [c09]=""
)

for TAG in k0 x03 x06 x09 c03 c06 c09; do
    OUT="../checkpoints_e8_${TAG}"
    if [ ! -f "$OUT/best_model.pt" ]; then
        echo "SKIP $TAG: no $OUT/best_model.pt"
        continue
    fi
    echo "=== RESCORE $TAG  (held-out: [${EXCL[$TAG]}])  beam=4 ==="
    python cross_subject_eval.py \
        --checkpoint "$OUT/best_model.pt" \
        --tokenizer_dir "$OUT/tokenizer" \
        --processed_dir ../processed_zuco2 \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --exclude_subjects "${EXCL[$TAG]}" \
        --beam_size 4 \
        --output "$OUT/test_by_subject.json" \
        || echo "WARN: rescore failed for $TAG"
done

echo "=== RE-AGGREGATE ==="
python aggregate_e8.py

echo "END=$(date)"
