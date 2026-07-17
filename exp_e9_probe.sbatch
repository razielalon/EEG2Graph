#!/bin/bash
### EEG2Graph E9 — bridge/encoder retrieval probe (RQ1/RQ2), POST-HOC, ONE job.
### Branch: exp/e9-retrieval (forks exp/tier2-grid so load_model sees the
###          subject/temporal/align arch flags the E5/E6/E8 checkpoints were
###          trained with).
###
### THE QUESTION E9 SETTLES
### E4 found a linear probe on the RAW EEG retrieves the read sentence at ~2.6x
### chance (top-10 0.394). E7 found the TRAINED bridge's InfoNCE alignment never
### beats its chance floor. Those read as a contradiction ("signal exists" vs
### "bridge learns nothing") only because they measure different things. E9 runs
### E4's *identical* Ridge->TF-IDF retrieval probe on the representation the model
### actually produces, changing ONLY the feature matrix:
###   raw            reference  (~0.394, reproduced here as a self-check)
###   random_bridge  control    (untrained Linear+LN; ~= raw by JL — a generic
###                              projection does NOT destroy the signal)
###   bridge         VERDICT    (a trained checkpoint's bridge output)
###   encoder        VERDICT    (REBEL encoder states over the bridge output —
###                              i.e. exactly what the frozen decoder cross-attends to)
###   text           upper bnd  (REBEL's own text-encoder states; must land ~1.0,
###                              reproducing E1's alpha=1 retrieval — a pooling sanity check)
###
### DECISION RULE
###   bridge/encoder ~= raw    -> training PRESERVED the signal; the wall is
###                              DOWNSTREAM (frozen decoder / objective can't consume
###                              it). "Information present but unusable" — a different,
###                              more tractable thesis than "no signal at all".
###   bridge/encoder ~= chance -> training DESTROYED the signal; the bridge collapsed
###                              and the objective actively harms a signal that was there.
### Unlike triplet-F1 (pinned at 0 by exact match) this metric has real dynamic range,
### so either outcome is a positive, quotable result for the thesis.
###
### This is inference-only over the checkpoints already on the cluster: no training,
### no retraining. It fits Ridge on TRAIN features and scores retrieval on TEST for
### every (source x checkpoint). Runtime is dominated by the REBEL encoder forward
### over ~10k train + ~1.2k test sentences per checkpoint.
###
### SUBMIT (from $HOME/EEG2Graph, after `git checkout exp/e9-retrieval`):
###   mkdir -p logs
###   sbatch exp_e9_probe.sbatch
### It auto-discovers every checkpoints_e5_* / e6_* / e8_* dir present. To probe a
### subset instead, edit CKPT_DIRS below (e.g. the cleanest core:
###   ../checkpoints_e8_k0 ../checkpoints_e5_s1t0aPL ../checkpoints_e5_s0t0aPW).

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-04:00:00
#SBATCH --job-name=eeg2graph-e9
#SBATCH --output=logs/eeg2graph-e9-%J.out
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

### Auto-discover the 19 E5/E6/E8 checkpoints (whatever is present). `|| true`
### keeps `set -e` happy when a glob has no match. Label = dir basename.
CKPT_DIRS=$(ls -d ../checkpoints_e5_* ../checkpoints_e6_* ../checkpoints_e8_* 2>/dev/null || true)
echo "CHECKPOINTS DISCOVERED:"
echo "$CKPT_DIRS" | tr ' ' '\n'

python bridge_retrieval_probe.py \
    --processed_dir ../processed_zuco2 \
    --sources raw,random_bridge,bridge,encoder,text \
    --checkpoints $CKPT_DIRS \
    --batch_size 16 \
    --out ../exp_logs/e9_retrieval.json

echo "END=$(date)"
