#!/bin/bash
### EEG2Graph EXPERIMENT E2: EEG-ablation eval (RQ3 causal test) — decoder_noise.
### Branch: exp/decoder-noise  (must be checked out — uses this branch's model code)
###
### Inference-only. Generates triplets under real / swap (deranged EEG) / zero and
### compares F1 + identical_to_real. Writes its OWN json so it can't race a
### concurrent subject_temporal run. Merge the per-run jsons when writing up.
###     git checkout exp/decoder-noise && sbatch exp_e2_run.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-00:30:00
#SBATCH --job-name=eeg2graph-e2-decoder-noise
#SBATCH --output=logs/eeg2graph-e2-decoder-noise-%J.out
#SBATCH --mail-user=zivcz@post.bgu.ac.il
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

python eeg_ablation_eval.py \
    --checkpoint ../checkpoints_decoder_noise/best_model.pt \
    --tokenizer_dir ../checkpoints_decoder_noise/tokenizer \
    --processed_dir ../processed_zuco2 \
    --triplets_path ../processed_zuco2/sentence_triplets.json \
    --split test --beam_size 1 \
    --name decoder_noise --out ../exp_logs/e2_decoder_noise.json

echo "END=$(date)"
