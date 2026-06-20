#!/bin/bash
### EEG2Graph EXPERIMENT E2: EEG-ablation eval (RQ3 causal test).
### Branch: exp/tier1-conclusions
###
### Inference-only. For each checkpoint, generate triplets under three EEG
### conditions — real / swap (deranged EEG) / zero — and compare. If swap and
### zero produce (near-)identical output to real and F1 barely moves, the
### decoder is ignoring the EEG (decoder shortcut). This is the causal
### confirmation of the correlational collapse measured in E3.
###
### Scores both runs that have completed checkpoints on the cluster, appending
### to one JSON keyed by --name. No training; short job.
###     sbatch exp_e2_ablation.sbatch

#SBATCH --qos=eliyanac
#SBATCH --partition=l40s
#SBATCH --time=0-00:30:00
#SBATCH --job-name=eeg2graph-e2-ablation
#SBATCH --output=logs/eeg2graph-e2-ablation-%J.out
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

for run in decoder_noise subject_temporal; do
    echo "=== E2 ablation: $run ==="
    python eeg_ablation_eval.py \
        --checkpoint ../checkpoints_${run}/best_model.pt \
        --tokenizer_dir ../checkpoints_${run}/tokenizer \
        --processed_dir ../processed_zuco2 \
        --triplets_path ../processed_zuco2/sentence_triplets.json \
        --split test --beam_size 1 \
        --name ${run} --out ../exp_logs/e2_eeg_ablation.json
done

echo "END=$(date)"
