#!/bin/bash
#SBATCH --job-name=tfidf_lr
#SBATCH --output=log/tfidf_lr_%j.out
#SBATCH --error=log/tfidf_lr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --partition=RM-shared

echo "Start!"
module load anaconda3/2024.10-1 
conda activate llm 
echo "activate conda environment"
echo "Running on node: $(hostname)"
echo "Detected CPUs: $(nproc)"

python tfidf_lr_baseline.py
echo "Done!"
