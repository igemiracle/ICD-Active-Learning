#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 8:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=icd_train_model_5000_samples
#SBATCH --output=log/train_model_5000_%j.log
#SBATCH --error=log/train_model_5000_%j.err

# Print job information
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Number of cores: $SLURM_NTASKS_PER_NODE"

# Load necessary modules 
echo "Start!"
module load anaconda3/2024.10-1 
conda activate llm 
echo "activate conda environment"

# Create directories for results
mkdir -p ../results/models
mkdir -p ../results/plots

# Show available CPU cores
echo "Available CPU cores: $(nproc)"

# Run the Python script
echo "Starting model training script..."
python train_model.py

#python train_model_fulldata.py

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training script failed with error code $? at $(date)"
fi

# Create a tarball of all results
JOB_ID=$SLURM_JOB_ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"

echo "Creating tarball of results..."
mkdir -p $RESULTS_DIR
cp -r ../logs/ ../results/ $RESULTS_DIR/
tar -czf ${RESULTS_DIR}.tar.gz $RESULTS_DIR
echo "Results archived to ${RESULTS_DIR}.tar.gz"

# Print summary of disk usage
echo "Disk usage summary:"
du -h --max-depth=1 .

echo "Job completed at $(date)"