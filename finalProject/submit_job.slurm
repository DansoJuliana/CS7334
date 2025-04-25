#!/bin/bash
#BATCH -J supercloud_train       # Job name
#SBATCH -o %x_%j.out             # Standard output log
#SBATCH -e %x_%j.err             # Standard error log
#SBATCH -p rtx              # Partition (use GPU nodes)
#SBATCH -N 2                     # Number of nodes
#SBATCH -n 8                     # Total MPI tasks (4 GPUs/node * 2 nodes)
#SBATCH -t 02:00:00              # Maximum runtime (2 hours)
#SBATCH -A ASC23013

# Configuration - Adjust these variables
#CONFIG="configs/deepspeed.json"  # Path to config file
#STRATEGY="deepspeed"             # Parallelization strategy (deepspeed|fsdp|ddp)
#RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
#OUTPUT_DIR="$SCRATCH/results/${RUN_ID}"  # Using $SCRATCH for output
#SCALING_TEST=0                   # Set to 1 for scaling test

# Load required modules
module load gcc/9.1.0
module load python3/3.8.2
module load cuda/12.2

# Activate your virtual environment
source $SCRATCH/pytorchenv/bin/activate

$HOME/CS7334_project/finalProject/configs/deepspeed.json 
# --- Create and encode hostfile ---
HOSTFILE="$SCRATCH/hostfile.$SLURM_JOB_ID"
GPUS_PER_NODE=4  # Must match --gres=gpu:X above

# Generate hostfile with slots information
scontrol show hostnames $SLURM_JOB_NODELIST | while read -r hostname; do
    echo "${hostname} slots=${GPUS_PER_NODE}" >> $HOSTFILE
done

# Base64 encode the hostfile content
ENCODED_WORLD_INFO=$(base64 -w 0 $HOSTFILE)

# --- Network Configuration for Frontera ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$(( RANDOM % 1000 + 29500 ))  # Random port in safe range
export NCCL_SOCKET_IFNAME=ib0    # Use InfiniBand on Frontera
export NCCL_DEBUG=INFO           # Verbose NCCL logging
export NCCL_IB_DISABLE=0         # Enable IB
export NCCL_IB_HCA=mlx5_0        # Mellanox HCA on Frontera

# --- Main Training Launch using srun ---
srun --mpi=pmi2 \
    --ntasks-per-node=$GPUS_PER_NODE \
    --distribution=block:block \
    --kill-on-bad-exit \
    python -u main.py \
    --deepspeed \
    --deepspeed_config $HOME/CS7334_project/finalProject/configs/deepspeed.json \
    --output_dir $SCRATCH/results/run_$SLURM_JOB_ID \
    --local_rank $SLURM_LOCALID \
    --world_info $ENCODED_WORLD_INFO \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT

# --- Cleanup ---
rm -f $HOSTFILE
deactivate
