#!/bin/bash
#SBATCH --nodes=4                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Important for distributed usage (1 task per node)
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8                  # 8 GPUs per node
##SBATCH --exclusive
#SBATCH --qos=normal

set -x -e

# (Optional) Set up Hugging Face cache directory
export HF_HOME=/path/to/user/cache

# Activate your conda environment
source /path/to/user/miniconda3/etc/profile.d/conda.sh
conda activate smolvlm

# Debug prints
echo "Python path: $(which python)"
python -c "import sys; print('Sys path:', sys.path)"
python -c "import torch; print('PyTorch version:', torch.__version__, '\nPyTorch location:', torch.__file__)"
python -c "import transformers; print('Transformers location:', transformers.__file__)"
which torchrun

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6001
export WORLD_SIZE=$((SLURM_NNODES * 8))
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=0

# User-defined variables
MODEL_NAME="HuggingFaceTB/SmolVLM-Instruct"
# DATA_PATH="scripts/mixtures/llava-onevision-video178k.yaml"
DATA_PATH="scripts/mixtures/llavaonevision-video.yaml"
#DATA_PATH="scripts/mixtures/llavaonevision-imagetext.yaml"
DATA_FOLDER="/path/to/user/apollo-dataset/"
RUN_NAME="firstrun"

# Debug prints for environment
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"

cd /path/to/user/smolvlmvideo

export PYTHONPATH="/path/to/user/smolvlmvideo:$PYTHONPATH"
# export TORCHELASTIC_ERROR_FILE="/path/to/user/smolvlmvideo/torchelastic_error_file.log"
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export DEEPSPEED_LOG_LEVEL=INFO
# # Add before srun
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_NSOCKS_PERTHREAD=8
# export NCCL_IGNORE_CPU_AFFINITY=1  # Important for multi-node

srun torchrun \
    --nproc_per_node=8 \
    --nnodes="${SLURM_NNODES}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    smolvlm/train/train_mem.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --deepspeed scripts/zero2.json \
    --model_max_length 4096 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --vision_tower_lr 0 \
    --connector_lr 5e-5 \
    --language_model_lr 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --peft_enable False \
    --logging_steps 1 \
    --data_mixture $DATA_PATH \
    --data_folder $DATA_FOLDER \
    --dataloader_drop_last True \
    --dataloader_num_workers 1 \
    --bf16 True \
    --ddp_find_unused_parameters True \
    --packed False \
    --video_target_size 384 \
    --image_target_size 1920 \
    --max_frames 25 \
    --fps 1 \
    --use_liger_kernel False \
    --report_to wandb \
    --run_name $RUN_NAME
#    --optim paged_adamw_8bit \