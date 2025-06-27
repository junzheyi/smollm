#!/bin/bash


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
export MASTER_PORT=6002
export WORLD_SIZE=$((SLURM_NNODES * 1))
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=0

# User-defined variables
MODEL_NAME="HuggingFaceTB/SmolVLM-500M-Instruct"
# DATA_PATH="scripts/mixtures/llava-onevision-video178k.yaml"
# DATA_PATH="scripts/mixtures/llavaonevision-video.yaml"
# DATA_PATH="scripts/mixtures/tiny-m4-llavaonevision-llamavideolong.yaml"
# DATA_PATH='scripts/mixtures/orr-new-mixture.yaml'
# DATA_PATH='scripts/mixtures/onevision_less_mammoth.yaml'
DATA_PATH="scripts/mixtures/onevision_less_mammoth_more_videos.yaml"
DATA_FOLDER="/path/to/user/apollo-dataset/"
RUN_NAME="firstrun500"

# Debug prints for environment
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"

cd /path/to/user/smolvlmvideo

export PYTHONPATH="/path/to/user/smolvlmvideo:$PYTHONPATH"
export TORCHELASTIC_ERROR_FILE="/path/to/user/smolvlmvideo/torchelastic_error_file.log"
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export DEEPSPEED_LOG_LEVEL=INFO
# # Add before srun
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_NSOCKS_PERTHREAD=8
# export NCCL_IGNORE_CPU_AFFINITY=1  # Important for multi-node
torchrun \
    --nproc_per_node=1 \
    --nnodes="${SLURM_NNODES}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    smolvlm/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --model_max_length 8192 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 6 \
    --vision_tower_lr 1e-10 \
    --connector_lr 1e-4 \
    --language_model_lr 2e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --num_train_epochs 0.37 \
    --add_media_intro_outro True \
    --peft_enable False \
    --logging_steps 1 \
    --data_mixture $DATA_PATH \
    --data_folder $DATA_FOLDER \
    --dataloader_drop_last True \
    --dataloader_num_workers 0 \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1 \
    --gradient_checkpointing True \
    --packed False \
    --mask_user_tokens True \
    --video_target_size 512 \
    --image_target_size 2048 \
    --max_frames 64 \
    --fps 1 \
    --use_liger_kernel False \
    --run_name $RUN_NAME


        # --frames_per_clip 2 \
