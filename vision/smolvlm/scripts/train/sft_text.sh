#!/bin/bash

MODEL_NAME="HuggingFaceTB/SmolVLM-Instruct"
DATA_PATH="scripts/mixtures/sft-text.yaml"
DATA_FOLDER="./data"

#source scripts/setups/train.sh

#torchrun \
#    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \

## NOTE: use deepspeed only for debug/single node. multi-node, only use torchrun. 

deepspeed \
    smolvlm/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --output_dir checkpoints/smolvlm-2b-sft \
    --model_max_length 6144 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --vision_tower_lr 0 \
    --connector_lr 5e-5 \
    --language_model_lr 1e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --peft_enable False \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj \
    --logging_steps 1 \
    --report_to wandb \
    --data_mixture $DATA_PATH \
    --data_folder $DATA_FOLDER \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --bf16 True \
    --gradient_checkpointing True \
    --packed False \
    --video_target_size 384 \
    --image_target_size 1536 \
    --max_frames 25 \
    --fps 1 \
    --use_liger_kernel False 



#    --optim paged_adamw_8bit \
