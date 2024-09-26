#!/bin/bash

export MODEL_NAME=CompVis/stable-diffusion-v1-4
export TRAIN_DIR=dataset
export OUTPUT_DIR=output_model
export train_steps=1000

# Use $1 as the first argument for the checkpoint (optional)
CHECKPOINT_PATH=$1
VAE_PATH=$2
# Construct the base command
CMD="accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --use_8bit_adam \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision=\"fp16\" \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler=\"constant\" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=200 \
  --caption_column=\"caption\""

# If a checkpoint path is provided, add the --resume_from_checkpoint argument
if [ -n "$CHECKPOINT_PATH" ]; then
  export train_steps=$((CHECKPOINT_PATH + 1000))
  CMD="$CMD --max_train_steps=$train_steps --resume_from_checkpoint=$OUTPUT_DIR/checkpoint-$CHECKPOINT_PATH"
else
  CMD="$CMD --max_train_steps=$train_steps"
fi

if [ -n "$VAE_PATH" ]; then
  CMD="$CMD --vae_model_name_or_path=$VAE_PATH"
fi
# Execute the command
eval $CMD
