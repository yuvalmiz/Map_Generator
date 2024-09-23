#!/bin/bash
export MODEL_NAME=stableDiffusion/output_model_full
export TRAIN_DIR=stableDiffusion/dataset2
export OUTPUT_DIR=stableDiffusion/output_model_full_longer
export train_steps=1000
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --use_8bit_adam \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=$train_steps \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --caption_column="caption" \
  --checkpointing_steps=200
