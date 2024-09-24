from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
from api import read_csv_to_list, log_to_csv
import os
import gc
from tqdm import tqdm
import random


trained_model_path = "stableDiffusion/output_model_full_longer"
vae = AutoencoderKL.from_pretrained('stableDiffusion/vaeFineTuned/vae_checkpoint_epoch_1')
pipe = StableDiffusionPipeline.from_pretrained(trained_model_path, torch_dtype=torch.float16, vae=vae)

prompt = "a game world map"

pipe(prompt).images[0].save("test_image.png")