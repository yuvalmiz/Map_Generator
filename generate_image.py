from diffusers import StableDiffusionPipeline
import torch
from api import read_csv_to_list, log_to_csv
import os
import gc
from tqdm import tqdm
import random


trained_model_path = "stableDiffusion/output_model_full_longer"
pipe = StableDiffusionPipeline.from_pretrained(trained_model_path, torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "a game world map"

pipe(prompt).images[0].save("test_image.png")