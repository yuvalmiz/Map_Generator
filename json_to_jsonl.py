import os
import json
from tqdm import tqdm
from api import read_csv_to_list

train_dir = "./stableDiffusion/dataset"
image_prompt_list = read_csv_to_list("image_prompt_list.csv", ['image', 'prompt'])
metadata_filename = os.path.join(train_dir, "metadata.jsonl")
with open(metadata_filename, "w") as f:
    for i, (image_path, prompt) in enumerate(tqdm(image_prompt_list)):
        # Save image
        img_filename = f"image_{i}.png"
        metadata = {"file_name": img_filename, "caption": prompt}
        f.write(json.dumps(metadata) + "\n")
