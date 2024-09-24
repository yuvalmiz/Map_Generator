import os
from api import read_csv_to_list
import json
from tqdm import tqdm

train_dir = "./stableDiffusion/dataset2"  # This should match the environment variable you will use in the bash script
metadata_filename = os.path.join("./", "metadata.jsonl")
image_prompt = read_csv_to_list("./image_prompt_shorter.csv",["image", "prompt"])
image_prompt_dict = {}


for image, prompt in tqdm(image_prompt):
    image_prompt_dict[image] = prompt


with open("./imageNumber_to_image.json") as f:
    image_nummber_to_image = json.load(f)

with open(metadata_filename, "a") as f:
    # Process images
    for key in image_nummber_to_image.keys():
            # Write metadata
            metadata = {"file_name": key, "caption": image_prompt_dict[image_nummber_to_image[key]]}
            f.write(json.dumps(metadata) + "\n")

