import os
import json
from PIL import Image
from tqdm import tqdm
from api import read_csv_to_list

# Define paths
train_dir = "dataset"  # This should match the environment variable you will use in the bash script
os.makedirs(train_dir, exist_ok=True)
image_folder = "downloaded_maps_final"
# image_prompt_list_path = "image_prompt_list.csv"
image_prompt_list_path = "image_prompt.csv"
finished_images_prompt_list_path = "finished_images_prompt_list.json"
last_processed_index_path = "last_processed_index.txt"

# Read image-prompt list
image_prompt_list = read_csv_to_list(image_prompt_list_path, ['image', 'prompt'])

# Read list of finished images, if it exists
finished_images_prompt_list = []
if os.path.exists(finished_images_prompt_list_path):
    with open(finished_images_prompt_list_path, "r") as json_file:
        finished_images_prompt_list = json.load(json_file)

# Read the last processed index, if it exists
if os.path.exists(last_processed_index_path):
    with open(last_processed_index_path, "r") as f:
        last_processed_index = int(f.read().strip())
else:
    last_processed_index = -1  # If no index file, start from -1 (will increment to 0 on first image)

# Initialize i from last processed index
i = last_processed_index + 1

# Open metadata file for appending (if interrupted, we continue appending)
metadata_filename = os.path.join(train_dir, "metadata.jsonl")
with open(metadata_filename, "a") as f:
    # Process images
    for image_path, prompt in tqdm(image_prompt_list):
        # Skip if the image has already been processed
        if image_path in finished_images_prompt_list:
            continue  # Skip already processed images

        try:
            # Save image
            full_image_path = os.path.join(image_folder, image_path)
            img = Image.open(full_image_path).convert("RGB")
            img_filename = f"image_{i}.png"  # Use the current value of i for file naming
            img.save(os.path.join(train_dir, img_filename))

            # Write metadata
            metadata = {"file_name": img_filename, "caption": prompt}
            f.write(json.dumps(metadata) + "\n")

            # Log the processed image
            finished_images_prompt_list.append(image_path)
            with open(finished_images_prompt_list_path, "w") as json_file:
                json.dump(finished_images_prompt_list, json_file)

            # Save the last processed index
            with open(last_processed_index_path, "w") as idx_file:
                idx_file.write(str(i))

            # Increment i for the next image
            i += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
