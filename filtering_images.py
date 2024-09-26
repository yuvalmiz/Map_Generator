# Import required libraries
from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
import csv
import requests
import os
import json
import re
from PIL import Image
import cairosvg
import transformers
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from math import e
import gc
# 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

# login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_to_csv(filename, data, headliners):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headliners)  # Write the headers
        writer.writerows(data)  # Write all the rows


# Load the CLIP model and processor
Clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to compute similarity between an image and a text prompt
def compute_similarity(model, processor, image_path, prompt):
    # Load and preprocess the image
    extension = image_path.split('.')[-1]
    if extension == 'svg':
        # Convert SVG to PNG
        png_path = "".join(image_path.split('.')[:-1]) + '.png'
        cairosvg.svg2png(url=image_path, write_to=png_path)
    else:
        png_path = image_path
    try:
      image = Image.open(png_path)

      # Preprocess the inputs (both image and text)
      inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

      # Forward pass through the model
      outputs = model(**inputs)

      # Extract image and text embeddings
      image_embeds = outputs.image_embeds
      text_embeds = outputs.text_embeds

      # Normalize the embeddings
      image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
      text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

      # Compute the cosine similarity between the image and text embeddings
      similarity = (image_embeds @ text_embeds.T).item()  # Get scalar value
    except:
      print(f"Failed to open {image_path}")
      return 0
    return similarity

# Function to filter images based on relevance to a prompt
def filter_images(image_folder, prompt, threshold=0.22, max_iteretions=None):
    relevant_images = []
    iter_num = 0
    if max_iteretions is None:
      always_continue = True
    else:
      always_continue = False
    # Loop through all images in the folder
    for image_name in tqdm(os.listdir(image_folder)):
        if not always_continue and iter_num >= max_iteretions:
          break
        if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'svg')):
            if image_name.lower().endswith(('svg')):
              continue
            iter_num += 1
            image_path = os.path.join(image_folder, image_name)
            similarity = compute_similarity(Clipmodel, processor,image_path, prompt)
            if similarity > threshold:
                relevant_images.append(image_name)

    return relevant_images



# Path to the folder containing your images
image_folder = "downloaded_maps_final"
image_descriptions = json.load(open(f"{image_folder}/image_descriptions.json")) # enter the directory where you saved the image discriptions!!!
# Your prompt to evaluate relevance
prompt1 = "Map of a earth"
prompt2 = "Map of a city"
# prompt3 = "geographical map"
prompt4 = "Map of a country"
prompt5 = "Map of game world"

max_iteretions = None
# Call the function to filter relevant images
relevant_images1 = filter_images(image_folder, prompt1, max_iteretions = max_iteretions)
print(f"first prompt gave {len(relevant_images1)} results")
relevant_images2 = filter_images(image_folder, prompt2, max_iteretions = max_iteretions)
print(f"second prompt gave {len(relevant_images2)} results")
# relevant_images3 = filter_images(image_folder, prompt3, max_iteretions = max_iteretions)
# print(f"third prompt gave {len(relevant_images3)} results")
relevant_images4 = filter_images(image_folder, prompt4, max_iteretions = max_iteretions)
print(f"fourth prompt gave {len(relevant_images4)} results")
relevant_images5 = filter_images(image_folder, prompt5, max_iteretions = max_iteretions)
print(f"fifth prompt gave {len(relevant_images5)} results")

# relevant_images_final = list(set().union(relevant_images1, relevant_images2, relevant_images3, relevant_images4, relevant_images5))
relevant_images_final = list(set().union(relevant_images1, relevant_images2, relevant_images4, relevant_images5))
images_labels = []
for image in relevant_images_final:
  if image not in image_descriptions:
    continue
  images_labels.append((image ,image_descriptions[image]["category"], image_descriptions[image]["image_description"]))
log_to_csv("filtered_images.csv", images_labels, ["image", "category", "image_description"])
# List of relevant images
