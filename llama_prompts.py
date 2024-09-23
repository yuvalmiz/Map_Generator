import transformers
import torch
import csv
from tqdm import tqdm
from api import log_to_csv, read_csv_to_list
import gc
import subprocess
import os

# filtered_images_path = 'filtered_images.csv'
# images_prompt_path = 'image_prompt.csv'
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# model = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     tokenizer=tokenizer)

# # Initialize an empty dictionary
# images_labels = read_csv_to_list(filtered_images_path, ['image', 'category', 'image_description'])

# def llama_generate_image_prompts_batch(image_data_batch):
#     # Prepare instructions for the entire batch
#     instructions = []
#     for image_name, category, description in image_data_batch:
#         instruction = [{
#             "role": "system",
#             "content": (
#                 f"You are given Image name, image description and image category. Based on the data you are given, "
#                 f"you must generate a prompt to create the type of the map that will be used with stable diffusion:\n\n"
#                 f"Image name: {image_name}.\n"
#                 f"Image description: {description}.\n"
#                 f"Image category: {category}.\n"
#                 f"Provide more attention to details related to category.\n"
#                 f"The answer should be one sentence with a few words, it must contain only the prompt, a positive example for an answer can be: [Generate map of earth with continents clearly labeled.]\n"
#                 f"A negative example for an answer can be: [Here's a prompt that can be used with Stable Diffusion to generate an image of a colorful world map with major countries, oceans, and continents clearly labeled.]"
#             )
#         }]
#         instructions.append(instruction)

#     # Generate prompts in batch using the model
#     generated = model(instructions, num_return_sequences=1, temperature=0.7, pad_token_id=128001, max_new_tokens=1000, batch_size=len(instructions))

#     # result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     # print(result)
#     # Extract and return the generated prompts
#     generated_prompts = [gen[0]["generated_text"][-1]['content'] for gen in generated]

#     # del instructions, generated  # Delete tensors to free memory
#     # torch.cuda.empty_cache()  # Clear CUDA memory cache
#     # gc.collect()  # Run garbage collection
#     return generated_prompts



# def process_with_constant_batch_size(images_labels, batch_size=16):

#     all_prompts = []  # List to save all the generated prompts
#     batch = []

#     for idx, (image, category, description) in enumerate(tqdm(images_labels)):
#         batch.append((image, category, description))

#         # If batch is full or we're at the last batch, process it
#         if len(batch) == batch_size or idx == len(images_labels) - 1:
#             try:
#                 # Generate prompts for the current batch
#                 prompts = llama_generate_image_prompts_batch(batch)

#                 # Prepare data for storing
#                 image_prompt_list = []
#                 for image, prompt in zip(batch, prompts):
#                     # Ensure that the prompt is a string before applying string methods
#                     if isinstance(prompt, str):
#                         cleaned_prompt = prompt.replace("[", "").replace("]", "")
#                         image_prompt_list.append((image[0], cleaned_prompt))
#                     else:
#                         raise ValueError("Prompt is not a string.")

#                 # Add to the master list of all prompts
#                 all_prompts.extend(image_prompt_list)

#                 # Clear the batch
#                 batch = []

#             except:
#                 batch = []
#                 continue

#     return all_prompts  # Return all collected prompts

# # Start processing with a constant batch size
# all_generated_prompts = process_with_constant_batch_size(images_labels, batch_size=16)

# # Write all prompts to the CSV at the end
# log_to_csv(images_prompt_path, all_generated_prompts, ["image", "prompt"])

# # print(f"Finished processing and saved {len(all_generated_prompts)} prompts to {csv_filename}")



filtered_images_path = 'filtered_images.csv'
images_prompt_path = 'image_prompt_shorter.csv'
checkpoint_path = 'checkpoint_shorter.csv'  # File to save intermediate progress
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load model and tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    tokenizer=tokenizer)

# Read the images and their labels
images_labels = read_csv_to_list(filtered_images_path, ['image', 'category', 'image_description'])

def llama_generate_image_prompts_batch(image_data_batch):
    instructions = []
    for image_name, category, description in image_data_batch:
        instruction = [{
            "role": "system",
            "content": (
                f"You are given Image name, image description and image category. Based on the data you are given, "
                f"you must generate a prompt to create the type of the map that will be used with stable diffusion:\n\n"
                f"Image name: {image_name}.\n"
                f"Image description: {description}.\n"
                f"Image category: {category}.\n"
                f"Provide more attention to details related to category.\n"
                f"The answer should be few words (4-8), it must contain only the prompt, a positive example for an answer can be: [map of earth with continents.]\n"
                f"A negative example for an answer can be: [Here's a prompt that can be used with Stable Diffusion to generate an image of a colorful world map with major countries, oceans, and continents clearly labeled.]"
            )
        }]
        instructions.append(instruction)

    generated = model(instructions, num_return_sequences=1, temperature=0.7, pad_token_id=128001, max_new_tokens=1000, batch_size=len(instructions))

    generated_prompts = [gen[0]["generated_text"][-1]['content'] for gen in generated]
    return generated_prompts

def load_checkpoint():
    if os.path.exists(checkpoint_path):
        return read_csv_to_list(checkpoint_path, ['image', 'prompt'])
    return []

def save_checkpoint(checkpoint_data):
    log_to_csv(checkpoint_path, checkpoint_data, ["image", "prompt"])

def process_with_constant_batch_size(images_labels, batch_size=16):
    all_prompts = load_checkpoint()  # Recover from the last checkpoint
    processed_images = set([image for image, prompt in all_prompts])
    
    batch = []
    
    for idx, (image, category, description) in enumerate(tqdm(images_labels)):
        if image in processed_images:
            continue  # Skip already processed images

        batch.append((image, category, description))

        # If batch is full or we're at the last batch, process it
        if len(batch) == batch_size or idx == len(images_labels) - 1:
            try:
                prompts = llama_generate_image_prompts_batch(batch)
                image_prompt_list = []
                for image_data, prompt in zip(batch, prompts):
                    if isinstance(prompt, str):
                        cleaned_prompt = prompt.replace("[", "").replace("]", "")
                        image_prompt_list.append((image_data[0], cleaned_prompt))

                all_prompts.extend(image_prompt_list)
                save_checkpoint(all_prompts)  # Save progress after each batch

                batch = []  # Clear the batch

            except Exception as e:
                print(f"Error processing batch: {e}")
                batch = []
                continue


    return all_prompts

# Start processing with a constant batch size and 4-hour time limit
all_generated_prompts = process_with_constant_batch_size(images_labels, batch_size=16)

# Write all prompts to the final CSV
log_to_csv(images_prompt_path, all_generated_prompts, ["image", "prompt"])
