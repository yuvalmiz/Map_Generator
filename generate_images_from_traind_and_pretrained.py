from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
from api import read_csv_to_list, log_to_csv
import os
import gc
from tqdm import tqdm
import random

seed = 1
random.seed(seed)
generator = torch.manual_seed(seed)
checkpoint = 2000
# Function to batch image prompt list
def batch_image_prompts(image_prompt_list, batch_size):
    for i in range(0, len(image_prompt_list), batch_size):
        yield image_prompt_list[i:i + batch_size]

# Parameters
image_prompt_list = read_csv_to_list("image_prompt_shorter.csv", ['image', 'prompt'])
image_folder = "downloaded_maps_final" # put the path to the dowloaded images after converting svg to png
generated_folder = f"generated_data"
batch_size = 16  # Set your desired batch size here
random_sample_size = 1000
image_prompt_list = random.sample(image_prompt_list, min(random_sample_size, len(image_prompt_list)))


if not os.path.exists(generated_folder):
    os.makedirs(generated_folder)

# Load the pre-trained model
pretrained_model_path = "CompVis/stable-diffusion-v1-4"  # Replace with your output model path
trained_model_path = "output_model"
trained_model_path_with_vae = "output_model_with_vae" # when trained the VAE make sure you used diffrent output path then the regular model

vae = AutoencoderKL.from_pretrained('vae_checkpoint_epoch_3', torch_dtype=torch.float16) #choose the VAE you used on training
pipe = StableDiffusionPipeline.from_pretrained(trained_model_path_with_vae, torch_dtype=torch.float16, vae=vae).to("cuda")

csv_dict = {}

# Process in batches
for i in tqdm(range(0, len(image_prompt_list), batch_size)):
    image_batch = image_prompt_list[i:i + batch_size]
    captions = [caption for _, caption in image_batch]
    images = pipe(captions, num_inference_steps=50, generator=generator).images  # Generate images for batch

    for (image_path, caption), generated_image in zip(image_batch, images):
        generated_path = os.path.join(generated_folder, f"generated_trained_with_VAE_{image_path}")
        generated_image.save(generated_path)
        csv_dict[image_path] = {"caption": caption, "trained_VAE_path": generated_path}

del pipe
del vae
torch.cuda.empty_cache()
gc.collect()



pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# Process in batches for fine-tuned model
for image_batch in tqdm(batch_image_prompts(image_prompt_list, batch_size)):
    captions = [caption for _, caption in image_batch]
    images = pipe(captions, num_inference_steps=50).images  # Generate images for batch

    for (image_path, caption), generated_image in zip(image_batch, images):
        generated_path = os.path.join(generated_folder, f"generated_pretrained_{image_path}")
        generated_image.save(generated_path)
        csv_dict[image_path]["pre_trained_path"] = generated_path

del pipe
torch.cuda.empty_cache()
gc.collect()

# Load the fine-tuned model
pipe = StableDiffusionPipeline.from_pretrained(trained_model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# Process in batches for fine-tuned model
for image_batch in tqdm(batch_image_prompts(image_prompt_list, batch_size)):
    captions = [caption for _, caption in image_batch]
    images = pipe(captions, num_inference_steps=50).images  # Generate images for batch

    for (image_path, caption), generated_image in zip(image_batch, images):
        generated_path = os.path.join(generated_folder, f"generated_trained_{image_path}")
        generated_image.save(generated_path)
        csv_dict[image_path]["trained_path"] = generated_path



# Convert csv_dict to list and log to CSV

csv_list = []
for image_path in csv_dict.keys():
    csv_list.append([image_path, csv_dict[image_path]["caption"], csv_dict[image_path]["pre_trained_path"], csv_dict[image_path]["trained_path"], csv_dict[image_path]["trained_VAE_path"]])

log_to_csv(f"generated_pre_and_trained_images_with_VAE.csv", csv_list, ["image_path", "caption", "pre_trained_path", "trained_path", "trained_VAE_path"])
