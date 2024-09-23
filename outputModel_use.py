from diffusers import StableDiffusionPipeline
import torch

# Load the fine-tuned model
model_path = "stableDiffusion/output_model"  # Replace with your output model path
# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# Move the pipeline to GPU
pipe.to("cuda")

prompt = "history map of earth"
image = pipe(prompt, num_inference_steps=50).images[0]

# Display the image
image.save('test2.png')