from diffusers import StableDiffusionPipeline, AutoencoderKL


trained_model_path = "stableDiffusion/output_model"
vae = AutoencoderKL.from_pretrained('vaeFineTuned/vae_checkpoint_epoch_1', torch_dtype=torch.float16) # change to your favorite VAE
pipe = StableDiffusionPipeline.from_pretrained(trained_model_path, torch_dtype=torch.float16, vae=vae)

prompt = "a game world map" # change prompt to what you want to generate

pipe(prompt).images[0].save("test_image.png")
