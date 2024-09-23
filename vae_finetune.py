import os
import json
import torch
from diffusers import StableDiffusionPipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
# Load the pre-trained Stable Diffusion model
model_dir = './stableDiffusion/output_model_full_longer'
pipe = StableDiffusionPipeline.from_pretrained(model_dir, revision="fp16", torch_dtype=torch.float16).to("cuda")

# Extract the VAE from the pipeline
vae = pipe.vae

# Load the metadata file and images
train_dir = './stableDiffusion/dataset2'
metadata_file = os.path.join(train_dir, 'metadata.jsonl')

# Set up the dataset and transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize images
])

class ImageDatasetWithPrompts:
    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.metadata = self.load_metadata(metadata_file)
        self.transform = transform

    def load_metadata(self, metadata_file):
        # Load metadata.jsonl
        data = []
        with open(metadata_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the image filename and the corresponding prompt
        img_entry = self.metadata[idx]
        img_path = os.path.join(self.image_dir, img_entry['file_name'])
        prompt = img_entry['caption']

        # Load the image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, prompt

# Create the dataset and DataLoader
train_dataset = ImageDatasetWithPrompts(train_dir, metadata_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define the optimizer
optimizer = AdamW(vae.parameters(), lr=1e-5)

# Training loop
def train_vae(epochs, train_loader):
    vae.train()
    for epoch in tqdm(range(epochs)):
        for step, (images, prompts) in tqdm(enumerate(train_loader)):
            images = images.to("cuda", dtype=torch.float16)

            # Forward pass through the VAE
            latents = vae.encode(images).latent_dist.sample()
            recon_images = vae.decode(latents).sample

            # Loss: reconstruction loss
            loss = ((recon_images - images) ** 2).mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item()}")

        # Save checkpoint after each epoch
        vae.save_pretrained(f"{model_dir}/vae_checkpoint_epoch_{epoch+1}")

# Run training
epochs = 10
train_vae(epochs, train_loader)
