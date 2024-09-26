import os
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy.linalg import sqrtm
import torch
from torchvision.models import inception_v3
from tqdm import tqdm

# Paths
pretrained_dir = 'generated_data'
real_images_dir = 'downloaded_maps_final'

# Helper function to load images
def load_images(image_dir, prefix, real_images_dir):
    images = []
    real_images = []
    for img_name in tqdm(os.listdir(image_dir)):
        if img_name.startswith(prefix):
            img_path = os.path.join(image_dir, img_name)
            real_img_name = img_name[len(prefix):]
            real_img_path = os.path.join(real_images_dir, real_img_name)
            if os.path.exists(real_img_path):
                img = Image.open(img_path).convert('RGB')
                real_img = Image.open(real_img_path).convert('RGB')
                images.append(img)
                real_images.append(real_img)
            else:
                print(f"Real image not found for {img_name}")
    return images, real_images

# Load pretrained model images
pretrained_images, _ = load_images(pretrained_dir, 'generated_pretrained_', real_images_dir, load_real=False)

# Load trained model images
trained_images, _ = load_images(pretrained_dir, 'generated_trained_', real_images_dir, load_real=False)

trained_images_with_VAE, real_images = load_images(pretrained_dir, 'generated_trained_with_VAE_', real_images_dir, load_real=True)

# Preprocess function for InceptionV3 model
def preprocess(images):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return torch.stack([preprocess(img) for img in images])

# Compute activations
def get_activations(images, model, batch_size=32, dims=2048, device='cuda'):
    model.eval()
    pred_arr = np.empty((len(images), dims))
    images = preprocess(images)
    n_batches = (len(images) + batch_size - 1) // batch_size  # Ceiling division

    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(images))
        batch = images[start:end].to(device)
        with torch.no_grad():
            pred = model(batch)  # Get the features from the model
        pred_arr[start:end] = pred.cpu().numpy()

    return pred_arr

# Function to calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    # Handle imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Function to compute FID between two image sets
def compute_fid(images1, images2, model, device):
    act1 = get_activations(images1, model, device=device)
    act2 = get_activations(images2, model, device=device)

    # Compute mean and covariance of activations
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    return calculate_fid(mu1, sigma1, mu2, sigma2)

# Load InceptionV3 model for feature extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modify the InceptionV3 model to output features suitable for FID
class InceptionV3ForFID(torch.nn.Module):
    def __init__(self):
        super(InceptionV3ForFID, self).__init__()
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # Remove the final classification layer
        self.model.Mixed_7c.register_forward_hook(self.output_hook)
        self.activations = None

    def output_hook(self, module, input, output):
        self.activations = output

    def forward(self, x):
        # Run the model
        _ = self.model(x)
        # Apply adaptive average pooling
        activations = self.activations
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.view(activations.size(0), -1)
        return activations

inception_model = InceptionV3ForFID().to(device)

# Compute FID for pretrained model
fid_pretrained = compute_fid(pretrained_images, pretrained_real_images, inception_model, device)

# Compute FID for pretrained model
fid_pretrained = compute_fid(pretrained_images, real_images, inception_model, device)

# Compute FID for trained model
fid_trained = compute_fid(trained_images, real_images, inception_model, device)
fid_trained_with_VAE = compute_fid(trained_images_with_VAE,real_images, inception_model, device)
print(f"FID for Pretrained Model: {fid_pretrained}")
print(f"FID for Trained Model: {fid_trained}")
print(f"FID for Trained Model with VAE: {fid_trained_with_VAE}")
