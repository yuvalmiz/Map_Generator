# MAP GENERATOR
## Overview
This repository contains a collection of scripts that work together to provide a comprehensive solution for collecting data from wikimedia, generating images, fine-tuning diffusion models, managing images and metadata. The project is fully deployable in Google Colab for ease of use, but it can also be run locally.

## Key Features:
- Image generation from trained and pretrained models.
- Fine-tuning of diffusion model with checkpoints and using various VAE's.
- data collection and Metadata menagment.
- Various utilities for filtering, dataset preparation, and prompt management.
- Integration with Wikipedia for data extraction.
## Installation
To get started, you need to install the required dependencies. You can do this using pip:

```
pip install -r requirements.txt
```
Make sure that you have Python 3.8+ installed.

## Files and Their Purpose
Each file in the repository serves a distinct purpose. Here's an overview of the key scripts:

- api.py: Defines API calls for interacting with the model.
- fid_calculation.py: Responsible for calculating Fréchet Inception Distance (FID) to evaluate image generation quality.
- filtering_images.py: Filters generated images based on CLIP model.
- finetune_checkpint_script.sh: A shell script to fine-tune models from a checkpoint(optional) using fine-tuned VAE(optinal).
- finetune_script.sh: A shell script to initiate model fine-tuning(without vae or checkpoints).
- generate_image.py: Script to generate a single image using the trained or pretrained model.
- generate_images_from_trained_and_pretrained.py: Generates multiple images using both trained and pretrained models, used for calculating FID.
- generate_metadata.py: Script to generate metadata for the generated images.
- get_examples.py: Retrieves example data for model input.
-ImageNumber_to_image.py: Maps image numbers to their respective image files (used in the proccess of creating the data to have a file that saves the image number in the dataset to the image original name).
- json_to_jsonl.py: Converts JSON files to JSON Lines format for efficient data handling.
- llama_prompts.py: generating prompt for the dataset using llama3.1-instruct-8b.
- outputModel_use.py: Manages the output from the trained model for post-processing.
- preparing_dataset.py: Prepares datasets for model training and evaluation.
- reading_prompts.py: Handles reading and interpreting prompts from various sources.
- svg_to_png.py: Converts SVG files to PNG format.
- vae_finetune.py: Fine-tunes a VAE (Variational AutoEncoder) model.
- wikipedia_reading_maps.py: Extracts and processes data from Wikipedia, mapping it to relevant use cases.
- link to the google cloud folder that containes models and data: https://drive.google.com/drive/folders/1zlCdTsY6wuZvNvW-LBtu64wRVWlGTEfi?usp=drive_link

## How to Use
You can run each of the scripts locally or in Google Colab.

Running Locally
Clone the repository:

```
git clone https://github.com/yuvalmiz/Map_Generator.git
cd Map_Generator
```
Install the dependencies:

```
pip install -r requirements.txt
```
Run individual scripts


## Google Colab Implementation
Link - https://colab.research.google.com/drive/1rgDpIDz50oNGeNaAQ45ytxrta5B28yku#scrollTo=43DER0QRwngI
This Colab notebook allows you to run everything in sequence without needing a local setup. Simply open the notebook and follow the instructions provided.
notice that you may need to add the folder with the data to your drive or at least add a link of the folder to your google drive.
this can be done by entering the link, click on the "deeplearning final project" title with right click. choose orgenize then add a shortcat

notice that you will need to change some of the paths in the code to the relevant files

## Project Workflow
The general workflow is as follows:

### Dataset Preparation:

- use wikipedia_reading_maps.py to collect specific category images from wikimedia (change the category inside)
- use svg_to_png.py to convert svg files to png
- use filtering.py tp filter images that are not relevant
- use llama_prompts.py to generate prompt for the data
- use preparing_dataset.py to preper dataset from dowloaded images. the dataset.

### Model Fine-tuning:

Fine-tune the model using finetune_script.sh or finetune_checkpint_script.sh if resuming from a checkpoint.

for using checkpoint and trained VAE 
```
finetune_checkpint_script.sh {checkpoint-number or ""} {trained_vae_path or keep blank}
```
notice that you may need to enter the file and change the paths to the train directory and output directory
### Image Generation:

Generate images with generate_image.py or generate_images_from_trained_and_pretrained.py.
### Evaluation:

Calculate FID scores using fid_calculation.py.
