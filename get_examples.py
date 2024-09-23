import csv
import os
import shutil

# Input CSV file containing image paths and details
csv_file = "generated_pre_and_trained_images2.csv"

# Directories
image_folder = "./gdrive/deeplearning final project/downloaded_maps3"
generated_folder = "./stableDiffusion/generated_data_longerModel2000"
output_folder = "./stableDiffusion/examples_checkpoint2000"

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the CSV and process each row
with open(csv_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    i = 0
    num_of_steps = 10
    start_step = 800
    for row in csv_reader:
        if i > num_of_steps + start_step:
            break
        if i < start_step:
            i+=1
            continue
        i +=1
        image_name = row['image_path']
        pre_trained_path = row['pre_trained_path']
        trained_path = row['trained_path']
        
        # Define the source paths for each image
        image_source_path = os.path.join(image_folder, image_name)
        pre_trained_source_path = pre_trained_path
        trained_source_path = trained_path
        
        # Define the destination paths for each image
        image_dest_path = os.path.join(output_folder, image_name)
        pre_trained_dest_path = os.path.join(output_folder, os.path.basename(pre_trained_source_path))
        trained_dest_path = os.path.join(output_folder, os.path.basename(trained_source_path))
        
        # Copy the original image if it exists
        if os.path.exists(image_source_path):
            shutil.copy(image_source_path, image_dest_path)
            print(f"Copied original image: {image_name}")
        else:
            print(f"Original image not found: {image_source_path}")
        
        # Copy the pre-trained image if it exists
        if os.path.exists(pre_trained_source_path):
            shutil.copy(pre_trained_source_path, pre_trained_dest_path)
            print(f"Copied pre-trained image: {os.path.basename(pre_trained_source_path)}")
        else:
            print(f"Pre-trained image not found: {pre_trained_source_path}")
        
        # Copy the trained image if it exists
        if os.path.exists(trained_source_path):
            shutil.copy(trained_source_path, trained_dest_path)
            print(f"Copied trained image: {os.path.basename(trained_source_path)}")
        else:
            print(f"Trained image not found: {trained_source_path}")
