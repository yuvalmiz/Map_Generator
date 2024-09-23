import pandas as pd
import json

# Load the CSV file
csv_file_path = 'image_prompt.csv'
csv_data = pd.read_csv(csv_file_path)

# Create a dictionary from the CSV
caption_to_image_path = dict(zip(csv_data['prompt'], csv_data['image']))

# Prepare the resulting dictionary
result = {}

# Load the JSONL file and process it
jsonl_file_path = 'stableDiffusion/dataset2/metadata.jsonl'
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        caption = data['caption']
        file_name = data['file_name']
        
        # If the caption exists in the dictionary, map it to the image path
        if caption in caption_to_image_path:
            result[file_name] = caption_to_image_path[caption]

# Save the result to a JSON file
output_file_path = 'imageNumber_to_image.json'
with open(output_file_path, 'w') as json_file:
    json.dump(result, json_file, indent=4)

print("JSON file created successfully.")
