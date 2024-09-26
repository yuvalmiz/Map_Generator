import requests
import os
import json
import re

# Function to get category data
def get_category_data(category):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmlimit': 'max',
    }

    response = requests.get(url, params=params)
    data = response.json()

    return data

# Function to get image info and download it
def get_image_info(file_title, category):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': file_title,
        'prop': 'imageinfo',
        'iiprop': 'url|extmetadata'
    }
    response = requests.get(url, params=params)
    data = response.json()

    pages = data['query']['pages']
    for page_id, page_data in pages.items():
        if 'imageinfo' in page_data:
            image_url = page_data['imageinfo'][0]['url']
            if 'ImageDescription' not in page_data['imageinfo'][0]['extmetadata']:
                return
            image_description = page_data['imageinfo'][0]['extmetadata']['ImageDescription']['value']
            if not image_description:
                return
            extension = image_url.split('.')[-1]
            if extension not in ['jpg', 'jpeg', 'png', 'svg']:
                return
            label = ''.join(page_data['title'].replace('File:', '').replace('/', '_').replace(':', '_').replace(' ', '_').split('.')[-2]).replace('.', '_') + '.' +extension
            
            label = re.sub(r'[<>:"/\\|?*]', '_', label)  # Replace invalid characters with '_'
            new_label = label
            if len(label) > 100:
                new_label = label[:100] + '.' + extension
            download_image(image_url, new_label, label, image_description, category)

# Function to download the image and save it correctly in binary mode
def download_image(url, new_label, label, image_description, category):
    headers = {
        'User-Agent': 'yuval32211@gmail.com'  # Customize with your email
    }

    # Download the image data in binary mode
    response = requests.get(url, stream=True, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Determine the file extension        
        # Create a file name with the label
        file_name = f"{new_label}"
        
        # Save the image in binary mode
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        url_discription_dict[label] = { 'category': category, 'image_description': image_description }
        with open('image_descriptions.json', 'w') as file:
            json.dump(url_discription_dict, file, indent=4)
    else:
        print(f"Failed to download {label}")

# Recursive function to read through subcategories
def read_category(category):
    if category in finished_categories or category in visited_categories:
        print(f"Skipping category: {category}")
        return
    visited_categories.append(category)
    data = get_category_data(category)
    category_members = data['query']['categorymembers']
    for item in category_members:
        if item['title'].startswith('Category:'):
            print(f"Reading category: {item['title']}")
            read_category(item['title'])
        else:
            get_image_info(item['title'], category)
    finished_categories.append(category)
    with open('finished_categories.json', 'w') as file:
        json.dump({"finished": finished_categories}, file, indent=4)
    

# Main category
category = 'Category:Maps by area'


url_discription_dict = {}
finished_categories = []
# Create a directory to store the images
if not os.path.exists('downloaded_maps_final'):
    os.makedirs('downloaded_maps_final')

# Change the current working directory to the download folder
os.chdir('downloaded_maps_final')


if os.path.exists('finished_categories.json'):
    with open('finished_categories.json', 'r') as file:
        data = json.load(file)
        finished_categories = data.get('finished', [])

if os.path.exists('image_descriptions.json'):
    with open('image_descriptions.json', 'r') as file:
        url_discription_dict = json.load(file)
else:
    url_discription_dict = {}

visited_categories = []
# Start reading the category
read_category(category)

