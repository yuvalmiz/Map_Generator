import json
import cairosvg
import os
from tqdm import tqdm
images_directory = "downloaded_maps_final/"
# images_directory = "./test/"
with open(f'{images_directory}image_descriptions.json', 'r') as file:
    url_discription_dict = json.load(file)

image_path_list = [image_path for image_path in  url_discription_dict.keys() if image_path.split('.')[-1] == "svg"]

for image_path in tqdm(image_path_list):
    image_path_png = f"{"".join(image_path.split('.')[:-1])}.png"
    png_path =  f"{images_directory}{image_path_png}"
    url_full_path = f"{images_directory}{image_path}"
    if os.path.isfile(url_full_path):
        try:
            cairosvg.svg2png(url=url_full_path, write_to=png_path)
        except:
             continue
        url_discription_dict[image_path_png] = url_discription_dict.pop(image_path)
        with open(f'{images_directory}image_descriptions.json', 'w') as file:
                json.dump(url_discription_dict, file, indent=4)
        os.remove(url_full_path)
    else:
        a = url_discription_dict.pop(image_path)
        if os.path.isfile(png_path):
            url_discription_dict[image_path_png] = a
        with open(f'{images_directory}image_descriptions.json', 'w') as file:
            json.dump(url_discription_dict, file, indent=4)
