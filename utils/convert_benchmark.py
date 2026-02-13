from datasets import load_dataset
import json
import os
from PIL import Image
from tqdm import tqdm

dataset = load_dataset(
    "inclusionAI/ZoomBench", split='test'
)

output_dir = "../ZoomBench"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

output_json = []
image_counter = 0

for item in tqdm(dataset):
    image = item['image'] 
    crop_image = item['crop_image']
    image_path = os.path.join(output_dir, "images", f"image_{image_counter}.png")
    crop_image_path = os.path.join(output_dir, "images", f"crop_image_{image_counter}.png")
    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        background.save(image_path)
    elif image.mode != 'RGB':
        image.convert('RGB').save(image_path)
    else:
        image.save(image_path)
    
    if crop_image is not None:
        if crop_image.mode == 'RGBA':
            background = Image.new('RGB', crop_image.size, (255, 255, 255))
            background.paste(crop_image, mask=crop_image.split()[3])
            background.save(crop_image_path)
        elif crop_image.mode != 'RGB':
            crop_image.convert('RGB').save(crop_image_path)
        else:
            crop_image.save(crop_image_path)
    
    json_entry = {
        'images': [image_path],
        'crop_images': [crop_image_path] if crop_image is not None else [],
        'query': item['prompt'] ,
        'response': item['answer']
    }
    output_json.append(json_entry)
    
    image_counter += 1

json_path = os.path.join(output_dir, "zoombench.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"Processed {image_counter} images")
print(f"Saved {len(output_json)} entries to {json_path}")
