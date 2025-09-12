"""
image 移动到了 /mnt/minyingqian/MME-RealWorld-Lite/data/imgs 下，因此要更改数据集的 image 路径
"""

NEW_DIR = "/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs"

import os

# read json
ORIGINAL_JSON = "MME-RealWorld-Lite.json"
OUTPUT_JSON = "MME-RealWorld-Lite_new.json"

import json
with open(ORIGINAL_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    original_image_path = item['Image']
    new_image_path = os.path.join(NEW_DIR, os.path.basename(original_image_path))
    item['Image_new'] = [os.path.join(NEW_DIR, os.path.basename(original_image_path))]

    print(f"Original: {original_image_path} -> New: {new_image_path}")


# save json
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    
print(f"Saved updated JSON to {OUTPUT_JSON}")
