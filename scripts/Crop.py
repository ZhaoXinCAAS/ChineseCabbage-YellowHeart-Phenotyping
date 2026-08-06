import cv2
import json
import numpy as np
import os

# Configuration
# Default paths for Demo dataset
json_folder_path = "./samples_images/results/SAM3_predicted"
image_folder_path = "./samples_images/results/SAM3_predicted"
output_root_dir = "./samples_images/results/annotations_head_region"

# Full dataset paths (uncomment when running Figshare dataset)
# json_folder_path = "./data/SAM3_predicted"
# image_folder_path = "./data/SAM3_predicted"
# output_root_dir = "./data/Annotations_head_region"

os.makedirs(output_root_dir, exist_ok=True)

for json_filename in os.listdir(json_folder_path):
    if not json_filename.endswith('.json'):
        continue

    json_path = os.path.join(json_folder_path, json_filename)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_filename}: {e}")
        continue

    img_base_name = os.path.splitext(json_filename)[0]
    target_image_path = None
    target_image_filename = None

    for file in os.listdir(image_folder_path):
        if os.path.splitext(file)[0] == img_base_name:
            target_image_path = os.path.join(image_folder_path, file)
            target_image_filename = file
            break

    if not target_image_path:
        print(f"Image not found for {img_base_name}")
        continue

    img = cv2.imread(target_image_path)
    if img is None:
        print(f"Failed to load image: {target_image_path}")
        continue

    object_count = 1
    shapes = data.get('shapes', [])
    shape_count = len(shapes)

    for region in shapes:
        coords = region['points']
        pts = np.array(coords, np.int32).reshape((-1, 1, 2))

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # Bounding box with 20% margin expansion
        x, y, w, h = cv2.boundingRect(pts)
        expand_ratio = 0.2
        dx = int(w * expand_ratio)
        dy = int(h * expand_ratio)
        x = max(x - dx, 0)
        y = max(y - dy, 0)
        w = min(w + 2 * dx, img.shape[1] - x)
        h = min(h + 2 * dy, img.shape[0] - y)

        roi = masked_img[y:y+h, x:x+w]

        if roi.size == 0:
            print(f"Warning: Cropped region {object_count} in {target_image_filename} is empty.")
            object_count += 1
            continue

        if shape_count == 1:
            output_filename = target_image_filename
        else:
            file_ext = os.path.splitext(target_image_filename)[1]
            output_filename = f"{img_base_name}_{object_count}{file_ext}"

        output_path = os.path.join(output_root_dir, output_filename)

        if cv2.imwrite(output_path, roi):
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to save: {output_path}")

        object_count += 1

print("Processing complete.")
