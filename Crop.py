import cv2
import json
import numpy as np
import os

# ===================== Configuration parameters / Path Configuration =====================
# [default path] point to GitHub The test set that comes with the warehouse (You can run it directly after cloning the project Demo)
json_folder_path = "./samples_images"
image_folder_path = "./samples_images"

# Results save output path
output_root_dir = "./results/CROP_black"

# -------------------------------------------------------------------------
# [Full data set operation guide / Full Dataset Instructions]
# If needed run from Figshare Download the full dataset,Please comment out the path above,and uncomment the three lines below:
# json_folder_path = "./data/SAM3_predicted"
# image_folder_path = "./data/SAM3_predicted"
# output_root_dir = "./data/CROP_black"
# =========================================================================

# Automatically create output folders(If it does not exist, it will be created automatically.,Ignore if present)
os.makedirs(output_root_dir, exist_ok=True)

# Traverse JSON each in the folder JSON File
for json_filename in os.listdir(json_folder_path):
    if json_filename.endswith('.json'):
        json_path = os.path.join(json_folder_path, json_filename)
        
        # read JSON File(Increase coding fault tolerance)
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            print(f"read JSON File failed {json_filename}: {str(e)}")
            continue
        
        # ========== core:Exactly matches the original image file name ==========
        img_base_name = os.path.splitext(json_filename)[0]
        target_image_path = None
        target_image_filename = None
        
        for file in os.listdir(image_folder_path):
            file_base = os.path.splitext(file)[0]
            if file_base == img_base_name:
                target_image_path = os.path.join(image_folder_path, file)
                target_image_filename = file
                break
        
        if not target_image_path:
            print(f"not found {img_base_name} Corresponding image file")
            continue
        
        # Read original image
        img = cv2.imread(target_image_path)
        if img is None:
            print(f"Unable to open or read file: {target_image_path}")
            continue

        # counter:Handling multi-labeled areas
        object_count = 1
        shape_count = len(data.get('shapes', []))

        # Extract the image part of each region
        for region in data.get('shapes', []):
            coords = region['points']
            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            
            # Create mask(Reserve target area,The background is black)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            # Compute bounding box and expand bounds(Proportion 0.2)
            x, y, w, h = cv2.boundingRect(pts)
            expand_ratio = 0.2
            dx = int(w * expand_ratio)
            dy = int(h * expand_ratio)
            x = max(x - dx, 0)
            y = max(y - dy, 0)
            w = min(w + 2 * dx, img.shape[1] - x)
            h = min(h + 2 * dy, img.shape[0] - y)

            # Crop area of interest (ROI)
            roi = masked_img[y:y+h, x:x+w]

            # Check ROI Is it empty
            if roi.size == 0:
                print(f"image {target_image_filename} area {object_count} Cropping result is empty")
                object_count += 1
                continue

            # ========== Keep file name format consistent ==========
            if shape_count == 1:
                # single area:Same as the original image file name
                output_filename = target_image_filename
            else:
                # multi-region:original name_No..Original suffix
                file_ext = os.path.splitext(target_image_filename)[1]
                output_filename = f"{img_base_name}_{object_count}{file_ext}"
            
            # splice output path
            output_path = os.path.join(output_root_dir, output_filename)

            # Save cropped image
            success = cv2.imwrite(output_path, roi)
            if success:
                print(f"Saved successfully: {output_path}")
            else:
                print(f"Save failed: {output_path}")

            object_count += 1

print("All files processed!")
