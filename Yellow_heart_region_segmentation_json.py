import os
import json
import cv2
import numpy as np
from pathlib import Path

# ---------------------- Configuration items(Modify according to actual situation) ----------------------
INPUT_DIR = r"./data/annotations_head_region" # pictures+JSONFolder
OUTPUT_DIR = r"./data/Auto_segment_Yellow_heart/Visualization"  # Visualization results saving folder
JSON_OUTPUT_DIR = r"./data/Auto_segment_Yellow_heart/json"  # JSONfile save folder
TARGET_LABEL = "Pan_center_contour_area"
EXR_THRESHOLD = 0.15          # ExRthreshold
LAB_B_LOW_THRESHOLD = 20      # Lab-blower limit(20)
LAB_B_HIGH_THRESHOLD = 70     # Lab-bupper limit(70)
EPS = 1e-6                    # Prevent zero errors
YELLOW_LABEL = "Yellow_area"  # new label name

# ---------------------- Helper function ----------------------
def round_and_clip_polygon(points, w, h):
    """Clipping polygon coordinates to image boundaries and converting to integers"""
    pts = []
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        x, y = float(p[0]), float(p[1])
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))
        pts.append((xi, yi))
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)

def compute_exr_per_pixel(img):
    """Pixel by pixel calculationExRvalue"""
    # separationBGRchannel(OpenCVWhat is read isBGR)
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)
    
    # CalculateExR: (1.4*R - G) / (R+G+B + EPS)
    S = R + G + B
    S_safe = np.where(S == 0, EPS, S)
    ExR = (1.4 * R - G) / (S_safe + EPS)
    return ExR

def compute_lab_b_channel(img):
    """CalculateLabcolor spacebchannel(Yellow-blue axis:Positive numbers are yellower,Negative numbers are bluer)"""
    # Convert toLabcolor space(OpenCVThe default isBGRinput)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # separationL(brightness),a(red-green),b(Yellow-blue)channel
    _, _, b_channel = cv2.split(lab)
    return b_channel

def load_json_and_get_mask(json_path, img_shape):
    """LoadJSONand generatePan_center_contour_areamask"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for shape in data.get("shapes", []):
        if shape.get("label") == TARGET_LABEL and shape.get("shape_type") == "polygon":
            points = shape.get("points", [])
            # Crop and convert polygon coordinates
            poly = round_and_clip_polygon(points, w, h)
            if poly.shape[0] >= 3:
                cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 255)
    return mask

def extract_yellow_contour(img, mask):
    """Extract within mask Lab-bin20~70between or ExR>0.15 The closed outline of the pixels"""
    # 1. CalculateExRandLab-bchannel
    exr = compute_exr_per_pixel(img)
    lab_b = compute_lab_b_channel(img)
    
    # 2. Keep only pixels within the mask(The mask peripheral is0/False)
    masked_exr = np.where(mask == 255, exr, 0)
    masked_lab_b = np.where(mask == 255, lab_b, 0)
    
    # 3. Generate yellow area binary map
    # 3.1 ExR>0.15 mask
    exr_mask = (masked_exr > EXR_THRESHOLD).astype(np.uint8) * 255
    # 3.2 Lab-bin20~70between mask(Core modifications:Range determination)
    lab_b_mask = np.where(
        (masked_lab_b >= LAB_B_LOW_THRESHOLD) & (masked_lab_b <= LAB_B_HIGH_THRESHOLD),
        255,
        0
    ).astype(np.uint8)
    
    # 3.3 Merge two masks(Bitwise OR:If one of the above conditions is met, it is considered yellow.)
    yellow_binary = cv2.bitwise_or(exr_mask, lab_b_mask)
    
    # 4. Morphological operations:Remove small noise,Avoid irrelevant pixel interference(Key optimization)
    kernel = np.ones((3, 3), np.uint8)
    yellow_binary = cv2.erode(yellow_binary, kernel, iterations=1)  # Erosion to remove small noise spots
    yellow_binary = cv2.dilate(yellow_binary, kernel, iterations=1) # Expansion recovery core area
    
    # 5. Find contours(Take only the outermost layer)
    contours, _ = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter the outline with the largest area(Avoid small noise)
    max_contour = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    return max_contour, yellow_binary

def visualize_contour_and_save(img, contour, output_path):
    """Draw a red outline on the original image and save it"""
    # Draw red outline(Line width2,red:BGR=(0,0,255))
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    # Create output directory and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Visualization saved: {output_path}")

# ---------------------- Start with the key modification areas ----------------------
def contour_to_json(contour, orig_json_path, img_path, img_shape):
    """read rawJSON,ReservePan_center_contour_areaand appendYellow_areaoutline"""
    h, w = img_shape[:2]
    
    # 1. Read the original JSON File content
    with open(orig_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        
    # 2. filter retain originalJSONin Pan_center_contour_area area
    preserved_shapes = []
    for shape in json_data.get("shapes", []):
        if shape.get("label") == TARGET_LABEL:
            preserved_shapes.append(shape)
            
    # will shapes List resets to include only reserved areas
    json_data["shapes"] = preserved_shapes
    
    # 3. Make sure the underlying metadata is correct
    json_data["imagePath"] = os.path.basename(img_path)
    json_data["imageHeight"] = h
    json_data["imageWidth"] = w
    
    # 4. If a new yellow area outline is extracted,build shape add in
    if contour is not None and len(contour) > 0:
        points = []
        for point in contour:
            x = float(point[0][0])
            y = float(point[0][1])
            points.append([x, y])
            
        yellow_shape = {
            "label": YELLOW_LABEL,
            "points": points,
            "group_id": None,
            "description": None,
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {}
        }
        # append to shapes in list
        json_data["shapes"].append(yellow_shape)
        
    return json_data

def save_yellow_area_json(contour, orig_json_path, img_path, img_shape, json_output_path):
    """Save an outline containing two regionsJSONFile"""
    # Generate updatedJSONdata
    json_data = contour_to_json(contour, orig_json_path, img_path, img_shape)
    # Create output directory
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    # saveJSONFile
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"savedJSONFile: {json_output_path}")
# ---------------------- End of key modification area ----------------------

def process_single_file(img_path, json_path, output_img_path, output_json_path):
    """Process a single image+JSONvisualization andJSONgenerate"""
    # 1. Read the original image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Unable to read image: {img_path}")
        return
    
    # 2. LoadJSONand generate a mask
    mask = load_json_and_get_mask(json_path, img.shape)
    if np.sum(mask) == 0:
        print(f"⚠️ Not found validPan_center_contour_areamask: {json_path}")
        # When there is no outline, save the original image directly and the image containing only the original area.JSON
        visualize_contour_and_save(img, None, output_img_path)
        save_yellow_area_json(None, json_path, img_path, img.shape, output_json_path)
        return
    
    # 3. Extract yellow area outline(Lab-b20~70 or ExR>0.15)
    contour, _ = extract_yellow_contour(img, mask)
    
    # 4. Visualize and save pictures
    visualize_contour_and_save(img, contour, output_img_path)
    
    # 5. Save containsPan_center_contour_areaandYellow_areaofJSONFile
    save_yellow_area_json(contour, json_path, img_path, img.shape, output_json_path)

def batch_process():
    """Process all images in batches+JSONFile"""
    input_path = Path(INPUT_DIR)
    # Supported image formats
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    # Iterate through all image files
    img_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]
    
    for img_file in img_files:
        # Splicing correspondenceJSONpath(Same name)
        json_file = input_path / f"{img_file.stem}.json"
        if not json_file.exists():
            print(f"⚠️ No correspondence foundJSONFile: {json_file}")
            continue
        
        # Splicing output picture path
        output_img_file = Path(OUTPUT_DIR) / img_file.name
        # splicing outputJSONpath
        output_json_file = Path(JSON_OUTPUT_DIR) / f"{img_file.stem}.json"
        
        # Process single file
        process_single_file(img_file, json_file, output_img_file, output_json_file)

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    # Batch processing
    batch_process()
    print("\n✅ All files processed!")