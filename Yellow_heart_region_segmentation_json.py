import os
import json
import cv2
import numpy as np
from pathlib import Path

# ===================== Configuration items / Path Configuration =====================
# [default path] point to GitHub Comes with the warehouse Mini test set (Can be run directly Demo)
INPUT_DIR = r"./samples_images/results/annotations_head_region" 
OUTPUT_DIR = r"./samples_images/results/Auto_segment_Yellow_heart/"
JSON_OUTPUT_DIR = r"./samples_images/results/Auto_segment_Yellow_heart/json"

# -----------------------------------------------------------------------
# [Full data set operation guide / Full Dataset Instructions]
# If needed run from Figshare Download the full dataset,Please comment out the three lines above,and uncomment the three lines below:
# INPUT_DIR = r"./data/annotations_head_region"
# OUTPUT_DIR = r"./data/Auto_segment_Yellow_heart/True_Yellow_heart/Visualisation"
# JSON_OUTPUT_DIR = r"./data/Auto_segment_Yellow_heart/True_Yellow_heart/json"
# =======================================================================

TARGET_LABEL = "Pan_center_contour_area"
EXR_THRESHOLD = 0.15          # ExRthreshold
LAB_B_LOW_THRESHOLD = 20      # Lab-blower limit(20)
LAB_B_HIGH_THRESHOLD = 70     # Lab-bupper limit(70)
EPS = 1e-6                    # Prevent zero errors
YELLOW_LABEL = "Yellow_area"  # Newly extracted yellow area label name


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
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)
    
    S = R + G + B
    S_safe = np.where(S == 0, EPS, S)
    ExR = (1.4 * R - G) / (S_safe + EPS)
    return ExR


def compute_lab_b_channel(img):
    """CalculateLabcolor spacebchannel(Yellow-blue axis:Positive numbers are yellower,Negative numbers are bluer)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
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
            poly = round_and_clip_polygon(points, w, h)
            if poly.shape[0] >= 3:
                cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 255)
    return mask


def extract_yellow_contour(img, mask):
    """Extract within mask Lab-bin20~70between or ExR>0.15 The closed outline of the pixels"""
    exr = compute_exr_per_pixel(img)
    lab_b = compute_lab_b_channel(img)
    
    masked_exr = np.where(mask == 255, exr, 0)
    masked_lab_b = np.where(mask == 255, lab_b, 0)
    
    exr_mask = (masked_exr > EXR_THRESHOLD).astype(np.uint8) * 255
    lab_b_mask = np.where(
        (masked_lab_b >= LAB_B_LOW_THRESHOLD) & (masked_lab_b <= LAB_B_HIGH_THRESHOLD),
        255,
        0
    ).astype(np.uint8)
    
    yellow_binary = cv2.bitwise_or(exr_mask, lab_b_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    yellow_binary = cv2.erode(yellow_binary, kernel, iterations=1)  # Erosion to remove small noise spots
    yellow_binary = cv2.dilate(yellow_binary, kernel, iterations=1) # Expansion recovery core area
    
    contours, _ = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    return max_contour, yellow_binary


def visualize_contour_and_save(img, contour, output_path):
    """Draw a red outline on the original image and save it"""
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Visualization saved: {output_path}")


def contour_to_json(contour, orig_json_path, img_path, img_shape):
    """read rawJSON,ReservePan_center_contour_areaand appendYellow_areaoutline"""
    h, w = img_shape[:2]
    
    with open(orig_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        
    preserved_shapes = []
    for shape in json_data.get("shapes", []):
        if shape.get("label") == TARGET_LABEL:
            preserved_shapes.append(shape)
            
    json_data["shapes"] = preserved_shapes
    json_data["imagePath"] = os.path.basename(img_path)
    json_data["imageHeight"] = h
    json_data["imageWidth"] = w
    
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
        json_data["shapes"].append(yellow_shape)
        
    return json_data


def save_yellow_area_json(contour, orig_json_path, img_path, img_shape, json_output_path):
    """Save an outline containing two regionsJSONFile"""
    json_data = contour_to_json(contour, orig_json_path, img_path, img_shape)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"savedJSONFile: {json_output_path}")


def process_single_file(img_path, json_path, output_img_path, output_json_path):
    """Process a single image+JSONvisualization andJSONgenerate"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Unable to read image: {img_path}")
        return
    
    mask = load_json_and_get_mask(json_path, img.shape)
    if np.sum(mask) == 0:
        print(f"⚠️ Not found validPan_center_contour_areamask: {json_path}")
        visualize_contour_and_save(img, None, output_img_path)
        save_yellow_area_json(None, json_path, img_path, img.shape, output_json_path)
        return
    
    contour, _ = extract_yellow_contour(img, mask)
    visualize_contour_and_save(img, contour, output_img_path)
    save_yellow_area_json(contour, json_path, img_path, img.shape, output_json_path)


def batch_process():
    """Process all images in batches+JSONFile"""
    input_path = Path(INPUT_DIR)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]
    
    for img_file in img_files:
        json_file = input_path / f"{img_file.stem}.json"
        if not json_file.exists():
            print(f"⚠️ No correspondence foundJSONFile: {json_file}")
            continue
        
        output_img_file = Path(OUTPUT_DIR) / img_file.name
        output_json_file = Path(JSON_OUTPUT_DIR) / f"{img_file.stem}.json"
        
        process_single_file(img_file, json_file, output_img_file, output_json_file)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    batch_process()
    print("\n✅ All files processed!")
