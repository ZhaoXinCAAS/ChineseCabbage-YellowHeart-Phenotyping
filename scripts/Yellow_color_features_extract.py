import os
import json
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

INPUT_FOLDER = "./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/"
OUTPUT_XLSX = "./full_dataset_tables/roi_10_color_features_with_ratio.xlsx"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_YELLOW = "Yellow_area"
LABEL_PAN = "Pan_center_contour_area"

def round_and_clip_polygon(points, w, h):
    pts = []
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        x, y = float(p[0]), float(p[1])
        xi = max(0, min(w - 1, int(round(x))))
        yi = max(0, min(h - 1, int(round(y))))
        pts.append((xi, yi))
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)

def circular_mean_degrees(h_vals_0_179):
    if len(h_vals_0_179) == 0:
        return float("nan")
    deg = np.asarray(h_vals_0_179, dtype=np.float64) * 2.0
    rad = np.deg2rad(deg)
    sinm = np.nanmean(np.sin(rad))
    cosm = np.nanmean(np.cos(rad))
    if np.isnan(sinm) or np.isnan(cosm):
        return float("nan")
    mean_rad = math.atan2(sinm, cosm)
    mean_deg = math.degrees(mean_rad)
    if mean_deg < 0:
        mean_deg += 360.0
    return float(mean_deg)

def mean_safe(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size > 0 else float("nan")

def compute_ExR_mean(R, G, B):
    R_f, G_f, B_f = R.astype(np.float64), G.astype(np.float64), B.astype(np.float64)
    sum_RGB = R_f + G_f + B_f
    sum_RGB[sum_RGB == 0] = 1.0 
    r, g = R_f / sum_RGB, G_f / sum_RGB
    ExR = 1.4 * r - g
    return mean_safe(ExR)

def process_image_pair(image_path, json_path):
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h, w = img.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    mask_yellow = np.zeros((h, w), dtype=np.uint8)
    mask_pan = np.zeros((h, w), dtype=np.uint8)

    for shape in jd.get("shapes", []):
        label = shape.get("label")
        pts = shape.get("points", [])
        poly = round_and_clip_polygon(pts, w, h)
        if poly.shape[0] < 3:
            continue
        
        if label == LABEL_YELLOW:
            cv2.fillPoly(mask_yellow, [poly.reshape((-1, 1, 2))], 255)
        elif label == LABEL_PAN:
            cv2.fillPoly(mask_pan, [poly.reshape((-1, 1, 2))], 255)

    coords_yellow = np.where(mask_yellow == 255)
    yellow_pixel_count = int(len(coords_yellow[0]))
    pan_pixel_count = int(np.sum(mask_pan == 255))
    yellow_ratio = yellow_pixel_count / pan_pixel_count if pan_pixel_count > 0 else 0.0

    if yellow_pixel_count == 0:
        raise ValueError(f"No yellow region pixels detected in {Path(image_path).name}")

    roi_pixels = img[coords_yellow]
    B_raw = roi_pixels[:, 0].astype(np.float64)
    G_raw = roi_pixels[:, 1].astype(np.float64)
    R_raw = roi_pixels[:, 2].astype(np.float64)

    R_mean, G_mean, B_mean = mean_safe(R_raw), mean_safe(G_raw), mean_safe(B_raw)
    ExR_mean = compute_ExR_mean(R_raw, G_raw, B_raw)

    hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[coords_yellow]
    lab_roi = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[coords_yellow]

    H_circular_mean_deg = circular_mean_degrees(hsv_roi[:, 0])
    S_mean = mean_safe(hsv_roi[:, 1].astype(np.float64) / 255.0)
    V_mean = mean_safe(hsv_roi[:, 2].astype(np.float64) / 255.0)
    L_mean = mean_safe(lab_roi[:, 0].astype(np.float64) / 255.0 * 100.0) 
    a_mean = mean_safe(lab_roi[:, 1].astype(np.float64) - 128.0)         
    b_mean = mean_safe(lab_roi[:, 2].astype(np.float64) - 128.0)         

    return {
        "filename": Path(image_path).name,
        "Yellow_pixel_count": yellow_pixel_count,
        "Pan_pixel_count": pan_pixel_count,
        "Yellow_Ratio": yellow_ratio, 
        "R_mean": R_mean, "G_mean": G_mean, "B_mean": B_mean,
        "H_circular_mean_deg": H_circular_mean_deg, "S_mean": S_mean, "V_mean": V_mean,
        "L_mean": L_mean, "a_mean": a_mean, "b_mean": b_mean,
        "ExR_mean": ExR_mean
    }

def main(input_folder, output_xlsx):
    input_p = Path(input_folder)
    rows = []
    files = sorted([p for p in input_p.iterdir() if p.suffix.lower() in VALID_EXTS])

    for img_path in files:
        json_path = input_p / f"{img_path.stem}.json"
        if not json_path.exists():
            continue

        try:
            rec = process_image_pair(str(img_path), str(json_path))
            rows.append(rec)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        priority_cols = ["filename", "Yellow_pixel_count", "Pan_pixel_count", "Yellow_Ratio"]
        other_cols = [c for c in df.columns if c not in priority_cols]
        os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
        df[priority_cols + other_cols].to_excel(output_xlsx, index=False)
        print(f"Feature extraction complete. Output saved to: {output_xlsx}")

if __name__ == "__main__":
    main(INPUT_FOLDER, OUTPUT_XLSX)
