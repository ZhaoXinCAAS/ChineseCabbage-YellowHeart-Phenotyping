"""Extract paired-mask features and transform both sets using one reference fit."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# JSON Tag name definition
LABEL_YELLOW_AUTO   = "Yellow_area"              # Automated yellow heart label
LABEL_YELLOW_MANUAL = "Yellow_area_manual"       # Artificial yellow heart label
LABEL_PAN           = "Pan_center_contour_area"  # Summary ball area label

# Need to be normalized 10 color feature name
COLOR_FEATURES = [
    "R_mean", "G_mean", "B_mean",
    "H_circular_mean_deg", "S_mean", "V_mean",
    "L_mean", "a_mean", "b_mean", "ExR_mean"
]

# ---------------------- Auxiliary algorithm function ----------------------

def round_and_clip_polygon(points, w, h):
    """Clipping polygon coordinates to image boundaries and converting to integers, Prevent out of bounds when filling mask"""
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
    """Calculate HSV in H Circular mean of channel, prevent 0/360degree mean error around"""
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
    """Safe calculation of array mean, ignore NaN and data other than finite values"""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size > 0 else float("nan")

def compute_ExR_mean(R, G, B):
    """Calculate the lighting self-normalization properties ExR mean (1.4*r - g)"""
    R_f, G_f, B_f = R.astype(np.float64), G.astype(np.float64), B.astype(np.float64)
    sum_RGB = R_f + G_f + B_f
    sum_RGB[sum_RGB == 0] = 1.0
    r, g = R_f / sum_RGB, G_f / sum_RGB
    ExR = 1.4 * r - g
    return mean_safe(ExR)

def extract_roi_features(img, mask_target, pan_pixel_count, filename):
    """Given a binary mask, Extract the pixel area ratio of the mask area and10item color features"""
    coords = np.where(mask_target == 255)
    target_pixel_count = int(len(coords[0]))
    if target_pixel_count == 0:
        return None

    yellow_ratio = target_pixel_count / pan_pixel_count if pan_pixel_count > 0 else 0.0

    roi_pixels = img[coords]  # (N, 3) BGR
    B_raw = roi_pixels[:, 0].astype(np.float64)
    G_raw = roi_pixels[:, 1].astype(np.float64)
    R_raw = roi_pixels[:, 2].astype(np.float64)

    R_mean, G_mean, B_mean = mean_safe(R_raw), mean_safe(G_raw), mean_safe(B_raw)
    ExR_mean = compute_ExR_mean(R_raw, G_raw, B_raw)

    # only for ROI Local pixel processing HSV/Lab Transformation to dramatically increase batch extraction speeds
    roi_bgr_3d = roi_pixels.reshape(-1, 1, 3)
    hsv_roi = cv2.cvtColor(roi_bgr_3d, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab_roi = cv2.cvtColor(roi_bgr_3d, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    H_circular_mean_deg = circular_mean_degrees(hsv_roi[:, 0])
    S_mean = mean_safe(hsv_roi[:, 1].astype(np.float64) / 255.0)
    V_mean = mean_safe(hsv_roi[:, 2].astype(np.float64) / 255.0)
    L_mean = mean_safe(lab_roi[:, 0].astype(np.float64) / 255.0 * 100.0)
    a_mean = mean_safe(lab_roi[:, 1].astype(np.float64) - 128.0)
    b_mean = mean_safe(lab_roi[:, 2].astype(np.float64) - 128.0)

    return {
        "filename": filename,
        "Yellow_pixel_count": target_pixel_count,
        "Pan_pixel_count": pan_pixel_count,
        "Yellow_Ratio": yellow_ratio,
        "R_mean": R_mean, "G_mean": G_mean, "B_mean": B_mean,
        "H_circular_mean_deg": H_circular_mean_deg, "S_mean": S_mean, "V_mean": V_mean,
        "L_mean": L_mean, "a_mean": a_mean, "b_mean": b_mean,
        "ExR_mean": ExR_mean
    }

# ---------------------- Single image annotation analysis ----------------------

def parse_image_json(image_path, json_path, require_manual=False):
    """
    Parse a single image with JSON mark: 
      - require_manual=False: used for 1319 benchmark set, Just extract Yellow_area
      - require_manual=True : used for 150 control set, Extract simultaneously Yellow_area_manual and Yellow_area
    """
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    h, w = img.shape[:2]
    fname = Path(image_path).name

    with open(json_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    mask_auto   = np.zeros((h, w), dtype=np.uint8)
    mask_manual = np.zeros((h, w), dtype=np.uint8)
    mask_pan    = np.zeros((h, w), dtype=np.uint8)

    for shape in jd.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        pts = shape.get("points", [])
        poly = round_and_clip_polygon(pts, w, h)
        if poly.shape[0] < 3:
            continue

        if label == LABEL_YELLOW_MANUAL:
            cv2.fillPoly(mask_manual, [poly.reshape((-1, 1, 2))], 255)
        elif label == LABEL_YELLOW_AUTO:
            cv2.fillPoly(mask_auto, [poly.reshape((-1, 1, 2))], 255)
        elif label == LABEL_PAN:
            cv2.fillPoly(mask_pan, [poly.reshape((-1, 1, 2))], 255)

    pan_pixel_count = int(np.sum(mask_pan == 255))

    auto_rec = extract_roi_features(img, mask_auto, pan_pixel_count, fname)
    if auto_rec is None:
        raise ValueError(f"Not here {fname} Valid automatic mask detected in ({LABEL_YELLOW_AUTO})")

    if not require_manual:
        return auto_rec, None

    manual_rec = extract_roi_features(img, mask_manual, pan_pixel_count, fname)
    if manual_rec is None:
        raise ValueError(f"Not here {fname} Valid artificial mask detected in ({LABEL_YELLOW_MANUAL})")

    return auto_rec, manual_rec

# ---------------------- Normalization and preservation functions ----------------------

def apply_scaler_and_save(df, fitted_scaler, output_xlsx, dataset_name):
    """Use global fitting Scaler execute transform, And output in the established column order Excel"""
    normalized_data = fitted_scaler.transform(df[COLOR_FEATURES])

    for i, col_name in enumerate(COLOR_FEATURES):
        df[f"{col_name}_norm"] = normalized_data[:, i]

    priority_cols = ["filename", "Yellow_pixel_count", "Pan_pixel_count", "Yellow_Ratio"]
    final_cols = priority_cols.copy()
    for col_name in COLOR_FEATURES:
        final_cols.append(col_name)
        final_cols.append(f"{col_name}_norm")

    Path(output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    df[final_cols].to_excel(output_xlsx, index=False)
    print(f"[{dataset_name}] total {len(df)} records, Saved to: \n   {output_xlsx}")

# ---------------------- Main control process ----------------------

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path, help="Reference images and automatic annotations.")
    parser.add_argument("paired_dir", type=Path, help="Images with paired automatic and manual labels.")
    parser.add_argument("output_dir", type=Path, help="Directory for the two feature workbooks.")
    args = parser.parse_args()

    # ==================== first step: Traverse 1319 Folder fitting global extrema ====================
    base_p = args.reference_dir
    if not base_p.exists():
        raise FileNotFoundError(f"not found1319Full folder path: {base_p}")

    base_files = sorted([p for p in base_p.iterdir() if p.suffix.lower() in VALID_EXTS])
    print(f"[1/3] scan 1319 Full folder, Found in total {len(base_files)} images, Start extracting global baseline features..")

    base_rows = []
    for idx, img_path in enumerate(base_files, 1):
        json_path = base_p / f"{img_path.stem}.json"
        if not json_path.exists():
            continue
        try:
            auto_rec, _ = parse_image_json(str(img_path), str(json_path), require_manual=False)
            base_rows.append(auto_rec)
            if idx % 200 == 0 or idx == len(base_files):
                print(f"   ..Progress {idx}/{len(base_files)}")
        except Exception as e:
            print(f"   [ERR] {img_path.name} Failed to extract: {e}")

    if not base_rows:
        raise RuntimeError("Never follow 1319 Extract any valid data from the folder, Please check the path or JSON label! ")

    df_1319 = pd.DataFrame(base_rows)
    global_scaler = MinMaxScaler()
    global_scaler.fit(df_1319[COLOR_FEATURES])

    print(f"\nAlready based on {len(df_1319)} full image [{LABEL_YELLOW_AUTO}] Fit global extremum benchmark (Min / Max): ")
    for idx, feat in enumerate(COLOR_FEATURES):
        print(f"   - {feat:<20}: Min = {global_scaler.data_min_[idx]:.4f}, Max = {global_scaler.data_max_[idx]:.4f}")

    # ==================== Step 2: Traverse 150 Extract double mask features against folders ====================
    paired_p = args.paired_dir
    if not paired_p.exists():
        raise FileNotFoundError(f"not found150Check folder path: {paired_p}")

    paired_files = sorted([p for p in paired_p.rglob('*') if p.suffix.lower() in VALID_EXTS])
    print(f"\n[2/3] scan 150 Check folder, Found in total {len(paired_files)} images, Start synchronously extracting manual and automatic mask features..")

    manual_150_rows = []
    auto_150_rows   = []

    for img_path in paired_files:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"   [WARN] Lack of corresponding JSON File: {json_path.name}")
            continue
        try:
            auto_rec, manual_rec = parse_image_json(str(img_path), str(json_path), require_manual=True)
            manual_150_rows.append(manual_rec)
            auto_150_rows.append(auto_rec)
            print(f"   [OK] {img_path.name} -> Manual Ratio: {manual_rec['Yellow_Ratio']:.4f} | Auto Ratio: {auto_rec['Yellow_Ratio']:.4f}")
        except Exception as e:
            print(f"   [ERR] {img_path.name} Failed to extract: {e}")

    if not manual_150_rows:
        raise RuntimeError("Never follow 150 Extract valid control data from the folder, please check JSON label! ")

    # ==================== Step 3: For unified use 1319 Global benchmark normalization and saving ====================
    print(f"\n[3/3] in use 1319 global benchmark pair {len(manual_150_rows)} Group paired samples perform normalization..")
    df_manual_150 = pd.DataFrame(manual_150_rows)
    df_auto_150   = pd.DataFrame(auto_150_rows)

    apply_scaler_and_save(df_manual_150, global_scaler, args.output_dir / "manual_mask_features.xlsx", f"artificial mask ({LABEL_YELLOW_MANUAL})")
    apply_scaler_and_save(df_auto_150,   global_scaler, args.output_dir / "automatic_mask_features.xlsx",   f"automatic mask ({LABEL_YELLOW_AUTO})")

    print("\n" + "=" * 65)
    print("All processing completed! Manual and automatic mask data aligned 1319 All extreme value benchmarks and exported separately. ")
    print("=" * 65)

if __name__ == "__main__":
    main()