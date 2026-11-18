"""Evaluate manual and automated yellow-heart polygons stored in the same JSON."""

from argparse import ArgumentParser
from pathlib import Path
import os
import json
import numpy as np
import cv2
import pandas as pd

# Label names for the two yellow-centered areas
LABEL_MANUAL = "Yellow_area_manual"
LABEL_AUTO = "Yellow_area"


def get_canvas_size(json_data, json_path):
    """Get canvas width and height,Prioritize from JSON of imageHeight/imageWidth read,If not, read the picture with the same name or adapt it according to the coordinates"""
    h = json_data.get("imageHeight")
    w = json_data.get("imageWidth")
    if h and w:
        return int(h), int(w)

    # Try to read the dimensions from an image with the same name in the same directory
    base_name = os.path.splitext(json_path)[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        img_path = base_name + ext
        if os.path.exists(img_path):
            # Support Chinese path reading
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.shape[:2]

    # If there is no picture,Calculate based on the maximum coordinates of all polygons
    max_x, max_y = 0, 0
    for shape in json_data.get("shapes", []):
        for pt in shape.get("points", []):
            max_x = max(max_x, int(np.ceil(pt[0])))
            max_y = max(max_y, int(np.ceil(pt[1])))
    return max_y + 50, max_x + 50

def create_mask_from_shapes(shapes, target_label, height, width):
    """will specify label All polygons are drawn as binary masks (0 and 1)"""
    mask = np.zeros((height, width), dtype=np.uint8)
    matched_count = 0

    for shape in shapes:
        raw_label = shape.get("label", "").strip()
        # Compatible with some software that may contain a confidence suffix when exporting(Such as "Yellow_area 1.00"),Strictly distinguish between manual
        if target_label == LABEL_MANUAL:
            is_match = (raw_label == LABEL_MANUAL or raw_label.startswith(LABEL_MANUAL + " "))
        else:
            is_match = (raw_label == LABEL_AUTO or (raw_label.startswith(LABEL_AUTO + " ") and "manual" not in raw_label))

        if is_match:
            pts = np.array(shape.get("points", []), dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
                matched_count += 1

    return mask, matched_count

def compute_iou_dice(mask1, mask2):
    """Calculate the pixel area of two binary masks,intersection,union,IoU and Dice"""
    area1 = int(np.sum(mask1))
    area2 = int(np.sum(mask2))

    intersection = int(np.sum((mask1 == 1) & (mask2 == 1)))
    union = int(np.sum((mask1 == 1) | (mask2 == 1)))

    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / (area1 + area2) if (area1 + area2) > 0 else 0.0

    return area1, area2, intersection, union, iou, dice

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Flat or grouped directory of paired-label JSONs.")
    parser.add_argument("output", type=Path, help="Path for the detailed and summary workbook.")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Annotation directory does not exist: {args.input_dir}")
    paths = sorted(args.input_dir.rglob("*.json"))
    if not paths:
        parser.error(f"No JSON annotations found: {args.input_dir}")
    records = []

    for json_path in paths:
        folder = json_path.parent.name if json_path.parent != args.input_dir else "all"
        file_id = json_path.stem
        jf = json_path.name

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        h, w = get_canvas_size(data, json_path)
        shapes = data.get("shapes", [])

        mask_manual, count_manual = create_mask_from_shapes(shapes, LABEL_MANUAL, h, w)
        mask_auto, count_auto = create_mask_from_shapes(shapes, LABEL_AUTO, h, w)

        area_m, area_a, inter, union, iou, dice = compute_iou_dice(mask_manual, mask_auto)

        status = "normal"
        if count_manual == 0 or count_auto == 0:
            status = f"Missing tag (manual:{count_manual}, auto:{count_auto})"
            print(f"  [{folder}] {jf} -> {status}")

        records.append({
            "Image_ID": file_id,
            "Yellow_Level": folder,
            "Manual_Area_px": area_m,
            "Auto_Area_px": area_a,
            "Intersection_px": inter,
            "Union_px": union,
            "IoU": round(iou, 6),
            "Dice": round(dice, 6),
            "Status": status
        })

    df_detail = pd.DataFrame(records)

    # Generate summary statistics table for each level(mean ± standard deviation)
    if not df_detail.empty:
        summary_rows = []
        for level, group in df_detail.groupby("Yellow_Level", sort=False):
            summary_rows.append({
                "Yellow_Level": level,
                "Sample_Count": len(group),
                "Mean_IoU": round(group["IoU"].mean(), 4),
                "Std_IoU": round(group["IoU"].std(), 4),
                "Mean_Dice": round(group["Dice"].mean(), 4),
                "Std_Dice": round(group["Dice"].std(), 4)
            })
        # add all 150 Zhang's total row
        summary_rows.append({
            "Yellow_Level": "All_Total",
            "Sample_Count": len(df_detail),
            "Mean_IoU": round(df_detail["IoU"].mean(), 4),
            "Std_IoU": round(df_detail["IoU"].std(), 4),
            "Mean_Dice": round(df_detail["Dice"].mean(), 4),
            "Std_Dice": round(df_detail["Dice"].std(), 4)
        })
        df_summary = pd.DataFrame(summary_rows)

        # write Excel(Contains two worksheets:detailed list + Summary statistics table)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            df_detail.to_excel(writer, sheet_name="Per_Image", index=False)
            df_summary.to_excel(writer, sheet_name="Summary_By_Level", index=False)

        print("\n" + "=" * 60)
        print(f"Calculation completed!Results have been saved to:\n{args.output}")
        print("\nPreview of average indicators for each level:")
        print(df_summary.to_string(index=False))
    else:
        print("Didn't read any JSON File,Please check if the path is correct.")

if __name__ == "__main__":
    main()