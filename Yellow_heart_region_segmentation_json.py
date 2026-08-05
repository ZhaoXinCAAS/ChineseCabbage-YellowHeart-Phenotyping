import os
import json
import cv2
import numpy as np
from pathlib import Path

# ===================== 配置项 / Path Configuration =====================
# [默认路径] 指向 GitHub 仓库自带的 Mini 测试集 (可直接运行 Demo)
INPUT_DIR = r"./samples_images" 
OUTPUT_DIR = r"./samples_images/CROP_black/Annotate/Visualisation"
JSON_OUTPUT_DIR = r"./samples_images/CROP_black/Annotate/json"

# -----------------------------------------------------------------------
# [全量数据集运行指南 / Full Dataset Instructions]
# 如果需要运行从 Figshare 下载的完整数据集，请注释掉上方三行，并取消下方三行的注释：
# INPUT_DIR = r"./data/annotations_head_region"
# OUTPUT_DIR = r"./data/Auto_segment_Yellow_heart/True_Yellow_heart/Visualisation"
# JSON_OUTPUT_DIR = r"./data/Auto_segment_Yellow_heart/True_Yellow_heart/json"
# =======================================================================

TARGET_LABEL = "Pan_center_contour_area"
EXR_THRESHOLD = 0.15          # ExR阈值
LAB_B_LOW_THRESHOLD = 20      # Lab-b下限（20）
LAB_B_HIGH_THRESHOLD = 70     # Lab-b上限（70）
EPS = 1e-6                    # 防除零错误
YELLOW_LABEL = "Yellow_area"  # 新提取的黄色区域标签名称


# ---------------------- 辅助函数 ----------------------
def round_and_clip_polygon(points, w, h):
    """裁剪多边形坐标到图像边界并转换为整数"""
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
    """逐像素计算ExR值"""
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)
    
    S = R + G + B
    S_safe = np.where(S == 0, EPS, S)
    ExR = (1.4 * R - G) / (S_safe + EPS)
    return ExR


def compute_lab_b_channel(img):
    """计算Lab颜色空间的b通道（黄-蓝轴：正数越黄，负数越蓝）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    _, _, b_channel = cv2.split(lab)
    return b_channel


def load_json_and_get_mask(json_path, img_shape):
    """加载JSON并生成Pan_center_contour_area的掩膜"""
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
    """提取掩膜内 Lab-b在20~70之间 或 ExR>0.15 的像素点的封闭轮廓"""
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
    yellow_binary = cv2.erode(yellow_binary, kernel, iterations=1)  # 腐蚀去小噪点
    yellow_binary = cv2.dilate(yellow_binary, kernel, iterations=1) # 膨胀恢复核心区域
    
    contours, _ = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    return max_contour, yellow_binary


def visualize_contour_and_save(img, contour, output_path):
    """在原图上绘制红色轮廓并保存"""
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"已保存可视化结果: {output_path}")


def contour_to_json(contour, orig_json_path, img_path, img_shape):
    """读取原始JSON，保留Pan_center_contour_area并追加Yellow_area轮廓"""
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
    """保存包含两个区域轮廓的JSON文件"""
    json_data = contour_to_json(contour, orig_json_path, img_path, img_shape)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"已保存JSON文件: {json_output_path}")


def process_single_file(img_path, json_path, output_img_path, output_json_path):
    """处理单张图片+JSON的可视化和JSON生成"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return
    
    mask = load_json_and_get_mask(json_path, img.shape)
    if np.sum(mask) == 0:
        print(f"⚠️ 未找到有效Pan_center_contour_area掩膜: {json_path}")
        visualize_contour_and_save(img, None, output_img_path)
        save_yellow_area_json(None, json_path, img_path, img.shape, output_json_path)
        return
    
    contour, _ = extract_yellow_contour(img, mask)
    visualize_contour_and_save(img, contour, output_img_path)
    save_yellow_area_json(contour, json_path, img_path, img.shape, output_json_path)


def batch_process():
    """批量处理所有图片+JSON文件"""
    input_path = Path(INPUT_DIR)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]
    
    for img_file in img_files:
        json_file = input_path / f"{img_file.stem}.json"
        if not json_file.exists():
            print(f"⚠️ 未找到对应JSON文件: {json_file}")
            continue
        
        output_img_file = Path(OUTPUT_DIR) / img_file.name
        output_json_file = Path(JSON_OUTPUT_DIR) / f"{img_file.stem}.json"
        
        process_single_file(img_file, json_file, output_img_file, output_json_file)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    batch_process()
    print("\n✅ 所有文件处理完成！")
