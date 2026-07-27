import os
import json
import cv2
import numpy as np
from pathlib import Path

# ---------------------- 配置项（根据实际修改） ----------------------
INPUT_DIR = r"ChineseCabbage-YellowHeart-Phenotyping\samples_images\Data\Entire_leafy_head_annotated_region"  # 图片+JSON所在文件夹
OUTPUT_DIR = r"ChineseCabbage-YellowHeart-Phenotyping\samples_images\Data\Entire_leafy_head_annotated_region\json"  # 可视化结果保存文件夹
JSON_OUTPUT_DIR = r"ChineseCabbage-YellowHeart-Phenotyping\samples_images\Data\Entire_leafy_head_annotated_region\json"  # JSON文件保存文件夹
TARGET_LABEL = "Pan_center_contour_area"
EXR_THRESHOLD = 0.15          # ExR阈值
LAB_B_LOW_THRESHOLD = 20      # Lab-b下限（20）
LAB_B_HIGH_THRESHOLD = 70     # Lab-b上限（70）
EPS = 1e-6                    # 防除零错误
YELLOW_LABEL = "Yellow_area"  # 新的标签名称

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
    # 分离BGR通道（OpenCV读取的是BGR）
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)
    
    # 计算ExR: (1.4*R - G) / (R+G+B + EPS)
    S = R + G + B
    S_safe = np.where(S == 0, EPS, S)
    ExR = (1.4 * R - G) / (S_safe + EPS)
    return ExR

def compute_lab_b_channel(img):
    """计算Lab颜色空间的b通道（黄-蓝轴：正数越黄，负数越蓝）"""
    # 转换为Lab颜色空间（OpenCV默认是BGR输入）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # 分离L(亮度)、a(红-绿)、b(黄-蓝)通道
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
            # 裁剪并转换多边形坐标
            poly = round_and_clip_polygon(points, w, h)
            if poly.shape[0] >= 3:
                cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 255)
    return mask

def extract_yellow_contour(img, mask):
    """提取掩膜内 Lab-b在20~70之间 或 ExR>0.15 的像素点的封闭轮廓"""
    # 1. 计算ExR和Lab-b通道
    exr = compute_exr_per_pixel(img)
    lab_b = compute_lab_b_channel(img)
    
    # 2. 仅保留掩膜内的像素（掩膜外设为0/False）
    masked_exr = np.where(mask == 255, exr, 0)
    masked_lab_b = np.where(mask == 255, lab_b, 0)
    
    # 3. 生成黄色区域二值图
    # 3.1 ExR>0.15 的掩码
    exr_mask = (masked_exr > EXR_THRESHOLD).astype(np.uint8) * 255
    # 3.2 Lab-b在20~70之间 的掩码（核心修改：范围判定）
    lab_b_mask = np.where(
        (masked_lab_b >= LAB_B_LOW_THRESHOLD) & (masked_lab_b <= LAB_B_HIGH_THRESHOLD),
        255,
        0
    ).astype(np.uint8)
    
    # 3.3 合并两个掩码（按位或：满足其一即视为黄色）
    yellow_binary = cv2.bitwise_or(exr_mask, lab_b_mask)
    
    # 4. 形态学操作：去除小噪点，避免无关像素干扰（关键优化）
    kernel = np.ones((3, 3), np.uint8)
    yellow_binary = cv2.erode(yellow_binary, kernel, iterations=1)  # 腐蚀去小噪点
    yellow_binary = cv2.dilate(yellow_binary, kernel, iterations=1) # 膨胀恢复核心区域
    
    # 5. 查找轮廓（只取最外层）
    contours, _ = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选面积最大的轮廓（避免小噪点）
    max_contour = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    return max_contour, yellow_binary

def visualize_contour_and_save(img, contour, output_path):
    """在原图上绘制红色轮廓并保存"""
    # 绘制红色轮廓（线宽2，红色：BGR=(0,0,255)）
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    # 创建输出目录并保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"已保存可视化结果: {output_path}")

# ---------------------- 重点修改区域开始 ----------------------
def contour_to_json(contour, orig_json_path, img_path, img_shape):
    """读取原始JSON，保留Pan_center_contour_area并追加Yellow_area轮廓"""
    h, w = img_shape[:2]
    
    # 1. 读取原 JSON 文件内容
    with open(orig_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        
    # 2. 筛选保留原JSON中的 Pan_center_contour_area 区域
    preserved_shapes = []
    for shape in json_data.get("shapes", []):
        if shape.get("label") == TARGET_LABEL:
            preserved_shapes.append(shape)
            
    # 将 shapes 列表重置为仅包含保留的区域
    json_data["shapes"] = preserved_shapes
    
    # 3. 确保基础元数据正确
    json_data["imagePath"] = os.path.basename(img_path)
    json_data["imageHeight"] = h
    json_data["imageWidth"] = w
    
    # 4. 如果提取到了新的黄色区域轮廓，构建 shape 追加进去
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
        # 追加到 shapes 列表中
        json_data["shapes"].append(yellow_shape)
        
    return json_data

def save_yellow_area_json(contour, orig_json_path, img_path, img_shape, json_output_path):
    """保存包含两个区域轮廓的JSON文件"""
    # 生成更新后的JSON数据
    json_data = contour_to_json(contour, orig_json_path, img_path, img_shape)
    # 创建输出目录
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    # 保存JSON文件
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"已保存JSON文件: {json_output_path}")
# ---------------------- 重点修改区域结束 ----------------------

def process_single_file(img_path, json_path, output_img_path, output_json_path):
    """处理单张图片+JSON的可视化和JSON生成"""
    # 1. 读取原图
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return
    
    # 2. 加载JSON并生成掩膜
    mask = load_json_and_get_mask(json_path, img.shape)
    if np.sum(mask) == 0:
        print(f"⚠️ 未找到有效Pan_center_contour_area掩膜: {json_path}")
        # 无轮廓时直接保存原图和仅包含原区域的JSON
        visualize_contour_and_save(img, None, output_img_path)
        save_yellow_area_json(None, json_path, img_path, img.shape, output_json_path)
        return
    
    # 3. 提取黄色区域轮廓（Lab-b20~70 或 ExR>0.15）
    contour, _ = extract_yellow_contour(img, mask)
    
    # 4. 可视化并保存图片
    visualize_contour_and_save(img, contour, output_img_path)
    
    # 5. 保存包含Pan_center_contour_area和Yellow_area的JSON文件
    save_yellow_area_json(contour, json_path, img_path, img.shape, output_json_path)

def batch_process():
    """批量处理所有图片+JSON文件"""
    input_path = Path(INPUT_DIR)
    # 支持的图片格式
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    # 遍历所有图片文件
    img_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]
    
    for img_file in img_files:
        # 拼接对应JSON路径（同名）
        json_file = input_path / f"{img_file.stem}.json"
        if not json_file.exists():
            print(f"⚠️ 未找到对应JSON文件: {json_file}")
            continue
        
        # 拼接输出图片路径
        output_img_file = Path(OUTPUT_DIR) / img_file.name
        # 拼接输出JSON路径
        output_json_file = Path(JSON_OUTPUT_DIR) / f"{img_file.stem}.json"
        
        # 处理单文件
        process_single_file(img_file, json_file, output_img_file, output_json_file)

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    # 批量处理
    batch_process()
    print("\n✅ 所有文件处理完成！")