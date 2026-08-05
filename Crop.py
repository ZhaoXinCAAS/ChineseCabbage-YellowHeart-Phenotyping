import cv2
import json
import numpy as np
import os

# ===================== 配置参数 / Path Configuration =====================
# [默认路径] 指向 GitHub 仓库自带的测试集 (克隆项目后可直接运行 Demo)
json_folder_path = "./samples_images/results/SAM3_predicted"
image_folder_path = "./samples_images/results/SAM3_predicted"

# 结果保存输出路径
output_root_dir = "./samples_images/results/annotations_head_region"

# -------------------------------------------------------------------------
# [全量数据集运行指南 / Full Dataset Instructions]
# 如果需要运行从 Figshare 下载的完整数据集，请注释掉上方路径，并取消下方三行的注释：
# json_folder_path = "./data/SAM3_predicted"
# image_folder_path = "./data/SAM3_predicted"
# output_root_dir = "./data/CROP_black"
# =========================================================================

# 自动创建输出文件夹（不存在则自动创建，存在则忽略）
os.makedirs(output_root_dir, exist_ok=True)

# 遍历 JSON 文件夹中的每个 JSON 文件
for json_filename in os.listdir(json_folder_path):
    if json_filename.endswith('.json'):
        json_path = os.path.join(json_folder_path, json_filename)
        
        # 读取 JSON 文件（增加编码容错）
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            print(f"读取 JSON 文件失败 {json_filename}: {str(e)}")
            continue
        
        # ========== 核心：精准匹配原始图像文件名 ==========
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
            print(f"未找到 {img_base_name} 对应的图像文件")
            continue
        
        # 读取原始图像
        img = cv2.imread(target_image_path)
        if img is None:
            print(f"无法打开或读取文件: {target_image_path}")
            continue

        # 计数器：处理多标注区域的情况
        object_count = 1
        shape_count = len(data.get('shapes', []))

        # 提取每个区域的图像部分
        for region in data.get('shapes', []):
            coords = region['points']
            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            
            # 创建掩膜（保留目标区域，背景为黑色）
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            # 计算包围盒并扩充边界（比例 0.2）
            x, y, w, h = cv2.boundingRect(pts)
            expand_ratio = 0.2
            dx = int(w * expand_ratio)
            dy = int(h * expand_ratio)
            x = max(x - dx, 0)
            y = max(y - dy, 0)
            w = min(w + 2 * dx, img.shape[1] - x)
            h = min(h + 2 * dy, img.shape[0] - y)

            # 裁剪感兴趣区域 (ROI)
            roi = masked_img[y:y+h, x:x+w]

            # 检查 ROI 是否为空
            if roi.size == 0:
                print(f"图像 {target_image_filename} 的区域 {object_count} 裁剪结果为空")
                object_count += 1
                continue

            # ========== 保持文件名格式统一 ==========
            if shape_count == 1:
                # 单区域：与原图像文件名一致
                output_filename = target_image_filename
            else:
                # 多区域：原名称_编号.原后缀
                file_ext = os.path.splitext(target_image_filename)[1]
                output_filename = f"{img_base_name}_{object_count}{file_ext}"
            
            # 拼接输出路径
            output_path = os.path.join(output_root_dir, output_filename)

            # 保存裁剪后的图像
            success = cv2.imwrite(output_path, roi)
            if success:
                print(f"成功保存: {output_path}")
            else:
                print(f"保存失败: {output_path}")

            object_count += 1

print("所有文件处理完成！")
