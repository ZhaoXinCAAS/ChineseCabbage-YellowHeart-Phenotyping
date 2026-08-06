# Quantitative Grading of the Yellow-Heart Trait in Chinese Cabbage via Multidimensional Color Space Analysis

## Description:
Open-source Python scripts and phenotypic pipelines for automated yellow-heart region segmentation, feature extraction, and multidimensional color space quantification, enabling objective quantitative grading of Chinese cabbage yellow-heart traits based on the Comprehensive Yellow Score (CYS).
1. Threshold Adjustment for Illumination Variations: 
Due to potential variations in image lighting and shooting environments, the threshold values for the Excess Red Index (ExR) and CIELAB color space b* may require empirical tuning. Users are encouraged to adjust these parameters flexibly based on their specific image datasets and actual experimental conditions.
2. Integration with SAM 3 for Batch Segmentation: 
For automated batch segmentation across diverse crop images, users can leverage the **Segment Anything Model 3 (SAM 3)** via [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling). The pretrained **SAM 3 ViT-H** model supports text-prompted segmentation (e.g., inputting `"cabbage"`), allowing high-throughput mask generation. Pretrained ONNX checkpoints and model configurations can be obtained from the [X-AnyLabeling Model Zoo](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/model_zoo.md).
3. Scope and Methodology Disclaimer: 
This repository provides the fundamental experimental framework and implementation scripts—including yellow-heart region segmentation, multidimensional color trait extraction, Gaussian Mixture Model (GMM) grading, and broad-sense heritability calculation—designed to offer a methodological reference and workflow pipeline for researchers. Specific parameter configurations, extracted phenotypic features, and final grading thresholds should be carefully calibrated and validated according to the specific biological characteristics of the target crop under investigation.

## Dataset & Project Structure
To ensure complete reproducibility, the full high-throughput image dataset and raw analysis results used in this study are hosted on Figshare. A mini test dataset (./samples_images/) is provided directly within this repository for a quick demo.

### Full Dataset Access: [Download Full Dataset on Figshare](https://figshare.com/s/b0fd423cf7abd8d44f1b)

## Requirements:
Install dependencies using pip
```bash
pip install -r requirements.txt
```

## 1. Downloading the Datasets
The full-scale dataset is deposited on Figshare ([https://doi.org/10.6084/m9.figshare.33154715](https://doi.org/10.6084/m9.figshare.33154715)). 

Please download the `data.zip` file (≈17.53 GB) and extract its contents into a folder named `data/` at the root directory of this repository.

### Expected folder structure after extraction:

```text
ChineseCabbage-YellowHeart-Phenotyping/
├── data/
│   ├── Raw_images/                     # Original unprocessed RGB cross-section images
│   ├── SAM3_predicted/                 # Initial binary mask predictions from SAM 3 inference
│   ├── Annotations_head_region/        # Manually curated polygon annotations (JSON format)
│   ├── Auto_segment_Yellow_heart/      # Automated yellow-heart segmentation evaluation dataset
│   │   ├── False_Yellow_heart/         # Incorrectly segmented cases
│   │   │   ├── Head_region_label/      # Cropped single-head images with manually annotated head polygon JSON files("Pan_center_contour_area")
│   │   │   ├── Visualization/          # Overlay visualizations of segmented yellow-heart regions
│   │   │   └── json/                   # Exported polygon mask annotations in JSON format
│   │   └── True_Yellow_heart/          # Correctly segmented cases
│   │       ├── Head_region_label/      # Cropped single-head images with manually annotated head polygon JSON files("Pan_center_contour_area")
│   │       ├── Visualization/          # Overlay visualizations of segmented yellow-heart regions
│   │       └── json/                   # Exported polygon mask annotations in JSON format
│   ├── Fisher_score_data/              # Categorized datasets for Fisher score evaluation based on yellow-heart intensity
│   │   ├── Deep_yellow/                # Cross-section images with deep yellow-heart intensity
│   │   ├── Light_yellow/               # Cross-section images with light yellow-heart intensity
│   │   └── yellow/                     # Cross-section images with moderate yellow-heart intensity
│   └── SAM3_onnx/                      # SAM 3 ONNX weights and X-AnyLabeling annotation tools
```

## 1. Batch Segmentation with SAM 3 (Text-prompt):
Utilizing SAM 3 with the text prompt "cabbage" to automatically perform high-throughput extraction of leaf-heading region masks from cross-section images.
* **Tools & Models**: The pre-trained ONNX model files and X-AnyLabeling executable software are provided in `./data/SAM3_onnx/`.
* **Input**: `./samples_images/results/Raw_images/` (Demo) | `./data/Raw_images/` (Full Dataset)
* **Output**: `./samples_images/results/SAM3_predicted/` (Demo) | `./data/SAM3_predicted/` (Full Dataset)
<p align="center">
  <img width="85%" alt="Yeqiu_predict" src="https://github.com/user-attachments/assets/2f37b9a9-11a8-4734-aed9-fb5d39ef488f" />
</p>

## 2. Individual Chinese cabbage Head Cropping & Extraction:
`Crop.py` is used to crop individual cabbage head ROIs from raw images based on JSON annotations.
```bash
python scripts/Crop.py
```
* **Input**: './samples_images/results/SAM3_predicted/' (Demo) | './data/SAM3_predicted/' (Full Dataset)
* **Output**: './samples_images/results/annotations_head_region/' (Demo) | './data/annotations_head_region/' (Full Dataset)

## 3. Manual Annotation of Chinese cabbage head region：
Use [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) to manually annotate the precise boundaries of both the head region and the short stem for each cropped single-cabbage cross-section image. The generated `.json` files are used for subsequent phenotypic feature and color analysis.

* **Tool**: X-AnyLabeling
* **Label Classes**: 
  * `Pan_center_contour_area` (Polygon annotation for the main leaf-head region)
  * `Short_stem` (Polygon annotation for the internal short stem)
* **Input**: `./samples_images/results/annotations_head_region/` (Demo) | `./data/annotations_head_region/` (Full Dataset)
* **Output**: `./samples_images/results/annotations_head_region/` (Demo) | `./data/annotations_head_region/` (Full Dataset)

## 4. Automated Yellow-Heart Region Extraction

Automated algorithm (`Yellow_heart_region_segmentation_json.py`) is used to extract the internal yellow-heart tissue within the annotated leaf-heading region (`Pan_center_contour_area`), excluding the short stem area.
```bash
python scripts/Yellow_heart_region_segmentation_json.py
```
* **Input**: `./samples_images/results/annotations_head_region` (Demo) | `./data/annotations_head_region/` (Full Dataset)
* **Output**: `./samples_images/results/Auto_segment_Yellow_heart/` (Demo) | `./data/Auto_segment_Yellow_heart/` (Full Dataset)

## 5. Chinese cabbage and yellow-heart region segmentation:
Automated Segmentation of Head Region and Yellow-Heart Trait
Figure: Demonstration of annotated head region masks and automated extraction of internal yellow-heart tissues.

<!-- 2. 下方 3 张动图：统一高度对齐排版 -->
<table border="0">
  <tr>
    <td width="33%" align="center" valign="middle">
      <img height="250px" src="https://github.com/user-attachments/assets/491e65cc-eb52-494b-aa68-cd3a766d9327" alt="Yeqiu_predict_3" />
    </td>
    <td width="33%" align="center" valign="middle">
      <img height="250px" src="https://github.com/user-attachments/assets/efe18338-9b2b-43fe-8f9d-ff1cc1ae6141" alt="Yeqiu_predict_2" />
    </td>
    <td width="33%" align="center" valign="middle">
      <img height="250px" src="https://github.com/user-attachments/assets/9f7cceca-dcd7-404b-8321-dbfbd9706ac7" alt="Yeqiu_predict_1" />
    </td>
  </tr>
</table>
<hr style="height: 1px; border: none; background-color: #e1e4e8; margin: 20px 0;" />

## 6. Yellow-Heart Phenotypic Trait Extraction:
`Yellow_color_features_extract.py` is used to automatically extract quantitative phenotypic traits (yellow-heart area ratio and 10 color space features) from segmented Chinese cabbage head regions.
```bash
python scripts/Yellow_color_features_extract.py
```
* **Input**: `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/` (Full Dataset)
* **Output**: `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx` (Full Dataset)

## 7. Phenotypic Data Min-Max Normalization:
`Min_Max.py` performs Min-Max normalization on the extracted raw phenotypic trait dataset to scale all feature values (e.g., area ratio and color parameters) into the range of [0, 1], eliminating scale differences for downstream analysis.
```bash
python scripts/Min_Max.py
```
* **Input**: `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx` (Full Dataset)
* **Output**: `./samples_images/results/normalized_data_1319.xlsx` (Demo)

## 8. Fisher Discriminant Score Calculation:
`Fisher_scores.py` computes the Fisher discriminant score for each extracted phenotypic feature, evaluating its power to differentiate between distinct yellow-heart phenotype categories.
```bash
python scripts/Fisher_scores.py
```

* **Input**: `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx` (Full Dataset) and `./data/Fisher_score_data/`
* **Output**: `./samples_images/results/fisher_weights_418.xlsx` (Demo)

## 9. GMM-Based Color Trait Modeling and Distribution Visualization:
`GMM_CYS.py` applies Gaussian Mixture Models (GMM) to model the CYS color distribution within heading Chinese cabbage yellow-heart regions and generates corresponding probability density distribution plots.
```bash
python scripts/GMM_CYS.py
```

* **Input**: `./samples_images/results/CYS_1319.xlsx`
* **Output**: `./samples_images/results/Plot/` (Demo)

## 10. Broad-Sense Heritability Calculation:
```bash
python scripts/H2.py
```

* **Input**: `./samples_images/results/CYS_1319.xlsx`
* **Output**: `./samples_images/results/Plot/` (Demo)
