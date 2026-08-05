# Quantitative Grading of the Yellow-Heart Trait in Chinese Cabbage via Multidimensional Color Space Analysis

## Description
Open-source Python scripts and phenotypic pipelines for automated yellow-heart region segmentation, feature extraction, and multidimensional color space quantification, enabling objective quantitative grading of Chinese cabbage yellow-heart traits based on the Comprehensive Yellow Score (CYS).
1. Threshold Adjustment for Illumination Variations: 
Due to potential variations in image lighting and shooting environments, the threshold values for the Excess Red Index (ExR) and CIELAB color space b* may require empirical tuning. Users are encouraged to adjust these parameters flexibly based on their specific image datasets and actual experimental conditions.
2. Integration with SAM 3 for Batch Segmentation: 
For automated batch segmentation across diverse crop images, users can leverage the **Segment Anything Model 3 (SAM 3)** via [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling). The pretrained **SAM 3 ViT-H** model supports text-prompted segmentation (e.g., inputting `"cabbage"`), allowing high-throughput mask generation. Pretrained ONNX checkpoints and model configurations can be obtained from the [X-AnyLabeling Model Zoo](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/model_zoo.md).
3. Scope and Methodology Disclaimer: 
This repository provides the fundamental experimental framework and implementation scripts—including yellow-heart region segmentation, multidimensional color trait extraction, Gaussian Mixture Model (GMM) grading, and broad-sense heritability calculation—designed to offer a methodological reference and workflow pipeline for researchers. Specific parameter configurations, extracted phenotypic features, and final grading thresholds should be carefully calibrated and validated according to the specific biological characteristics of the target crop under investigation.

## Requirements

Install dependencies using pip
```bash
pip install -r requirements.txt
```
## Batch Segmentation with SAM 3 (Text-prompt)
Utilizing SAM 3 with the text prompt "cabbage" to automatically perform high-throughput extraction of leaf-heading region masks from cross-section images.

<p align="center">
  <img width="85%" alt="Yeqiu_predict" src="https://github.com/user-attachments/assets/2f37b9a9-11a8-4734-aed9-fb5d39ef488f" />
</p>

## Chinese cabbage and yellow-heart region segmentation
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

