<p align="center">
  <img width="85%" alt="Yeqiu_predict" src="https://github.com/user-attachments/assets/2f37b9a9-11a8-4734-aed9-fb5d39ef488f" />
</p>

<!-- 2. 下方 3 张图片一字排开 -->
<table border="0">
  <tr>
    <td width="33%" align="center">
      <img src="https://github.com/user-attachments/assets/491e65cc-eb52-494b-aa68-cd3a766d9327" alt="Yeqiu_predict_3" />
    </td>
    <td width="33%" align="center">
      <img src="https://github.com/user-attachments/assets/efe18338-9b2b-43fe-8f9d-ff1cc1ae6141" alt="Yeqiu_predict_2" />
    </td>
    <td width="33%" align="center">
      <img src="https://github.com/user-attachments/assets/9f7cceca-dcd7-404b-8321-dbfbd9706ac7" alt="Yeqiu_predict_1" />
    </td>
  </tr>
</table>
# Quantitative Grading of the Yellow-Heart Trait in Chinese Cabbage via Multidimensional Color Space Analysis
Open-source Python scripts and phenotypic pipelines for automated yellow-heart region segmentation, feature extraction, and multidimensional color space quantification, enabling objective quantitative grading of Chinese cabbage yellow-heart traits based on the Comprehensive Yellow Score (CYS).
1. Threshold Adjustment for Illumination Variations: 
Due to potential variations in image lighting and shooting environments, the threshold values for the Excess Red Index (ExR) and CIELAB color space b* may require empirical tuning. Users are encouraged to adjust these parameters flexibly based on their specific image datasets and actual experimental conditions.
2. Integration with SAM 2 for Batch Segmentation: 
For automated batch segmentation across diverse crop images, users can leverage the **Segment Anything Model 2 (SAM 2)**. The official repository and pretrained checkpoints can be found at [facebookresearch/sam2](https://github.com/facebookresearch/sam2.git). You can select appropriate model checkpoints and fine-tune mask generation parameters to best fit your target crop species.
3. Scope and Methodology Disclaimer: 
This repository provides the fundamental experimental framework and implementation scripts—including yellow-heart region segmentation, multidimensional color trait extraction, Gaussian Mixture Model (GMM) grading, and broad-sense heritability calculation—designed to offer a methodological reference and workflow pipeline for researchers. Specific parameter configurations, extracted phenotypic features, and final grading thresholds should be carefully calibrated and validated according to the specific biological characteristics of the target crop under investigation.
