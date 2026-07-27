# ChineseCabbage-YellowHeart-Phenotyping
Open-source Python scripts and phenotypic pipelines for automated yellow-heart region segmentation, feature extraction, and multidimensional color space quantification, enabling objective quantitative grading of Chinese cabbage yellow-heart traits based on the Comprehensive Yellow Score (CYS).
Threshold Adjustment for Illumination Variations
Due to potential variations in image lighting and shooting environments, the threshold values for the Excess Red Index (ExR) and Lab color space $b$-channel (e.g., $EXR\_THRESHOLD$ and $LAB\_B\_THRESHOLD$) may require empirical tuning. Users are encouraged to adjust these parameters flexibly based on their specific image datasets and actual experimental conditions.

### 2. Integration with SAM 2 for Batch Segmentation
For automated batch segmentation across diverse crop images, users can leverage the **Meta Segment Anything Model 2 (SAM 2)**. The official repository and pretrained checkpoints can be found at [facebookresearch/sam2](https://github.com/facebookresearch/sam2.git). You can select appropriate model checkpoints and fine-tune mask generation parameters to best fit your target crop species.

### 3. Scope and Methodology Disclaimer
This repository provides the fundamental experimental framework and implementation scripts—including yellow-heart region segmentation, multidimensional color trait extraction, Gaussian Mixture Model (GMM) grading, and broad-sense heritability calculation—designed to offer a methodological reference and workflow pipeline for researchers. Specific parameter configurations, extracted phenotypic features, and final grading thresholds should be carefully calibrated and validated according to the specific biological characteristics of the target crop under investigation.
