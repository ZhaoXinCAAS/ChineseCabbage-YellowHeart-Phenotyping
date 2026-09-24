# Quantitative Grading of the Yellow-Heart Trait in Chinese Cabbage via Multidimensional Color Space Analysis

## Description

Open-source Python scripts and phenotypic pipelines for automated
yellow-heart region segmentation, multidimensional color-space feature
extraction, and objective quantitative grading of the Chinese cabbage
yellow-heart trait based on the **Comprehensive Yellow Score (CYS)**.

This repository provides the fundamental experimental framework and
implementation scripts — yellow-heart region segmentation, multidimensional
color trait extraction, Gaussian Mixture Model (GMM) grading, and broad-sense
heritability calculation — as a methodological reference and reproducible
workflow for plant phenomics researchers.

### Methodological notes

1. **Threshold Adjustment for Illumination Variations** — Due to potential
   variations in image lighting and shooting environments, the threshold
   values for the Excess Red Index (ExR) and CIELAB color space b\* may require
   empirical tuning. Users are encouraged to adjust these parameters flexibly
   based on their specific image datasets and actual experimental conditions.
2. **Integration with SAM 3 for Batch Segmentation** — For automated batch
   segmentation across diverse crop images, users can leverage the
   **Segment Anything Model 3 (SAM 3)** via
   [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling). The pretrained
   **SAM 3 ViT-H** model supports text-prompted segmentation (e.g., inputting
   `"cabbage"`), allowing high-throughput mask generation. Pretrained ONNX
   checkpoints and model configurations can be obtained from the
   [X-AnyLabeling Model Zoo](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/model_zoo.md).
3. **Scope and Methodology Disclaimer** — Specific parameter configurations,
   extracted phenotypic features, and final grading thresholds should be
   carefully calibrated and validated according to the specific biological
   characteristics of the target crop under investigation.

## Dataset & Project Structure

To ensure complete reproducibility, the full high-throughput image dataset and
raw analysis results used in this study are hosted on Figshare. A mini test
dataset (`./samples_images/`) is provided directly within this repository for
a quick demo.

### Full Dataset Access

[Download Full Dataset on Figshare](https://figshare.com/s/b0fd423cf7abd8d44f1b)

### Demo dataset layout (`./samples_images/`)

```
samples_images/
└── results/
    ├── Raw_images/                 # step 1: original cross-section photos
    ├── SAM3_predicted/             # step 1: SAM 3 masks + JSON
    ├── annotations_head_region/   # steps 2–3: cropped heads + head-region JSON
    ├── Auto_segment_Yellow_heart/  # step 4: segmented yellow-heart output
    │   ├── json/                   #       (images + Yellow_area JSON used by step 6)
    │   ├── Head_region_label/
    │   └── Visualization/
    └── Plot/                       # steps 9–10: GMM & heritability figures
```

## Requirements

Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/) (recommended;
`pyproject.toml` + `uv.lock` give fully reproducible environments):

```bash
uv sync --extra dev   # runtime + pytest/ruff for contributors
```

or, with plain `pip` (no `requirements.txt` needed — metadata lives in
`pyproject.toml`):

```bash
pip install -e .          # runtime only
pip install -e ".[dev]"   # + pytest/ruff
```

> **Note on SAM 3:** Step 1 (batch segmentation with SAM 3) is performed in
> the external **X-AnyLabeling** GUI, not by the Python scripts in this
> repository. If you also need to run SAM 3 / ONNX inference from Python,
> install the optional extra: `pip install -e ".[sam]"`.

All downstream scripts accept command-line arguments so the Demo and full
Figshare datasets can be selected without editing source code.

```bash
# Demo dataset (defaults — no flags needed)
uv run --extra dev <command>

# Full Figshare dataset — pass paths explicitly
uv run --extra dev <command> --input <full_path> --output <full_path>
```

## 1. Batch Segmentation with SAM 3 (Text-prompt)

Utilizing SAM 3 with the text prompt "cabbage" to automatically perform
high-throughput extraction of leaf-heading region masks from cross-section
images.
* **Tools & Models**: The pre-trained ONNX model files and X-AnyLabeling
  executable software are provided in `./data/SAM3_onnx/`.
* **Input**: `./samples_images/results/Raw_images/` (Demo) | `./data/Raw_images/` (Full Dataset)
* **Output**: `./samples_images/results/SAM3_predicted/` (Demo) | `./data/SAM3_predicted/` (Full Dataset)

<p align="center">
  <img width="85%" alt="Yeqiu_predict" src="https://github.com/user-attachments/assets/2f37b9a9-11a8-4734-aed9-fb5d39ef488f" />
</p>

## 2. Individual Chinese cabbage Head Cropping & Extraction

`crop_heads.py` crops individual cabbage head ROIs from raw images based on JSON
annotations.

```bash
# Demo
crop-heads
# Full dataset
crop-heads \
  --input ./data/SAM3_predicted \
  --images ./data/SAM3_predicted \
  --output ./data/annotations_head_region
```

* **Input**: `./samples_images/results/SAM3_predicted/` (Demo) | `./data/SAM3_predicted/` (Full Dataset)
* **Output**: `./samples_images/results/annotations_head_region/` (Demo) | `./data/annotations_head_region/` (Full Dataset)

## 3. Manual Annotation of Chinese cabbage head region

Use [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) to manually
annotate the precise boundaries of both the head region and the short stem for
each cropped single-cabbage cross-section image. The generated `.json` files
are used for subsequent phenotypic feature and color analysis.

* **Tool**: X-AnyLabeling
* **Label Classes**:
  * `Pan_center_contour_area` (Polygon annotation for the main leaf-head region)
  * `Short_stem` (Polygon annotation for the internal short stem)
* **Input**: './samples_images/results/annotations_head_region/' (Demo) | './data/Annotations_head_region/' (Full Dataset)
* **Output**: './samples_images/results/annotations_head_region/' (Demo) | './data/Annotations_head_region/' (Full Dataset)

## 6. Automated Yellow-Heart Region Extraction

`segment_yellow_heart.py` extracts the internal yellow-heart
tissue within the annotated leaf-heading region (`Pan_center_contour_area`),
excluding the short stem area.

```bash
# Demo
segment-yellow-heart
# Full dataset
segment-yellow-heart \
  --input ./data/annotations_head_region \
  --output ./data/Auto_segment_Yellow_heart/True_Yellow_heart/Visualisation \
  --json-output ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json
```

* **Input**: `./samples_images/results/annotations_head_region/` (Demo) | `./data/annotations_head_region/` (Full Dataset)
* **Output**: `./samples_images/results/Auto_segment_Yellow_heart/` (Demo) | `./data/Auto_segment_Yellow_heart/` (Full Dataset)

## 5. Chinese cabbage and yellow-heart region segmentation

Automated Segmentation of Head Region and Yellow-Heart Trait.
Figure: Demonstration of annotated head region masks and automated extraction of
internal yellow-heart tissues.

<!-- 2. Below: 3 GIFs at uniform height -->
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

## 6. Yellow-Heart Phenotypic Trait Extraction

`extract_color_features.py` automatically extracts quantitative
phenotypic traits (yellow-heart area ratio and 10 color-space features) from
segmented Chinese cabbage head regions.

```bash
# Demo
extract-color-features
# Full dataset
extract-color-features \
  --input ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/ \
  --output ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx
```

* **Input**: `./samples_images/results/Auto_segment_Yellow_heart/json/` (Demo) | `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/` (Full Dataset)
* **Output**: `./samples_images/results/roi_10_color_features_with_ratio.xlsx` (Demo) | `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx` (Full Dataset)

## 7. Phenotypic Data Min-Max Normalization

`normalize_minmax.py` performs Min-Max normalization on the extracted raw phenotypic
trait dataset to scale all feature values (e.g., area ratio and color
parameters) into the range of [0, 1], eliminating scale differences for
downstream analysis.

```bash
# Demo
minmax-normalize
# Full dataset
minmax-normalize \
  --input ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx \
  --output ./data/normalized_data_1319.xlsx
```

* **Input**: `./samples_images/results/roi_10_color_features_with_ratio.xlsx` (Demo) | `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx` (Full Dataset)
* **Output**: `./samples_images/results/normalized_data_1319.xlsx` (Demo) | `./data/normalized_data_1319.xlsx` (Full Dataset)

## 8. Fisher Discriminant Score Calculation

`fisher_scores.py` computes the Fisher discriminant score for each extracted
phenotypic feature, evaluating its power to differentiate between distinct
yellow-heart phenotype categories.

```bash
# Demo
fisher-scores
# Full dataset
fisher-scores \
  --features ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx \
  --labeled-root ./data/Fisher_score_data/ \
  --output ./data/fisher_weights_418.xlsx
```

* **Input**: `./samples_images/results/roi_10_color_features_with_ratio.xlsx` (Demo) | `./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx` (Full Dataset), plus `./data/Fisher_score_data/` phenotype-label folders
* **Output**: `./samples_images/results/fisher_weights_418.xlsx` (Demo) | `./data/fisher_weights_418.xlsx` (Full Dataset)

## 9. GMM-Based Color Trait Modeling and Distribution Visualization

`gmm_grading.py` applies Gaussian Mixture Models (GMM) to model the CYS color
distribution within heading Chinese cabbage yellow-heart regions and generates
corresponding probability density distribution plots.

```bash
# Demo
gmm-cys
# Full dataset
gmm-cys \
  --data ./data/CYS_1319.xlsx \
  --output ./data/Plot

# Force a specific number of GMM components (default: BIC-optimal)
gmm-cys --force-components 3
```

* **Input**: `./samples_images/results/CYS_1319.xlsx` (Demo) | `./data/CYS_1319.xlsx` (Full Dataset)
* **Output**: `./samples_images/results/Plot/` (Demo) | `./data/Plot/` (Full Dataset)

## 10. Broad-Sense Heritability Calculation

`heritability.py` fits a random-intercept Linear Mixed Model per trait and reports
broad-sense heritability $H^2 = \sigma^2_G / (\sigma^2_G + \sigma^2_E/n_h)$,
where $n_h$ is the harmonic mean of replicate counts per genotype.

```bash
# Demo
broad-heritability
# Full dataset
broad-heritability \
  --data ./data/CYS_1319.xlsx \
  --output ./data/Plot/
```

* **Input**: `./samples_images/results/CYS_1319.xlsx` (Demo) | `./data/CYS_1319.xlsx` (Full Dataset)
## 11. Genome-Wide Association Study (GWAS) Interface

`gwas.py` interfaces with the external [PLINK](https://www.cog-genomics.org/plink/) binary to perform association analysis mapping the quantitative CYS trait against genotype data (binary PED format: `.bed`, `.bim`, `.fam`).

```bash
gwas \
  --geno-prefix ./data/genotypes \
  --pheno ./samples_images/results/CYS_1319.xlsx \
  --trait-col CYS \
  --output ./data/gwas_results.csv \
  --alpha 0.05
```

* **Input**: Genotype prefix (`.bed`/`.bim`/`.fam`), phenotype table (`.csv` or `.xlsx` containing `sample_id` and trait column)
* **Output**: Filtered association results CSV (`SNP`, `CHR`, `BP`, `P`, etc.)

## 12. Manhattan Plot Visualization

`manhattan_plot.py` creates publication-quality Manhattan plots from GWAS association results, displaying significance ($5\times 10^{-8}$) and suggestive ($1\times 10^{-5}$) threshold lines, alternating chromosome colors, and annotating top candidate SNPs.

```bash
manhattan-plot \
  --input ./data/gwas_results.csv \
  --output ./data/Plot/manhattan_plot.png \
  --title "Chinese Cabbage Yellow-Heart (CYS) GWAS"
```

* **Input**: GWAS results CSV (validated via `GwasColumns` contract: `SNP`, `CHR`, `BP`, `P`)
* **Output**: Manhattan plot image (`.png` / `.pdf`)

## 13. Candidate Gene Annotation

`annotate_genes.py` maps significant GWAS peak SNPs against a reference gene annotation file (GTF/GFF or tab-delimited format: `chr`, `start`, `end`, `gene`) within a configurable flanking window (default: $\pm 500\,\text{kb}$).

```bash
annotate-genes \
  --gwas ./data/gwas_results.csv \
  --annotation ./data/genes_annotation.gtf \
  --sig-threshold 5e-8 \
  --window-bp 500000 \
  --output ./data/annotated_candidate_genes.csv
```

* **Input**: GWAS results CSV and genomic annotation file
* **Output**: Annotated CSV reporting nearest candidate genes, distance to gene boundaries, and all genes located within the flanking window

## Reproducibility

* The number of GMM components defaults to the BIC-optimal choice and is
  controlled by `--random-state` (default `42`) for reproducibility.
* The Demo dataset under `./samples_images/results/` lets the downstream steps
  (4, 6, 7, 9, 10) run end-to-end out of the box. Step 8 (Fisher scores)
  requires phenotype-label subfolders under `./samples_images/Fisher_score_data/`
  (or `--labeled-root`); on the Demo bundle without these folders the script
  exits cleanly with a diagnostic message.

## Citation

If you use this pipeline or data in your research, please cite this repository
and the Figshare dataset.
