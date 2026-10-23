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

## 4. Automated Yellow-Heart Region Extraction

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

## 7. Phenotypic Data Normalization

`normalize_minmax.py` scales the ten extracted colour traits with Min-Max
(default) or Z-score standardization. Min-Max maps each feature to [0, 1];
Z-score centres on the mean and scales to unit SD, matching the GWAS-ready
tables from the revision scripts.

```bash
# Demo
minmax-normalize
# Full dataset
minmax-normalize \
  --input ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx \
  --output ./data/normalized_data_1319.xlsx

# Z-score variant
minmax-normalize --method zscore \
  --input ./data/roi_10_color_features_with_ratio.xlsx \
  --output ./data/color_features_zscore.xlsx
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

# Cross-check with the one-way ANOVA moment estimator
broad-heritability --data ./data/CYS_1319.xlsx --output ./data/Plot_ANOVA --method anova
```

* **Input**: `./samples_images/results/CYS_1319.xlsx` (Demo) | `./data/CYS_1319.xlsx` (Full Dataset)
* **Estimator**: REML LMM by default; `--method anova` uses the unbalanced-design one-way ANOVA estimator (`r_eff` replaces `n_h`) from the revision scripts. Both write `variance_components.csv` and `h2_model_spec.txt`.
## 11. Genome-Wide Association Study (GWAS) Interface

`gwas.py` supports two execution engines to map the quantitative CYS trait against genotype data:
1. **Native Python Engine** (zero external dependencies): uses `scikit-allel` and `numpy` for chunked streaming regression directly on `.vcf` or `.vcf.gz` files. Supports interactive progress bars (`tqdm`), automatic sample matching, incremental checkpointing (`--checkpoint-interval`), and resuming from interruption (`--resume`).
2. **PLINK Engine**: interfaces with an external [PLINK](https://www.cog-genomics.org/plink/) binary for binary PED format (`.bed`, `.bim`, `.fam`).

```bash
# 1. Native Python GWAS (VCF input with progress & checkpoint saving)
gwas \
  --method native \
  --vcf ./GWASData/SNPs/AA_AABB.snp.clean.rapa.maf0.05.recode.vcf.gz \
  --pheno ./full_dataset_tables/CYS_1319.xlsx \
  --trait-col CYS \
  --sample-col QR \
  --output ./GWASData/gwas_results.csv \
  --checkpoint-interval 5000 \
  --alpha 0.05

# Resume interrupted run
gwas --method native --vcf ... --pheno ... --output ./GWASData/gwas_results.csv --resume

# 2. PLINK Binary GWAS (BED/BIM/FAM input)
gwas \
  --method plink \
  --geno-prefix ./data/genotypes \
  --pheno ./samples_images/results/CYS_1319.xlsx \
  --trait-col CYS \
  --output ./data/gwas_results.csv \
  --alpha 0.05
```

* **Input**: VCF file (`.vcf`/`.vcf.gz`) or PLINK binary genotype prefix (`.bed`/`.bim`/`.fam`), phenotype table (`.csv` or `.xlsx` containing `sample_id` or `QR` and trait column)
* **Output**: Filtered association results CSV (`CHR`, `SNP`, `BP`, `A1`, `BETA`, `SE`, `P`)

## 12. Manhattan Plot Visualization

`manhattan_plot.py` creates publication-quality Manhattan plots from GWAS association results, displaying significance ($5\times 10^{-8}$) and suggestive ($1\times 10^{-5}$) threshold lines, qualitative chromosome color palettes, and annotating top candidate SNPs.

Supports dual rendering backends unified under HoloViews:
- **Static Publication Mode (`HoloViews` + `matplotlib`)**: outputs high-DPI `.png` or vector `.pdf` figures with automated peak labels and publication layout.
- **Interactive Web Mode (`HoloViews` + `bokeh`)**: outputs interactive `.html` files supporting pan, zoom, and mouse hover tooltips (`SNP`, `CHR`, `BP`, `P`) for all variants with dense, continuous background sampling.

```bash
# 1. Publication static figure with qualitative palette (default: Set2; also supports Dark2, tab10, Paired, GAPIT)
manhattan-plot \
  --input ./data/gwas_results.csv \
  --output ./data/Plot/manhattan_plot.png \
  --palette Set2 \
  --title "Chinese Cabbage Yellow-Heart (CYS) GWAS"

# 2. Interactive HTML visualization via HoloViews + Bokeh (inferred from .html or --backend bokeh)
manhattan-plot \
  --input ./data/gwas_results.csv \
  --output ./data/Plot/manhattan_plot.html \
  --palette Set2 \
  --title "Interactive CYS GWAS Manhattan Plot"
```

* **Input**: GWAS results CSV (validated via `GwasColumns` contract: `SNP`, `CHR`, `BP`, `P`)
* **Output**: Manhattan plot image (`.png` / `.pdf`) or interactive HTML (`.html`)

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

## 14. CYS Formulation Comparison

`cys_model_compare.py` rebuilds four candidate CYS formulations from
`Yellow_Ratio` and `Yellow_score`, fits the same variance-component model to
each one, and exports an Excel report. It answers whether the multiplicative
form over-penalizes small but deeply yellow heads, and whether the choice of
formula changes the accession ranking.

| Model | Formula |
| --- | --- |
| `multiply` | `YS × YR` (paper default) |
| `additive` | `0.5 × YS + 0.5 × YR` |
| `geometric` | `√(YS × YR)` |
| `harmonic` | `2 × YS × YR / (YS + YR)` |

```bash
# Full dataset (defaults)
cys-model-compare
# Explicit paths and a plant-level sensitivity check
cys-model-compare \
  --data ./data/CYS_1319.xlsx \
  --output ./data/CYS_models_comparison.xlsx \
  --level aggregate --replicate-col plant_id
```

* **Input**: table with `Yellow_Ratio`, `Yellow_score`, and the genotype column (`QR` by default)
* **Output**: `CYS_models_comparison.xlsx` with `Model_Heritability_H2`, `Spearman_Correlation`, and `Per_Sample_CYS_Values` sheets

## 15. Segmentation Accuracy and Grade Benchmark

`evaluate_segmentation.py` compares predicted `Yellow_area` masks against
independent manual ground truth, reporting IoU, Dice, mask discrepancies, and
failure rates. Three grade modes reproduce the per-phenotype-tier benchmark
table (Grade 2–5 plus `ALL`):

| Mode | Layout | `grade` column |
| --- | --- | --- |
| `--gt-subdirs` | graded GT tree (`gt/Grade_2/*.json`), flat predictions | GT folder |
| `--grade-from pred` | graded prediction tree (`pred/Grade_2/*.json`), flat GT — the revision benchmark layout | prediction folder |
| no flag / `--grade-from none` | no grade information | empty |

```bash
# Standard evaluation with difficulty strata
evaluate-segmentation --pred-dir ./data/Auto_segment \
  --gt-dir ./annotations/manual_gt --strata ./annotations/test_strata.csv \
  --output-csv ./results/evaluation.csv \
  --output-summary ./results/evaluation_summary.csv

# Revision layout: predictions already sit in grade folders, ground truth is flat
evaluate-segmentation --pred-dir ./data/ByGrade_5_100 \
  --gt-dir ./annotations/manual_gt --grade-from pred \
  --grades Grade_2,Grade_3,Grade_4,Grade_5 \
  --output-csv ./results/evaluation.csv \
  --output-summary ./results/evaluation_summary.csv \
  --output-grade-summary ./results/evaluation_by_grade.csv \
  --excel ./results/segmentation_metrics_summary.xlsx

# Second annotator: inter-annotator IoU/Dice, the measurement ceiling
evaluate-segmentation --pred-dir ./data/ByGrade_5_100 \
  --gt-dir ./annotations/annotator_1 --gt2-dir ./annotations/annotator_2 \
  --output-summary ./results/summary.csv \
  --output-agreement ./results/agreement.csv
```

* **`--grades`**: keeps only the named folders (`Grade_2,Grade_3,...`); a requested grade with no samples raises instead of silently producing a short row.
* **`--gt2-dir` / `--gt2-label`**: compares two independent annotation sets on their shared stems and writes `<output-agreement>.csv` (summary with `mean ± SD` and `share_iou_ge_0.5`) plus `<output-agreement>_samples.csv`.
* **Output**: per-image CSV, stratum summary, per-grade summary (`grade, grade_label, n, mean_iou, sd_iou, mean_dice, sd_dice, iou_formatted, dice_formatted, failure_rate`), and an optional Excel workbook with `Table_Summary`, `Per_Sample_Details`, and — when `--gt2-dir` is given — `Annotator_Agreement` / `Annotator_Agreement_Samples`.
* **SD convention**: population SD (`ddof=0`), matching the revision script's `np.std` output.
* **Grade labels**: `Grade_2...Grade_5` map to the paper names (Light Yellow / Yellow / Deep Yellow / Extreme Yellow); any other folder keeps its raw name, so an unexpected grade is never mislabelled.

## 16. Annotation Overlay Visualization

`visualize_annotations.py` renders every polygon in an image + JSON pair as a
semi-transparent fill with a closed outline and label tag. It is useful for
reviewing head and yellow-heart masks before extraction.

```bash
# Demo
visualize-annotations
# Full dataset, selected labels only
visualize-annotations --input ./data/Auto_segment_Yellow_heart/json \
  --output ./data/Annotation_visualization \
  --labels Yellow_area,Short_stem --alpha 0.35
```

* **Input**: folder of images with same-stem JSON sidecars (add `--recursive` to walk grade subfolders)
* **Output**: `<stem>_vis.<ext>` overlays; files without JSON sidecars are reported and skipped

## 17. Stratified Image Subset Sampling

`sample_images.py` copies up to N images from each class/grade subfolder into
a flat folder with a group prefix and writes a manifest. With a fixed seed the
same subset is reproduced, which is how the 30-per-grade illumination
robustness set is assembled.

```bash
sample-images --input ./data/Graded_Images \
  --output ./data/Robustness_Graded_Images_Dataset --per-folder 30 --seed 42
```

* **Input**: root folder of class/grade subfolders (optionally restricted with `--groups Grade_2,Grade_3`)
* **Output**: copied images plus `sampling_manifest.csv` (`group, filename, source, output`)

## 18. Nested-Ellipse Spatially Weighted Traits

`ellipse_weighted_traits.py` fits one ellipse to the head contour plus two
scaled copies (1.0, 0.75, 0.5) anchored at the bottom, top, or centre. For each
ellipse it measures the overlap with the yellow region and the ten colour
traits inside that overlap, then combines them across ellipses.

| Column group | Meaning |
| --- | --- |
| `Ellipse_i_scale`, `Ellipse_i_ratio` | Scale factor and yellow inclusion ratio of ellipse *i* |
| `Ellipse_i_<trait>` | Colour traits measured inside ellipse *i* ∩ yellow region |
| `Weighted_<trait>` | `Σ ratio_i × trait_i` (raw spatial sum) |
| `WeightedNorm_<trait>` | `Σ ratio_i × trait_i / Σ ratio_i` (ratio-weighted mean) |
| `Total_yellow_ratio` | `Σ ratio_i`; overlapping ellipses can push this above 1 |

```bash
# Demo
ellipse-weighted-traits
# Full dataset
ellipse-weighted-traits --input ./data/Auto_segment_Yellow_heart/json \
  --output ./data/ellipse_weighted_features.xlsx \
  --alignment bottom \
  --visualization-dir ./data/ellipse_visualization
```

* **Input**: images with same-stem JSON containing `Pan_center_contour_area` (and `Yellow_area` or a referenced yellow mask)
* **Output**: trait table, `<output>.status.csv`, and per-image ellipse overlays

## 19. Fisher-Weighted Otsu Segmentation

`segment_fisher_score.py` is an alternative to the fixed ExR threshold: it
builds per-pixel maps for the ten canonical traits, normalizes each inside the
annotated head, combines them with Fisher-score weights (negative-direction
traits use `1 - x`), and applies Otsu within the head. The largest component is
kept and morphologically cleaned.

```bash
fisher-scores   # produces the weights table used below
fisher-segment --input ./data/annotations_head_region \
  --output ./data/Fisher_score_segmentation \
  --fisher ./data/fisher_weights_418.xlsx
```

* **Input**: cropped head images with `Pan_center_contour_area` JSON and a Fisher weight table (all ten features required)
* **Output**: original image, `masks/*_yellow_mask.png`, `masks/*_head_mask.png`, annotation JSON with both mask paths, and `segmentation_status.csv`
* **Note**: outputs feed directly into `extract-color-features`; validate the masks against independent annotations before using the traits in a paper table

## 20. PCA/KMeans Color Clustering

`pca_cluster.py` standardizes the ten colour traits, projects samples onto
PC1/PC2, clusters them with KMeans, and writes the explained-variance scatter
plot plus a table with `PC1`, `PC2`, and `Cluster`.

```bash
pca-cluster --input ./data/roi_10_color_features_with_ratio.xlsx \
  --output-xlsx ./data/color_pca_kmeans.xlsx \
  --output-plot ./data/color_pca_kmeans.png --clusters 3
```

* **Input**: table with the ten canonical colour features
* **Output**: Excel table and PNG/SVG scatter

## Reproducibility

### Quality gates

The same three checks run locally and in CI (`.github/workflows/ci.yml`):

```bash
uv run ruff check src tests   # lint, line length 100
uv run pyrefly check          # type check, 0 errors expected
uv run pytest -q              # full suite under tests/
```

CI installs the project with `uv sync --extra dev` on Python 3.13 and adds the
OpenCV system libraries (`libgl1`, `libglib2.0-0`) needed by the Ubuntu runner.

### Revision validation workflow

The revised segmentation retains the **whole yellow-heart envelope** as the
primary region and explicitly excludes `Short_stem`. The color rule is
**ExR only** (`ExR > 0.15`), retaining the core-localizing behavior of the
historical sample results. This is an engineering baseline, not an established
biological definition of yellow heart. Historical CYS tables and GMM thresholds
must not be treated as validated for newly generated masks. The filled-envelope
definition, treatment of veins and gaps, and ambiguous boundaries need agreement
before independent annotation and accuracy evaluation.

Always use a fresh output directory. Batch segmentation refuses directories
containing previous annotation outputs or a status report, preventing stale
predictions from being mixed into a new run. For example:

```bash
segment-yellow-heart --input samples_images/results/annotations_head_region \
  --output results/revision_v21/visualization \
  --json-output results/revision_v21/annotations --remove-gaps

extract-color-features --input results/revision_v21/annotations \
  --output results/revision_v21/features_envelope.xlsx

extract-color-features --input results/revision_v21/annotations \
  --region tissue --output results/revision_v21/features_tissue.xlsx
```

Each annotation preserves head and stem polygons, references lossless PNG
masks, and records algorithm version, thresholds, region definition, OpenCV
version and processing status. The masks are:

| Mask | Definition |
| --- | --- |
| `valid_head` | Annotated head minus short stem; the area-ratio denominator |
| `envelope` | Largest external yellow contour filled and clipped to valid head |
| `pixels` | Components selected by the chosen color rule, opened and clipped to valid head |
| `tissue` | Optional envelope minus dark-gap candidates; an exploratory comparison |

`--remove-gaps` uses `L* < 50` and local black-hat deficit `> 15`, with an
elliptical kernel radius of 8 pixels at image long side 1024, scaled with image
size (minimum radius 2). These candidates can include tissue shadows and miss
bright gaps. White veins are not excluded based on color. Gap removal is not
enabled by default and requires manual validation before adoption.

Feature extraction and segmentation evaluation read the referenced raster
mask rather than refilling polygons, so exclusions and holes survive export.
The illumination robustness script also resegments each perturbed image before
extracting features and compares whole envelopes. It no longer measures color
inside a reused baseline yellow mask. This remains a simulation of segmentation
and feature stability, not an external accuracy test or a complete CYS/grade test.
Original images are copied alongside output annotations; visualization images
are kept separately. Legacy polygon-only files remain readable. Segmentation
writes `segmentation_status.csv` for every input image, including missing
annotations and processing failures. Extraction writes a companion
`*.status.csv`. An empty detected yellow region has area zero and undefined
color features (NaN); it is **not** automatically classified as biologically
white. CYS preserves missing measurements rather than replacing them with zero.

For independent evaluation, manually annotate the same biological definition
(whole yellow-heart envelope including internal veins, excluding short stem).
Use a separate ground-truth directory covering the complete evaluation sample
set, including images with absent predictions:

```bash
evaluate-segmentation --pred-dir results/revision_v21/annotations \
  --gt-dir annotations/manual_test --strata annotations/test_strata.csv \
  --output-csv results/revision_v21/evaluation.csv \
  --output-summary results/revision_v21/evaluation_summary.csv
```

Prediction/GT self-comparison and canvas-size mismatches are rejected. Missing
prediction files count as failures even for empty GT. Predictions without GT
are rejected rather than silently discarded. Difficulty strata must be assigned
independently in a `filename,stratum` CSV (`simple`, `typical`, `hard`, `blurry`);
otherwise rows are `unassigned`. Cause labels describe mask discrepancies and
require human review; small regions do not establish biological causes. Reports
include signed pixel-area error and relative area error (undefined for empty GT).

Baseline comparison now requires accession IDs and uses the same
`StratifiedGroupKFold` splits for LDA, logistic regression, RF, RBF-SVM and
linear SVM. Scaling is fitted on training folds only; each model's top-4
discriminative features (mean absolute coefficient or impurity importance)
are written in the `top_features` column. The source CYS weighting method is
a univariate Fisher-score weighting, not this LDA classifier.

```bash
baseline-compare --features results/revision_v21/features_envelope.xlsx \
  --labeled-root annotations/color_classes \
  --sample-sheet annotations/sample_sheet.csv --group-col QR \
  --output results/revision_v21/baseline_metrics.csv \
  --confusion results/revision_v21/baseline_confusion.csv
```

Recover the image, accession and plant hierarchy from experimental records.
Record `filename,QR,plant_id,cut_id,batch_id,harvest_time,imaging_time,stratum,review_cause`
in the sample sheet. Fill in real
IDs; do not infer biological replicates from filename numbering. The baseline
join uses `filename` and `QR`; the other fields document the experimental design.
With confirmed biological plant IDs, use `broad-heritability --level both
--replicate-col plant_id` on a table retaining `QR` and `plant_id` to compare
image-level and plant-aggregated estimates. CYS construction preserves these
IDs when they are supplied in its input. No new heritability or external validity
claim follows from these code changes alone.

* The number of GMM components defaults to the BIC-optimal choice and is
  controlled by `--random-state` (default `42`) for reproducibility.
* The Demo dataset under `./samples_images/results/` lets the downstream steps
  (4, 6, 7, 9, 10) run end-to-end out of the box. Step 8 (Fisher scores)
  requires phenotype-label subfolders under `./samples_images/Fisher_score_data/`
  (or `--labeled-root`); on the Demo bundle without these folders the script
  exits cleanly with a diagnostic message.

## Script migration map

The standalone scripts under `E:\Phenotype_analysis` were consolidated into
the package as follows.

| Source script | Integrated as | Notes |
| --- | --- | --- |
| `1_revise_Iou_Dice_Yellow_heart.py` | `evaluate-segmentation --gt-subdirs` | Per-grade mean ± SD IoU/Dice, overall row, optional Excel table |
| `2_CYS.py` | `cys-model-compare` | Four formulations including harmonic mean; LMM or ANOVA H²; Spearman matrix |
| `20260617_CYS_fenbu.py` | `visualize-annotations` | Semi-transparent polygon overlays with label tags |
| `5_FDA_SVM.py` | `baseline-compare` | Linear SVM and top-4 feature reporting added; accession-grouped CV replaces plain stratified CV |
| `6_White_balance.py` | `color-calibrate` / `robustness-test` | Source file is empty; gray-world white balance already exists in `color_calibrate` |
| `Fisher_scores.py` | `fisher-scores` | Already present; folder labels use the canonical names |
| `H2.py` | `broad-heritability --method anova` | One-way ANOVA estimator; REML LMM remains the default |
| `test.py` | `sample-images` | Deterministic stratified subset with a manifest |
| `1A_YEQIU_Github/*` | existing commands | Older versions of segmentation, extraction, normalization, Fisher, GMM, and H² |
| `Cabbage_color/cabbage_color_10_HSV_Lab.py` | `extract-color-features` | Duplicate colour extraction |
| `Chinese_cabbage_Github_scripts/yellow_heart_segmentation.py` | `segment-yellow-heart` | Earlier ExR segmentation |
| `Chinese_cabbage_Github_scripts/pca.py` | `pca-cluster` | PCA + KMeans on the ten colour traits |
| `Yeqiu_tuoyuannihe/tuoyuan_*.py`, `Yellow_area_pan_area_ratio*.py`, `Pan_area_Yellow_area_ratio_fenbie_10_color.py` | `ellipse-weighted-traits` | Nested-ellipse spatial weighting, per-ellipse and normalized columns |
| `Yeqiu_tuoyuannihe/yeqiu_yellow_10_json.py` | `fisher-segment` | Fisher-weighted score + Otsu, lossless mask output |
| `Yeqiu_tuoyuannihe/Z-score.py` | `minmax-normalize --method zscore` | Z-score standardization |
| `Yeqiu_tuoyuannihe/Yellow_area_RGB_HSV_Lab.py`, `Yellow_area_Gray.py` | `extract-color-features` | Duplicate extraction; gray statistics are outside the canonical ten traits |
| `1_A_leaf_label_model/*`, `2_A_leaf_edge_qifu/*` | not integrated | Leaf area/petiole and leaf-tip ellipse analyses are outside the yellow-heart scope |
| `yeqiu_rgb_hsv_lab_holoviews/hv_hist.py` | not integrated | Interactive Panel/HoloViews notebook view |
| `manuscript.original.md`, `manuscript.revised.md` | not code | Manuscript text is kept outside the package |

Legacy phenotype folders `yellow_low / yellow_medium / yellow_many` map to the
canonical `Light_yellow / yellow / Deep_yellow`. Legacy ground-truth JSON files
labelled `yellow_area` can be read with `--gt-label yellow_area`.

## Citation

If you use this pipeline or data in your research, please cite this repository
and the Figshare dataset.
