# Data and outputs

## Bundled data

`samples_images/results/annotations_head_region/` contains cropped RGB images
and LabelMe JSON annotations. The manuscript demo matches images to JSON by
filename stem. Only image/annotation pairs are processed. The remaining demo
directory holds example segmentation outputs. Use a fresh output directory when
rerunning the demo to avoid mixing generated and bundled files.

[`full_dataset_tables/`](../full_dataset_tables/) contains study-derived tables.
The table below gives their roles and identifying fields.

| Workbook | Contents |
| --- | --- |
| `roi_10_color_features_with_ratio.xlsx` | Image-level color descriptors, pixel counts and yellow area ratio, keyed by `filename`. |
| `normalized_data_1319.xlsx` | Archived normalized table. It retains unsuffixed descriptor headers such as `ExR_mean`. The manuscript scoring script expects the `_norm` headers written by the current extraction script. |
| `fisher_weights_418.xlsx` | Feature-level `Fisher_Score`, `Response_Direction`, and `Final_Weight` indexed by `Feature_Name`. |
| `CYS_1319.xlsx` | Image-level yellow score (`Yellow_score` or YS), yellow area ratio (`Yellow_Ratio` or YR), and composite score (`CYS`). `QR` is the accession identifier. |
| `Plot/CYS_1319_CYS_GMM_5Classes.xlsx` | Archived image-level scores with `Scientific_Grade` assignments. |

The analysis unit in these image-level workbooks is the individual head image.
Use `filename` to join image-level tables. Use `QR` where present to group images
by accession. `Yellow_pixel_count` and `Pan_pixel_count` are counts in pixels.
`Yellow_Ratio` is a fraction, and YS and CYS are unitless appearance scores.

## Study archive

The manuscript's Data and Code Availability statement gives the study-data
access route. The Figshare record identifier is `10.6084/m9.figshare.33154715`.
The public DOI did not resolve at the time this guide was prepared. If needed,
follow the access instructions in the manuscript or request access from its
corresponding author. Use the archive's terms for reuse of its data and images.

The full image and annotation sets, original GMM fit input, paired-mask tables,
and expert-rating workbooks are obtained through that route. Follow the
[manuscript command table](../README.md#manuscript-reproduction-scripts) for
the expected inputs. For the historical GMM fit, use the archived `CYS.xlsx` in
its original row order. Use `GMM_thresholds` in the original or reproduced
`CYS_GMM_Analysis_Results.xlsx` for full-precision grade assignment. See the
[reproduction notes](../README.md#reproduction-notes) for the grading and
validation output sheets.
