"""Chinese cabbage yellow-heart phenotyping toolkit.

Reproducible pipeline for yellow-heart region segmentation, multidimensional
color-space trait extraction, and quantitative grading via the Comprehensive
Yellow Score (CYS):

* ``imaging`` — raw photos to trait tables: head cropping (``crop_heads``),
  yellow-heart segmentation (``segment_yellow_heart``), color-trait extraction
  (``extract_color_features``), illumination calibration (``color_calibrate``),
  acquisition provenance (``validate_metadata``), and segmentation accuracy
  (``evaluate_segmentation``);
* ``scoring`` — trait tables to CYS grades: normalization
  (``normalize_minmax``), Fisher weighting (``fisher_scores``), CYS synthesis
  (``cys_compute``), GMM grading (``gmm_grading``), plus method validation
  (``baseline_compare``, ``validate_thresholds``, ``robustness_test``);
* ``genetics`` — grades to genetics: broad-sense heritability
  (``heritability``), GWAS (``gwas``), Manhattan plots (``manhattan_plot``),
  and gene annotation (``annotate_genes``).

Shared infrastructure (``_typing``, ``constants``) lives at this level. Every
stage is exposed as a console script (see ``pyproject.toml``
``[project.scripts]``); see ``README.md`` for the end-to-end workflow.
"""

__version__ = "0.1.0"
