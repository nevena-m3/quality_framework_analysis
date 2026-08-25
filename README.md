# Paper 1 Final Analysis

This repository is the clean analysis layer for the frozen Paper 1 recording-quality feature release. It reads the governed artifacts in the adjacent `Paper_1_Final_Cleanup/outputs/` folder and never recalculates or edits upstream quality measures.

## Input contract

By default, the repositories are adjacent:

```text
Paper_1/
  Paper_1_Final_Cleanup/
    outputs/data/frozen_bamboo_recordings.csv
    outputs/segments/frozen_segmentation_decisions.csv
    outputs/features/recording_features.csv
    outputs/features/feature_registry.csv
  Paper_1_Final_Analysis/
```

The feature release must contain 519 unique recordings and 20 registered analysis features. Human-QC annotations default to `Nevena_project/Data_13072026/Bamboo_passage_HumanQC/`.

Override the two roots without editing code:

```powershell
$env:PAPER1_FINAL_CLEANUP_ROOT = "D:\path\to\Paper_1_Final_Cleanup"
$env:PAPER1_HUMAN_QC_ROOT = "D:\path\to\Bamboo_passage_HumanQC"
```

No participant-level inputs or generated outputs are committed to GitHub.

## What is analyzed

The pipeline produces:

1. Feature definitions, availability, zero prevalence, and distribution summaries.
2. First-versus-second recording persistence by participant. This is explicitly not described as technical test-retest reliability.
3. Complete recording-level Spearman dependence among all 20 retained features.
4. Standardized, median-imputed PCA with loadings and cumulative explained variance.
5. Exploratory participant-weighted ALS-versus-control contrasts using Mann–Whitney tests, rank-biserial effects, and Benjamini–Hochberg correction. These are acquisition/confounding analyses, not diagnostic prediction.
6. Correspondence between all signal features and seven independent human-QC dimensions, retaining construct-matched and off-target comparisons with multiplicity correction.

## Setup and run

Python 3.11 or 3.12 is supported.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks,dev]"
python scripts\preflight.py
python -m pytest
python scripts\run_final_analysis.py
```

Or open the single canonical notebook:

```powershell
.\scripts\setup\open_jupyter_lab.ps1
```

## Outputs

Generated results are local under `outputs/`:

```text
outputs/
  run_manifest.json
  tables/
    feature_summary.csv
    participant_persistence.csv
    feature_spearman_matrix.csv
    pca_loadings.csv
    pca_explained_variance.csv
    diagnosis_contrasts_exploratory.csv
    human_correspondence.csv
  figures/
    01_feature_availability.png/.svg
    02_feature_correlation_heatmap.png/.svg
    03_pca_explained_variance.png/.svg
    04_participant_persistence.png/.svg
    05_human_correspondence_heatmap.png/.svg
  logs/human_qc_parse_failures.csv
```

Every run manifest records exact input paths and SHA-256 hashes, cohort dimensions, human-QC coverage, and the complete table/figure inventory.

## Repository organization

```text
notebooks/01_run_final_analysis.ipynb  Interactive canonical workflow
scripts/preflight.py                  Input and notebook contract checks
scripts/run_final_analysis.py         Non-interactive complete execution
scripts/setup/open_jupyter_lab.ps1    Notebook launcher
src/rq_analysis/final_pipeline.py     Reusable analysis implementation
tests/test_final_pipeline.py          Core contract/statistical tests
outputs/                              Local results; ignored by Git
```
