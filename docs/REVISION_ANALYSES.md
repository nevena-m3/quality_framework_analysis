# Revision analyses (JBHI submission)

`notebooks/08_revision_analyses.ipynb` adds four light analyses requested in review of the
manuscript. It reuses Notebook 02's frozen inputs, transformations, and model routes, and Notebook
03's pairwise correlation table. Nothing upstream is recomputed or changed.

| Section | Question | Main output |
|---|---|---|
| R1 | Do ALS–control differences persist when ALS recordings are restricted to normal bulbar function (ALSFRS-R bulbar 11–12)? | `R1_normal_bulbar_vs_controls.csv` |
| R2 | Does measure availability differ by cohort and bulbar severity group? | `R2_availability_by_group.csv`, `R2_availability_models.csv` |
| R3 | How are cohort/bulbar effects and ICC persistence distributed across separability classes A/B/C, against the base rate? | `R3_effects_by_class.csv`, `R3_base_rate_test.csv`, `R3_persistence_icc_by_measure.csv` |
| R4 | Is the within- vs between-family contrast robust to excluding shared-support pairs and to leaving out each family? | `R4_family_contrast_sensitivity.csv` |

All outputs are aggregate tables (no participant-level rows). The file
`outputs/revision_analyses/revision_summary.md` collects the numbers needed for the manuscript.

## How to run (Windows PowerShell)

1. Update the repository and switch to the revision branch:

   ```powershell
   cd C:\path\to\analysis_recording_quality
   git fetch origin
   git switch revision/jbhi-v17-analyses
   ```

2. Activate the existing environment (no new packages are needed):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python scripts\preflight.py
   ```

   If your inputs are not in the default locations, set the same `PAPER1_*` environment
   variables you use for Notebooks 02 and 03 (see the README) in this PowerShell window first.

3. Make sure Notebooks 02 and 03 have completed final runs (`FINAL_RUN = True`), so that these
   folders exist and contain final tables:

   ```text
   outputs\extent_distribution_persistence\tables\
   outputs\multidimensional_structure\tables\
   ```

   If they were run before, there is no need to run them again.

4. Run the revision notebook, either in JupyterLab (`.\scripts\setup\open_jupyter_lab.ps1`, then
   open `notebooks/08_revision_analyses.ipynb` and choose *Run → Run All Cells*), or headless:

   ```powershell
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 `
     --output 08_revision_analyses_executed.ipynb notebooks\08_revision_analyses.ipynb
   ```

   Expected run time is a few minutes; R4's 10,000-permutation loops take most of it.

5. Check the outputs:
   * `outputs\revision_analyses\revision_summary.md` exists and its last section (R4) is filled in;
   * the R3 count check prints `matches = True` for cohort (6), between-person bulbar (4), and
     within-person bulbar (2). If not, the notebook warns and the summary shows this run's values;
   * R4 printed "Original contrast reproduces Notebook 03 at both levels."

6. Share `outputs\revision_analyses\revision_summary.md` and the `outputs\revision_analyses\tables\`
   folder (aggregate only) for manuscript writing.

## Notes on the analyses

* **R1** uses the full-cohort IQR scaling from Notebook 02, so subset and full-cohort estimates are
  on the same scale. The primary subset is recording-level (bulbar 11–12 at that recording);
  the sensitivity subset uses participants whose mean matched bulbar score is ≥ 11.
* **R2** carries over Notebook 02's clustered availability models and adds odds ratios and a
  descriptive table by bulbar group.
* **R3** reads Notebook 02's primary cohort and bulbar models for the 14 main outputs and uses its
  q-values. Class assignments are listed in the notebook and in `R0_feature_roles_and_classes.csv`.
* **R4** reproduces Notebook 03's observed contrast exactly (the notebook stops if it does not),
  then reruns the label permutation for each variant.
