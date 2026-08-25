# Detailed analysis guide

This document explains what each notebook does, how it is run, what it writes, and
which claims it supports. Notebook code remains the executable specification; the
frozen plan documents in `docs/plans/` record the prespecified decisions.

## 1. Analytical validation synthesis

`notebooks/01_validation_synthesis.ipynb`

This stage scans the six governed family freezes using `config/families.yml` and
standardizes otherwise heterogeneous family artifacts into registries for feature
decisions, validation gates, dashboards, checklists, figure sources, and recording
tables. File hashes and selected source paths are retained for provenance. The stage
then creates the Section 3.1 analytical-validation figure from controlled-response,
repeated-recording-persistence, and final-role evidence.

It does not recompute feature estimators, use diagnosis labels, or override a family’s
final decision. Important outputs are the registries in `data/derived/registries/` and
publication figure/source-data files in `outputs/figures/`.

## 2. Extent, distribution, persistence, cohort and bulbar structure

`notebooks/02_extent_distribution_persistence.ipynb`

This is the primary recording/participant characterization analysis. It discovers and
hashes the frozen data and family tables, harmonizes them into a common matrix, and
blocks execution if the population, diagnosis, date, or bulbar linkage contracts fail.

The notebook distinguishes continuous, bounded, binary-presence, sparse-event, and
availability outcomes. It preserves valid zeros, applies frozen transformations and
winsorization rules, and scales continuous effects using declared analysis units.
It reports recording- and participant-level distributions, availability, positive-only
extent for sparse features, and recordings per participant.

Repeated-recording persistence is participant-aware and is not described as formal
test–retest reliability. Cohort models use participant random intercepts where
supported, with participant-clustered GEE fallbacks. Binary outcomes use clustered
binomial GEE. Bulbar associations are restricted by prespecified linkage and support
gates. Sensitivities include participant-collapsed summaries, one-recording-per-
participant resampling, cohort alternatives, and longitudinal formulations.

Final settings use seed `20260805`, 2,000 ICC bootstrap replicates, and 1,000
one-recording resamples. Inspect `logs/completion_checklist.csv`, population audits,
model diagnostics, and the frozen specification before using figures or coefficients.

## 3. Multidimensional structure

`notebooks/03_multidimensional_structure.ipynb`

This stage asks whether reviewed indicators are distinct, overlapping, or redundant.
It uses the same governed population and transformation contract as Notebook 2.
Pairwise associations retain pair-specific support rather than forcing a global
complete-case matrix. Between-participant associations use participant summaries;
within-participant associations isolate repeated-recording covariation.

The analysis includes participant-aware correlation bootstrap intervals, family
contrasts, permutation tests, cohort-stratified and ALS time-adjusted sensitivities,
random-recording-pair sensitivity, and signed QCHAN sensitivity. PCA is secondary and
runs only when complete-case participant support passes its gate. Parallel analysis,
loading alignment, bootstrap stability, and feature-set sensitivity are reported.

Final settings include 2,000 correlation/family/PCA bootstrap replicates, 10,000
family permutations, and 1,000 parallel-analysis simulations. Interpret PCA only when
the support audit and stability gates pass. Principal components are descriptive
structure, not latent causal constructs.

## 4. Human annotation ingestion and audit

`notebooks/04_human_annotation_audit.ipynb`

This stage inventories all rater files, validates schemas, reconstructs recording
identities, parses JSON annotations into an interval ledger, and audits segment
geometry, duplicates, taxonomy, metadata resolution, and frozen-population coverage.
It constructs recording/item/family summaries and evaluates interval-merge tolerance
sensitivity.

The stage is deliberately audit-only: correspondence models are disabled. Its outputs
form the governed input to Notebook 5. Resolve every blocking checklist item before
continuing. Do not infer reliability from distributed main ratings because most main
recordings were not rated by all four raters.

## 5. Human-reference reliability and harmonization

`notebooks/05_human_reference_reliability.ipynb`

This stage selects the complete crossed four-rater subset and constructs governed
presence matrices at family and item levels. It reports prevalence by rater,
positive/negative agreement, chance-adjusted reliability coefficients, pairwise
agreement, and temporal overlap. Primary interval merging uses zero tolerance, with a
50-ms sensitivity analysis.

The notebook uses 2,000 bootstrap replicates in final mode with seed `20260805`.
Machine comparisons remain disabled so human-reference roles are frozen without
looking at feature correspondence. Low-prevalence and prevalence-imbalance effects
must be considered alongside chance-adjusted coefficients.

## 6. Family-level human correspondence

`notebooks/06_family_human_correspondence.ipynb`

This stage joins the governed Stage-1/Stage-1B human outputs to the 519-recording
signal matrix. The construct map is frozen before modelling. Reliability summaries
provide context but are not used to select favorable signal results.

Models are participant-aware and routed by outcome/predictor type. Execution requires
minimum recording, positive/negative, participant, rater, and predictor-variation
support. Multiplicity correction is applied to the declared family-level tests.
Participant-collapsed resampling is a sensitivity analysis, not a replacement for the
primary recording-level clustered model. The analysis quantifies correspondence; it
does not turn human labels into ground-truth causal source labels.

Final mode uses 1,000 subject-collapsed resamples and seed `20260805`. Review support
gates, convergence/diagnostic tables, linkage audits, and interpretation guardrails.

## 7. Human–machine item mapping atlas

`notebooks/07_item_mapping_atlas.ipynb`

This exploratory stage expands from family-level correspondence to the full inventory
of human items and machine indicators. It freezes indicator direction/scaling,
constructs joint and rater-specific analysis matrices, and reports raw within-family,
off-family, and task-negative-control associations. Adjusted within-family models are
run only when joint and rater-specific support gates pass.

Outputs include prevalence/support tables, human item phi/Jaccard matrices, machine
indicator Spearman structure, complete item-feature correlation atlases, adjusted
models, mapping-class summaries, and rater-stratified direction-consistency checks.
These results are exploratory mapping diagnostics, not proof that a machine feature
detects a uniquely named perceptual artifact.

## Completion standard

For every stage, completion requires all of the following:

1. governed input paths and hashes recorded;
2. expected population and unique-key audits passed;
3. model/support gates recorded rather than silently bypassed;
4. diagnostics and sensitivity outputs written;
5. source data saved for every publication figure;
6. blocking completion checklist passed;
7. run manifest and frozen analysis specification saved.
