# Analysis plan and execution order

The analysis pipeline is implemented as seven canonical notebooks. Run them in numeric order after completing the repository preflight. Detailed methods, inputs, outputs, and interpretation guidance are maintained in [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md); input schemas and path overrides are documented in [DATA_CONTRACT.md](DATA_CONTRACT.md).

1. `01_validation_synthesis.ipynb` — build the governed input registries and validation synthesis.
2. `02_extent_distribution_persistence.ipynb` — quantify feature availability, distributions, zero mass, and repeated-recording persistence.
3. `03_multidimensional_structure.ipynb` — assess within- and cross-family dependence, redundancy, and latent structure.
4. `04_human_annotation_audit.ipynb` — audit human-QC annotation coverage, labels, and missingness before correspondence analyses.
5. `05_human_reference_reliability.ipynb` — estimate agreement and reliability of the human reference annotations.
6. `06_family_human_correspondence.ipynb` — test prespecified family-to-human construct correspondence with multiplicity control.
7. `07_item_mapping_atlas.ipynb` — provide the item-level mapping atlas used to interpret family-level results.

Generated results belong under `outputs/` and are intentionally excluded from Git because they are reproducible and may contain participant-level information. The repository retains only the pipeline, documentation, configuration, and empty output structure.
