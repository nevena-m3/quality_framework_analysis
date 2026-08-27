# Paper 1 analysis reproducibility

## Release identity and environment

- Branch: `release/paper1-publication-freeze`
- Audited software baseline: `dcfd4e1ec75aff7c04f44a301f4e7f3767501fb2`
- Immutable release identity: tag `paper1-analysis-publication-v1.0.0`
- Python: 3.12.10; dependency lock: `requirements-publication-lock.txt`
- Core reproduced packages: NumPy 2.5.2, pandas 2.3.3, SciPy 1.18.1, statsmodels 0.14.6, matplotlib 3.11.1, scikit-learn 1.9.0, PyArrow 22.0.0

## Inputs and clean-room execution

All seven canonical notebooks were executed from cleared generated-output directories against the freshly regenerated feature release. The five retained-family Parquet inputs were byte-identical to the governed publication inputs:

- metadata: `96d68c52b5f36ef3ba1a3d0ee6ca879fce036fb449c5a689a2024377bf706e78`
- QADD: `d24e799778bb10faa68667b365a62ae94b9d89c984a1fa06d760e9c1c0af136a`
- QGAIN: `af829b293582ad5d50eb12eff7212a211b2ef4673b28011e18b8f24115dd947e`
- QREV: `449e52d238cdcfcfa5920e7507c0fb8ae6699908019cbc10d6d2955e222acc88`
- QCHAN: `6cbea32330994aba9bcfcf933e94d8e5844ded015d54d49b18bbde908f89259b`
- QDIST: `e013d82cdfed9a2c696eaf4fd4eb31c148eada24c2c2ed1ff8a5ddf448175a39`
- QTEMP exploratory, explicitly reused and non-primary: `76b428c5461b8ccb877fd99ac92a6f6600582be66466cc2c9e64cc181b772c97`

Governed HumanQC annotations and family validation evidence were reused. They were not regenerated or committed. The analysis preflight passed, all 11 tests passed, all seven notebooks completed, and `scripts/verify_publication_outputs.py` passed 22/22 golden aggregate checks.

Notebook 02 reproduced 519 recordings, 224 participants, 2,000 participant-cluster ICC bootstraps, and 1,000 one-recording-per-participant resamples. Notebook 03 reproduced the frozen multilevel correlation/PCA results. Notebooks 04–07 reproduced the annotation audit, reliability population, family-human correspondence, and independent item-atlas verification.

## Commands

Set the numerical input roots to the fresh feature handoff and the validation root to the governed validation evidence, then run:

```powershell
python scripts\preflight.py
python -m pytest -q
jupyter nbconvert --to notebook --execute notebooks\01_validation_synthesis.ipynb
jupyter nbconvert --to notebook --execute notebooks\02_extent_distribution_persistence.ipynb
jupyter nbconvert --to notebook --execute notebooks\03_multidimensional_structure.ipynb
jupyter nbconvert --to notebook --execute notebooks\04_human_annotation_audit.ipynb
jupyter nbconvert --to notebook --execute notebooks\05_human_reference_reliability.ipynb
jupyter nbconvert --to notebook --execute notebooks\06_family_human_correspondence.ipynb
jupyter nbconvert --to notebook --execute notebooks\07_item_mapping_atlas.ipynb
python scripts\verify_publication_outputs.py --outputs-root outputs
```

The required environment variables and privacy rules are documented in `DATA_CONTRACT.md`. Generated participant-level tables, HumanQC data, and raw media remain local-only.

## Scientifically justified claim

The governed raw-audio-to-reviewed-feature pipeline and reviewed-feature-to-results analysis were independently reproduced in the recorded publication environments. This claim explicitly includes reuse of governed human adjudications, HumanQC annotations, the QDIST legacy comparison baseline, validation evidence, and the non-primary QTEMP exploratory table; it does not claim those human/governance inputs were independently regenerated.
