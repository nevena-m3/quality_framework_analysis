# Data contract

## Governed sources

The analysis consumes the immutable 519-recording data freeze and the latest approved
family freezes from the feature pipeline. The recording key is
`logical_recording_id`; participant clustering uses the governed participant/subject
identifier. All recording-level joins must be one-to-one unless an analysis explicitly
constructs a participant-level table.

The six family folders are:

- `additive_interference` (QADD)
- `gain_dynamics` (QGAIN)
- `reverberation` (QREV)
- `channel_device` (QCHAN)
- `nonlinear_distortion` (QDIST)
- `temporal_discontinuity` (QTEMP)

QTEMP has no retained primary feature. Its outputs can appear only where the frozen
analysis specification explicitly permits monitoring, exploratory, or availability
descriptions.

## Human annotations

The annotation design contains distributed main ratings and a crossed reliability
subset rated by four raters. These designs are never pooled as if every recording had
four ratings. Stage 1 reconstructs interval identities and geometry; Stage 1B defines
the complete reliability population before machine features are inspected.

## Missingness and support

Unavailable measurements remain missing. Valid measured zeros remain zero. The
analysis does not impute values merely to obtain a complete matrix. Model-specific
support, class-count, participant-count, and variation gates are written to output
tables and can block interpretation.

## Privacy and Git

Input freezes, raw annotations, participant-level derived tables, and generated output
folders are local-only. Git contains code, configurations, empty directory markers,
and documentation—not participant-level data.
