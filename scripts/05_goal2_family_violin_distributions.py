
# Goal 2: family-wise ALS/control violin distributions with participant-balanced statistics
# Run inside the executed Goal 2 notebook after `analysis` and `FEATURE_REGISTRY` exist.

from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.stats.multitest import multipletests

required_objects = ["analysis", "FEATURE_REGISTRY"]
missing_objects = [name for name in required_objects if name not in globals()]
if missing_objects:
    raise RuntimeError(
        "Run the Goal 2 notebook from the top first. Missing objects: "
        + ", ".join(missing_objects)
    )

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
VIOLIN_FAMILY_ORDER = ["QADD", "QGAIN", "QREV", "QCHAN", "QDIST", "QTEMP"]

FAMILY_TITLES = {
    "QADD": "QADD — additive-interference observables",
    "QGAIN": "QGAIN — recorded level and level dynamics",
    "QREV": "QREV — residual-tail and modulation-smearing observables",
    "QCHAN": "QCHAN — reference-relative spectral observables",
    "QDIST": "QDIST — hard-plateau clipping morphology",
    "QTEMP": "QTEMP — temporal-discontinuity monitoring outputs",
}

DISPLAY_LABELS = {
    "qadd_pause_ac_level_dbfs_median": "Pause-region level",
    "qadd_pause_level_iqr_db": "Pause-level IQR",
    "qadd_speech_pause_level_contrast_db": "Speech–pause contrast",
    "qadd_pause_spectral_flatness": "Pause spectral flatness",
    "qadd_mains_hum_comb_score_db": "Hum-comb prominence",
    "qgain_typical_speech_level_dbfs": "Typical speech level",
    "qgain_within_segment_iqr_db": "Within-segment level IQR",
    "qgain_between_segment_mad_db": "Between-segment level MAD",
    "qgain_abs_drift_db_per_min": "Absolute level drift",
    "qrev_tail_excess_100ms_db": "Early residual-tail excess",
    "qrev_tail_persistence_median_sec": "Tail persistence",
    "qrev_downward_decay_rate_db_per_sec": "Downward decay rate",
    "qrev_srmr_norm": "Normalized-fast SRMR",
    "qchan_ltas_distance_db": "LTAS distance",
    "qchan_rolloff95_deficit_hz": "Rolloff-95 deficit",
    "qchan_highband_ratio_deficit": "High-band ratio deficit",
    "qchan_tilt_steepening_db_per_oct": "Spectral-tilt steepening",
    "qdist_hard_clipped_sample_fraction": "Hard-clipped sample fraction",
    "qdist_hard_clip_event_rate_per_min": "Hard-clipping episode rate",
    "qtemp_dropout_duration_fraction": "Dropout-like duration fraction",
    "qtemp_dropout_event_rate_per_min": "Dropout-like event rate",
    "qtemp_frozen_audio_duration_fraction": "Frozen-audio duration fraction",
    "qtemp_frozen_audio_event_rate_per_min": "Frozen-audio event rate",
}

ROLE_GROUP = {
    "primary_contextual": "core",
    "primary_mixed": "core",
    "primary_conditional": "core",
    "established_comparator": "core",
    "primary_nonordinal": "core",
    "primary_one_sided": "core",
    "primary_sparse_burden": "core",
    "secondary": "supporting",
    "secondary_mixed": "supporting",
    "secondary_nonordinal": "supporting",
    "targeted_conditional": "supporting",
    "secondary_conditional_nonindependent": "supporting",
    "secondary_nonindependent": "supporting",
    "secondary_sparse_event": "supporting",
    "exploratory_contextual": "exploratory",
    "exploratory_conditional": "exploratory",
    "exploratory_phenotype_sensitive": "exploratory",
    "exploratory_not_primary": "exploratory",
    "monitoring_zero_variation": "monitoring",
    "conditional_audit": "audit",
}

MIN_DENSE_POSITIVE_RECORDINGS = 20
MIN_DENSE_POSITIVE_PARTICIPANTS = 10
MIN_SMALLER_BINARY_STATE_PARTICIPANTS = 10
SEED_LOCAL = int(globals().get("SEED", 20260805))

if "ANALYSIS_ROOT" in globals():
    violin_root = Path(ANALYSIS_ROOT) / "outputs" / "goal2_cohort_characterization"
else:
    violin_root = Path.cwd().resolve() / "outputs" / "goal2_cohort_characterization"

violin_figure_dir = violin_root / "figures" / "family_violins"
violin_table_dir = violin_root / "tables"
violin_source_dir = violin_root / "figures" / "source_data"
for directory in [violin_figure_dir, violin_table_dir, violin_source_dir]:
    directory.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
COHORT_COLORS = {"CONTROL": cycle[0], "ALS": cycle[1]}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _export(frame: pd.DataFrame, path: Path) -> None:
    if "export_dataframe" in globals():
        export_dataframe(frame, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def _participant_weights(frame: pd.DataFrame) -> pd.Series:
    """Each participant contributes equal total weight while all recordings remain."""
    cluster_size = (
        frame.groupby("participant_id")["participant_id"]
        .transform("size")
        .astype(float)
    )
    weights = 1.0 / cluster_size
    return weights / weights.mean()


def _weighted_quantile(
    values: np.ndarray,
    quantiles: list[float] | np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values)


def _weighted_kde(
    values: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Simple weighted Gaussian KDE, avoiding unweighted participant overrepresentation."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if values.size < 3 or np.unique(values).size < 2:
        return np.zeros_like(grid)

    weights = weights / weights.sum()
    mean = np.sum(weights * values)
    variance = np.sum(weights * (values - mean) ** 2)
    std = np.sqrt(max(variance, 1e-12))
    q25, q75 = _weighted_quantile(values, [0.25, 0.75], weights)
    robust_sigma = (q75 - q25) / 1.349 if np.isfinite(q75 - q25) else std
    scale = min(std, robust_sigma) if robust_sigma > 0 else std

    effective_n = 1.0 / np.sum(weights**2)
    bandwidth = 0.9 * scale * effective_n ** (-1 / 5)
    if not np.isfinite(bandwidth) or bandwidth <= 1e-6:
        bandwidth = max(std * 0.15, 0.05)

    standardized = (grid[:, None] - values[None, :]) / bandwidth
    kernels = np.exp(-0.5 * standardized**2)
    density = kernels @ weights
    density /= bandwidth * np.sqrt(2 * np.pi)
    return density


def _fit_gee(
    frame: pd.DataFrame,
    *,
    outcome: str,
    family,
    target: str = "cohort_als",
    binary: bool = False,
) -> dict | None:
    required = [outcome, "participant_id", "cohort", target]
    data = frame.dropna(subset=required).copy()
    if data.empty or data["participant_id"].nunique() < 2:
        return None

    if binary:
        if data[outcome].nunique() < 2:
            return None
        state_participants = (
            data.groupby(outcome)["participant_id"]
            .nunique()
            .reindex([0, 1], fill_value=0)
        )
        if state_participants.min() < MIN_SMALLER_BINARY_STATE_PARTICIPANTS:
            return None

        cohort_state = pd.crosstab(data["cohort"], data[outcome]).reindex(
            index=["CONTROL", "ALS"],
            columns=[0, 1],
            fill_value=0,
        )
        if (cohort_state == 0).any().any():
            return None

    data["participant_weight"] = _participant_weights(data)
    formula = f"{outcome} ~ {target}"

    attempts = [
        (Exchangeable(), "bias_reduced"),
        (Exchangeable(), "robust"),
        (Independence(), "bias_reduced"),
        (Independence(), "robust"),
    ]

    errors = []
    for covariance, covariance_type in attempts:
        try:
            model = sm.GEE.from_formula(
                formula,
                groups="participant_id",
                data=data,
                family=family,
                cov_struct=covariance,
                weights=data["participant_weight"],
            )
            result = model.fit(cov_type=covariance_type)
            low, high = result.conf_int().loc[target].astype(float)
            return {
                "estimate": float(result.params[target]),
                "standard_error": float(result.bse[target]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "p_value": float(result.pvalues[target]),
                "n_recordings": int(len(data)),
                "n_participants": int(data["participant_id"].nunique()),
                "model": (
                    f"Participant-balanced GEE; "
                    f"{covariance.__class__.__name__}; {covariance_type}"
                ),
                "converged": True,
                "notes": (
                    "All recordings retained; each participant contributes equal total weight."
                ),
            }
        except Exception as exc:
            errors.append(
                f"{covariance.__class__.__name__}/{covariance_type}: "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "estimate": np.nan,
        "standard_error": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "p_value": np.nan,
        "n_recordings": int(len(data)),
        "n_participants": int(data["participant_id"].nunique()),
        "model": "GEE failed",
        "converged": False,
        "notes": " | ".join(errors),
    }


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _format_effect(row: pd.Series) -> str:
    q_text = _format_p(float(row["q_value"])) if "q_value" in row else "NA"
    if row["analysis"] == "continuous":
        return (
            f"Δ={row['estimate']:+.2f} "
            f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]"
            f"; q={q_text}"
        )
    if row["analysis"] == "presence":
        odds_ratio = np.exp(row["estimate"])
        low = np.exp(row["ci95_low"])
        high = np.exp(row["ci95_high"])
        return (
            f"OR={odds_ratio:.2f} [{low:.2f}, {high:.2f}]"
            f"; q={q_text}"
        )
    return "descriptive only"


# ---------------------------------------------------------------------
# Select complete family portfolios
# ---------------------------------------------------------------------
registry = FEATURE_REGISTRY.copy()

if "figure_role" not in registry.columns:
    registry["figure_role"] = (
        registry["validated_role"].map(ROLE_GROUP).fillna("other")
    )

if "display_label" not in registry.columns:
    registry["display_label"] = (
        registry["feature"].map(DISPLAY_LABELS).fillna(registry["label"])
    )
else:
    registry["display_label"] = (
        registry["feature"].map(DISPLAY_LABELS).fillna(registry["display_label"])
    )

if "feature_order" not in registry.columns:
    registry["feature_order"] = np.arange(len(registry))

# Audit-only outputs are never plotted as cohort distributions.
plot_registry = registry.loc[
    ~registry["figure_role"].isin(["audit"])
    & registry["family"].isin(VIOLIN_FAMILY_ORDER)
].copy()

# Keep one numerical ledger view where outputs are deterministic companions.
# QDIST frame occupancy remains audit-only; QTEMP duration and rate are retained
# in the table, but only plotted if empirical support makes a violin meaningful.
plot_registry = plot_registry.sort_values(["family", "feature_order"])

analysis_local = analysis.copy()
if "cohort_als" not in analysis_local.columns:
    analysis_local["cohort_als"] = analysis_local["cohort"].eq("ALS").astype(int)

# ---------------------------------------------------------------------
# Fit cohort differences
# ---------------------------------------------------------------------
stats_rows = []
plot_eligibility_rows = []

for row in plot_registry.itertuples(index=False):
    feature = row.feature
    z_col = f"{feature}__z"
    available_col = f"{feature}__available"

    if feature not in analysis_local.columns or z_col not in analysis_local.columns:
        continue

    if available_col in analysis_local.columns:
        data = analysis_local.loc[analysis_local[available_col]].copy()
    else:
        data = analysis_local.loc[analysis_local[feature].notna()].copy()

    if data.empty:
        continue

    positive = data[feature].gt(0)
    positive_recordings = int(positive.sum())
    positive_participants = int(
        data.loc[positive, "participant_id"].nunique()
    )
    positive_cohorts = int(data.loc[positive, "cohort"].nunique())

    zero_heavy = bool(
        row.outcome_class in {
            "one_sided_zero",
            "sparse_event_extent",
            "sparse_event_rate",
        }
    )

    violin_eligible = True
    exclusion_reason = ""

    if row.outcome_class in {"sparse_event_extent", "sparse_event_rate"}:
        violin_eligible = bool(
            positive_recordings >= MIN_DENSE_POSITIVE_RECORDINGS
            and positive_participants >= MIN_DENSE_POSITIVE_PARTICIPANTS
            and positive_cohorts == 2
        )
        if not violin_eligible:
            exclusion_reason = (
                "Sparse event distribution: violin would be dominated by zeros."
            )

    if data[z_col].nunique(dropna=True) < 2:
        violin_eligible = False
        exclusion_reason = "No empirical variation."

    plot_eligibility_rows.append({
        "feature": feature,
        "family": row.family,
        "label": row.display_label,
        "validated_role": row.validated_role,
        "outcome_class": row.outcome_class,
        "n_available_recordings": int(len(data)),
        "n_available_participants": int(data["participant_id"].nunique()),
        "n_positive_recordings": positive_recordings,
        "n_positive_participants": positive_participants,
        "positive_cohorts": positive_cohorts,
        "violin_eligible": violin_eligible,
        "exclusion_reason": exclusion_reason,
    })

    if row.outcome_class in {"continuous", "censored_continuous"}:
        fitted = _fit_gee(
            data,
            outcome=z_col,
            family=sm.families.Gaussian(),
            binary=False,
        )
        if fitted is not None:
            stats_rows.append({
                "feature": feature,
                "family": row.family,
                "label": row.display_label,
                "validated_role": row.validated_role,
                "figure_role": row.figure_role,
                "outcome_class": row.outcome_class,
                "analysis": "continuous",
                **fitted,
            })

    elif row.outcome_class == "one_sided_zero":
        data["event_presence"] = positive.astype(int)
        fitted = _fit_gee(
            data,
            outcome="event_presence",
            family=sm.families.Binomial(),
            binary=True,
        )
        if fitted is not None:
            stats_rows.append({
                "feature": feature,
                "family": row.family,
                "label": row.display_label,
                "validated_role": row.validated_role,
                "figure_role": row.figure_role,
                "outcome_class": row.outcome_class,
                "analysis": "presence",
                **fitted,
            })

        # Positive extent is exported but not used as the primary subplot annotation.
        positive_data = data.loc[positive].copy()
        if (
            positive_data["participant_id"].nunique()
            >= MIN_DENSE_POSITIVE_PARTICIPANTS
            and positive_data["cohort"].nunique() == 2
        ):
            fitted_extent = _fit_gee(
                positive_data,
                outcome=z_col,
                family=sm.families.Gaussian(),
                binary=False,
            )
            if fitted_extent is not None:
                stats_rows.append({
                    "feature": feature,
                    "family": row.family,
                    "label": row.display_label,
                    "validated_role": row.validated_role,
                    "figure_role": row.figure_role,
                    "outcome_class": row.outcome_class,
                    "analysis": "positive_extent",
                    **fitted_extent,
                })

    elif row.outcome_class in {"sparse_event_extent", "sparse_event_rate"}:
        if violin_eligible:
            data["event_presence"] = positive.astype(int)
            fitted = _fit_gee(
                data,
                outcome="event_presence",
                family=sm.families.Binomial(),
                binary=True,
            )
            if fitted is not None:
                stats_rows.append({
                    "feature": feature,
                    "family": row.family,
                    "label": row.display_label,
                    "validated_role": row.validated_role,
                    "figure_role": row.figure_role,
                    "outcome_class": row.outcome_class,
                    "analysis": "presence",
                    **fitted,
                })

stats_table = pd.DataFrame(stats_rows)
eligibility_table = pd.DataFrame(plot_eligibility_rows)

if not stats_table.empty:
    stats_table["inference_tier"] = np.where(
        stats_table["figure_role"].eq("core"),
        "core",
        "secondary_exploratory",
    )
    stats_table["fdr_family"] = (
        stats_table["analysis"] + "_" + stats_table["inference_tier"]
    )
    stats_table["q_value"] = np.nan

    for _, indices in stats_table.groupby("fdr_family").groups.items():
        p_values = stats_table.loc[indices, "p_value"].to_numpy(dtype=float)
        finite = np.isfinite(p_values)
        if finite.any():
            adjusted = multipletests(
                p_values[finite],
                method="fdr_bh",
            )[1]
            stats_table.loc[np.asarray(indices)[finite], "q_value"] = adjusted

_export(
    stats_table,
    violin_table_dir / "goal2_family_violin_cohort_tests.csv",
)
_export(
    eligibility_table,
    violin_table_dir / "goal2_family_violin_eligibility.csv",
)

# ---------------------------------------------------------------------
# Build one uncrowded figure per family
# ---------------------------------------------------------------------
primary_annotation = (
    stats_table.loc[
        stats_table["analysis"].isin(["continuous", "presence"])
    ]
    .drop_duplicates(subset=["feature"], keep="first")
    .set_index("feature")
    if not stats_table.empty
    else pd.DataFrame()
)

generated_figures = []
source_rows = []

for family in VIOLIN_FAMILY_ORDER:
    family_registry = plot_registry.loc[
        plot_registry["family"] == family
    ].copy()

    family_eligibility = eligibility_table.loc[
        eligibility_table["family"] == family
    ].set_index("feature")

    eligible_features = [
        feature
        for feature in family_registry["feature"]
        if (
            feature in family_eligibility.index
            and bool(family_eligibility.loc[feature, "violin_eligible"])
        )
    ]

    if not eligible_features:
        print(
            f"{family}: no statistically meaningful violin plot generated. "
            "See goal2_family_violin_eligibility.csv."
        )
        continue

    family_registry = family_registry.loc[
        family_registry["feature"].isin(eligible_features)
    ].sort_values("feature_order")

    n_rows = len(family_registry)
    figure_height = max(2.8, 0.78 * n_rows + 1.25)
    fig, axis = plt.subplots(
        figsize=(7.15, figure_height),
        constrained_layout=False,
    )

    # Common family x-range on transformed/IQR-standardized scale.
    pooled_values = []
    for feature in eligible_features:
        z_col = f"{feature}__z"
        values = analysis_local[z_col].dropna().to_numpy(dtype=float)
        pooled_values.extend(values.tolist())

    pooled_values = np.asarray(pooled_values, dtype=float)
    if pooled_values.size:
        x_low, x_high = np.quantile(pooled_values, [0.005, 0.995])
        span = max(x_high - x_low, 1.0)
        x_low -= 0.05 * span
        x_high += 0.05 * span
    else:
        x_low, x_high = -2.0, 2.0

    x_grid = np.linspace(x_low, x_high, 500)
    y_positions = np.arange(n_rows)[::-1]
    axis.axvline(0, color="0.50", linewidth=0.75, linestyle="--", zorder=0)

    for y_value, row in zip(
        y_positions,
        family_registry.itertuples(index=False),
    ):
        feature = row.feature
        z_col = f"{feature}__z"
        available_col = f"{feature}__available"

        if available_col in analysis_local.columns:
            feature_data = analysis_local.loc[
                analysis_local[available_col]
            ].copy()
        else:
            feature_data = analysis_local.loc[
                analysis_local[feature].notna()
            ].copy()

        for cohort, sign in [("CONTROL", +1), ("ALS", -1)]:
            cohort_data = feature_data.loc[
                feature_data["cohort"] == cohort
            ].dropna(subset=[z_col]).copy()

            values = cohort_data[z_col].to_numpy(dtype=float)
            weights = _participant_weights(cohort_data).to_numpy(dtype=float)
            density = _weighted_kde(values, weights, x_grid)

            if density.max() > 0:
                width = 0.31 * density / density.max()
                axis.fill_between(
                    x_grid,
                    y_value,
                    y_value + sign * width,
                    color=COHORT_COLORS[cohort],
                    alpha=0.36,
                    linewidth=0,
                    zorder=1,
                )
                axis.plot(
                    x_grid,
                    y_value + sign * width,
                    color=COHORT_COLORS[cohort],
                    linewidth=0.8,
                    zorder=2,
                )

            q1, median, q3 = _weighted_quantile(
                values,
                [0.25, 0.50, 0.75],
                weights,
            )
            summary_y = y_value + sign * 0.10
            axis.plot(
                [q1, q3],
                [summary_y, summary_y],
                color=COHORT_COLORS[cohort],
                linewidth=1.8,
                solid_capstyle="round",
                zorder=4,
            )
            axis.scatter(
                [median],
                [summary_y],
                s=22,
                color=COHORT_COLORS[cohort],
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )

            source_rows.append({
                "family": family,
                "feature": feature,
                "label": row.display_label,
                "cohort": cohort,
                "n_recordings": int(len(cohort_data)),
                "n_participants": int(
                    cohort_data["participant_id"].nunique()
                ),
                "weighted_q1_z": q1,
                "weighted_median_z": median,
                "weighted_q3_z": q3,
                "zero_percent": (
                    100 * float(cohort_data[feature].eq(0).mean())
                    if len(cohort_data)
                    else np.nan
                ),
            })

        # Right-side statistical annotation.
        if (
            isinstance(primary_annotation, pd.DataFrame)
            and feature in primary_annotation.index
        ):
            annotation = _format_effect(primary_annotation.loc[feature])
        else:
            annotation = "descriptive only"

        axis.text(
            1.015,
            y_value,
            annotation,
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.7,
            clip_on=False,
        )

        if row.outcome_class in {
            "one_sided_zero",
            "sparse_event_extent",
            "sparse_event_rate",
        }:
            control_data = feature_data.loc[
                feature_data["cohort"] == "CONTROL"
            ]
            als_data = feature_data.loc[
                feature_data["cohort"] == "ALS"
            ]
            zero_note = (
                f"zero: {100*control_data[feature].eq(0).mean():.0f}% C; "
                f"{100*als_data[feature].eq(0).mean():.0f}% ALS"
            )
            axis.text(
                0.99,
                y_value - 0.27,
                zero_note,
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=5.9,
                color="0.40",
            )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(family_registry["display_label"])
    for label, role in zip(
        axis.get_yticklabels(),
        family_registry["figure_role"],
    ):
        if role == "core":
            label.set_fontweight("bold")
        elif role == "exploratory":
            label.set_fontstyle("italic")

    axis.set_xlim(x_low, x_high)
    axis.set_ylim(-0.55, n_rows - 0.45)
    axis.set_xlabel(
        "Recording value after frozen transform and IQR standardization"
    )
    axis.set_title(
        FAMILY_TITLES[family],
        loc="left",
        fontweight="bold",
        pad=10,
    )
    axis.grid(axis="x", linewidth=0.45, alpha=0.18)
    axis.tick_params(axis="y", length=0)

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=COHORT_COLORS["CONTROL"],
            linewidth=5,
            alpha=0.55,
            label="Control",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=COHORT_COLORS["ALS"],
            linewidth=5,
            alpha=0.55,
            label="ALS",
        ),
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
    )

    fig.text(
        0.015,
        0.012,
        "All recordings are retained. Violin densities and cohort models use inverse recording-count "
        "weights so each participant contributes equal total weight. Δ denotes the participant-balanced "
        "ALS−control difference on the transformed/IQR scale; OR denotes the odds ratio for a detected "
        "one-sided state. Core labels are bold and exploratory labels italic.",
        fontsize=6.2,
    )
    fig.subplots_adjust(
        left=0.26,
        right=0.75,
        top=0.88,
        bottom=0.14,
    )

    output_stem = (
        violin_figure_dir
        / f"Figure_goal2_{family.lower()}_ALS_control_violins"
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    generated_figures.append(str(output_stem.with_suffix(".png")))
    plt.show()

source_table = pd.DataFrame(source_rows)
_export(
    source_table,
    violin_source_dir / "goal2_family_violin_source_data.csv",
)

manifest = {
    "figures": generated_figures,
    "distribution_unit": "recording",
    "participant_balance": (
        "inverse recording-count weighting within feature and cohort; "
        "each participant contributes equal total weight"
    ),
    "continuous_test": "participant-balanced Gaussian GEE",
    "one_sided_test": (
        "participant-balanced binomial GEE for positive-state presence; "
        "positive extent exported separately when supported"
    ),
    "fdr": (
        "Benjamini-Hochberg within analysis type and core versus "
        "secondary/exploratory tier"
    ),
    "sparse_rule": {
        "minimum_positive_recordings": MIN_DENSE_POSITIVE_RECORDINGS,
        "minimum_positive_participants": MIN_DENSE_POSITIVE_PARTICIPANTS,
        "both_cohorts_required": True,
    },
}
(
    violin_figure_dir / "goal2_family_violin_manifest.json"
).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\nGenerated family violin figures:")
for figure_path in generated_figures:
    print(" -", figure_path)

print(
    "\nQDIST/QTEMP are automatically omitted from violin plotting when their "
    "distributions are too sparse or degenerate; their exact eligibility and "
    "support are saved in goal2_family_violin_eligibility.csv."
)

display(stats_table)
display(eligibility_table)
