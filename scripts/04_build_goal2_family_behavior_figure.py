
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Uses the active Matplotlib color cycle; no family score is constructed.
_cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
COHORT_COLORS = {"CONTROL": _cycle[0], "ALS": _cycle[1]}
FAMILY_COLORS = {
    family: _cycle[index % len(_cycle)]
    for index, family in enumerate(["QADD", "QGAIN", "QREV", "QCHAN", "QDIST", "QTEMP"])
}
ROLE_MARKERS = {"core": "o", "supporting": "s", "exploratory": "^"}
FAMILY_ORDER_LOCAL = ["QADD", "QGAIN", "QREV", "QCHAN"]

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.7,
    "ytick.labelsize": 6.7,
    "legend.fontsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _style_tick_labels(axis, ordered_registry: pd.DataFrame) -> None:
    roles = ordered_registry["figure_role"].tolist()
    for label, role in zip(axis.get_yticklabels(), roles):
        if role == "core":
            label.set_fontweight("bold")
        elif role == "exploratory":
            label.set_fontstyle("italic")


def _family_positions(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    positions: dict[str, float] = {}
    centers: dict[str, float] = {}
    cursor = 0.0
    for family in FAMILY_ORDER_LOCAL:
        family_frame = frame.loc[frame["family"] == family].sort_values("feature_order")
        start = cursor
        for feature in family_frame["feature"]:
            positions[feature] = cursor
            cursor += 1.0
        if not family_frame.empty:
            centers[family] = (start + cursor - 1.0) / 2.0
            cursor += 0.75
    return positions, centers


def _add_family_labels(axis, centers: dict[str, float], x_fraction: float = -0.42) -> None:
    for family, center in centers.items():
        axis.text(
            x_fraction,
            center,
            family,
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontweight="bold",
            color=FAMILY_COLORS[family],
            clip_on=False,
        )


def _effect_star(q_value: float) -> str:
    return "*" if np.isfinite(q_value) and q_value < 0.05 else ""


# =====================================================================
# Figure canvas
# =====================================================================
fig = plt.figure(figsize=(7.25, 12.2), constrained_layout=False)
outer = fig.add_gridspec(
    4,
    1,
    height_ratios=[3.6, 3.35, 3.25, 1.35],
    hspace=0.62,
)

fig.suptitle(
    "Family-wise behavior of recording-quality indicators across cohorts and bulbar function",
    x=0.50,
    y=0.992,
    fontsize=11.2,
    fontweight="bold",
)

# =====================================================================
# A. Participant-balanced family profiles
# =====================================================================
gs_a = outer[0].subgridspec(2, 2, hspace=0.55, wspace=0.58)
axes_a = {}
for index, family in enumerate(FAMILY_ORDER_LOCAL):
    axes_a[family] = fig.add_subplot(gs_a[index // 2, index % 2])

profile_values = panel_a_profile["participant_median_z"].dropna().to_numpy()
if profile_values.size:
    global_low, global_high = np.quantile(profile_values, [0.005, 0.995])
    span = max(global_high - global_low, 1.0)
    global_low -= 0.05 * span
    global_high += 0.05 * span
else:
    global_low, global_high = -2.0, 2.0

profile_rng = np.random.default_rng(SEED)

for family, axis in axes_a.items():
    registry_family = (
        portfolio_registry.loc[portfolio_registry["family"] == family]
        .sort_values("feature_order")
        .copy()
    )
    features = registry_family["feature"].tolist()
    y_positions = np.arange(len(features))[::-1]

    axis.axvline(0, linewidth=0.7, linestyle="--", color="0.55", zorder=0)

    for row_index, (feature, y_value) in enumerate(zip(features, y_positions)):
        role = registry_family.loc[
            registry_family["feature"] == feature,
            "figure_role",
        ].iloc[0]
        marker = ROLE_MARKERS.get(role, "o")

        for cohort, offset in [("CONTROL", 0.13), ("ALS", -0.13)]:
            values = panel_a_profile.loc[
                (panel_a_profile["feature"] == feature)
                & (panel_a_profile["cohort"] == cohort),
                "participant_median_z",
            ].dropna().to_numpy()

            if not values.size:
                continue

            plotted = np.clip(values, global_low, global_high)
            jitter = profile_rng.normal(0, 0.025, size=len(plotted))
            axis.scatter(
                plotted,
                np.full(len(plotted), y_value + offset) + jitter,
                s=5,
                alpha=0.16,
                color=COHORT_COLORS[cohort],
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

            q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            q1 = np.clip(q1, global_low, global_high)
            median = np.clip(median, global_low, global_high)
            q3 = np.clip(q3, global_low, global_high)

            axis.plot(
                [q1, q3],
                [y_value + offset, y_value + offset],
                color=COHORT_COLORS[cohort],
                linewidth=1.6,
                solid_capstyle="round",
                zorder=3,
            )
            axis.scatter(
                [median],
                [y_value + offset],
                s=23,
                marker=marker,
                color=COHORT_COLORS[cohort],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(registry_family["display_label"])
    _style_tick_labels(axis, registry_family)
    axis.set_xlim(global_low, global_high)
    axis.set_ylim(-0.6, len(features) - 0.4)
    axis.set_title(
        family,
        loc="left",
        fontweight="bold",
        color=FAMILY_COLORS[family],
        pad=5,
    )
    axis.grid(axis="x", linewidth=0.45, alpha=0.18)
    axis.tick_params(axis="y", length=0)

for family in ["QREV", "QCHAN"]:
    axes_a[family].set_xlabel(
        "Participant median, transformed and IQR-standardized"
    )

fig.text(
    0.055,
    0.965,
    "A",
    fontsize=10,
    fontweight="bold",
)
fig.text(
    0.085,
    0.965,
    "Participant-balanced distributions across complete family portfolios",
    fontsize=9.2,
    fontweight="bold",
)
fig.text(
    0.085,
    0.943,
    "One median per participant; higher values retain the raw indicator direction shown in each label. "
    "Core labels are bold; exploratory labels are italic.",
    fontsize=6.7,
)

cohort_handles = [
    mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=COHORT_COLORS["CONTROL"],
        label="Control",
        markersize=4.5,
    ),
    mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=COHORT_COLORS["ALS"],
        label="ALS",
        markersize=4.5,
    ),
]
fig.legend(
    handles=cohort_handles,
    loc="upper right",
    bbox_to_anchor=(0.965, 0.967),
    frameon=False,
    ncol=2,
)

# =====================================================================
# B. Cohort effects
# =====================================================================
gs_b = outer[1].subgridspec(1, 2, width_ratios=[3.5, 1.45], wspace=0.75)
ax_b_cont = fig.add_subplot(gs_b[0, 0])
ax_b_bin = fig.add_subplot(gs_b[0, 1])

b_cont = panel_b_continuous.sort_values(["family", "feature_order"]).copy()
pos_b, centers_b = _family_positions(b_cont)
b_cont["y"] = b_cont["feature"].map(pos_b)

ax_b_cont.axvline(0, linewidth=0.75, linestyle="--", color="0.50")
for row in b_cont.itertuples(index=False):
    marker = ROLE_MARKERS.get(row.figure_role, "o")
    color = FAMILY_COLORS[row.family]
    alpha = 1.0 if row.figure_role != "exploratory" else 0.65
    ax_b_cont.errorbar(
        row.estimate,
        row.y,
        xerr=np.array(
            [[row.estimate - row.ci95_low], [row.ci95_high - row.estimate]]
        ),
        fmt=marker,
        color=color,
        markerfacecolor=(
            color if row.figure_role == "core" else "white"
        ),
        markeredgecolor=color,
        markersize=4.2,
        linewidth=1.0,
        capsize=2.0,
        alpha=alpha,
        zorder=3,
    )
    if _effect_star(row.q_value):
        ax_b_cont.text(
            row.ci95_high,
            row.y,
            " *",
            va="center",
            ha="left",
            color=color,
            fontsize=7,
        )

ordered_b = b_cont.sort_values("y")
ax_b_cont.set_yticks(ordered_b["y"])
ax_b_cont.set_yticklabels(ordered_b["display_label"])
_style_tick_labels(
    ax_b_cont,
    ordered_b[["figure_role"]],
)
_add_family_labels(ax_b_cont, centers_b)
ax_b_cont.set_ylim(
    max(pos_b.values()) + 0.65 if pos_b else 1,
    -0.65,
)
ax_b_cont.set_xlabel(
    "ALS − control difference (transformed, per IQR)"
)
ax_b_cont.set_title(
    "Continuous and bounded outputs",
    loc="left",
    fontweight="bold",
)
ax_b_cont.grid(axis="x", linewidth=0.45, alpha=0.20)
ax_b_cont.tick_params(axis="y", length=0)

b_bin = panel_b_presence.sort_values("feature_order").copy()
if b_bin.empty:
    ax_b_bin.text(
        0.5,
        0.5,
        "No estimable directional-presence models",
        ha="center",
        va="center",
        transform=ax_b_bin.transAxes,
    )
    ax_b_bin.set_axis_off()
else:
    b_bin["odds_ratio"] = np.exp(b_bin["estimate"])
    b_bin["or_low"] = np.exp(b_bin["ci95_low"])
    b_bin["or_high"] = np.exp(b_bin["ci95_high"])
    y_bin = np.arange(len(b_bin))[::-1]
    ax_b_bin.axvline(1, linewidth=0.75, linestyle="--", color="0.50")
    for y_value, row in zip(y_bin, b_bin.itertuples(index=False)):
        marker = ROLE_MARKERS.get(row.figure_role, "o")
        color = FAMILY_COLORS[row.family]
        ax_b_bin.errorbar(
            row.odds_ratio,
            y_value,
            xerr=np.array(
                [[row.odds_ratio - row.or_low], [row.or_high - row.odds_ratio]]
            ),
            fmt=marker,
            color=color,
            markerfacecolor=(
                color if row.figure_role == "core" else "white"
            ),
            markeredgecolor=color,
            markersize=4.2,
            linewidth=1.0,
            capsize=2.0,
        )
        if _effect_star(row.q_value):
            ax_b_bin.text(
                row.or_high,
                y_value,
                " *",
                va="center",
                ha="left",
                color=color,
                fontsize=7,
            )

    ax_b_bin.set_xscale("log")
    ax_b_bin.set_yticks(y_bin)
    ax_b_bin.set_yticklabels(b_bin["display_label"])
    _style_tick_labels(ax_b_bin, b_bin)
    ax_b_bin.set_xlabel("ALS/control odds ratio")
    ax_b_bin.set_title(
        "Directional-deficit presence",
        loc="left",
        fontweight="bold",
    )
    ax_b_bin.grid(axis="x", linewidth=0.45, alpha=0.20)
    ax_b_bin.tick_params(axis="y", length=0)

fig.text(0.055, 0.655, "B", fontsize=10, fontweight="bold")
fig.text(
    0.085,
    0.655,
    "Cohort differences across family portfolios",
    fontsize=9.2,
    fontweight="bold",
)
fig.text(
    0.085,
    0.637,
    "Participant-balanced GEE estimates; positive continuous effects indicate a higher raw indicator value in ALS. "
    "Odds ratios >1 indicate more frequent directional deficits in ALS.",
    fontsize=6.7,
)

# =====================================================================
# C. Bulbar effects
# =====================================================================
gs_c = outer[2].subgridspec(1, 2, width_ratios=[3.5, 1.45], wspace=0.75)
ax_c_cont = fig.add_subplot(gs_c[0, 0])
ax_c_bin = fig.add_subplot(gs_c[0, 1])

c_cont = panel_c_continuous.copy()
if not c_cont.empty:
    # Re-express effects per one-point lower score (worse function).
    c_cont["display_estimate"] = -c_cont["estimate"]
    c_cont["display_low"] = -c_cont["ci95_high"]
    c_cont["display_high"] = -c_cont["ci95_low"]

    c_registry = (
        portfolio_registry.loc[
            portfolio_registry["outcome_class"].isin(
                ["continuous", "censored_continuous"]
            )
        ]
        .sort_values(["family", "feature_order"])
        .copy()
    )
    pos_c, centers_c = _family_positions(c_registry)
    c_cont["y_base"] = c_cont["feature"].map(pos_c)

    ax_c_cont.axvline(0, linewidth=0.75, linestyle="--", color="0.50")
    effect_offsets = {"bulbar_between": 0.12, "bulbar_within": -0.12}
    effect_markers = {"bulbar_between": "o", "bulbar_within": "D"}

    for row in c_cont.itertuples(index=False):
        y_value = row.y_base + effect_offsets[row.analysis]
        color = FAMILY_COLORS[row.family]
        ax_c_cont.errorbar(
            row.display_estimate,
            y_value,
            xerr=np.array(
                [[row.display_estimate - row.display_low],
                 [row.display_high - row.display_estimate]]
            ),
            fmt=effect_markers[row.analysis],
            color=color,
            markerfacecolor=(
                color if row.analysis == "bulbar_between" else "white"
            ),
            markeredgecolor=color,
            markersize=4.1,
            linewidth=1.0,
            capsize=2.0,
            alpha=(0.68 if row.figure_role == "exploratory" else 1.0),
        )
        if _effect_star(row.q_value):
            ax_c_cont.text(
                row.display_high,
                y_value,
                " *",
                va="center",
                ha="left",
                color=color,
                fontsize=7,
            )

    c_ordered = c_registry.copy()
    c_ordered["y"] = c_ordered["feature"].map(pos_c)
    c_ordered = c_ordered.sort_values("y")
    ax_c_cont.set_yticks(c_ordered["y"])
    ax_c_cont.set_yticklabels(c_ordered["display_label"])
    _style_tick_labels(ax_c_cont, c_ordered)
    _add_family_labels(ax_c_cont, centers_c)
    ax_c_cont.set_ylim(
        max(pos_c.values()) + 0.65 if pos_c else 1,
        -0.65,
    )
    ax_c_cont.set_xlabel(
        "Effect per one-point lower bulbar score (standardized units)"
    )
    ax_c_cont.set_title(
        "Continuous and bounded outputs",
        loc="left",
        fontweight="bold",
    )
    ax_c_cont.grid(axis="x", linewidth=0.45, alpha=0.20)
    ax_c_cont.tick_params(axis="y", length=0)

c_bin = panel_c_presence.copy()
if c_bin.empty:
    ax_c_bin.text(
        0.5,
        0.5,
        "No estimable directional-presence models",
        ha="center",
        va="center",
        transform=ax_c_bin.transAxes,
    )
    ax_c_bin.set_axis_off()
else:
    # Per one-point lower score: reverse the fitted log-odds coefficient.
    c_bin["display_log_or"] = -c_bin["estimate"]
    c_bin["display_low"] = -c_bin["ci95_high"]
    c_bin["display_high"] = -c_bin["ci95_low"]
    c_bin["odds_ratio"] = np.exp(c_bin["display_log_or"])
    c_bin["or_low"] = np.exp(c_bin["display_low"])
    c_bin["or_high"] = np.exp(c_bin["display_high"])

    binary_registry = (
        portfolio_registry.loc[
            portfolio_registry["outcome_class"].eq("one_sided_zero")
        ]
        .sort_values("feature_order")
        .copy()
    )
    y_map = {
        feature: y
        for feature, y in zip(
            binary_registry["feature"],
            np.arange(len(binary_registry))[::-1],
        )
    }
    offsets = {
        "bulbar_between_presence": 0.12,
        "bulbar_within_presence": -0.12,
    }
    markers = {
        "bulbar_between_presence": "o",
        "bulbar_within_presence": "D",
    }

    ax_c_bin.axvline(1, linewidth=0.75, linestyle="--", color="0.50")
    for row in c_bin.itertuples(index=False):
        y_value = y_map[row.feature] + offsets[row.analysis]
        color = FAMILY_COLORS[row.family]
        ax_c_bin.errorbar(
            row.odds_ratio,
            y_value,
            xerr=np.array(
                [[row.odds_ratio - row.or_low],
                 [row.or_high - row.odds_ratio]]
            ),
            fmt=markers[row.analysis],
            color=color,
            markerfacecolor=(
                color
                if row.analysis == "bulbar_between_presence"
                else "white"
            ),
            markeredgecolor=color,
            markersize=4.1,
            linewidth=1.0,
            capsize=2.0,
            alpha=(0.68 if row.figure_role == "exploratory" else 1.0),
        )
        if _effect_star(row.q_value):
            ax_c_bin.text(
                row.or_high,
                y_value,
                " *",
                va="center",
                ha="left",
                color=color,
                fontsize=7,
            )

    ax_c_bin.set_xscale("log")
    y_ticks = [
        y_map[feature]
        for feature in binary_registry["feature"]
    ]
    ax_c_bin.set_yticks(y_ticks)
    ax_c_bin.set_yticklabels(binary_registry["display_label"])
    _style_tick_labels(ax_c_bin, binary_registry)
    ax_c_bin.set_xlabel(
        "Odds ratio per one-point lower score"
    )
    ax_c_bin.set_title(
        "Directional-deficit presence",
        loc="left",
        fontweight="bold",
    )
    ax_c_bin.grid(axis="x", linewidth=0.45, alpha=0.20)
    ax_c_bin.tick_params(axis="y", length=0)

bulbar_handles = [
    mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color="0.25",
        markerfacecolor="0.25",
        label="Between participant",
        markersize=4.5,
    ),
    mpl.lines.Line2D(
        [0],
        [0],
        marker="D",
        linestyle="",
        color="0.25",
        markerfacecolor="white",
        label="Within participant",
        markersize=4.2,
    ),
]
ax_c_cont.legend(
    handles=bulbar_handles,
    frameon=False,
    loc="lower right",
)

fig.text(0.055, 0.367, "C", fontsize=10, fontweight="bold")
fig.text(
    0.085,
    0.367,
    "Associations with worse bulbar function within ALS",
    fontsize=9.2,
    fontweight="bold",
)
fig.text(
    0.085,
    0.349,
    "Between- and within-participant effects are separated. Positive values indicate that a lower ALSFRS-R bulbar score "
    "is associated with a higher raw indicator value.",
    fontsize=6.7,
)

# =====================================================================
# D. Sparse event families
# =====================================================================
ax_d = fig.add_subplot(outer[3])
sparse_order = [
    ("QDIST", "Hard clipping"),
    ("QTEMP", "Dropout-like events"),
    ("QTEMP", "Frozen-audio events"),
]
y_sparse = np.arange(len(sparse_order))[::-1]
max_prevalence = max(
    float(panel_d_sparse["recording_prevalence_percent"].max()),
    0.5,
)
x_limit = max(2.0, max_prevalence * 1.45)

for y_value, (family, construct) in zip(y_sparse, sparse_order):
    rows = panel_d_sparse.loc[
        (panel_d_sparse["family"] == family)
        & (panel_d_sparse["construct"] == construct)
    ]
    for cohort, offset in [("CONTROL", 0.10), ("ALS", -0.10)]:
        row = rows.loc[rows["cohort"] == cohort]
        if row.empty:
            continue
        row = row.iloc[0]
        x_value = float(row["recording_prevalence_percent"])
        ax_d.scatter(
            x_value,
            y_value + offset,
            s=24,
            color=COHORT_COLORS[cohort],
            zorder=3,
        )
        ax_d.text(
            min(x_value + 0.04 * x_limit, 0.82 * x_limit),
            y_value + offset,
            f'{int(row["n_positive_recordings"])}/{int(row["n_recordings"])}',
            va="center",
            ha="left",
            fontsize=6.5,
            color=COHORT_COLORS[cohort],
        )

    pooled = rows.copy()
    positive_rows = pooled.loc[pooled["n_positive_recordings"] > 0]
    if positive_rows.empty:
        note = "No detected events"
    else:
        extent = positive_rows["positive_extent_median"].dropna()
        rate = positive_rows["positive_rate_median"].dropna()
        if not extent.empty and not rate.empty:
            note = (
                f"positive median: {extent.median():.3g} "
                f'{positive_rows["extent_unit"].iloc[0]}; '
                f"{rate.median():.3g} {positive_rows['rate_unit'].iloc[0]}"
            )
        else:
            note = "Detected events; positive extent retained descriptively"

    ax_d.text(
        1.02,
        y_value,
        note,
        transform=ax_d.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=6.5,
        clip_on=False,
    )

ax_d.set_yticks(y_sparse)
ax_d.set_yticklabels(
    [f"{family}: {construct}" for family, construct in sparse_order]
)
for label, (family, _) in zip(ax_d.get_yticklabels(), sparse_order):
    label.set_color(FAMILY_COLORS[family])
    label.set_fontweight("bold")
ax_d.set_xlim(0, x_limit)
ax_d.set_xlabel("Recordings with a detected event (%)")
ax_d.set_title(
    "D   Sparse QDIST and QTEMP behavior — descriptive only",
    loc="left",
    fontweight="bold",
)
ax_d.grid(axis="x", linewidth=0.45, alpha=0.20)
ax_d.tick_params(axis="y", length=0)

# =====================================================================
# Global legends and export
# =====================================================================
role_handles = [
    mpl.lines.Line2D(
        [0],
        [0],
        marker=ROLE_MARKERS[role],
        linestyle="",
        color="0.25",
        markerfacecolor=("0.25" if role == "core" else "white"),
        label=label,
        markersize=4.5,
    )
    for role, label in [
        ("core", "Core"),
        ("supporting", "Supporting"),
        ("exploratory", "Exploratory"),
    ]
]
fig.legend(
    handles=role_handles,
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.965, 0.642),
    ncol=3,
)

fig.text(
    0.055,
    0.014,
    "No family-level or global score was formed. Each row is an analytically retained observable; "
    "asterisks denote q<0.05 within the prespecified core or secondary/exploratory FDR family.",
    fontsize=6.4,
)

fig.subplots_adjust(
    left=0.255,
    right=0.87,
    top=0.925,
    bottom=0.055,
)

output_stem = FIGURE_MAIN_DIR / "Figure_goal2_family_behavior_PUBLICATION"
fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
plt.show()

caption = """
Figure X | Family-wise behavior of recording-quality indicators across cohorts and bulbar function.
a, Participant-level distributions of every analytically retained QADD, QGAIN, QREV, and QCHAN output.
Each participant contributes one median per indicator, preventing participants with more recordings from
dominating the display. Values retain the raw indicator direction and are transformed and scaled by the
frozen recording-level interquartile range. b, Participant-balanced population-average cohort effects.
Continuous and bounded outputs are reported as ALS–control differences per interquartile range; one-sided
QCHAN outputs are decomposed into the odds of a directional deficit. c, Between- and within-participant
associations with recording-matched ALSFRS-R bulbar scores, re-expressed per one-point lower score so that
positive values indicate a higher raw indicator value with worse bulbar function. d, QDIST and QTEMP
event occurrence, reported descriptively because positive support was insufficient for stable inference.
Bold labels denote core outputs and italic labels denote exploratory outputs. No family or global score
was calculated.
""".strip()
(FIGURE_MAIN_DIR / "Figure_goal2_family_behavior_PUBLICATION_caption.txt").write_text(
    caption,
    encoding="utf-8",
)
