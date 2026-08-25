from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FAMILY_ORDER = ["QADD", "QGAIN", "QREV", "QCHAN", "QDIST", "QTEMP"]
ROLE_ORDER = [
    "principal_or_contextual",
    "supporting_or_comparator",
    "exploratory_monitoring_or_audit",
    "dropped",
]
ROLE_LABELS = {
    "principal_or_contextual": "Principal / contextual",
    "supporting_or_comparator": "Supporting / comparator",
    "exploratory_monitoring_or_audit": "Exploratory / monitoring / audit",
    "dropped": "Dropped",
}
# Okabe-Ito palette: colorblind-safe and publication friendly.
COLORS = {
    "primary": "#0072B2",
    "reference": "#6F6F6F",
    "principal_or_contextual": "#0072B2",
    "supporting_or_comparator": "#E69F00",
    "exploratory_monitoring_or_audit": "#009E73",
    "dropped": "#CC79A7",
    "grid": "#D9D9D9",
    "text": "#1A1A1A",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find analysis_recording_quality from the notebook or script location."""
    start = (start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (
            (candidate / "data" / "derived" / "registries" / "family_registry.csv").exists()
            and (candidate / "data" / "derived" / "registries" / "feature_registry.csv").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate analysis_recording_quality. Run the registry-building cells first."
    )


def read_family_registry(root: Path) -> pd.DataFrame:
    path = root / "data" / "derived" / "registries" / "family_registry.csv"
    return pd.read_csv(path).set_index("family")


def read_feature_registry(root: Path) -> pd.DataFrame:
    path = root / "data" / "derived" / "registries" / "feature_registry.csv"
    return pd.read_csv(path)


def source_path(registry: pd.DataFrame, family: str, column: str) -> Path:
    value = registry.loc[family, column]
    if pd.isna(value):
        raise FileNotFoundError(f"No {column} was registered for {family}.")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_construct_data(registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    qadd = pd.read_csv(source_path(registry, "QADD", "construct_source_file"))
    qadd = qadd.loc[qadd["control"].eq("pause_noise_dose")].copy()
    result["QADD"] = qadd[["pause_noise_dbfs", "qadd_pause_ac_level_dbfs_median"]].dropna()

    qgain = pd.read_csv(source_path(registry, "QGAIN", "construct_source_file"))
    qgain = qgain.loc[qgain["mechanism"].eq("amplitude_modulation")].copy()
    result["QGAIN"] = qgain[["dose", "response"]].dropna()

    qrev = pd.read_csv(source_path(registry, "QREV", "construct_source_file"))
    used = qrev["used_for_ordering"].astype(str).str.lower().eq("true")
    qrev = qrev.loc[qrev["section"].eq("boundary_rir") & used].copy()
    result["QREV"] = qrev[["rt60_sec", "qrev_tail_persistence_median_sec"]].dropna()

    qchan = pd.read_csv(source_path(registry, "QCHAN", "construct_source_file"))
    qchan = qchan.loc[qchan["condition"].eq("lowpass")].copy()
    qchan["upper_band_reduction_hz"] = qchan["dose"].max() - qchan["dose"]
    result["QCHAN"] = qchan[["upper_band_reduction_hz", "qchan_rolloff95_deficit_hz"]].dropna()

    qdist = pd.read_csv(source_path(registry, "QDIST", "construct_source_file"))
    result["QDIST"] = qdist[["realized_fraction", "estimated_sample_fraction"]].dropna()

    return result


def _extract_persistence_row(source: Path, feature: str) -> tuple[float, int]:
    frame = pd.read_csv(source)
    feature_column = "feature" if "feature" in frame.columns else "metric"
    row = frame.loc[frame[feature_column].astype(str).eq(feature)]
    if row.empty:
        raise KeyError(f"Feature {feature!r} was not found in {source}.")
    row = row.iloc[0]

    rho_col = next(
        (name for name in ("first_second_spearman", "positive_part_spearman_rho") if name in frame.columns),
        None,
    )
    n_col = next(
        (
            name
            for name in (
                "paired_subject_count",
                "first_second_pair_count",
                "participant_pair_count",
                "subjects_with_repeats",
            )
            if name in frame.columns
        ),
        None,
    )
    if rho_col is None or n_col is None:
        raise KeyError(f"Persistence columns were not recognized in {source}.")
    return float(row[rho_col]), int(row[n_col])


def fisher_ci(rho: float, n: int, z_critical: float = 1.959963984540054) -> tuple[float, float]:
    """Approximate 95% CI using Fisher's z transform, applied uniformly."""
    if n <= 3 or not np.isfinite(rho):
        return np.nan, np.nan
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    return float(np.tanh(z - z_critical * se)), float(np.tanh(z + z_critical * se))


def prepare_persistence_data(registry: pd.DataFrame) -> pd.DataFrame:
    selected = [
        ("QADD", "qadd_pause_ac_level_dbfs_median", "QADD pause-region level"),
        ("QGAIN", "qgain_within_segment_iqr_db", "QGAIN within-segment level IQR"),
        ("QREV", "qrev_tail_excess_100ms_db", "QREV early residual-tail excess"),
        ("QCHAN", "qchan_ltas_distance_db", "QCHAN LTAS distance"),
        ("QCHAN", "qchan_rolloff95_deficit_hz", "QCHAN rolloff-95 deficit"),
    ]
    rows = []
    for family, feature, label in selected:
        source = source_path(registry, family, "persistence_source_file")
        rho, n = _extract_persistence_row(source, feature)
        lower, upper = fisher_ci(rho, n)
        rows.append(
            {
                "family": family,
                "feature": feature,
                "label": label,
                "spearman_rho": rho,
                "ci_lower": lower,
                "ci_upper": upper,
                "paired_participants": n,
                "source_file": str(source),
            }
        )
    return pd.DataFrame(rows)


def prepare_role_counts(feature_registry: pd.DataFrame) -> pd.DataFrame:
    companions = feature_registry["is_companion_status"].astype(str).str.lower().isin({"true", "1", "yes"})
    frame = feature_registry.loc[~companions].copy()
    table = (
        frame.groupby(["family", "role_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(FAMILY_ORDER, fill_value=0)
    )
    for role in ROLE_ORDER:
        if role not in table.columns:
            table[role] = 0
    return table[ROLE_ORDER].reset_index()


def panel_letter(ax: mpl.axes.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.12,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.2, length=3, width=0.7)
    ax.xaxis.label.set_size(7.4)
    ax.yaxis.label.set_size(7.4)
    ax.title.set_size(8.6)


def build_publication_figure(project_root: Path | None = None) -> dict[str, Path]:
    root = find_project_root(project_root)
    family_registry = read_family_registry(root)
    feature_registry = read_feature_registry(root)
    construct = prepare_construct_data(family_registry)
    persistence = prepare_persistence_data(family_registry)
    roles = prepare_role_counts(feature_registry)

    output_main = root / "outputs" / "figures" / "main"
    output_source = root / "outputs" / "figures" / "source_data"
    output_main.mkdir(parents=True, exist_ok=True)
    output_source.mkdir(parents=True, exist_ok=True)

    basename = "Figure_3_1_analytical_validation_PUBLICATION"

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    # Designed at a standard two-column journal width.
    fig = plt.figure(figsize=(7.15, 9.35), constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.72, 0.74, 0.78],
        hspace=0.52,
        left=0.11,
        right=0.985,
        top=0.905,
        bottom=0.065,
    )

    # ------------------------------------------------------------------
    # A. Representative controlled responses
    # ------------------------------------------------------------------
    top = outer[0].subgridspec(2, 3, wspace=0.50, hspace=0.72)
    axes = [fig.add_subplot(top[i, j]) for i in range(2) for j in range(3)]

    blue = COLORS["primary"]
    gray = COLORS["reference"]

    data = construct["QADD"].sort_values("pause_noise_dbfs")
    axes[0].scatter(
        data["pause_noise_dbfs"],
        data["qadd_pause_ac_level_dbfs_median"],
        s=24,
        color=blue,
        zorder=3,
    )
    lo = float(min(data.min()))
    hi = float(max(data.max()))
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color=gray, label="Identity")
    axes[0].set_title("QADD")
    axes[0].set_xlabel("Added pause-region level (dBFS)")
    axes[0].set_ylabel("Estimated pause-region level (dBFS)")
    axes[0].set_xlim(lo - 1.5, hi + 1.5)
    axes[0].set_ylim(lo - 1.5, hi + 1.5)

    data = construct["QGAIN"].sort_values("dose")
    axes[1].plot(data["dose"], data["response"], marker="o", markersize=4.2, linewidth=1.35, color=blue)
    axes[1].set_title("QGAIN")
    axes[1].set_xlabel("Imposed modulation dose (dB)")
    axes[1].set_ylabel("Within-segment level IQR (dB)")

    data = construct["QREV"].sort_values("rt60_sec")
    axes[2].plot(
        data["rt60_sec"],
        data["qrev_tail_persistence_median_sec"],
        marker="o",
        markersize=4.2,
        linewidth=1.35,
        color=blue,
    )
    axes[2].axhline(0.6, linestyle="--", linewidth=0.9, color=gray)
    axes[2].text(
        0.98,
        0.92,
        "0.6-s observation horizon",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=gray,
    )
    axes[2].set_title("QREV")
    axes[2].set_xlabel("Simulated RIR decay setting (s)")
    axes[2].set_ylabel("Tail persistence (s)")
    axes[2].set_ylim(-0.015, 0.645)

    data = construct["QCHAN"].sort_values("upper_band_reduction_hz")
    axes[3].plot(
        data["upper_band_reduction_hz"],
        data["qchan_rolloff95_deficit_hz"],
        marker="o",
        markersize=4.2,
        linewidth=1.35,
        color=blue,
    )
    axes[3].set_title("QCHAN")
    axes[3].set_xlabel("Applied upper-frequency reduction (Hz)")
    axes[3].set_ylabel("Rolloff-95 deficit (Hz)")

    data = construct["QDIST"]
    qdist_x = 100.0 * data["realized_fraction"]
    qdist_y = 100.0 * data["estimated_sample_fraction"]
    axes[4].scatter(
        qdist_x,
        qdist_y,
        s=14,
        alpha=0.62,
        color=blue,
        edgecolors="none",
        rasterized=True,
    )
    limit = float(max(qdist_x.max(), qdist_y.max()))
    axes[4].plot([0, limit], [0, limit], linestyle="--", linewidth=1.0, color=gray)
    axes[4].set_title("QDIST")
    axes[4].set_xlabel("True altered samples (%)")
    axes[4].set_ylabel("Estimated clipped samples (%)")

    axes[5].axis("off")
    axes[5].text(
        0.5,
        0.73,
        "QTEMP",
        transform=axes[5].transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["text"],
    )
    axes[5].text(
        0.5,
        0.46,
        "No primary indicator retained\n2/519 dropout-like positive\n0/519 frozen-audio positive",
        transform=axes[5].transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        linespacing=1.25,
        color=COLORS["text"],
    )
    axes[5].text(
        0.5,
        0.15,
        "Completed negative feature-selection result",
        transform=axes[5].transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color=gray,
        style="italic",
    )

    for ax in axes[:5]:
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.5)
    fig.text(0.035, 0.935, "A", fontsize=12, fontweight="bold", ha="left", va="top")
    fig.text(0.11, 0.935, "Representative controlled responses", fontsize=9.2, fontweight="semibold", ha="left", va="top")

    # ------------------------------------------------------------------
    # B. Repeated-recording persistence with uncertainty
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[1])
    display = persistence.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    lower_error = display["spearman_rho"] - display["ci_lower"]
    upper_error = display["ci_upper"] - display["spearman_rho"]
    ax_b.errorbar(
        display["spearman_rho"],
        y,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        markersize=5.2,
        color=blue,
        ecolor=blue,
        elinewidth=1.15,
        capsize=2.5,
        capthick=1.0,
        zorder=3,
    )
    ax_b.axvline(0.5, color=gray, linestyle=":", linewidth=0.9)
    ax_b.set_yticks(y, display["label"])
    ax_b.set_xlim(0.25, 0.92)
    ax_b.set_xlabel("First-versus-second recording Spearman correlation (approximate 95% CI)")
    ax_b.set_title("Repeated-recording persistence of principal indicators", loc="left", fontsize=9.2, pad=7)
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax_b.grid(axis="y", visible=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(labelsize=7.5)
    for idx, row in display.iterrows():
        ax_b.text(
            min(row["ci_upper"] + 0.018, 0.89),
            idx,
            f"ρ={row['spearman_rho']:.2f}; n={int(row['paired_participants'])}",
            va="center",
            ha="left",
            fontsize=6.9,
            color=COLORS["text"],
        )
    panel_letter(ax_b, "B")

    # ------------------------------------------------------------------
    # C. Final analytical roles
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[2])
    left = np.zeros(len(roles), dtype=float)
    for role in ROLE_ORDER:
        values = roles[role].to_numpy(dtype=float)
        ax_c.barh(
            roles["family"],
            values,
            left=left,
            height=0.68,
            label=ROLE_LABELS[role],
            color=COLORS[role],
            edgecolor="white",
            linewidth=0.6,
        )
        for row_index, (start, value) in enumerate(zip(left, values)):
            if value > 0:
                ax_c.text(
                    start + value / 2,
                    row_index,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color="black",
                )
        left += values
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, max(5.15, float(left.max()) + 0.15))
    ax_c.set_xlabel("Number of candidate outputs")
    ax_c.set_title("Final analytical roles assigned after validation", loc="left", fontsize=9.2, pad=7)
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax_c.grid(axis="y", visible=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.tick_params(labelsize=7.5)
    ax_c.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=6.6,
        columnspacing=1.25,
        handlelength=1.55,
    )
    panel_letter(ax_c, "C")

    fig.suptitle(
        "Analytical validation and final role assignment of recording-quality indicators",
        fontsize=11.2,
        fontweight="semibold",
        y=0.985,
    )

    outputs: dict[str, Path] = {}
    for suffix in ("png", "svg", "pdf"):
        path = output_main / f"{basename}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else 300, bbox_inches="tight")
        outputs[suffix] = path
    plt.close(fig)

    for family, frame in construct.items():
        frame.to_csv(output_source / f"{basename}_panelA_{family}.csv", index=False)
    persistence.to_csv(output_source / f"{basename}_panelB_persistence_with_ci.csv", index=False)
    roles.to_csv(output_source / f"{basename}_panelC_roles.csv", index=False)

    caption = (
        "Figure X. Analytical validation and final role assignment of signal-derived recording-quality indicators. "
        "(A) Representative controlled-response results. QADD pause-region level tracked added pause-region energy; "
        "QGAIN within-segment variability increased with imposed modulation; QREV tail persistence increased with the "
        "simulated decay setting and was bounded by its 0.6-s observation horizon; QCHAN rolloff-95 deficit increased "
        "with applied upper-frequency reduction; and QDIST estimated sample burden followed the exact altered-sample "
        "burden. QTEMP is shown as a completed negative feature-selection result because no candidate was retained for "
        "primary analysis. (B) First-versus-second recording Spearman correlations for prespecified principal or contextual "
        "indicators. Error bars show approximate 95% confidence intervals obtained uniformly using Fisher's z transformation. "
        "These estimates describe persistence across real recording sessions rather than technical test-retest reliability. "
        "QDIST positive-part persistence was not estimable because positive recordings were sparse. (C) Final analytical "
        "roles assigned after validation. Supporting, exploratory, monitoring, and audit roles preserve measurements with "
        "narrower evidential scope; the QTEMP splice-discontinuity candidate was removed."
    )
    caption_path = output_main / f"{basename}.caption.md"
    caption_path.write_text(caption, encoding="utf-8")
    outputs["caption"] = caption_path

    report = {
        "project_root": str(root),
        "figure": {name: str(path) for name, path in outputs.items()},
        "construct_rows": {family: int(len(frame)) for family, frame in construct.items()},
        "persistence_rows": int(len(persistence)),
        "role_rows": int(len(roles)),
        "ci_method": "Fisher z approximation applied uniformly to Spearman correlations",
    }
    report_path = output_main / f"{basename}.build.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs["build_report"] = report_path

    print("Publication figure created:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


if __name__ == "__main__":
    root_arg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    build_publication_figure(root_arg)
