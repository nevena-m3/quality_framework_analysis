from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import ProjectPaths
from .registry import load_yaml


FAMILY_ORDER = ["QADD", "QGAIN", "QREV", "QCHAN", "QDIST", "QTEMP"]


def _read_family_registry(paths: ProjectPaths) -> pd.DataFrame:
    file_path = paths.registries / "family_registry.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Run 01_build_input_registries.py first: {file_path}")
    return pd.read_csv(file_path).set_index("family")


def _source_path(registry: pd.DataFrame, family: str, column: str) -> Path:
    value = registry.loc[family, column]
    if pd.isna(value):
        raise FileNotFoundError(f"No {column} registered for {family}")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_construct_data(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    registry = _read_family_registry(paths)
    result: dict[str, pd.DataFrame] = {}

    qadd = pd.read_csv(_source_path(registry, "QADD", "construct_source_file"))
    qadd = qadd.loc[qadd["control"].eq("pause_noise_dose")].copy()
    result["QADD"] = qadd[["pause_noise_dbfs", "qadd_pause_ac_level_dbfs_median"]].dropna()

    qgain = pd.read_csv(_source_path(registry, "QGAIN", "construct_source_file"))
    qgain = qgain.loc[qgain["mechanism"].eq("amplitude_modulation")].copy()
    result["QGAIN"] = qgain[["dose", "response"]].dropna()

    qrev = pd.read_csv(_source_path(registry, "QREV", "construct_source_file"))
    qrev = qrev.loc[qrev["section"].eq("boundary_rir") & qrev["used_for_ordering"].astype(str).str.lower().eq("true")].copy()
    result["QREV"] = qrev[["rt60_sec", "qrev_tail_persistence_median_sec"]].dropna()

    qchan = pd.read_csv(_source_path(registry, "QCHAN", "construct_source_file"))
    qchan = qchan.loc[qchan["condition"].eq("lowpass")].copy()
    qchan["upper_band_reduction_hz"] = qchan["dose"].max() - qchan["dose"]
    result["QCHAN"] = qchan[["upper_band_reduction_hz", "qchan_rolloff95_deficit_hz"]].dropna()

    qdist = pd.read_csv(_source_path(registry, "QDIST", "construct_source_file"))
    result["QDIST"] = qdist[["realized_fraction", "estimated_sample_fraction"]].dropna()

    return result


def _extract_persistence_row(source: Path, feature: str) -> dict[str, Any]:
    frame = pd.read_csv(source)
    feature_column = "feature" if "feature" in frame.columns else "metric"
    row = frame.loc[frame[feature_column].astype(str).eq(feature)]
    if row.empty:
        return {"feature": feature, "spearman": np.nan, "n": np.nan}
    row = row.iloc[0]
    spearman_col = next(
        (column for column in ("first_second_spearman", "positive_part_spearman_rho") if column in frame.columns),
        None,
    )
    n_col = next(
        (
            column
            for column in (
                "paired_subject_count",
                "first_second_pair_count",
                "participant_pair_count",
                "subjects_with_repeats",
            )
            if column in frame.columns
        ),
        None,
    )
    return {
        "feature": feature,
        "spearman": row[spearman_col] if spearman_col else np.nan,
        "n": row[n_col] if n_col else np.nan,
    }


def prepare_persistence_data(paths: ProjectPaths, config: dict[str, Any]) -> pd.DataFrame:
    registry = _read_family_registry(paths)
    rows = []
    for item in config["panel_b"]["features"]:
        family = item["family"]
        source = _source_path(registry, family, "persistence_source_file")
        values = _extract_persistence_row(source, item["feature"])
        rows.append({"family": family, "label": item["label"], **values, "source_file": str(source)})
    return pd.DataFrame(rows)


def prepare_role_counts(paths: ProjectPaths) -> pd.DataFrame:
    feature_registry = pd.read_csv(paths.registries / "feature_registry.csv")
    feature_registry = feature_registry.loc[~feature_registry["is_companion_status"].astype(bool)].copy()
    categories = [
        "principal_or_contextual",
        "supporting_or_comparator",
        "exploratory_monitoring_or_audit",
        "dropped",
    ]
    table = (
        feature_registry.groupby(["family", "role_group"]).size().unstack(fill_value=0).reindex(FAMILY_ORDER, fill_value=0)
    )
    for category in categories:
        if category not in table.columns:
            table[category] = 0
    table = table[categories].reset_index()
    return table


def _panel_letter(fig: plt.Figure, x: float, y: float, letter: str) -> None:
    fig.text(x, y, letter, fontsize=14, fontweight="bold", va="top", ha="left")


def build_figure31(project_root: str | Path) -> dict[str, Path]:
    paths = ProjectPaths.from_root(project_root)
    paths.ensure()
    config = load_yaml(paths.root / "config" / "figure31.yml")
    construct = prepare_construct_data(paths)
    persistence = prepare_persistence_data(paths, config)
    roles = prepare_role_counts(paths)

    figure_cfg = config["figure"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(figure_cfg["width_in"], figure_cfg["height_in"]))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[1.08, 0.92], hspace=0.42, wspace=0.35)
    top = outer[0, :].subgridspec(2, 3, hspace=0.55, wspace=0.38)

    # A: representative controlled responses
    axes = [fig.add_subplot(top[i, j]) for i in range(2) for j in range(3)]

    data = construct["QADD"]
    axes[0].plot(data["pause_noise_dbfs"], data["qadd_pause_ac_level_dbfs_median"], marker="o")
    axes[0].set_title("QADD")
    axes[0].set_xlabel(config["panel_a"]["qadd"]["x_label"])
    axes[0].set_ylabel(config["panel_a"]["qadd"]["y_label"])

    data = construct["QGAIN"]
    axes[1].plot(data["dose"], data["response"], marker="o")
    axes[1].set_title("QGAIN")
    axes[1].set_xlabel(config["panel_a"]["qgain"]["x_label"])
    axes[1].set_ylabel(config["panel_a"]["qgain"]["y_label"])

    data = construct["QREV"]
    axes[2].plot(data["rt60_sec"], data["qrev_tail_persistence_median_sec"], marker="o")
    axes[2].set_title("QREV")
    axes[2].set_xlabel(config["panel_a"]["qrev"]["x_label"])
    axes[2].set_ylabel(config["panel_a"]["qrev"]["y_label"])

    data = construct["QCHAN"]
    axes[3].plot(data["upper_band_reduction_hz"], data["qchan_rolloff95_deficit_hz"], marker="o")
    axes[3].set_title("QCHAN")
    axes[3].set_xlabel(config["panel_a"]["qchan"]["x_label"])
    axes[3].set_ylabel(config["panel_a"]["qchan"]["y_label"])

    data = construct["QDIST"]
    axes[4].scatter(data["realized_fraction"], data["estimated_sample_fraction"], s=12, alpha=0.65)
    limit = float(max(data["realized_fraction"].max(), data["estimated_sample_fraction"].max()))
    axes[4].plot([0, limit], [0, limit], linestyle="--", linewidth=1)
    axes[4].set_title("QDIST")
    axes[4].set_xlabel(config["panel_a"]["qdist"]["x_label"])
    axes[4].set_ylabel(config["panel_a"]["qdist"]["y_label"])

    axes[5].axis("off")
    axes[5].text(
        0.5,
        0.62,
        "QTEMP",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=axes[5].transAxes,
    )
    axes[5].text(
        0.5,
        0.36,
        "No primary indicator retained\n2/519 dropout-like positive\n0/519 frozen-audio positive",
        ha="center",
        va="center",
        fontsize=10,
        transform=axes[5].transAxes,
    )
    for axis in axes[:5]:
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.grid(alpha=0.2)

    # B: persistence
    ax_b = fig.add_subplot(outer[1, 0])
    display = persistence.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    ax_b.scatter(display["spearman"], y, s=42)
    for idx, row in display.iterrows():
        ax_b.text(row["spearman"] + 0.02, idx, f"ρ={row['spearman']:.2f}; n={int(row['n'])}", va="center", fontsize=8)
    ax_b.set_yticks(y, display["label"])
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xlabel("First-versus-second recording Spearman correlation")
    ax_b.set_title(config["panel_b"]["title"], loc="left", fontsize=11)
    ax_b.grid(axis="x", alpha=0.2)
    ax_b.text(
        0,
        -0.22,
        "QDIST positive-part persistence was not estimable; QTEMP contributed no primary indicator.",
        transform=ax_b.transAxes,
        fontsize=8,
        va="top",
    )

    # C: final role counts
    ax_c = fig.add_subplot(outer[1, 1])
    categories = config["panel_c"]["categories"]
    labels = {
        "principal_or_contextual": "Principal/contextual",
        "supporting_or_comparator": "Supporting/comparator",
        "exploratory_monitoring_or_audit": "Exploratory/monitoring/audit",
        "dropped": "Dropped",
    }
    left = np.zeros(len(roles))
    for category in categories:
        values = roles[category].to_numpy(dtype=float)
        ax_c.barh(roles["family"], values, left=left, label=labels[category])
        for row_idx, (start, value) in enumerate(zip(left, values)):
            if value > 0:
                ax_c.text(start + value / 2, row_idx, str(int(value)), ha="center", va="center", fontsize=8)
        left += values
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Number of candidate outputs")
    ax_c.set_title(config["panel_c"]["title"], loc="left", fontsize=11)
    ax_c.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax_c.grid(axis="x", alpha=0.2)
    ax_c.text(
        0,
        -0.39,
        "QDIST occurrence is a companion status and is not counted as an independent output.",
        transform=ax_c.transAxes,
        fontsize=8,
        va="top",
    )

    fig.suptitle(figure_cfg["title"], fontsize=14, y=0.985)
    _panel_letter(fig, 0.015, 0.955, "A")
    _panel_letter(fig, 0.015, 0.405, "B")
    _panel_letter(fig, 0.555, 0.405, "C")

    basename = figure_cfg["basename"]
    outputs = {}
    for suffix in ("png", "svg", "pdf"):
        output = paths.figures_main / f"{basename}.{suffix}"
        fig.savefig(output, dpi=figure_cfg["dpi"], bbox_inches="tight")
        outputs[suffix] = output
    plt.close(fig)

    # Save source data for complete editability and audit.
    for family, frame in construct.items():
        frame.to_csv(paths.figures_source / f"{basename}_panelA_{family}.csv", index=False)
    persistence.to_csv(paths.figures_source / f"{basename}_panelB_persistence.csv", index=False)
    roles.to_csv(paths.figures_source / f"{basename}_panelC_roles.csv", index=False)

    caption = (
        "Figure 3.1. Analytical validation of signal-derived recording-quality indicators. "
        "(A) Representative controlled-response results for the five families contributing retained primary or supporting measurements; "
        "QTEMP is shown as a completed negative feature-selection result. "
        "(B) Repeated-recording persistence of prespecified principal indicators, summarized using first-versus-second recording Spearman correlations. "
        "These values describe empirical persistence across sessions rather than technical test-retest reliability. "
        "(C) Final analytical roles assigned after validation. Supporting, exploratory, monitoring, and audit roles preserve measurements with narrower evidential scope, while failed candidates were removed."
    )
    caption_path = paths.figures_main / f"{basename}.caption.md"
    caption_path.write_text(caption, encoding="utf-8")
    outputs["caption"] = caption_path

    report = {
        "figure": {key: str(value) for key, value in outputs.items()},
        "construct_rows": {family: len(frame) for family, frame in construct.items()},
        "persistence_rows": len(persistence),
        "role_rows": len(roles),
    }
    report_path = paths.figures_main / f"{basename}.build.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs["build_report"] = report_path
    return outputs
