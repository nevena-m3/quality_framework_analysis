from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HUMAN_COLUMNS = {
    "Environmental noise": "human_environmental_noise_fraction",
    "Competing speech": "human_competing_speech_fraction",
    "Volume unstable": "human_volume_unstable_fraction",
    "Clipping": "human_clipping_fraction",
    "Reverberation/echo": "human_reverberation_echo_fraction",
    "Platform effects": "human_platform_effects_fraction",
    "Temporal discontinuities": "human_temporal_discontinuities_fraction",
}

HUMAN_FAMILY = {
    "human_environmental_noise_fraction": "QADD",
    "human_competing_speech_fraction": "QADD",
    "human_volume_unstable_fraction": "QGAIN",
    "human_reverberation_echo_fraction": "QREV",
    "human_platform_effects_fraction": "QCHAN",
    "human_clipping_fraction": "QDIST",
    "human_temporal_discontinuities_fraction": "QTEMP",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result.loc[order] = np.clip(ranked, 0, 1)
    return result


def _intervals(value: object) -> list[tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    output = []
    for entries in payload.values():
        for item in entries or []:
            if len(item) == 2 and float(item[1]) > float(item[0]):
                output.append((float(item[0]), float(item[1])))
    return output


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = [list(item) for item in sorted(intervals)]
    output = [merged[0]]
    for start, end in merged[1:]:
        if start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], end)
        else:
            output.append([start, end])
    return float(sum(end - start for start, end in output))


def load_human_qc(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, failures = [], []
    for path in sorted(root.rglob("*_segments.csv")):
        try:
            frame = pd.read_csv(path)
            duration = float(frame["offset_seconds_absolute"].max())
            row = {
                "logical_recording_id": path.stem.removesuffix("_segments"),
                "human_rater": path.parent.name,
            }
            for source, target in HUMAN_COLUMNS.items():
                intervals = [interval for value in frame[source] for interval in _intervals(value)]
                burden = _union_duration(intervals)
                row[target] = burden / duration if duration > 0 else np.nan
            rows.append(row)
        except Exception as exc:  # failure ledger is part of the output contract
            failures.append({"path": str(path), "error": repr(exc)})
    long = pd.DataFrame(rows)
    human = [column for column in long if column.startswith("human_") and column != "human_rater"]
    consensus = long.groupby("logical_recording_id", as_index=False)[human].mean()
    counts = long.groupby("logical_recording_id").size().rename("human_rater_count")
    return consensus.merge(counts, on="logical_recording_id"), pd.DataFrame(failures)


def resolve_inputs(project_root: Path) -> dict[str, Path]:
    final_root = Path(
        os.environ.get("PAPER1_FINAL_CLEANUP_ROOT", project_root.parent / "Paper_1_Final_Cleanup")
    ).resolve()
    human_root = Path(
        os.environ.get(
            "PAPER1_HUMAN_QC_ROOT",
            project_root.parent.parent / "Data_13072026" / "Bamboo_passage_HumanQC",
        )
    ).resolve()
    paths = {
        "final_root": final_root,
        "features": final_root / "outputs" / "features" / "recording_features.csv",
        "registry": final_root / "outputs" / "features" / "feature_registry.csv",
        "data_freeze": final_root / "outputs" / "data" / "frozen_bamboo_recordings.csv",
        "segmentation": final_root / "outputs" / "segments" / "frozen_segmentation_decisions.csv",
        "human_qc": human_root,
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing governed inputs: {missing}")
    return paths


def _feature_summary(data: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.itertuples(index=False):
        values = pd.to_numeric(data[item.feature], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append({
            "family": item.family_code,
            "feature": item.feature,
            "role": item.role,
            "unit": item.unit,
            "measurement_version": item.measurement_version,
            "n": len(data),
            "available_n": len(finite),
            "available_fraction": len(finite) / len(data),
            "zero_fraction_among_available": float(finite.eq(0).mean()) if len(finite) else np.nan,
            "mean": finite.mean(),
            "std": finite.std(),
            "median": finite.median(),
            "q1": finite.quantile(0.25),
            "q3": finite.quantile(0.75),
            "min": finite.min(),
            "max": finite.max(),
        })
    return pd.DataFrame(rows)


def _persistence(data: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["SubjectID", "recording_date_analysis", "logical_recording_id"])
    repeated = ordered.groupby("SubjectID").filter(lambda frame: len(frame) >= 2)
    # Align the two observations by participant, not by their original row
    # indices.  ``GroupBy.nth`` preserves source indices, so constructing a
    # frame directly from its outputs otherwise yields no paired rows.
    first = repeated.groupby("SubjectID").nth(0).set_index("SubjectID")
    second = repeated.groupby("SubjectID").nth(1).set_index("SubjectID")
    rows = []
    for item in registry.itertuples(index=False):
        pair = pd.DataFrame({"first": first[item.feature], "second": second[item.feature]}).dropna()
        rho = p = np.nan
        if len(pair) >= 10 and pair.nunique().min() > 1:
            result = stats.spearmanr(pair["first"], pair["second"])
            rho, p = float(result.statistic), float(result.pvalue)
        rows.append({"family": item.family_code, "feature": item.feature, "participant_pairs": len(pair), "spearman_rho": rho, "p": p})
    result = pd.DataFrame(rows)
    result["p_bh"] = _bh(result["p"])
    return result


def _cohort_contrasts(data: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.itertuples(index=False):
        participant = data.groupby(["SubjectID", "diagnosis_analysis"], as_index=False)[item.feature].median()
        als = participant.loc[participant["diagnosis_analysis"].eq("ALS"), item.feature].dropna()
        controls = participant.loc[participant["diagnosis_analysis"].eq("CONTROLS"), item.feature].dropna()
        statistic = p = effect = np.nan
        if len(als) >= 10 and len(controls) >= 10:
            test = stats.mannwhitneyu(als, controls, alternative="two-sided")
            statistic, p = float(test.statistic), float(test.pvalue)
            effect = 2 * statistic / (len(als) * len(controls)) - 1
        rows.append({"family": item.family_code, "feature": item.feature, "als_participants": len(als), "control_participants": len(controls), "als_median": als.median(), "control_median": controls.median(), "rank_biserial_als_minus_control": effect, "p": p})
    result = pd.DataFrame(rows)
    result["p_bh"] = _bh(result["p"])
    return result


def _human_correspondence(data: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    targets = [column for column in HUMAN_FAMILY if column in data]
    rows = []
    for item in registry.itertuples(index=False):
        for target in targets:
            pair = data[[item.feature, target]].dropna()
            rho = p = np.nan
            if len(pair) >= 20 and pair.nunique().min() > 1:
                result = stats.spearmanr(pair[item.feature], pair[target])
                rho, p = float(result.statistic), float(result.pvalue)
            rows.append({"feature_family": item.family_code, "feature": item.feature, "human_dimension": target, "human_family": HUMAN_FAMILY[target], "relationship": "construct_matched" if item.family_code == HUMAN_FAMILY[target] else "off_target_control", "n": len(pair), "spearman_rho": rho, "p": p})
    result = pd.DataFrame(rows)
    result["p_bh"] = _bh(result["p"])
    return result


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str, cmap: str = "coolwarm") -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def run_final_analysis(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    paths = resolve_inputs(root)
    output = root / "outputs"
    tables, figures, logs = output / "tables", output / "figures", output / "logs"
    for folder in (tables, figures, logs):
        folder.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(paths["features"])
    registry = pd.read_csv(paths["registry"])
    metadata = pd.read_csv(paths["data_freeze"])
    segmentation = pd.read_csv(paths["segmentation"])
    human, human_failures = load_human_qc(paths["human_qc"])
    feature_names = registry.loc[registry["analysis_eligible"].astype(bool), "feature"].tolist()
    if len(features) != 519 or features["logical_recording_id"].nunique() != 519:
        raise RuntimeError("The canonical feature release must contain 519 unique recordings.")
    if len(feature_names) != 20:
        raise RuntimeError(f"Expected 20 analysis features; found {len(feature_names)}")

    metadata_columns = ["logical_recording_id", "diagnosis_analysis", "ALSFRS bulbar subscore", "Speaking rate", "Sentence intelligibility percent"]
    metadata = metadata[[column for column in metadata_columns if column in metadata]].drop_duplicates("logical_recording_id")
    segmentation = segmentation[["logical_recording_id", "segmentation_analysis_eligible", "decision"]].drop_duplicates("logical_recording_id")
    data = features.merge(metadata, on="logical_recording_id", how="left", validate="one_to_one")
    data = data.merge(segmentation, on="logical_recording_id", how="left", validate="one_to_one")
    data = data.merge(human, on="logical_recording_id", how="left", validate="one_to_one")

    summary = _feature_summary(data, registry)
    persistence = _persistence(data, registry)
    contrasts = _cohort_contrasts(data, registry)
    human_matrix = _human_correspondence(data, registry)
    correlations = data[feature_names].corr(method="spearman", min_periods=30)

    x_frame = data[feature_names].apply(pd.to_numeric, errors="coerce")
    x_frame = x_frame.fillna(x_frame.median())
    scale = x_frame.std(ddof=0).replace(0, 1)
    x = ((x_frame - x_frame.mean()) / scale).to_numpy(dtype=float)
    _u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    variance = singular_values**2 / max(1, len(x) - 1)
    variance_ratio = variance / variance.sum()
    loadings = pd.DataFrame(vt.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(feature_names))])
    explained = pd.DataFrame({"component": loadings.columns, "explained_variance_ratio": variance_ratio, "cumulative_explained_variance": np.cumsum(variance_ratio)})

    summary.to_csv(tables / "feature_summary.csv", index=False)
    persistence.to_csv(tables / "participant_persistence.csv", index=False)
    contrasts.to_csv(tables / "diagnosis_contrasts_exploratory.csv", index=False)
    human_matrix.to_csv(tables / "human_correspondence.csv", index=False)
    correlations.to_csv(tables / "feature_spearman_matrix.csv")
    loadings.to_csv(tables / "pca_loadings.csv")
    explained.to_csv(tables / "pca_explained_variance.csv", index=False)
    human_failures.to_csv(logs / "human_qc_parse_failures.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    for family, frame in summary.groupby("family", sort=False):
        ax.scatter(frame["available_fraction"], frame["feature"], label=family, s=45)
    ax.set(xlabel="Available fraction", ylabel="Feature", xlim=(0, 1.02), title="Feature availability in the 519-recording reviewed release")
    ax.grid(axis="x", alpha=0.25); ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(figures / "01_feature_availability.png", dpi=300); fig.savefig(figures / "01_feature_availability.svg"); plt.close(fig)

    _save_heatmap(correlations, figures / "02_feature_correlation_heatmap.png", "Recording-level feature structure")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(explained) + 1), explained["cumulative_explained_variance"], marker="o")
    ax.axhline(0.8, color="grey", linestyle="--"); ax.set(xlabel="Principal component", ylabel="Cumulative explained variance", ylim=(0, 1.02), title="PCA dimensionality of retained quality features"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "03_pca_explained_variance.png", dpi=300); fig.savefig(figures / "03_pca_explained_variance.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    display = persistence.sort_values("spearman_rho")
    ax.barh(display["feature"], display["spearman_rho"], color="#4C78A8")
    ax.axvline(0, color="black", linewidth=0.8); ax.set(xlabel="First-versus-second recording Spearman rho", title="Participant-level persistence (not technical reliability)"); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "04_participant_persistence.png", dpi=300); fig.savefig(figures / "04_participant_persistence.svg"); plt.close(fig)

    human_pivot = human_matrix.pivot(index="feature", columns="human_dimension", values="spearman_rho")
    _save_heatmap(human_pivot, figures / "05_human_correspondence_heatmap.png", "Feature correspondence with independent human QC")

    manifest = {
        "status": "complete",
        "recordings": len(data),
        "participants": int(data["SubjectID"].nunique()),
        "features": len(feature_names),
        "human_qc_recordings": int(data["human_rater_count"].notna().sum()),
        "inputs": {name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None} for name, path in paths.items()},
        "tables": sorted(path.name for path in tables.glob("*.csv")),
        "figures": sorted(path.name for path in figures.glob("*.*")),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    root = Path(os.environ.get("PAPER1_ANALYSIS_ROOT", Path.cwd())).resolve()
    print(json.dumps(run_final_analysis(root), indent=2))
    return 0
