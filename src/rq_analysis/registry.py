from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .paths import ProjectPaths


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _first_match(root: Path, patterns: Iterable[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(root.glob(pattern)))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    # Prefer non-source and final files when several variants exist.
    def score(path: Path) -> tuple[int, int, int, str]:
        text = str(path).lower()
        return (
            int("final" in text or "frozen" in text),
            int("source_" not in text and "source_candidate" not in text),
            -len(path.parts),
            str(path),
        )
    return sorted(candidates, key=score, reverse=True)[0]


def _resolve_family_root(freezes_root: Path, folder: str, version_prefix: str) -> Path:
    family_dir = freezes_root / folder
    if not family_dir.exists():
        raise FileNotFoundError(f"Missing family folder: {family_dir}")
    candidates = sorted(
        [p for p in family_dir.iterdir() if p.is_dir() and p.name.startswith(version_prefix)]
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {version_prefix} directory under {family_dir}; found {candidates}"
        )
    return candidates[0]


def _standardize_feature_decisions(family: str, source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    lower = {column.lower(): column for column in frame.columns}

    def col(*names: str) -> pd.Series:
        for name in names:
            if name.lower() in lower:
                return frame[lower[name.lower()]]
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")

    feature = col("feature")
    display_name = col("display_name")
    display_name = display_name.where(display_name.notna(), feature)
    decision = col("final_decision", "decision")
    role = col("publication_role", "final_role", "role")
    default = col("default_manuscript_inclusion", "model_default", "primary_validated_set")
    analysis_priority = col("analysis_priority", "analysis_use")
    unit = col("unit")
    rationale = col("rationale", "publication_claim")
    permitted = col("permitted_interpretation", "permitted_claim")
    prohibited = col("prohibited_interpretation", "prohibited_claim", "claim_limit")

    out = pd.DataFrame(
        {
            "family": family,
            "feature": feature,
            "display_name": display_name,
            "decision": decision,
            "publication_role": role,
            "default_manuscript_inclusion": default,
            "analysis_priority": analysis_priority,
            "unit": unit,
            "rationale": rationale,
            "permitted_interpretation": permitted,
            "prohibited_interpretation": prohibited,
            "source_file": str(source),
        }
    )
    out["default_manuscript_inclusion"] = out["default_manuscript_inclusion"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if pd.notna(value)
        else False
    )
    out["role_group"] = [classify_role(family, row) for _, row in out.iterrows()]
    out["is_companion_status"] = out["role_group"].eq("companion_status")
    return out


def classify_role(family: str, row: pd.Series) -> str:
    decision = str(row.get("decision", "")).strip().lower()
    role = str(row.get("publication_role", "")).strip().lower()
    priority = str(row.get("analysis_priority", "")).strip().lower()
    text = " ".join((decision, role, priority))
    # Do not classify the word "dropout" as a dropped feature. Only explicit
    # DROP decisions or a role explicitly labelled dropped are removed.
    if decision.startswith("drop") or role == "dropped" or "failed_analytical_validation" in decision:
        return "dropped"
    if "companion" in text or ("status" in text and family == "QDIST"):
        return "companion_status"
    if any(token in text for token in ("exploratory", "monitoring", "audit")):
        return "exploratory_monitoring_or_audit"
    if any(token in text for token in ("secondary", "comparator", "targeted")):
        return "supporting_or_comparator"
    if any(token in text for token in ("primary", "contextual")):
        return "principal_or_contextual"
    return "other"


def _standardize_gate_table(family: str, source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    lower = {column.lower(): column for column in frame.columns}
    gate_col = next((lower[key] for key in ("gate", "domain", "id") if key in lower), None)
    status_col = next(
        (lower[key] for key in ("status", "state", "final_state", "final status") if key in lower),
        None,
    )
    conclusion_col = next(
        (lower[key] for key in ("conclusion", "evidence", "scientific_conclusion") if key in lower),
        None,
    )
    out = pd.DataFrame(
        {
            "family": family,
            "gate": frame[gate_col] if gate_col else pd.Series(range(1, len(frame) + 1)),
            "status": frame[status_col] if status_col else pd.NA,
            "conclusion": frame[conclusion_col] if conclusion_col else pd.NA,
            "source_file": str(source),
        }
    )
    return out


def _standardize_dashboard(family: str, source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    lower = {column.lower(): column for column in frame.columns}
    domain_col = next(
        (lower[key] for key in ("domain", "validation_domain", "scientific_domain") if key in lower),
        None,
    )
    number_col = next((lower[key] for key in ("domain_number", "#", "number") if key in lower), None)
    status_col = next(
        (lower[key] for key in ("status", "state", "final_status", "final status") if key in lower),
        None,
    )
    conclusion_col = next(
        (lower[key] for key in ("conclusion", "scientific_conclusion") if key in lower), None
    )
    return pd.DataFrame(
        {
            "family": family,
            "domain_number": frame[number_col] if number_col else range(1, len(frame) + 1),
            "domain": frame[domain_col] if domain_col else pd.NA,
            "status": frame[status_col] if status_col else pd.NA,
            "conclusion": frame[conclusion_col] if conclusion_col else pd.NA,
            "source_file": str(source),
        }
    )


def _checklist_summary(family: str, source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    status_col = next((c for c in frame.columns if c.lower() in {"status", "state"}), None)
    if status_col is None:
        return pd.DataFrame(
            [{"family": family, "status": "UNKNOWN", "n": len(frame), "source_file": str(source)}]
        )
    counts = frame[status_col].fillna("BLANK").astype(str).value_counts(dropna=False)
    return pd.DataFrame(
        {
            "family": family,
            "status": counts.index,
            "n": counts.values,
            "source_file": str(source),
        }
    )


def build_registries(project_root: str | Path) -> dict[str, Path]:
    paths = ProjectPaths.from_root(project_root)
    paths.ensure()
    config = load_yaml(paths.root / "config" / "families.yml")["families"]

    family_rows: list[dict[str, Any]] = []
    feature_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    dashboard_frames: list[pd.DataFrame] = []
    checklist_frames: list[pd.DataFrame] = []
    figure_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []

    for family, cfg in config.items():
        family_root = _resolve_family_root(
            paths.family_freezes, cfg["folder"], cfg["expected_version_prefix"]
        )
        version = family_root.name
        sources: dict[str, Path | None] = {}
        for source_type, pattern_key in (
            ("feature_decisions", "feature_decision_patterns"),
            ("gate_summary", "gate_patterns"),
            ("checklist", "checklist_patterns"),
            ("dashboard", "dashboard_patterns"),
            ("construct_source", "construct_source_patterns"),
            ("persistence_source", "persistence_source_patterns"),
        ):
            sources[source_type] = _first_match(family_root, cfg.get(pattern_key, []))

        family_rows.append(
            {
                "family": family,
                "display_name": cfg["display_name"],
                "version": version,
                "family_root": str(family_root),
                **{f"{key}_file": str(value) if value else pd.NA for key, value in sources.items()},
            }
        )

        if sources["feature_decisions"]:
            feature_frames.append(_standardize_feature_decisions(family, sources["feature_decisions"]))
        if sources["gate_summary"]:
            gate_frames.append(_standardize_gate_table(family, sources["gate_summary"]))
        if sources["dashboard"]:
            dashboard_frames.append(_standardize_dashboard(family, sources["dashboard"]))
        if sources["checklist"]:
            checklist_frames.append(_checklist_summary(family, sources["checklist"]))

        for file_path in family_root.rglob("*"):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            relative = file_path.relative_to(family_root)
            lower_path = str(relative).lower().replace("\\", "/")
            if "/figures/" in f"/{lower_path}" or lower_path.startswith("figures/"):
                figure_rows.append(
                    {
                        "family": family,
                        "version": version,
                        "relative_path": str(relative),
                        "artifact_type": suffix.lstrip("."),
                        "is_source_data": suffix in {".csv", ".parquet", ".json"},
                        "is_vector": suffix in {".svg", ".pdf"},
                        "is_raster": suffix in {".png", ".jpg", ".jpeg"},
                        "absolute_path": str(file_path),
                    }
                )
            if suffix in {".csv", ".parquet"} and any(
                token in lower_path for token in ("recording_feature", "model_interface", "measurements_long", "exploratory_features")
            ):
                input_rows.append(
                    {
                        "family": family,
                        "version": version,
                        "relative_path": str(relative),
                        "absolute_path": str(file_path),
                        "format": suffix.lstrip("."),
                        "size_bytes": file_path.stat().st_size,
                    }
                )

    outputs: dict[str, Path] = {}
    registry_map = {
        "family_registry.csv": pd.DataFrame(family_rows),
        "feature_registry.csv": pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(),
        "gate_registry.csv": pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame(),
        "dashboard_registry.csv": pd.concat(dashboard_frames, ignore_index=True) if dashboard_frames else pd.DataFrame(),
        "checklist_summary.csv": pd.concat(checklist_frames, ignore_index=True) if checklist_frames else pd.DataFrame(),
        "figure_registry.csv": pd.DataFrame(figure_rows),
        "recording_table_registry.csv": pd.DataFrame(input_rows),
    }
    for filename, frame in registry_map.items():
        output = paths.registries / filename
        frame.to_csv(output, index=False)
        outputs[filename] = output

    build_report = {
        "family_count": len(family_rows),
        "feature_count": int(len(registry_map["feature_registry.csv"])),
        "figure_artifact_count": int(len(registry_map["figure_registry.csv"])),
        "recording_table_candidate_count": int(len(registry_map["recording_table_registry.csv"])),
        "files": {name: str(path) for name, path in outputs.items()},
    }
    report_path = paths.registries / "registry_build_report.json"
    report_path.write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    outputs["registry_build_report.json"] = report_path
    return outputs
