from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


FAMILIES = (
    "additive_interference",
    "gain_dynamics",
    "reverberation",
    "channel_device",
    "nonlinear_distortion",
    "temporal_discontinuity",
)


def analysis_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pipeline_root(root: Path) -> Path:
    return root.parents[1] / "paper_1_cleanup"


def validate_notebook(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    code_cells = 0
    for index, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = "".join(cell.get("source", []))
        # IPython line/shell magics are valid notebook syntax but not Python grammar.
        python_source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith("%")
            and not (
                line.lstrip().startswith("!")
                and not line.lstrip().startswith("!=")
            )
        )
        compile(python_source, f"{path.name}:cell-{index}", "exec")
    return {"notebook": path.name, "code_cells": code_cells, "status": "PASS"}


def run_preflight(root: Path) -> dict[str, object]:
    pipeline = Path(os.environ.get("PAPER1_PIPELINE_ROOT", default_pipeline_root(root))).resolve()
    data_freeze = Path(
        os.environ.get(
            "PAPER1_DATA_FREEZE_DIR",
            pipeline / "MAIN outputs" / "00_DATA_FREEZE" / "v1",
        )
    ).resolve()
    family_freeze = Path(
        os.environ.get(
            "PAPER1_FAMILY_FREEZE_DIR",
            pipeline / "MAIN outputs" / "02_FEATURE_REVIEWED" / "06_family_freezes",
        )
    ).resolve()
    human_qc = Path(
        os.environ.get(
            "PAPER1_HUMAN_QC_ROOT",
            root.parents[2] / "Data_13072026" / "Bamboo_passage_HumanQC",
        )
    ).resolve()

    required = {
        "pipeline_root": pipeline,
        "data_freeze": data_freeze,
        "family_freeze": family_freeze,
        "human_qc": human_qc,
    }
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    family_missing = [name for name in FAMILIES if not (family_freeze / name).is_dir()]
    notebooks = [validate_notebook(path) for path in sorted((root / "notebooks").glob("*.ipynb"))]
    if len(notebooks) != 7:
        raise RuntimeError(f"Expected 7 canonical notebooks; found {len(notebooks)}")
    result = {
        "analysis_root": str(root),
        "paths": {name: str(path) for name, path in required.items()},
        "missing_paths": missing,
        "missing_family_folders": family_missing,
        "notebooks": notebooks,
        "status": "PASS" if not missing and not family_missing else "FAIL",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate analysis inputs and notebook syntax")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    result = run_preflight(analysis_root())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Preflight: {result['status']}")
        for name, path in result["paths"].items():
            print(f"  {name}: {path}")
        print(f"  notebooks: {len(result['notebooks'])} valid")
        if result["missing_paths"]:
            print("  missing paths:", result["missing_paths"])
        if result["missing_family_folders"]:
            print("  missing families:", result["missing_family_folders"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
