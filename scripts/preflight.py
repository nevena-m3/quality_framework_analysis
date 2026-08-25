from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_analysis.final_pipeline import resolve_inputs


def main() -> int:
    inputs = resolve_inputs(ROOT)
    notebook = ROOT / "notebooks" / "01_run_final_analysis.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    compile_checks = []
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook.name}:cell-{index}", "exec")
            compile_checks.append(index)
    report = {
        "status": "PASS",
        "analysis_root": str(ROOT),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "canonical_notebook": str(notebook),
        "compiled_code_cells": compile_checks,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
