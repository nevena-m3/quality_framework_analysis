from __future__ import annotations

import argparse
import json
from pathlib import Path

from rq_analysis.publication_verify import (
    PublicationVerificationError,
    verify_publication_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify frozen Paper-1 publication results against golden aggregate assertions."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root containing canonical notebook output directories.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    try:
        result = verify_publication_outputs(args.outputs_root)
    except (PublicationVerificationError, FileNotFoundError, KeyError, ValueError) as exc:
        result = {
            "analysis_outputs_root": str(args.outputs_root.resolve()),
            "checks": [],
            "n_checks": 0,
            "n_passed": 0,
            "status": "FAIL",
            "error": str(exc),
        }

    text = json.dumps(result, indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
