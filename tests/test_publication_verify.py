from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rq_analysis.publication_verify import (
    EXPECTED_INPUT_SHA256,
    PublicationVerificationError,
    _close,
    _verify_hashes,
    verify_publication_outputs,
)


def test_close_uses_absolute_tolerance_only() -> None:
    assert _close(0.0918244, 0.091824, atol=1e-6)
    assert not _close(0.0919, 0.091824, atol=1e-6)


def test_hash_check_is_path_independent() -> None:
    table = pd.DataFrame(
        [
            {
                "input": name,
                "path": f"C:/arbitrary/{name}",
                "sha256": digest,
                "bytes": 1,
            }
            for name, digest in EXPECTED_INPUT_SHA256.items()
        ]
    )
    report: list[dict[str, object]] = []
    _verify_hashes(report, table, "fixture")
    assert report == [
        {
            "check": "fixture: frozen input hashes",
            "passed": True,
            "detail": f"observed={EXPECTED_INPUT_SHA256}",
        }
    ]


def test_missing_output_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(PublicationVerificationError, match="Missing required output"):
        verify_publication_outputs(tmp_path)

def test_serialized_corrected_permutation_values() -> None:
    # Notebook 03 uses (1 + exceedances) / (1 + valid permutations), while
    # its CSV output is serialized to 8 decimal places.
    assert _close(0.00169983, 17 / 10001, atol=5e-9)
    assert _close(0.00019998, 2 / 10001, atol=5e-9)
    # The manuscript-rounded values are intentionally too coarse for this gate.
    assert not _close(0.0017, 17 / 10001, atol=5e-9)
    assert not _close(0.0002, 2 / 10001, atol=5e-9)

