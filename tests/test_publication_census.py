from __future__ import annotations

import pandas as pd
import pytest

from rq_analysis.registry import (
    PUBLICATION_EXPECTED_CANDIDATE_COUNTS,
    classify_role,
    publication_candidate_registry,
)


def make_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, count in PUBLICATION_EXPECTED_CANDIDATE_COUNTS.items():
        for index in range(count):
            rows.append(
                {
                    "family": family,
                    "feature": f"{family.lower()}_{index}",
                    "role_group": (
                        "exploratory_monitoring_or_audit"
                        if family == "QTEMP"
                        else "principal_or_contextual"
                    ),
                    "is_companion_status": False,
                }
            )

    rows.append(
        {
            "family": "QDIST",
            "feature": "qdist_occurrence_status",
            "role_group": "companion_status",
            "is_companion_status": True,
        }
    )
    rows.append(
        {
            "family": "QTEMP",
            "feature": "qtemp_splice_discontinuity_rate",
            "role_group": "dropped",
            "is_companion_status": False,
        }
    )
    return pd.DataFrame(rows)


def test_publication_candidate_registry_is_exactly_24() -> None:
    full = make_registry()
    assert len(full) == 26

    publication = publication_candidate_registry(full)

    assert len(publication) == 24
    assert publication.groupby("family").size().to_dict() == (
        PUBLICATION_EXPECTED_CANDIDATE_COUNTS
    )
    assert "qdist_occurrence_status" not in set(publication["feature"])
    assert "qtemp_splice_discontinuity_rate" not in set(publication["feature"])


def test_publication_candidate_registry_fails_closed_on_wrong_census() -> None:
    broken = make_registry().query("feature != 'qadd_0'").copy()
    with pytest.raises(RuntimeError, match="candidate census mismatch"):
        publication_candidate_registry(broken)


def test_dropout_name_is_not_misclassified_as_dropped() -> None:
    row = pd.Series(
        {
            "decision": "retain",
            "publication_role": "exploratory monitoring",
            "analysis_priority": "dropout-like event monitor",
        }
    )
    assert classify_role("QTEMP", row) == "exploratory_monitoring_or_audit"


def test_explicit_drop_is_classified_as_dropped() -> None:
    row = pd.Series(
        {
            "decision": "DROP_failed_analytical_validation",
            "publication_role": "dropped",
            "analysis_priority": "",
        }
    )
    assert classify_role("QTEMP", row) == "dropped"
