from pathlib import Path

import numpy as np
import pandas as pd

from rq_analysis.final_pipeline import _bh, _intervals, _union_duration, resolve_inputs


def test_bh_is_bounded_and_rank_monotone() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03])
    adjusted = _bh(p_values)
    assert adjusted.between(0, 1).all()
    assert np.all(np.diff(adjusted.loc[p_values.sort_values().index]) >= 0)


def test_interval_union_avoids_double_counting() -> None:
    assert _intervals('{"r": [[0, 2], [1, 3]]}') == [(0.0, 2.0), (1.0, 3.0)]
    assert _union_duration([(0, 2), (1, 3), (5, 6)]) == 4


def test_final_cleanup_contract_is_resolvable() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = resolve_inputs(root)
    assert paths["features"].name == "recording_features.csv"
    assert paths["registry"].name == "feature_registry.csv"
