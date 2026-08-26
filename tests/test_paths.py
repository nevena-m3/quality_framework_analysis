from __future__ import annotations

from pathlib import Path

from rq_analysis.paths import ProjectPaths


def test_validation_freeze_environment_override(monkeypatch, tmp_path: Path) -> None:
    governed = tmp_path / "governed_validation"
    governed.mkdir()

    monkeypatch.setenv("PAPER1_VALIDATION_FREEZE_DIR", str(governed))
    paths = ProjectPaths.from_root(tmp_path / "analysis")

    assert paths.family_freezes == governed.resolve()


def test_missing_external_validation_freeze_is_not_created(
    monkeypatch, tmp_path: Path
) -> None:
    governed = tmp_path / "does_not_exist"
    analysis = tmp_path / "analysis"

    monkeypatch.setenv("PAPER1_VALIDATION_FREEZE_DIR", str(governed))
    paths = ProjectPaths.from_root(analysis)
    paths.ensure()

    assert not governed.exists()
    assert paths.registries.is_dir()
    assert paths.figures_main.is_dir()


def test_local_validation_freeze_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PAPER1_VALIDATION_FREEZE_DIR", raising=False)
    root = tmp_path / "analysis"

    paths = ProjectPaths.from_root(root)

    assert paths.family_freezes == (root / "data" / "raw" / "family_freezes").resolve()
