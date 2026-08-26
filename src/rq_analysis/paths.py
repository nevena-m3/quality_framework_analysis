from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    family_freezes: Path
    registries: Path
    figures_main: Path
    figures_source: Path
    tables: Path
    logs: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root = Path(root).resolve()
        configured_validation_freeze = os.environ.get(
            "PAPER1_VALIDATION_FREEZE_DIR", ""
        ).strip()
        family_freezes = (
            Path(configured_validation_freeze).expanduser().resolve()
            if configured_validation_freeze
            else root / "data" / "raw" / "family_freezes"
        )
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            family_freezes=family_freezes,
            registries=root / "data" / "derived" / "registries",
            figures_main=root / "outputs" / "figures" / "main",
            figures_source=root / "outputs" / "figures" / "source_data",
            tables=root / "outputs" / "tables",
            logs=root / "outputs" / "logs",
        )

    def ensure(self) -> None:
        # Governed input roots are never created implicitly. A missing
        # validation freeze must fail loudly rather than becoming an empty
        # directory that looks valid.
        for path in (
            self.data_raw,
            self.registries,
            self.figures_main,
            self.figures_source,
            self.tables,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
