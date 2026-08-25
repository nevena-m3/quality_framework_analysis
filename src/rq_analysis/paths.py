from __future__ import annotations

from dataclasses import dataclass
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
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            family_freezes=root / "data" / "raw" / "family_freezes",
            registries=root / "data" / "derived" / "registries",
            figures_main=root / "outputs" / "figures" / "main",
            figures_source=root / "outputs" / "figures" / "source_data",
            tables=root / "outputs" / "tables",
            logs=root / "outputs" / "logs",
        )

    def ensure(self) -> None:
        for path in (
            self.data_raw,
            self.family_freezes,
            self.registries,
            self.figures_main,
            self.figures_source,
            self.tables,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
