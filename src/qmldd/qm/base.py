from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class QMResult:
    backend: str
    method: str
    energy_ev: float
    forces_ev_a: list[list[float]] | None
    converged: bool
    wall_seconds: float
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + n_atoms]
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def xyz_block(symbols: Sequence[str], coords_a: np.ndarray) -> str:
    return "\n".join(
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coords_a)
    )
