from __future__ import annotations

import time
from pathlib import Path

from .base import QMResult, read_xyz


def run_xtb(
    xyz_path: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
) -> QMResult:
    try:
        from ase import Atoms
        from xtb.ase.calculator import XTB
    except ImportError as exc:
        raise RuntimeError("Install xtb-python and ASE to use the xTB backend") from exc

    symbols, coords = read_xyz(xyz_path)
    uhf = max(0, int(multiplicity) - 1)
    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.calc = XTB(method=method, charge=int(charge), uhf=uhf, accuracy=float(accuracy))

    start = time.perf_counter()
    try:
        energy_ev = float(atoms.get_potential_energy())
        forces = atoms.get_forces().tolist()
        elapsed = time.perf_counter() - start
        return QMResult("xtb", method, energy_ev, forces, True, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return QMResult("xtb", method, float("nan"), None, False, elapsed, str(exc))
