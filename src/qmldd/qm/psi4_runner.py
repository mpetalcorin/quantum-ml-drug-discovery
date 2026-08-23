from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import QMResult, read_xyz, xyz_block

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def run_psi4(
    xyz_path: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "b3lyp",
    basis: str = "def2-svp",
) -> QMResult:
    try:
        import psi4
    except ImportError as exc:
        raise RuntimeError("Psi4 is not installed in this environment") from exc

    symbols, coords = read_xyz(xyz_path)
    geometry = f"{int(charge)} {int(multiplicity)}\n{xyz_block(symbols, coords)}\nunits angstrom\nno_reorient\nno_com"
    molecule = psi4.geometry(geometry)
    level = f"{functional}/{basis}"
    psi4.core.set_output_file("psi4_output.dat", False)

    start = time.perf_counter()
    try:
        energy_h = float(psi4.energy(level, molecule=molecule))
        grad = np.asarray(psi4.gradient(level, molecule=molecule), dtype=float)
        force_ev_a = -grad * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        elapsed = time.perf_counter() - start
        return QMResult("psi4", level, energy_h * HARTREE_TO_EV, force_ev_a.tolist(), True, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return QMResult("psi4", level, float("nan"), None, False, elapsed, str(exc))
