from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import QMResult, read_xyz

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def run_pyscf(
    xyz_path: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "b3lyp",
    basis: str = "def2-svp",
    grid_level: int = 2,
    max_cycle: int = 100,
) -> QMResult:
    try:
        from pyscf import dft, gto
    except ImportError as exc:
        raise RuntimeError("Install PySCF to use this DFT backend") from exc

    symbols, coords = read_xyz(xyz_path)
    atom_spec = [(s, tuple(map(float, xyz))) for s, xyz in zip(symbols, coords)]
    spin = int(multiplicity) - 1
    mol = gto.M(atom=atom_spec, unit="Angstrom", charge=int(charge), spin=spin, basis=basis, verbose=0)
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = functional
    mf.grids.level = int(grid_level)
    mf.max_cycle = int(max_cycle)

    start = time.perf_counter()
    try:
        e_h = float(mf.kernel())
        converged = bool(mf.converged)
        grad_h_bohr = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
        force_ev_a = -grad_h_bohr * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        elapsed = time.perf_counter() - start
        return QMResult(
            "pyscf",
            f"{functional}/{basis}",
            e_h * HARTREE_TO_EV,
            force_ev_a.tolist(),
            converged,
            elapsed,
            "" if converged else "SCF did not report convergence",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return QMResult("pyscf", f"{functional}/{basis}", float("nan"), None, False, elapsed, str(exc))
