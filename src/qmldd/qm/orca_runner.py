from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .base import QMResult, read_xyz

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def _parse_engrad(path: Path, n_atoms: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "# The current gradient in Eh/bohr"
    if marker not in text:
        raise ValueError("Could not find gradient section in ORCA .engrad")
    tail = text.split(marker, 1)[1]
    values = []
    for line in tail.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            values.append(float(s.replace("D", "E")))
        except ValueError:
            if values:
                break
        if len(values) >= 3 * n_atoms:
            break
    if len(values) < 3 * n_atoms:
        raise ValueError("Incomplete ORCA gradient")
    return np.asarray(values[: 3 * n_atoms], dtype=float).reshape(n_atoms, 3)


def run_orca(
    xyz_path: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "B3LYP",
    basis: str = "def2-SVP",
    nprocs: int = 4,
) -> QMResult:
    exe = shutil.which("orca")
    if exe is None:
        raise RuntimeError("ORCA executable was not found on PATH")

    xyz_path = Path(xyz_path).resolve()
    symbols, _ = read_xyz(xyz_path)
    workdir = xyz_path.parent / f"orca_{xyz_path.stem}"
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / "job.inp"
    inp.write_text(
        f"! {functional} {basis} TightSCF EnGrad\n%pal nprocs {int(nprocs)} end\n* xyzfile {int(charge)} {int(multiplicity)} {xyz_path}\n",
        encoding="utf-8",
    )

    start = time.perf_counter()
    try:
        proc = subprocess.run([exe, str(inp)], cwd=workdir, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - start
        output = proc.stdout + "\n" + proc.stderr
        match = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", output)
        if match is None:
            raise ValueError("ORCA final energy not found")
        energy_h = float(match.group(1))
        engrad = workdir / "job.engrad"
        grad = _parse_engrad(engrad, len(symbols))
        force_ev_a = -grad * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        converged = proc.returncode == 0 and "ORCA TERMINATED NORMALLY" in output
        return QMResult("orca", f"{functional}/{basis}", energy_h * HARTREE_TO_EV, force_ev_a.tolist(), converged, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return QMResult("orca", f"{functional}/{basis}", float("nan"), None, False, elapsed, str(exc))
