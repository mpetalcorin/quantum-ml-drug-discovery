#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from qmldd.io import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--backend", choices=["pyscf", "psi4", "orca"], default="pyscf")
args = p.parse_args()
cfg = load_config(args.config)

if args.backend == "pyscf":
    from qmldd.qm.pyscf_runner import run_pyscf
    def runner(row):
        q = cfg["pyscf"]
        return run_pyscf(row.xyz_path, row.charge, row.multiplicity, q["functional"], q["basis"], q["grid_level"], q["max_cycle"])
elif args.backend == "psi4":
    from qmldd.qm.psi4_runner import run_psi4
    def runner(row):
        q = cfg["pyscf"]
        return run_psi4(row.xyz_path, row.charge, row.multiplicity, q["functional"], q["basis"])
else:
    from qmldd.qm.orca_runner import run_orca
    def runner(row):
        q = cfg["pyscf"]
        return run_orca(row.xyz_path, row.charge, row.multiplicity, q["functional"], q["basis"])

manifest = pd.read_csv(Path(cfg["paths"]["conformers"]) / "conformer_manifest.csv")
rows = []
valid = manifest[manifest.status == "ok"]
for row in tqdm(valid.itertuples(index=False), total=len(valid)):
    result = runner(row)
    rows.append({
        "molecule_id": row.molecule_id,
        "conformer_id": row.conformer_id,
        "energy_ev": result.energy_ev,
        "forces_json": json.dumps(result.forces_ev_a) if result.forces_ev_a is not None else "",
        "converged": result.converged,
        "wall_seconds": result.wall_seconds,
        "method": result.method,
        "message": result.message,
    })
out = Path(cfg["paths"]["qm"])
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"dft_{args.backend}.csv", index=False)
