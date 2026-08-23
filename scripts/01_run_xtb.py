#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from qmldd.io import load_config
from qmldd.qm.xtb_runner import run_xtb

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/quick.yaml")
args = p.parse_args()
cfg = load_config(args.config)
manifest = pd.read_csv(Path(cfg["paths"]["conformers"]) / "conformer_manifest.csv")
rows = []
for row in tqdm(manifest[manifest.status == "ok"].itertuples(index=False), total=(manifest.status == "ok").sum()):
    result = run_xtb(row.xyz_path, row.charge, row.multiplicity, cfg["xtb"]["method"], cfg["xtb"]["accuracy"])
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
pd.DataFrame(rows).to_csv(out / "xtb.csv", index=False)
