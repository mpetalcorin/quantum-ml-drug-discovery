#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from qmldd.dataset import ELEMENT_TO_INDEX
from qmldd.io import load_config
from qmldd.models.e3nn_delta import E3NNDeltaEnergy
from qmldd.qm.base import read_xyz

p = argparse.ArgumentParser(description="Score an xTB-only candidate pool without using DFT labels.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--manifest", required=True, help="Conformer manifest for the unlabelled pool")
p.add_argument("--xtb", required=True, help="xTB result CSV for the same pool")
p.add_argument("--out", default="results/pool_predictions.csv")
args = p.parse_args()
cfg = load_config(args.config)
manifest = pd.read_csv(args.manifest)
xtb = pd.read_csv(args.xtb)
pool = manifest.merge(xtb, on=["molecule_id", "conformer_id"], how="inner", suffixes=("", "_xtb"))
mcfg = cfg["model"]
models = []
for member in range(int(mcfg["ensemble_size"])):
    ckpt = torch.load(Path(cfg["paths"]["models"]) / f"e3nn_delta_member_{member}.pt", map_location="cpu")
    model = E3NNDeltaEnergy(
        hidden_irreps=mcfg["hidden_irreps"], layers=mcfg["layers"], cutoff=mcfg["cutoff"],
        radial_basis=mcfg["radial_basis"], radial_layers=mcfg["radial_layers"], radial_neurons=mcfg["radial_neurons"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    models.append(model)

records = []
for row in pool.itertuples(index=False):
    symbols, coords = read_xyz(row.xyz_path)
    element_index = [ELEMENT_TO_INDEX[s] for s in symbols]
    xtb_forces = np.asarray(json.loads(row.forces_json), dtype=float) if isinstance(row.forces_json, str) and row.forces_json else None
    preds = []
    for model in models:
        data = Data(
            pos=torch.tensor(coords, dtype=torch.get_default_dtype()),
            element_index=torch.tensor(element_index, dtype=torch.long),
            xtb_energy=torch.tensor([float(row.energy_ev)], dtype=torch.get_default_dtype()),
        )
        if xtb_forces is not None:
            data.xtb_forces = torch.tensor(xtb_forces, dtype=torch.get_default_dtype())
        with torch.no_grad():
            preds.append(float(model(data)[0]))
    records.append({
        "molecule_id": row.molecule_id,
        "conformer_id": int(row.conformer_id),
        "xtb_energy_ev": float(row.energy_ev),
        "pred_delta_ev": float(np.mean(preds)),
        "pred_dft_ev": float(row.energy_ev) + float(np.mean(preds)),
        "uncertainty_ev": float(np.std(preds, ddof=1 if len(preds) > 1 else 0)),
        "xyz_path": row.xyz_path,
    })

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(out, index=False)
print(f"Wrote {len(records)} pool predictions to {out}")
