#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from qmldd.dataset import load_graphs
from qmldd.evaluate import conformer_ranking_metrics, regression_metrics, speedup
from qmldd.io import load_config, write_json
from qmldd.models.e3nn_delta import E3NNDeltaEnergy

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/quick.yaml")
args = p.parse_args()
cfg = load_config(args.config)
paired_path = Path(cfg["paths"]["processed"]) / "paired.csv"
paired = pd.read_csv(paired_path)
test_graphs = load_graphs(paired_path, "test")
loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
mcfg = cfg["model"]
models = []
for member in range(int(mcfg["ensemble_size"])):
    ckpt = torch.load(Path(cfg["paths"]["models"]) / f"e3nn_delta_member_{member}.pt", map_location="cpu")
    model = E3NNDeltaEnergy(
        hidden_irreps=mcfg["hidden_irreps"],
        layers=mcfg["layers"],
        cutoff=mcfg["cutoff"],
        radial_basis=mcfg["radial_basis"],
        radial_layers=mcfg["radial_layers"],
        radial_neurons=mcfg["radial_neurons"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    models.append(model)

records = []
force_true_flat = []
force_pred_flat = []
for data in loader:
    member_delta = []
    member_force = []
    elapsed = []
    for model in models:
        # Coordinates must require gradients because force is -dE/dR.
        data.pos = data.pos.detach().clone().requires_grad_(True)
        start = time.perf_counter()
        total_energy, total_force = model.corrected_energy_and_forces(data, create_graph=False)
        elapsed.append(time.perf_counter() - start)
        member_delta.append(float(total_energy.detach()[0] - data.xtb_energy.detach()[0]))
        member_force.append(total_force.detach().cpu().numpy())

    mean_delta = float(np.mean(member_delta))
    std_delta = float(np.std(member_delta, ddof=1 if len(member_delta) > 1 else 0))
    pred_dft = float(data.xtb_energy.detach()[0]) + mean_delta
    pred_force = np.mean(np.stack(member_force, axis=0), axis=0)

    molecule_id = data.molecule_id[0] if isinstance(data.molecule_id, (list, tuple)) else str(data.molecule_id)
    conformer_id = int(data.conformer_id[0]) if hasattr(data.conformer_id, "__len__") else int(data.conformer_id)

    dft_force_mae = float("nan")
    if hasattr(data, "dft_forces"):
        truth_force = data.dft_forces.detach().cpu().numpy()
        dft_force_mae = float(np.mean(np.abs(pred_force - truth_force)))
        force_true_flat.extend(truth_force.reshape(-1).tolist())
        force_pred_flat.extend(pred_force.reshape(-1).tolist())

    pair_row = paired[(paired.molecule_id.astype(str) == str(molecule_id)) & (paired.conformer_id == conformer_id)]
    dft_wall = float(pair_row.iloc[0].dft_wall_seconds) if len(pair_row) else float("nan")
    xtb_wall = float(pair_row.iloc[0].xtb_wall_seconds) if len(pair_row) else float("nan")

    records.append({
        "molecule_id": str(molecule_id),
        "conformer_id": conformer_id,
        "dft_energy_ev": float(data.dft_energy.detach()[0]),
        "xtb_energy_ev": float(data.xtb_energy.detach()[0]),
        "pred_delta_ev": mean_delta,
        "pred_dft_ev": pred_dft,
        "uncertainty_ev": std_delta,
        "force_mae_ev_a": dft_force_mae,
        "xtb_wall_seconds": xtb_wall,
        "dft_wall_seconds": dft_wall,
        "ml_wall_seconds": float(np.mean(elapsed)),
    })

res = pd.DataFrame(records)
for col in ["dft_energy_ev", "xtb_energy_ev", "pred_dft_ev"]:
    res[f"relative_{col}"] = res[col] - res.groupby("molecule_id")[col].transform("min")

metrics = {}
metrics.update(regression_metrics(res.dft_energy_ev, res.xtb_energy_ev, "xtb_absolute_"))
metrics.update(regression_metrics(res.dft_energy_ev, res.pred_dft_ev, "delta_ml_absolute_"))
metrics.update(regression_metrics(res.relative_dft_energy_ev, res.relative_xtb_energy_ev, "xtb_relative_"))
metrics.update(regression_metrics(res.relative_dft_energy_ev, res.relative_pred_dft_ev, "delta_ml_relative_"))
metrics.update(conformer_ranking_metrics(res, "dft_energy_ev", "pred_dft_ev"))
if len(force_true_flat):
    metrics.update(regression_metrics(force_true_flat, force_pred_flat, "force_"))
if len(res) > 2 and res.uncertainty_ev.nunique() > 1:
    metrics["uncertainty_error_spearman"] = float(
        res.assign(abs_error=(res.pred_dft_ev - res.dft_energy_ev).abs())[["uncertainty_ev", "abs_error"]].corr(method="spearman").iloc[0, 1]
    )
else:
    metrics["uncertainty_error_spearman"] = float("nan")
metrics["median_ml_speedup_over_dft"] = speedup(res.dft_wall_seconds, res.ml_wall_seconds)
metrics["median_xtb_speedup_over_dft"] = speedup(res.dft_wall_seconds, res.xtb_wall_seconds)

out = Path(cfg["paths"]["results"])
out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / "test_predictions.csv", index=False)
write_json(metrics, out / "metrics.json")
print(json.dumps(metrics, indent=2))
