#!/usr/bin/env python
import argparse

from qmldd.conformers import generate_conformers
from qmldd.io import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--molecules", default=None, help="Optional CSV override")
p.add_argument("--out-dir", default=None, help="Optional conformer output directory override")
args = p.parse_args()
cfg = load_config(args.config)
molecules = args.molecules or cfg["paths"]["molecules"]
out_dir = args.out_dir or cfg["paths"]["conformers"]
manifest = generate_conformers(
    molecules,
    out_dir,
    n_conformers=cfg["conformers"]["n_conformers"],
    prune_rms_thresh=cfg["conformers"]["prune_rms_thresh"],
    optimize_mmff=cfg["conformers"]["optimize_mmff"],
    seed=cfg["seed"],
)
print(manifest["status"].value_counts(dropna=False))
