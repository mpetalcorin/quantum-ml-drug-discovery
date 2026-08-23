#!/usr/bin/env python
import argparse
from pathlib import Path

from qmldd.dataset import build_paired_table, molecule_split, scaffold_split
from qmldd.io import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--dft-backend", default="pyscf")
args = p.parse_args()
cfg = load_config(args.config)
processed = Path(cfg["paths"]["processed"])
processed.mkdir(parents=True, exist_ok=True)
paired_path = processed / "paired.csv"
df = build_paired_table(
    Path(cfg["paths"]["conformers"]) / "conformer_manifest.csv",
    Path(cfg["paths"]["qm"]) / "xtb.csv",
    Path(cfg["paths"]["qm"]) / f"dft_{args.dft_backend}.csv",
    paired_path,
)
if cfg["split"].get("mode", "molecule") == "scaffold":
    df = scaffold_split(df, cfg["split"]["train"], cfg["split"]["val"], cfg["seed"])
else:
    df = molecule_split(df, cfg["split"]["train"], cfg["split"]["val"], cfg["seed"])
df.to_csv(paired_path, index=False)
print(df.groupby("split")["molecule_id"].nunique())
