#!/usr/bin/env python
import argparse
from pathlib import Path

from qmldd.io import load_config
from qmldd.train import train_ensemble

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/quick.yaml")
args = p.parse_args()
cfg = load_config(args.config)
paths = train_ensemble(cfg, Path(cfg["paths"]["processed"]) / "paired.csv")
print("Saved models:")
for path in paths:
    print(path)
