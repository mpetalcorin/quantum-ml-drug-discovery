#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd

from qmldd.active_learning import select_by_uncertainty
from qmldd.io import load_config

p = argparse.ArgumentParser(description="Select xTB-only conformers for the next expensive DFT labelling round.")
p.add_argument("--config", default="configs/quick.yaml")
p.add_argument("--scores", default="results/pool_predictions.csv", help="Predictions for an unlabelled xTB-only pool")
p.add_argument("--out", default="results/active_learning_acquisition.csv")
args = p.parse_args()
cfg = load_config(args.config)
scores_path = Path(args.scores)
if not scores_path.exists():
    raise FileNotFoundError(
        f"{scores_path} does not exist. First score an xTB-only pool with scripts/05b_score_pool.py. "
        "Do not use the benchmark test set for reported active-learning acquisition."
    )
scores = pd.read_csv(scores_path)
selected = select_by_uncertainty(scores, cfg["active_learning"]["acquire_n"])
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(out, index=False)
print(selected[["molecule_id", "conformer_id", "uncertainty_ev"]].to_string(index=False))
