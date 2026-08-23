#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results", default="results")
args = p.parse_args()
root = Path(args.results)
figdir = root / "figures"
figdir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(root / "test_predictions.csv")

# 1. DFT parity plot
fig, ax = plt.subplots(figsize=(6.5, 6.0))
ax.scatter(df.dft_energy_ev, df.pred_dft_ev, alpha=0.8)
lo = min(df.dft_energy_ev.min(), df.pred_dft_ev.min())
hi = max(df.dft_energy_ev.max(), df.pred_dft_ev.max())
ax.plot([lo, hi], [lo, hi], linestyle="--")
ax.set_xlabel("DFT energy (eV)")
ax.set_ylabel("xTB + ΔML predicted energy (eV)")
ax.set_title("Higher-fidelity energy reconstruction")
fig.tight_layout()
fig.savefig(figdir / "01_dft_parity.png", dpi=220)
plt.close(fig)

# 2. Relative conformer energies
fig, ax = plt.subplots(figsize=(7.0, 5.5))
plot_df = df.sort_values(["molecule_id", "relative_dft_energy_ev"]).copy()
plot_df["rank_index"] = plot_df.groupby("molecule_id").cumcount()
for molecule_id, g in plot_df.groupby("molecule_id"):
    ax.plot(g.rank_index, g.relative_dft_energy_ev, marker="o", alpha=0.65, label=f"{molecule_id} DFT")
    ax.plot(g.rank_index, g.relative_pred_dft_ev, marker="x", linestyle="--", alpha=0.65, label=f"{molecule_id} ML")
ax.set_xlabel("Conformer order within molecule")
ax.set_ylabel("Relative energy (eV)")
ax.set_title("Conformer-energy profiles")
if plot_df.molecule_id.nunique() <= 4:
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figdir / "02_conformer_profiles.png", dpi=220)
plt.close(fig)

# 3. Uncertainty versus absolute error
fig, ax = plt.subplots(figsize=(6.5, 5.5))
err = np.abs(df.pred_dft_ev - df.dft_energy_ev)
ax.scatter(df.uncertainty_ev, err, alpha=0.8)
ax.set_xlabel("Ensemble uncertainty (eV)")
ax.set_ylabel("Absolute energy error (eV)")
ax.set_title("Does uncertainty identify difficult molecules?")
fig.tight_layout()
fig.savefig(figdir / "03_uncertainty_vs_error.png", dpi=220)
plt.close(fig)

# 4. Method error comparison
xtb_err = np.abs(df.xtb_energy_ev - df.dft_energy_ev)
ml_err = np.abs(df.pred_dft_ev - df.dft_energy_ev)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.boxplot([xtb_err, ml_err], tick_labels=["xTB baseline", "xTB + ΔML"])
ax.set_ylabel("Absolute error versus DFT (eV)")
ax.set_title("Does Δ-learning improve the baseline?")
fig.tight_layout()
fig.savefig(figdir / "04_error_comparison.png", dpi=220)
plt.close(fig)

# 5. Compute-time comparison on logarithmic scale
fig, ax = plt.subplots(figsize=(6.5, 5.2))
medians = [df.xtb_wall_seconds.median(), df.ml_wall_seconds.median(), df.dft_wall_seconds.median()]
ax.bar(["xTB", "ΔML inference", "DFT"], medians)
ax.set_yscale("log")
ax.set_ylabel("Median wall time per conformer (s, log scale)")
ax.set_title("Computational cost")
fig.tight_layout()
fig.savefig(figdir / "05_compute_time.png", dpi=220)
plt.close(fig)

print(f"Wrote figures to {figdir}")
