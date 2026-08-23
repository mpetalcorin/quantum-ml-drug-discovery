# Quantum ML Drug Discovery

**Physics-grounded machine learning for small-molecule energetics, forces, conformer ranking, uncertainty, active learning, and a PARP1 DNA-repair case study.**
<img width="1448" height="1086" alt="Quantum ML Drug Discovery" src="https://github.com/user-attachments/assets/4f215975-450c-49af-b3b8-ae7449f49e0d" />
This repository is designed as a focused portfolio project for machine-learning roles at the interface of quantum chemistry and small-molecule drug discovery. The core idea is **delta learning**:

\[
E_{\mathrm{DFT}} = E_{\mathrm{xTB}} + \Delta E_{\mathrm{ML}}
\]

Instead of asking a neural network to learn the full quantum-chemical energy surface from scratch, a fast semi-empirical method supplies a physically meaningful baseline and ML learns the correction toward a higher-level DFT target.

## Scientific workflow

```text
SMILES
  ↓
RDKit ETKDG conformer generation
  ↓
xTB (GFN2-xTB): energy + forces
  ↓
PySCF / Psi4 / ORCA: DFT energy + gradients
  ↓
Paired QM dataset
  ↓
Δ target = E_DFT - E_xTB
  ↓
e3nn E(3)-equivariant graph network
  ↓
E_DFT ≈ E_xTB + ΔE_ML
  ↓
Forces from -∂E/∂R
  ↓
Conformer ranking + uncertainty + OOD tests
  ↓
Active learning
  ↓
PARP1-focused translational case study
```

## What this demonstrates

- RDKit-based 3D molecular conformer generation.
- Semi-empirical quantum chemistry with GFN2-xTB.
- DFT backends for PySCF, Psi4, and ORCA.
- Reproducible generation of energies, gradients, and forces.
- Physics-grounded Δ-learning rather than black-box property fitting.
- E(3)-equivariant molecular graph modelling with e3nn.
- Energy-conserving force prediction through automatic differentiation.
- Molecule-level train/validation/test splitting to avoid conformer leakage.
- Scaffold/OOD evaluation.
- Deep-ensemble uncertainty and calibration analysis.
- Active-learning acquisition based on predictive uncertainty.
- Conformer-ranking metrics and wall-clock speed comparisons.
- A PARP1 inhibitor case study that connects molecular energetics to DNA-repair biology without claiming that gas-phase QM energy is equivalent to binding affinity or cellular efficacy.

## Repository layout

```text
quantum-ml-drug-discovery/
├── configs/
│   ├── quick.yaml
│   └── production.yaml
├── data/seed/
│   ├── molecules.csv
│   └── parp1_ligands.csv
├── docs/
│   ├── INTERVIEW_GUIDE.md
│   └── REFERENCES.md
├── notebooks/
│   └── 01_end_to_end_demo.ipynb
├── scripts/
│   ├── 00_make_conformers.py
│   ├── 01_run_xtb.py
│   ├── 02_run_dft.py
│   ├── 03_build_dataset.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 06_active_learn.py
│   └── 07_parp1_case_study.py
├── src/qmldd/
│   ├── active_learning.py
│   ├── conformers.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── io.py
│   ├── parp1.py
│   ├── train.py
│   ├── uncertainty.py
│   ├── models/e3nn_delta.py
│   └── qm/
│       ├── base.py
│       ├── orca_runner.py
│       ├── psi4_runner.py
│       ├── pyscf_runner.py
│       └── xtb_runner.py
└── tests/
```

## Installation

The most reliable route for xTB is conda-forge.

```bash
conda env create -f environment.yml
conda activate quantum-ml-dd
pip install -e .
```

If you only want the Python/ML layer first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[ml]"
```

Psi4 and ORCA are optional backends. ORCA itself is an external executable and is not distributed by this repository.

## Quick start

Generate 3D conformers:

```bash
python scripts/00_make_conformers.py --config configs/quick.yaml
```

Run GFN2-xTB:

```bash
python scripts/01_run_xtb.py --config configs/quick.yaml
```

Run DFT with PySCF:

```bash
python scripts/02_run_dft.py --config configs/quick.yaml --backend pyscf
```

Build paired Δ-learning dataset:

```bash
python scripts/03_build_dataset.py --config configs/quick.yaml
```

Train an ensemble of equivariant models:

```bash
python scripts/04_train.py --config configs/quick.yaml
```

Evaluate MAE, RMSE, force error, conformer ranking, uncertainty calibration, speedup, and OOD behavior:

```bash
python scripts/05_evaluate.py --config configs/quick.yaml
```

Score an xTB-only, DFT-unlabelled candidate pool and run uncertainty-guided active learning:

```bash
python scripts/05b_score_pool.py \
  --config configs/quick.yaml \
  --manifest path/to/pool/conformer_manifest.csv \
  --xtb path/to/pool/xtb.csv \
  --out results/pool_predictions.csv

python scripts/06_active_learn.py --config configs/quick.yaml --scores results/pool_predictions.csv
```

The reported benchmark test set is kept separate from acquisition decisions.

Generate the PARP1 conformer panel and case-study report:

```bash
python scripts/00_make_conformers.py --config configs/parp1.yaml
python scripts/07_parp1_case_study.py
```

Generate benchmark figures after evaluation:

```bash
python scripts/08_make_figures.py --results results
```

## Recommended experiment design

### Stage 1, small proof of concept

Use 20–50 neutral organic molecules with 5–10 conformers each. Run GFN2-xTB on all conformers and a modest DFT level on a subset. Demonstrate that Δ-learning reduces error relative to the xTB baseline.

### Stage 2, drug-like expansion

Increase molecular diversity and use molecule-level or scaffold-level splits. Quantify whether performance degrades on chemical classes absent from training.

### Stage 3, active learning

Start with a small DFT-labelled seed set. Train an ensemble, score unlabelled xTB conformers by uncertainty, then select the most informative conformers for DFT labelling.

### Stage 4, translational PARP1 layer

Apply the trained surrogate to conformer ensembles of known PARP inhibitors. Compare intramolecular conformational energetics and uncertainty. Treat this as a molecular-energetics case study, not as a direct predictor of PARP trapping, biochemical potency, or clinical response.

## Key metrics

The evaluation code reports:

- energy MAE and RMSE,
- Δ-energy MAE and RMSE,
- force MAE and RMSE,
- Spearman conformer-ranking correlation,
- top-1 conformer recovery,
- uncertainty–error correlation,
- empirical coverage of prediction intervals,
- inference throughput and estimated speedup over DFT,
- in-distribution versus OOD error.

## Why the model is physics-aware

The e3nn network uses spherical harmonics and tensor products so intermediate features transform consistently under 3D rotations and reflections. The final molecular energy is a scalar. Forces are obtained as

\[
\mathbf F_i = -\frac{\partial E}{\partial \mathbf R_i},
\]

which guarantees an energy-conserving force field when the predicted energy is differentiable with respect to coordinates.

## Important scientific limitations

1. Gas-phase conformer energetics are not binding free energies.
2. DFT accuracy depends on functional, basis set, dispersion treatment, charge state, spin state, and molecular system.
3. A model trained on a narrow chemical space should not be trusted on OOD molecules without explicit uncertainty analysis.
4. Protein–ligand binding requires solvation, protein flexibility, entropy, protonation/tautomer states, and often much more extensive sampling.
5. PARP inhibitor trapping and cytotoxicity are mechanistically richer than affinity alone.

## Description

> I built a reproducible quantum-ML pipeline in which GFN2-xTB supplies a fast physical baseline, DFT supplies higher-fidelity labels, and an E(3)-equivariant graph network learns the correction. I derive forces through the energy gradient, evaluate conformer ranking and OOD behavior, quantify ensemble uncertainty, and use that uncertainty to decide which molecular geometries should receive the next expensive DFT calculations. I then connect the workflow to PARP1 inhibitor chemistry while explicitly separating intramolecular energetics from biological efficacy.

