from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_parp1_report(ligand_csv: str | Path = "data/seed/parp1_ligands.csv") -> str:
    ligands = pd.read_csv(ligand_csv)
    names = ", ".join(ligands["name"].tolist())
    return f"""# PARP1 translational case study

Ligand panel: {names}.

## Scientific question

Can a quantum-ML surrogate reproduce relative intramolecular conformer energetics for chemically distinct PARP inhibitors, quantify uncertainty, and identify geometries that require higher-level DFT labelling?

## What is measured

- GFN2-xTB energy and forces for conformer ensembles.
- DFT energy and gradients for a labelled subset.
- ML correction from xTB to DFT.
- Relative conformer energies within each ligand.
- Uncertainty and out-of-distribution flags.

## What is deliberately *not* claimed

This workflow does not equate gas-phase conformer energy with PARP1 binding affinity, PARP trapping, biochemical potency, synthetic lethality, or clinical efficacy. Those outcomes additionally depend on protein structure, solvation, protonation, entropy, kinetics, cellular context, DNA damage state, and many other factors.

## Translational extension

A later protein-ligand module could use experimentally resolved PARP1 structures, explicit protonation/tautomer handling, constrained conformer generation, protein-ligand interaction features, alchemical or end-point free-energy calculations, and experimental biochemical labels. That is intentionally separated from the present intramolecular QM benchmark so that validation remains scientifically interpretable.
"""
