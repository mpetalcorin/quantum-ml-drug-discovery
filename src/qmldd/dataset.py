from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .qm.base import read_xyz

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ELEMENT_TO_INDEX = {e: i for i, e in enumerate(ELEMENTS)}


def build_paired_table(conformer_manifest: str | Path, xtb_csv: str | Path, dft_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    conf = pd.read_csv(conformer_manifest)
    xtb = pd.read_csv(xtb_csv).rename(columns={"energy_ev": "xtb_energy_ev", "wall_seconds": "xtb_wall_seconds"})
    dft = pd.read_csv(dft_csv).rename(columns={"energy_ev": "dft_energy_ev", "wall_seconds": "dft_wall_seconds"})
    keys = ["molecule_id", "conformer_id"]
    merged = conf.merge(xtb[keys + ["xtb_energy_ev", "xtb_wall_seconds", "forces_json"]], on=keys, how="inner")
    merged = merged.merge(dft[keys + ["dft_energy_ev", "dft_wall_seconds", "forces_json"]].rename(columns={"forces_json": "dft_forces_json"}), on=keys, how="inner")
    merged = merged.rename(columns={"forces_json": "xtb_forces_json"})
    merged["delta_energy_ev"] = merged["dft_energy_ev"] - merged["xtb_energy_ev"]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def molecule_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15, seed: int = 17) -> pd.DataFrame:
    groups = df["molecule_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
    train_idx, rest_idx = next(splitter.split(df, groups=groups))
    out = df.copy()
    out["split"] = ""
    out.loc[out.index[train_idx], "split"] = "train"
    rest = out.iloc[rest_idx]
    rel_val = val / max(1e-12, (1.0 - train))
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1)
    val_local, test_local = next(splitter2.split(rest, groups=rest["molecule_id"]))
    out.loc[rest.index[val_local], "split"] = "val"
    out.loc[rest.index[test_local], "split"] = "test"
    return out


def scaffold_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15, seed: int = 17) -> pd.DataFrame:
    """Split by Bemis-Murcko scaffold so test scaffolds are absent from training."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError("RDKit is required for scaffold splitting") from exc

    mol_table = df[["molecule_id", "smiles"]].drop_duplicates().copy()
    scaffolds = []
    for row in mol_table.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            scaffold = f"INVALID::{row.molecule_id}"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if not scaffold:
                scaffold = f"ACYCLIC::{row.molecule_id}"
        scaffolds.append(scaffold)
    mol_table["scaffold"] = scaffolds

    unique = mol_table["scaffold"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n = len(unique)
    n_train = max(1, int(round(train * n)))
    n_val = max(1, int(round(val * n))) if n >= 3 else 0
    train_scaf = set(unique[:n_train])
    val_scaf = set(unique[n_train:n_train+n_val])
    test_scaf = set(unique[n_train+n_val:])
    if not test_scaf and val_scaf:
        test_scaf.add(val_scaf.pop())

    scaffold_to_split = {**{s: "train" for s in train_scaf}, **{s: "val" for s in val_scaf}, **{s: "test" for s in test_scaf}}
    mol_table["split"] = mol_table["scaffold"].map(scaffold_to_split)
    out = df.merge(mol_table[["molecule_id", "scaffold", "split"]], on="molecule_id", how="left")
    return out


def row_to_tensors(row):
    import torch
    from torch_geometric.data import Data

    symbols, coords = read_xyz(row.xyz_path)
    z = []
    for symbol in symbols:
        if symbol not in ELEMENT_TO_INDEX:
            raise ValueError(f"Unsupported element: {symbol}")
        z.append(ELEMENT_TO_INDEX[symbol])
    pos = torch.tensor(coords, dtype=torch.get_default_dtype())
    data = Data(
        pos=pos,
        element_index=torch.tensor(z, dtype=torch.long),
        delta_energy=torch.tensor([float(row.delta_energy_ev)], dtype=torch.get_default_dtype()),
        xtb_energy=torch.tensor([float(row.xtb_energy_ev)], dtype=torch.get_default_dtype()),
        dft_energy=torch.tensor([float(row.dft_energy_ev)], dtype=torch.get_default_dtype()),
    )
    if isinstance(row.dft_forces_json, str) and row.dft_forces_json:
        dft_forces = np.asarray(json.loads(row.dft_forces_json), dtype=float)
        data.dft_forces = torch.tensor(dft_forces, dtype=torch.get_default_dtype())
    if isinstance(row.xtb_forces_json, str) and row.xtb_forces_json:
        xtb_forces = np.asarray(json.loads(row.xtb_forces_json), dtype=float)
        data.xtb_forces = torch.tensor(xtb_forces, dtype=torch.get_default_dtype())
        if hasattr(data, "dft_forces"):
            data.delta_forces = data.dft_forces - data.xtb_forces
    data.molecule_id = str(row.molecule_id)
    data.conformer_id = int(row.conformer_id)
    return data


def load_graphs(paired_csv: str | Path, split: str | None = None):
    df = pd.read_csv(paired_csv)
    if split is not None:
        df = df[df["split"] == split]
    return [row_to_tensors(row) for row in df.itertuples(index=False)]
