from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_conformers(
    molecules_csv: str | Path,
    out_dir: str | Path,
    n_conformers: int = 10,
    prune_rms_thresh: float = 0.35,
    optimize_mmff: bool = True,
    seed: int = 17,
) -> pd.DataFrame:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for conformer generation") from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    molecules = pd.read_csv(molecules_csv)
    records: list[dict] = []

    for row in molecules.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            records.append({"molecule_id": row.molecule_id, "status": "invalid_smiles"})
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.pruneRmsThresh = float(prune_rms_thresh)
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))

        if optimize_mmff and AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            for cid in conf_ids:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=int(cid), maxIters=500)
                except Exception:
                    pass
        else:
            props = None

        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for rank, cid in enumerate(conf_ids):
            conf = mol.GetConformer(int(cid))
            xyz_path = out_dir / f"{row.molecule_id}__c{rank:03d}.xyz"
            with open(xyz_path, "w", encoding="utf-8") as handle:
                handle.write(f"{mol.GetNumAtoms()}\n")
                handle.write(f"{row.molecule_id} conformer {rank}\n")
                for atom_idx, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom_idx)
                    handle.write(f"{symbol:2s} {p.x: .10f} {p.y: .10f} {p.z: .10f}\n")

            mmff_energy = None
            if props is not None:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))
                    mmff_energy = float(ff.CalcEnergy())
                except Exception:
                    pass

            records.append(
                {
                    "molecule_id": row.molecule_id,
                    "name": row.name,
                    "smiles": row.smiles,
                    "charge": int(row.charge),
                    "multiplicity": int(row.multiplicity),
                    "conformer_id": rank,
                    "xyz_path": str(xyz_path),
                    "mmff_energy_kcal_mol": mmff_energy,
                    "status": "ok",
                }
            )

    manifest = pd.DataFrame(records)
    manifest.to_csv(out_dir / "conformer_manifest.csv", index=False)
    return manifest
