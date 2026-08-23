import pandas as pd

from qmldd.dataset import molecule_split


def test_molecule_split_prevents_group_leakage():
    df = pd.DataFrame({
        "molecule_id": [f"m{i}" for i in range(20) for _ in range(3)],
        "conformer_id": list(range(3)) * 20,
    })
    out = molecule_split(df, train=0.7, val=0.15, seed=1)
    groups = {split: set(out.loc[out.split == split, "molecule_id"]) for split in ["train", "val", "test"]}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
