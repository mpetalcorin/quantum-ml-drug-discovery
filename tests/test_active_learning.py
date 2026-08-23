import pandas as pd

from qmldd.active_learning import select_by_uncertainty


def test_uncertainty_acquisition_is_descending():
    df = pd.DataFrame({
        "molecule_id": ["a", "b", "c"],
        "conformer_id": [0, 0, 0],
        "uncertainty_ev": [0.1, 0.4, 0.2],
    })
    out = select_by_uncertainty(df, 2)
    assert out.molecule_id.tolist() == ["b", "c"]
