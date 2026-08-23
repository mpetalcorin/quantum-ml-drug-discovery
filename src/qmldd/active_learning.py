from __future__ import annotations

import pandas as pd


def select_by_uncertainty(scores: pd.DataFrame, acquire_n: int = 20) -> pd.DataFrame:
    required = {"molecule_id", "conformer_id", "uncertainty_ev"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return scores.sort_values("uncertainty_ev", ascending=False).head(int(acquire_n)).copy()
