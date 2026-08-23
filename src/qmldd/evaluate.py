from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def conformer_ranking_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> dict[str, float]:
    rhos = []
    top1 = []
    for _, group in df.groupby("molecule_id"):
        if len(group) < 2:
            continue
        rho = spearmanr(group[truth_col], group[pred_col]).statistic
        if np.isfinite(rho):
            rhos.append(rho)
        top1.append(int(group[truth_col].idxmin() == group[pred_col].idxmin()))
    return {
        "mean_spearman_conformer_rank": float(np.mean(rhos)) if rhos else float("nan"),
        "top1_conformer_recovery": float(np.mean(top1)) if top1 else float("nan"),
    }


def speedup(dft_seconds, ml_seconds) -> float:
    dft = np.asarray(dft_seconds, dtype=float)
    ml = np.asarray(ml_seconds, dtype=float)
    return float(np.nanmedian(dft) / max(np.nanmedian(ml), 1e-12))
