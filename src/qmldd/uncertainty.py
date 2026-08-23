from __future__ import annotations

import numpy as np
import torch


def ensemble_predict(models, data):
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(model(data).detach().cpu().numpy())
    arr = np.stack(preds, axis=0)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1 if len(models) > 1 else 0)


def interval_coverage(y_true, mean, std, z: float = 1.96) -> float:
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    std = np.asarray(std)
    lo = mean - z * std
    hi = mean + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
