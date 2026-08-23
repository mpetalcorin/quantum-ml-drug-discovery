from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .dataset import load_graphs
from .models.e3nn_delta import E3NNDeltaEnergy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epoch(model, loader, optimizer=None, force_weight: float = 0.0):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    n = 0
    for batch in loader:
        batch.pos.requires_grad_(force_weight > 0 and hasattr(batch, "delta_forces"))
        pred_delta = model(batch)
        target_delta = batch.delta_energy.reshape(-1)
        energy_loss = torch.mean((pred_delta - target_delta) ** 2)
        loss = energy_loss

        if force_weight > 0 and hasattr(batch, "delta_forces"):
            pred_delta_forces = -torch.autograd.grad(pred_delta.sum(), batch.pos, create_graph=training, retain_graph=training)[0]
            force_loss = torch.mean((pred_delta_forces - batch.delta_forces) ** 2)
            loss = loss + force_weight * force_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def train_ensemble(config: dict, paired_csv: str | Path):
    train_graphs = load_graphs(paired_csv, "train")
    val_graphs = load_graphs(paired_csv, "val")
    mcfg = config["model"]
    out_dir = Path(config["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for member in range(int(mcfg["ensemble_size"])):
        set_seed(int(config["seed"]) + member)
        model = E3NNDeltaEnergy(
            hidden_irreps=mcfg["hidden_irreps"],
            layers=mcfg["layers"],
            cutoff=mcfg["cutoff"],
            radial_basis=mcfg["radial_basis"],
            radial_layers=mcfg["radial_layers"],
            radial_neurons=mcfg["radial_neurons"],
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(mcfg["learning_rate"]), weight_decay=float(mcfg["weight_decay"]))
        train_loader = DataLoader(train_graphs, batch_size=int(mcfg["batch_size"]), shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=int(mcfg["batch_size"]), shuffle=False)
        best_val = float("inf")
        best_path = out_dir / f"e3nn_delta_member_{member}.pt"

        for epoch in range(int(mcfg["epochs"])):
            train_loss = _epoch(model, train_loader, optimizer, float(mcfg["force_weight"]))
            val_loss = _epoch(model, val_loader, None, 0.0)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch, "val_loss": val_loss}, best_path)
            if epoch % 20 == 0:
                print(f"member={member} epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")
        saved.append(best_path)
    return saved
