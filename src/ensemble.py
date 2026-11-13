"""Softmax-average ensemble of E1 seed replicates, with optional flip TTA.

Used by the prespecified selection in results/seed_25d_prespecification.md
(step 3: "the final model of the adopted arm is the softmax-average ensemble of
its seeds; horizontal-flip TTA is added only if it raises the ensemble's
validation macro Dice").

`SoftmaxEnsemble` is an ordinary nn.Module whose output is the averaged class
probability map. Every evaluation path in this repo takes
`argmax(model(image), dim=1)`, and argmax of averaged probabilities is exactly
the ensemble decision, so the ensemble drops into `evaluate_cases`,
`predict_case_reconstructed` and the external-cohort scripts unchanged.

Flip axis. Slices are reoriented to RAS before preprocessing
(src/transforms.py), so array axis 0 of a slice is the left-right axis and
axis 1 is anterior-posterior. In a (N, C, H, W) batch that is dim 2 (H). The
left-right flip is the anatomically symmetric one; an anterior-posterior flip
would move the posterior peripheral zone to the front and is never used.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LEFT_RIGHT_DIM = 2  # (N, C, H, W): H is RAS axis 0 = left-right


class SoftmaxEnsemble(nn.Module):
    def __init__(self, models: Sequence[nn.Module], tta_flip: bool = False):
        super().__init__()
        if not models:
            raise ValueError("SoftmaxEnsemble needs at least one model")
        self.members = nn.ModuleList(models)
        self.tta_flip = bool(tta_flip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        n = 0
        for m in self.members:
            views = [F.softmax(m(x), dim=1)]
            if self.tta_flip:
                flipped = m(torch.flip(x, dims=[LEFT_RIGHT_DIM]))
                views.append(torch.flip(F.softmax(flipped, dim=1), dims=[LEFT_RIGHT_DIM]))
            for p in views:
                total = p if total is None else total + p
                n += 1
        return total / n


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_member(path: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild one checkpoint exactly as src/evaluate.run_evaluation does."""
    from src.model import build_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    info = {
        "path": path,
        "sha256": sha256_file(path),
        "epoch": int(ckpt.get("epoch", -1)),
        "seed": cfg.get("experiment", {}).get("seed", cfg.get("seed")),
        "context_slices": int(cfg.get("data", {}).get("context_slices", 0)),
        "in_channels": int(cfg.get("model", {}).get("in_channels", 1)),
        "preprocessing": cfg.get("preprocessing", {}),
        "config": cfg,
    }
    return model, info


FINAL_SELECTION = "results/e1_seed_selection/selection.json"


def is_selection_file(path: str) -> bool:
    return str(path).replace("\\", "/").endswith("selection.json")


def resolve_member_path(repo: str, recorded: str) -> str:
    """Turn a member path from the selection record into a path on this machine.

    The record was written on Windows, so its relative paths carry "\\"
    separators, which Linux (Colab) reads as part of a file name. Recorded
    relative paths are normalised to "/" and joined to the repo root; the record
    itself is never rewritten.
    """
    rel = str(recorded).replace("\\", "/")
    if os.path.isabs(rel) or (len(rel) > 1 and rel[1] == ":"):
        return rel
    return os.path.join(repo, *rel.split("/"))


def load_final_model(selection_path: str = FINAL_SELECTION, device: str = "cpu"):
    """Load the model frozen by the prespecified selection.

    Every member's weights are checked against the SHA-256 recorded at
    selection time, so a swapped file cannot pass silently. Returns
    (module, ckpt_cfg, identity): `ckpt_cfg` is the shared training config of
    the members (architecture, preprocessing, data layout), `identity` is what
    reports should cite instead of a single checkpoint hash.
    """
    import json

    with open(selection_path) as f:
        sel = json.load(f)
    final = sel["final_model"]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [resolve_member_path(repo, m["path"]) for m in final["members"]]
    module, infos = load_ensemble(paths, tta_flip=bool(final["tta_flip_left_right"]), device=device)
    for m, info in zip(final["members"], infos):
        if info["sha256"] != m["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {m['path']}: {info['sha256']} != {m['sha256']}")
    ensemble_id = hashlib.sha256(
        ("|".join(m["sha256"] for m in final["members"]) + f"|tta={final['tta_flip_left_right']}").encode()
    ).hexdigest()
    identity = {
        "kind": "softmax_ensemble",
        "arm": final["arm"],
        "seeds": [m["seed"] for m in final["members"]],
        "epochs": [m["epoch"] for m in final["members"]],
        "member_sha256": [m["sha256"] for m in final["members"]],
        "tta_flip_left_right": bool(final["tta_flip_left_right"]),
        "ensemble_id_sha256": ensemble_id,
        "selection_file": selection_path,
    }
    return module, infos[0]["config"], identity


def load_model_or_ensemble(path: str, device: str = "cpu"):
    """`path` is either one checkpoint (.pt) or the selection.json of the final ensemble.

    Returns (module, ckpt_cfg, identity) in both cases, so scripts that took a
    single checkpoint can take the final ensemble with no other change.
    """
    if is_selection_file(path):
        return load_final_model(path, device)
    model, info = load_member(path, device)
    identity = {"kind": "single_checkpoint", "path": path, "sha256": info["sha256"],
                "epoch": info["epoch"]}
    return model, info["config"], identity


def load_ensemble(paths: Sequence[str], tta_flip: bool = False, device: str = "cpu"):
    """Load members and check they share input layout and preprocessing."""
    models: List[nn.Module] = []
    infos: List[Dict[str, Any]] = []
    for p in paths:
        m, info = load_member(p, device)
        models.append(m)
        infos.append(info)
    ref = infos[0]
    for info in infos[1:]:
        for key in ("context_slices", "in_channels", "preprocessing"):
            if info[key] != ref[key]:
                raise ValueError(f"Ensemble members differ in {key}: {ref['path']} vs {info['path']}")
    ens = SoftmaxEnsemble(models, tta_flip=tta_flip).to(device)
    ens.eval()
    return ens, infos
