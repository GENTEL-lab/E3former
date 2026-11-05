"""Portable checkpoints carry the complete model and feature configuration."""

import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from e3former.config import ROOT
from e3former.model import Model


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest():
    return json.loads((ROOT / "checkpoints/manifest.json").read_text())


def resolve_checkpoint(value):
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    entries = manifest()
    if value not in entries:
        raise FileNotFoundError(f"Unknown checkpoint {value!r}. Available: {', '.join(entries)}")
    path = ROOT / "checkpoints" / entries[value]["file"]
    if checksum(path) != entries[value]["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return path


def load_checkpoint(value, device="cpu"):
    path = resolve_checkpoint(value)
    payload = torch.load(path, map_location="cpu")
    config = payload.get("config", payload.get("hyper_parameters", {}).get("cfg"))
    if config is None or "dataset_name" not in config:
        raise ValueError("Use a release checkpoint or a checkpoint written by this training CLI.")
    cfg = OmegaConf.create(config)
    model = Model(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), cfg, path
