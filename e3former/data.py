"""Read the released processed proteins and fixed split manifests."""

import hashlib
import json
import os
from pathlib import Path

import torch
from graphein.protein.tensor.dataloader import ProteinDataLoader
from torch.utils.data import Dataset

from e3former.config import DATASETS, ROOT
from proteinworkshop.features.sequence_features import amino_acid_one_hot

PPBS_SPLITS = ("70", "homology", "topology", "none")


def data_root(value=None):
    return Path(value or os.environ.get("E3FORMER_DATA", ROOT / "data")).expanduser().resolve()


def split_ids(root, dataset, split="test", subset="all"):
    directory = Path(root) / DATASETS[dataset]
    if dataset in {"cath", "cath-af"}:
        with (directory / "chain_set_splits.json").open() as f:
            ids = json.load(f)[split]
        ids = [name.replace(".", "_") for name in ids]
    else:
        suffixes = (
            [""] if split == "train" else (list(PPBS_SPLITS) if subset == "all" else [subset])
        )
        ids = []
        for suffix in suffixes:
            stem = split if not suffix else f"{split}_{suffix}"
            ids.extend((directory / "labels" / f"labels_{stem}_ids.txt").read_text().splitlines())
        ids = [name.strip() for name in ids if name.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate protein IDs in {dataset}/{split}/{subset}.")
    return ids


class ProcessedProteins(Dataset):
    def __init__(self, root, dataset, split="test", subset="all", limit=None):
        self.directory = Path(root) / DATASETS[dataset] / "processed"
        self.ids = split_ids(root, dataset, split, subset)
        if limit is not None:
            self.ids = self.ids[:limit]
        if not self.ids:
            raise ValueError(f"Empty split: {dataset}/{split}/{subset}")
        missing = [name for name in self.ids if not (self.directory / f"{name}.pt").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} processed proteins missing in {self.directory}: {missing[:10]}"
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        data = torch.load(self.directory / f"{self.ids[index]}.pt", map_location="cpu")
        data.x = torch.zeros(data.coords.shape[0])
        data.amino_acid_one_hot = amino_acid_one_hot(data)
        data.seq_pos = torch.arange(data.coords.shape[0]).unsqueeze(-1)
        data.id = self.ids[index]
        return data


def loader(
    root, dataset, split="test", subset="all", batch_size=8, workers=0, limit=None, shuffle=False
):
    return ProteinDataLoader(
        ProcessedProteins(root, dataset, split, subset, limit),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
    )


def audit(root):
    result = {}
    for dataset, folder in DATASETS.items():
        result[dataset] = {}
        for split in ("train", "validation", "test"):
            ids = split_ids(root, dataset, split)
            missing = [
                name
                for name in ids
                if not (Path(root) / folder / "processed" / f"{name}.pt").is_file()
            ]
            result[dataset][split] = {
                "proteins": len(ids),
                "missing": missing,
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            }
        sets = {s: set(split_ids(root, dataset, s)) for s in result[dataset]}
        result[dataset]["overlap"] = {
            "train_validation": len(sets["train"] & sets["validation"]),
            "train_test": len(sets["train"] & sets["test"]),
            "validation_test": len(sets["validation"] & sets["test"]),
        }
    return result
