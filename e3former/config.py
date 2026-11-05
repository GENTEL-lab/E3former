"""Task-aware configuration shared by training and checkpoint evaluation."""

from pathlib import Path

from omegaconf import OmegaConf

from proteinworkshop import register_custom_omegaconf_resolvers
from proteinworkshop.models.utils import get_input_dim

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "proteinworkshop/config"
DATASETS = {"cath": "cath", "cath-af": "cathAF", "ppbs": "binding_site", "ppbs-af": "bindingsiteAF"}
FEATURES = {"seq": "ca_seq", "angles": "ca_angles", "backbone": "ca_bb"}
ENCODERS = tuple(sorted(p.stem for p in (CONFIG / "encoder").glob("*.yaml")))


def make_config(dataset, features="seq", encoder="e3former", base_radius=None, energy_radius=None):
    if not OmegaConf.has_resolver("resolve_feature_config_dim"):
        register_custom_omegaconf_resolvers()
    inverse = dataset in {"cath", "cath-af"}
    feat = OmegaConf.load(CONFIG / "features" / f"{FEATURES[features]}.yaml")
    if inverse:
        feat.scalar_node_features.remove("amino_acid_one_hot")
    output = "residue_type" if inverse else "node_label"
    classes = 23 if inverse else 2
    decoder = OmegaConf.load(CONFIG / "decoder" / f"{output}.yaml")
    decoder[output].out_dim = classes
    cfg = OmegaConf.create(
        {
            "dataset_name": dataset,
            "feature_set": features,
            "encoder_name": encoder,
            "features": feat,
            "encoder": OmegaConf.load(CONFIG / "encoder" / f"{encoder}.yaml"),
            "decoder": decoder,
            "dataset": {"num_classes": classes},
            "task": {
                "task": "inverse_folding" if inverse else "classification",
                "classification_type": "multiclass" if inverse else "binary",
                "metric_average": "micro",
                "losses": {output: "cross_entropy" if inverse else "bce"},
                "label_smoothing": 0.0,
                "output": [output],
                "supervise_on": [output],
            },
            "metrics": {},
            "optimiser": OmegaConf.load(CONFIG / "optimiser/adam.yaml"),
            "scheduler": None,
        }
    )
    if encoder == "e3former":
        cfg.encoder.base_radius = (
            base_radius if base_radius is not None else (9.5 if inverse else 7.0)
        )
        cfg.encoder.energy_radius = (
            energy_radius if energy_radius is not None else (2.5 if inverse else 3.0)
        )
        cfg.encoder.node_s_nums = get_input_dim(cfg.features, "scalar_node_features", cfg.task)
    elif base_radius is not None or energy_radius is not None:
        raise ValueError("Radius options are specific to E3former.")
    if encoder == "gcpnet":
        from proteinworkshop.configs.config import validate_gcpnet_config

        validate_gcpnet_config(cfg)
    OmegaConf.resolve(cfg)
    return cfg
