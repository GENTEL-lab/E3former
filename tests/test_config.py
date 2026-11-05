import contextlib
import io
import unittest

import torch
from omegaconf import OmegaConf

from e3former.config import ROOT, make_config
from proteinworkshop.configs.config import (
    ExperimentConfigurationError,
    validate_supervision_config,
)
from proteinworkshop.models.base import BenchMarkModel


class ConfigTests(unittest.TestCase):
    def test_invalid_supervision_raises_without_debug_output(self):
        cfg = OmegaConf.create({"task": {"supervise_on": "residue_type"}})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ExperimentConfigurationError, "must be a list"):
                validate_supervision_config(cfg)
        self.assertEqual(output.getvalue(), "")

    def test_valid_supervision_is_unchanged(self):
        cfg = make_config("cath-af", "seq")
        expected = OmegaConf.to_container(cfg, resolve=True)
        self.assertIs(validate_supervision_config(cfg), cfg)
        self.assertEqual(OmegaConf.to_container(cfg, resolve=True), expected)

    def test_plateau_scheduler_uses_validation_loss(self):
        model = torch.nn.Linear(1, 1)
        model.config = make_config("cath-af", "seq")
        self.assertIsNone(model.config.scheduler)
        model.config.scheduler = OmegaConf.load(
            ROOT / "proteinworkshop/config/scheduler/plateau.yaml"
        )
        configured = BenchMarkModel.configure_optimizers(model)
        self.assertEqual(configured["lr_scheduler"]["monitor"], "val/loss")
        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        optimizer = configured["optimizer"]
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(scheduler.patience + 2):
            scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], initial_lr * 0.6)


if __name__ == "__main__":
    unittest.main()
