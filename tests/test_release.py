import unittest

import hydra
import torch
from torch_geometric.data import Batch, Data

from e3former.config import DATASETS, FEATURES, make_config
from e3former.data import ProcessedProteins, audit, data_root
from e3former.evaluate import evaluate


class ReleaseTests(unittest.TestCase):
    def test_feature_dimensions_and_no_inverse_sequence_leakage(self):
        for dataset in DATASETS:
            for features in FEATURES:
                cfg = make_config(dataset, features)
                data = ProcessedProteins(data_root(), dataset, limit=2)
                batch = Batch.from_data_list([data[0], data[1]])
                featuriser = hydra.utils.instantiate(cfg.features)
                original = featuriser(batch.clone()).x
                self.assertEqual(original.shape[-1], cfg.encoder.node_s_nums)
                if dataset.startswith("cath"):
                    batch.residue_type = (batch.residue_type + 1) % 23
                    self.assertTrue(torch.equal(original, featuriser(batch).x))

    def test_complete_nonoverlapping_splits(self):
        result = audit(data_root())
        for entry in result.values():
            for split in ("train", "validation", "test"):
                self.assertEqual(entry[split]["missing"], [])
            self.assertEqual(sum(entry["overlap"].values()), 0)

    def test_ppl_is_residue_weighted_and_final_batch_is_counted(self):
        class FakeModel:
            config = make_config("cath")

            def eval(self):
                return self

            def featurise(self, batch):
                return batch

            def __call__(self, batch):
                return {"residue_type": batch.x}

        a = Data(x=torch.tensor([[3.0, 0.0]]), residue_type=torch.tensor([0]), id="a")
        b = Data(
            x=torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]),
            residue_type=torch.tensor([0, 0, 0]),
            id="b",
        )
        batches = [Batch.from_data_list([a]), Batch.from_data_list([b])]
        result = evaluate(FakeModel(), batches, "cpu")
        expected = (
            torch.nn.functional.cross_entropy(
                torch.cat([a.x, b.x]), torch.cat([a.residue_type, b.residue_type])
            )
            .exp()
            .item()
        )
        self.assertAlmostEqual(result["ppl"], expected, places=5)
        self.assertEqual(result["proteins"], 2)
        self.assertEqual(result["residues"], 4)


if __name__ == "__main__":
    unittest.main()
