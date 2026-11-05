import unittest

import torch
from e3nn import o3

from e3former.config import DATASETS, FEATURES, make_config
from proteinworkshop.models.graph_encoders.components.elastic_ssm import (
    HighOrderLift,
    QuadraticProjection,
)


class ElasticSSMTests(unittest.TestCase):
    def test_new_training_uses_version_two_for_all_features(self):
        for dataset in DATASETS:
            for features in FEATURES:
                self.assertEqual(make_config(dataset, features).encoder.ssm_version, 2)

    def test_lift_and_projection_have_equivariant_input_paths(self):
        torch.manual_seed(52)
        low = o3.Irreps("8x0e+4x1e+2x2e")
        high = o3.Irreps("2x3e+2x4e+1x5e+1x6e")
        output = o3.Irreps("4x0e+2x1e+1x2e")
        lift = HighOrderLift(low, high).double()
        projection = QuadraticProjection(high, output).double()
        self.assertGreater(projection.tp.weight.numel(), 0)
        self.assertTrue(torch.equal(projection.tp.weight, torch.zeros_like(projection.tp.weight)))
        with torch.no_grad():
            projection.tp.weight.normal_(std=0.1)
        x = torch.randn(5, low.dim, dtype=torch.float64, requires_grad=True)
        features = lift(x)
        for section in high.slices():
            self.assertGreater(features[:, section].abs().max().item(), 0)
        result = projection(features)
        result.square().sum().backward()
        self.assertGreater(x.grad.abs().max().item(), 0)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = lift(x.detach() @ low.D_from_matrix(rotation).T)
        torch.testing.assert_close(
            rotated,
            features.detach() @ high.D_from_matrix(rotation).T,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            projection(rotated),
            result.detach() @ output.D_from_matrix(rotation).T,
            atol=2e-5,
            rtol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
