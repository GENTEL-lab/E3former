import unittest

import torch

from proteinworkshop.models.graph_encoders.components.e3formertool import (
    calculate_interaction_energy,
)


class EnergyTests(unittest.TestCase):
    def test_defaults_preserve_values_and_masking(self):
        for dtype in (torch.float32, torch.float64):
            distance = torch.cat(
                [
                    torch.linspace(1.0, 20.0, 257, dtype=dtype),
                    torch.tensor([0.0, float("nan"), float("inf")], dtype=dtype),
                ]
            )
            ratio = 3.8 / distance
            expected = 1.0 * (torch.pow(ratio, 12) - 2 * torch.pow(ratio, 6))
            expected[torch.isnan(distance) | (distance == 0)] = 0.0
            expected = torch.abs(expected)
            actual = calculate_interaction_energy(distance)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(actual.dtype, distance.dtype)
            self.assertEqual(actual.device, distance.device)

    def test_epsilon_and_sigma_are_used(self):
        distance = torch.tensor([2.0, 3.8, 6.0], dtype=torch.float64)
        default = calculate_interaction_energy(distance)
        torch.testing.assert_close(
            calculate_interaction_energy(distance, epsilon=2.0), 2 * default
        )
        self.assertTrue(
            torch.equal(calculate_interaction_energy(distance, epsilon=0.0), 0 * default)
        )
        ratio = 1.0 / distance
        expected = torch.abs(2.0 * (ratio.pow(12) - 2 * ratio.pow(6)))
        actual = calculate_interaction_energy(distance, epsilon=2.0, sigma=1.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, default))
        torch.testing.assert_close(
            calculate_interaction_energy(torch.tensor([1.0]), epsilon=2.0, sigma=1.0),
            torch.tensor([2.0]),
        )

    def test_custom_parameters_preserve_distance_gradients(self):
        distance = torch.tensor([2.0, 3.8, 6.0], dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: calculate_interaction_energy(x, epsilon=2.0, sigma=1.0),
                (distance,),
            )
        )


if __name__ == "__main__":
    unittest.main()
