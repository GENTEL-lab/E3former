import unittest

import torch

from proteinworkshop.models.graph_encoders.components.e3formertool import custom_radius


def reference_radius(x, radii, batch, cap):
    rows, cols = [], []
    for i in range(x.size(0)):
        distances = torch.norm(x - x[i], dim=1)
        indices = (distances <= radii[i]).nonzero().flatten()
        if batch is not None:
            indices = indices[batch[indices] == batch[i]]
            indices = indices[indices != i]
        if indices.numel() > cap:
            indices = indices[torch.topk(distances, cap, largest=False).indices]
        indices = indices[indices != i]
        rows.append(torch.full_like(indices, i))
        cols.append(indices)
    return torch.stack((torch.cat(rows), torch.cat(cols)))


class RadiusTests(unittest.TestCase):
    def test_edges_and_order(self):
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(577, 3, generator=generator) * 8
        radii = torch.rand(577, generator=generator) * 5 + 1
        batches = [None, torch.arange(577) % 3, torch.arange(577) // 200]
        for batch in batches:
            with self.subTest(batch=batch is not None):
                torch.testing.assert_close(
                    custom_radius(x, radii, batch, 1000),
                    reference_radius(x, radii, batch, 1000),
                    atol=0,
                    rtol=0,
                )

    def test_radius_boundary_and_zero_degree(self):
        x = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [20.0, 0.0, 0.0]])
        radii = torch.tensor([5.0, 4.999, 0.0])
        torch.testing.assert_close(custom_radius(x, radii), torch.tensor([[0], [1]]))

    def test_existing_truncation_is_unchanged(self):
        x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        radii = torch.full((3,), 100.0)
        for batch in (None, torch.zeros(3, dtype=torch.long)):
            torch.testing.assert_close(
                custom_radius(x, radii, batch, 2), reference_radius(x, radii, batch, 2)
            )

    def test_overfull_batch_preserves_existing_error(self):
        x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        radii = torch.full((5,), 100.0)
        batch = torch.zeros(5, dtype=torch.long)
        for function in (reference_radius, custom_radius):
            with self.assertRaises(IndexError):
                function(x, radii, batch, 1)


if __name__ == "__main__":
    unittest.main()
