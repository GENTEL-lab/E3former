"""Equivariant high-order features and input projection for Elastic-SSM."""

import torch
from e3nn import o3

from proteinworkshop.models.graph_encoders.layers import e3former as blocks
from proteinworkshop.models.graph_encoders.layers.e3tp.tensor_product_rescale import (
    FullyConnectedTensorProductRescale,
    LinearRS,
)


class HighOrderLift(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        latent = o3.Irreps("4x1e+4x2e")
        first = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l <= 4])
        last = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 4])
        if first + last != self.irreps_out or not first.dim or not last.dim:
            raise ValueError("HighOrderLift requires ordered degree-3 through degree-6 features.")
        self.norm = blocks.get_norm_layer("layer")(self.irreps_in)
        self.left = LinearRS(self.irreps_in, latent, bias=False)
        self.right = LinearRS(self.irreps_in, latent, bias=False)
        self.first = FullyConnectedTensorProductRescale(latent, latent, first, bias=False)
        self.last = FullyConnectedTensorProductRescale(first, latent, last, bias=False)

    def forward(self, features):
        features = self.norm(features)
        left, right = self.left(features), self.right(features)
        first = self.first(left, right)
        high = torch.cat((first, self.last(first, right)), dim=-1)
        return high / (high.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()


class QuadraticProjection(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out):
        super().__init__(irreps_in, irreps_in, irreps_out, bias=True)
        # Start with no quadratic contribution.
        torch.nn.init.zeros_(self.tp.weight)

    def forward(self, features):
        return super().forward(features, features)
