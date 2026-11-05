import e3nn
import torch
from torch_scatter import scatter

from proteinworkshop.models.graph_encoders.components import irreps_tools


class TensorProductConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_irreps,
        out_irreps,
        sh_irreps,
        edge_feats_dim,
        mlp_dim,
        aggr="add",
        batch_norm=True,
        gate=False,
        dropout=0.1,
    ):
        """Tensor Field Network GNN Layer in e3nn

        Implements a Tensor Field Network equivariant GNN layer for higher-order tensors, using e3nn.
        Implementation adapted from: https://github.com/gcorso/DiffDock/

        Paper: Tensor Field Networks, Thomas, Smidt et al.

        Args:
            in_irreps: (e3nn.o3.Irreps) Input irreps dimensions
            out_irreps: (e3nn.o3.Irreps) Output irreps dimensions
            sh_irreps: (e3nn.o3.Irreps) Spherical harmonic irreps dimensions
            edge_feats_dim: (int) Edge feature dimensions
            mlp_dim: (int) Hidden dimension of MLP for computing tensor product weights
            aggr: (str) Message passing aggregator
            batch_norm: (bool) Whether to apply equivariant batch norm
            gate: (bool) Whether to apply gated non-linearity
            dropout: (float) Dropout rate
        """
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.edge_feats_dim = edge_feats_dim
        self.aggr = aggr

        if gate:
            (
                irreps_scalars,
                irreps_gates,
                irreps_gated,
            ) = irreps_tools.irreps2gate(e3nn.o3.Irreps(out_irreps))
            act_scalars = [torch.nn.functional.silu for _, ir in irreps_scalars]
            act_gates = [torch.sigmoid for _, ir in irreps_gates]
            if irreps_gated.num_irreps == 0:
                self.gate = e3nn.nn.Activation(out_irreps, acts=[torch.nn.functional.silu])
            else:
                self.gate = e3nn.nn.Gate(
                    irreps_scalars,
                    act_scalars,
                    irreps_gates,
                    act_gates,
                    irreps_gated,
                )

                self.out_irreps = out_irreps = self.gate.irreps_in
        else:
            self.gate = None

        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(edge_feats_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_dim, self.tp.weight_numel),
        )

        self.batch_norm = e3nn.nn.BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_feat):
        src, dst = edge_index

        tp = self.tp(node_attr[dst], edge_attr, self.fc(edge_feat))

        out = scatter(tp, src, dim=0, reduce=self.aggr)

        if self.gate:
            out = self.gate(out)
        if self.batch_norm:
            out = self.batch_norm(out)
        return out
