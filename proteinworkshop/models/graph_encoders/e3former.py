"""Energy-adaptive equivariant encoder with Elastic-SSM depth aggregation."""

from typing import Set

import torch
from e3nn import o3

from proteinworkshop.custom_types import EncoderOutput
from proteinworkshop.models.graph_encoders.components import e3formertool
from proteinworkshop.models.graph_encoders.components.elastic_ssm import (
    HighOrderLift,
    QuadraticProjection,
)
from proteinworkshop.models.graph_encoders.components.eq_radial_basis import RadialBasis
from proteinworkshop.models.graph_encoders.layers import e3former as blocks
from proteinworkshop.models.graph_encoders.layers.e3tp.drop import EquivariantDropout
from proteinworkshop.models.graph_encoders.layers.e3tp.fast_activation import Activation, Gate
from proteinworkshop.models.graph_encoders.layers.e3tp.tensor_product_rescale import (
    LinearRS,
    irreps2gate,
)
from proteinworkshop.models.utils import get_aggregation

_AVG_DEGREE = 15.0
_AVG_NUM_NODES = 10.0


class E3FormerModel(torch.nn.Module):
    def __init__(
        self,
        node_s_nums,
        node_v_nums=0,
        irreps_in=None,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_add_embedding="128x0e+64x1e+32x2e+16x3e+8x4e+4x5e+2x6e",
        irreps_node_leaky_embedding="16x3e+8x4e+4x5e+2x6e",
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        base_radius=9.5,
        energy_radius=2.5,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=(64, 64),
        irreps_feature="512x0e",
        irreps_head="32x0e+32x1o+16x2e",
        num_heads=4,
        irreps_pre_attn=None,
        irreps_node_ssm_compress="30x0e",
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="128x0e+128x1e+64x2e",
        norm_layer="layer",
        irreps_node_ssm_mid="32x0e+16x1e+8x2e",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        deterministic_radius=False,
        ssm_version=1,
        mean=None,
        std=None,
        scale=None,
        atomref=None,
    ):
        super().__init__()
        if node_s_nums < 1 or node_v_nums != 0:
            raise ValueError("E3former expects scalar residue features (node_v_nums=0).")
        if num_layers < 2 or base_radius <= 0 or energy_radius < 0:
            raise ValueError("Use at least two layers and nonnegative radius parameters.")
        if ssm_version not in (1, 2):
            raise ValueError("Unsupported Elastic-SSM checkpoint version.")
        self.ssm_version = ssm_version
        self.node_s_nums = node_s_nums
        self.readout = get_aggregation("sum")
        self.base_radius = base_radius
        self.energy_radius = energy_radius
        self.deterministic_radius = deterministic_radius
        self.num_layers = num_layers
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_node_add_embedding = o3.Irreps(irreps_node_add_embedding)
        self.irreps_node_leaky_embedding = o3.Irreps(irreps_node_leaky_embedding)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.irreps_node_ssm_mid = o3.Irreps(irreps_node_ssm_mid)
        self.irreps_node_ssm_compress = o3.Irreps(irreps_node_ssm_compress)
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh)
            if irreps_sh is not None
            else o3.Irreps.spherical_harmonics(self.irreps_node_embedding.lmax)
        )
        self.irreps_combine_feature = (
            self.irreps_feature + self.irreps_node_ssm_compress * (num_layers - 1)
        ).simplify()
        self.register_buffer("atomref", atomref)
        self.atom_embed = blocks.NodeEmbeddingNetwork(
            self.irreps_node_embedding, node_s_nums, node_v_nums
        )
        cutoff = base_radius + energy_radius
        if basis_type == "gaussian":
            self.rbf = e3formertool.GaussianRadialBasisLayer(number_of_basis, cutoff)
        elif basis_type == "bessel":
            self.rbf = RadialBasis(number_of_basis, cutoff=cutoff, rbf={"name": "spherical_bessel"})
        else:
            raise ValueError(f"Unknown radial basis: {basis_type}")
        fc_neurons = [number_of_basis, *fc_neurons]
        self.edge_deg_embed = blocks.EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding, self.irreps_edge_attr, fc_neurons, _AVG_DEGREE
        )
        self.blocks = torch.nn.ModuleList(
            [
                blocks.TransBlock(
                    irreps_node_input=self.irreps_node_embedding,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_edge_attr=self.irreps_edge_attr,
                    irreps_node_output=(
                        self.irreps_node_add_embedding
                        if i < num_layers - 1
                        else self.irreps_feature
                    ),
                    fc_neurons=fc_neurons,
                    irreps_head=o3.Irreps(irreps_head),
                    num_heads=num_heads,
                    irreps_pre_attn=irreps_pre_attn,
                    rescale_degree=rescale_degree,
                    nonlinear_message=nonlinear_message,
                    alpha_drop=alpha_drop,
                    proj_drop=proj_drop,
                    drop_path_rate=drop_path_rate,
                    irreps_mlp_mid=o3.Irreps(irreps_mlp_mid),
                    norm_layer=norm_layer,
                )
                for i in range(num_layers)
            ]
        )
        self._build_ssm()
        self.norm = blocks.get_norm_layer(norm_layer)(self.irreps_feature)
        self.out_dropout = EquivariantDropout(self.irreps_feature, out_drop) if out_drop else None
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=True),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps("32x0e"), rescale=True),
        )
        self.apply(self._init_weights)
        self.ssmh_dtp = blocks.FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_ssm_mid,
            self.irreps_node_ssm_mid,
            self.irreps_node_ssm_mid,
            bias=False,
            internal_weights=None,
        )
        self.comblock = LinearRS(self.irreps_combine_feature, self.irreps_feature, rescale=True)
        if self.ssm_version == 2:
            self.high_order_lift = HighOrderLift(
                self.irreps_node_embedding, self.irreps_node_leaky_embedding
            )
            self.WB[0] = QuadraticProjection(
                self.irreps_node_leaky_embedding, self.WB[0].irreps_out
            )

    def _build_ssm(self):
        scalars, gates, gated = irreps2gate(self.irreps_node_ssm_mid)
        gate = (
            Gate(
                scalars,
                [torch.nn.SiLU() for _ in scalars],
                gates,
                [torch.sigmoid for _ in gates],
                gated,
            )
            if gated.num_irreps
            else Activation(self.irreps_node_ssm_mid, acts=[torch.sigmoid])
        )
        mid_output = (scalars + gates + gated).simplify()
        self.WB = torch.nn.Sequential(
            LinearRS(self.irreps_node_leaky_embedding, mid_output, rescale=True), gate
        )
        self.WC = torch.nn.Sequential(
            LinearRS(self.irreps_node_ssm_mid, mid_output, rescale=True),
            gate,
            LinearRS(self.irreps_node_ssm_mid, self.irreps_node_ssm_compress, rescale=True),
        )
        self.WA = torch.nn.Sequential(
            LinearRS(self.irreps_node_ssm_mid, mid_output, rescale=True), gate
        )
        self.WH = torch.nn.Sequential(
            LinearRS(self.irreps_node_ssm_mid, mid_output, rescale=True), gate
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "batch"}

    def forward(self, batch) -> EncoderOutput:
        if batch.x.shape[-1] != self.node_s_nums:
            raise ValueError(
                f"Expected {self.node_s_nums} residue features, got {batch.x.shape[-1]}."
            )
        energy, *_ = e3formertool.calculate_energy_sum(
            self.base_radius, self.energy_radius, batch.pos, batch.batch
        )
        radii = e3formertool.set_radius(
            energy,
            self.base_radius,
            self.energy_radius,
            batch.batch,
            deterministic=self.deterministic_radius,
        )
        src, dst = e3formertool.energy_radius_graph(
            batch.pos, radii, batch.batch, max_num_neighbors=100
        )
        edge_vec = batch.pos[src] - batch.pos[dst]
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, normalize=True, normalization="component"
        )
        edge_rbf = self.rbf(edge_vec.norm(dim=1))
        embedded = self.atom_embed(batch.x, batch.x)
        features = embedded + self.edge_deg_embed(
            embedded, edge_sh, edge_rbf, src, dst, batch.batch
        )
        attrs = torch.ones_like(features[:, :1])
        summaries = []
        state = None
        width = self.irreps_node_leaky_embedding.dim
        for i, block in enumerate(self.blocks):
            features = block(
                node_input=features,
                node_attr=attrs,
                edge_src=src,
                edge_dst=dst,
                edge_attr=edge_sh,
                edge_scalars=edge_rbf,
                batch=batch.batch,
            )
            if i < self.num_layers - 1:
                features = features[:, :-width]
                # Checkpoints without a version retain their original feature routing.
                source = (
                    self.high_order_lift(features)
                    if self.ssm_version == 2
                    else features[:, -width:]
                )
                ssm_input = self.WB(source)
                if state is None:
                    state = ssm_input
                state = self.WH(self.ssmh_dtp(ssm_input, self.WA(state)))
                summaries.append(self.WC(state))
        features = self.comblock(torch.cat([features, *summaries], dim=1))
        features = self.norm(features, batch=batch.batch)
        if self.out_dropout is not None:
            features = self.out_dropout(features)
        outputs = self.head(features) / _AVG_NUM_NODES
        return EncoderOutput(
            {"node_embedding": outputs, "graph_embedding": self.readout(outputs, batch.batch)}
        )
