import torch
import torch.distributions.beta as beta


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )


def adaptive_prob_distribution(param_value):
    my_alpha = 1.0 + param_value * 8.0
    my_beta = 1.0 + (1.0 - param_value) * 8.0
    prob_dist = beta.Beta(my_alpha, my_beta)
    sample_values = prob_dist.sample()

    return sample_values


def calculate_interaction_energy(distance, epsilon=1.0, sigma=3.8):
    r_by_sigma = sigma / distance
    r_by_sigma_6 = torch.pow(r_by_sigma, 6)
    r_by_sigma_12 = torch.pow(r_by_sigma, 12)
    energy = epsilon * (r_by_sigma_12 - 2 * r_by_sigma_6)
    mask = torch.isnan(distance) | (distance == 0)
    energy[mask] = 0.0
    return torch.abs(energy)


def calculate_energy_sum(base_radius, energy_radius, pos, batch):
    radius = base_radius + energy_radius
    num_samples = pos.size(0)
    max_batch = batch.max().item()
    energy_sum = torch.zeros(num_samples, device=pos.device)
    num_neighbors = torch.zeros(num_samples, device=pos.device, dtype=torch.int32)

    batch_energy_max = torch.zeros(max_batch + 1, device=pos.device)
    batch_energy_min = torch.zeros(max_batch + 1, device=pos.device)
    batch_energy_avg = torch.zeros(max_batch + 1, device=pos.device)
    batch_energy_var = torch.zeros(max_batch + 1, device=pos.device)

    for i in range(max_batch + 1):
        mask = batch == i

        if mask.sum().item() == 0:
            continue

        pos_batch = pos[mask]
        dist = torch.norm(pos_batch[:, None] - pos_batch, dim=2)

        neighbor_mask = dist <= radius

        dist[dist >= radius] = 999999
        energy = calculate_interaction_energy(dist, epsilon=1.0, sigma=3.8)
        energy_batch_sum = energy.sum(dim=1)

        energy_sum[mask] = energy_batch_sum
        num_neighbors[mask] = neighbor_mask.sum(dim=1).int()

        batch_energy_max[i] = energy_batch_sum.max().item()
        batch_energy_min[i] = energy_batch_sum.min().item()
        batch_energy_avg[i] = energy_batch_sum.mean().item()
        batch_energy_var[i] = energy_batch_sum.var().item()

    return (
        energy_sum,
        batch_energy_max,
        batch_energy_min,
        batch_energy_avg,
        batch_energy_var,
        num_neighbors,
    )


def set_radius(
    energy,
    base_radius,
    energy_radius,
    batch,
    deterministic=False,
):
    unique_batches = torch.unique(batch)
    radius_list = torch.empty_like(energy)

    for b in unique_batches:
        mask = batch == b
        energy_batch = energy[mask]

        energy_max = energy_batch.max()
        energy_min = energy_batch.min()
        energy_normalized = torch.log(energy_batch - energy_min + 1) / torch.log(
            energy_max - energy_min + 1
        )

        radius_min = base_radius
        radius_max = base_radius + energy_radius

        if deterministic:
            radius_weight = energy_normalized
            radius_batch = radius_min + (radius_max - radius_min) * radius_weight
        else:
            sameple_values = adaptive_prob_distribution(energy_normalized)
            radius_batch = radius_min + (radius_max - radius_min) * sameple_values

        radius_list[mask] = radius_batch

    return radius_list


def energy_radius_graph(
    x,
    radius_list,
    batch=None,
    loop=False,
    max_num_neighbors=50,
    flow="source_to_target",
):
    assert flow in ["source_to_target", "target_to_source"]
    assert len(radius_list) == x.size(0)

    row, col = custom_radius(x, radius_list, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == "source_to_target" else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def custom_radius(x, radius_list, batch=None, max_num_neighbors=50):
    assert len(radius_list) == x.size(0)

    if batch is None:
        groups = [torch.arange(x.size(0), device=x.device)]
    else:
        groups = [(batch == group).nonzero().flatten() for group in torch.unique(batch)]

    edges = []
    for indices in groups:
        positions = x[indices]
        for start in range(0, indices.numel(), 256):
            centers = indices[start : start + 256]
            distances = torch.norm(x[centers, None] - positions[None], dim=-1)
            inside = distances <= radius_list[centers, None]
            if batch is not None:
                inside &= centers[:, None] != indices[None]
            if torch.any(inside.sum(dim=1) > max_num_neighbors):
                # Keep the original truncation behavior for overfull neighborhoods.
                return _custom_radius_overflow(x, radius_list, batch, max_num_neighbors)
            row, col = inside.nonzero(as_tuple=True)
            row, col = centers[row], indices[col]
            keep = row != col
            edges.append(torch.stack((row[keep], col[keep])))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    result = torch.cat(edges, dim=1)
    order = torch.argsort(result[0] * x.size(0) + result[1])
    return result[:, order]


def _custom_radius_overflow(x, radius_list, batch, max_num_neighbors):
    row_list = []
    col_list = []
    for i in range(x.size(0)):
        center = x[i]

        distances = torch.norm(x - center, dim=1)
        mask = distances <= radius_list[i]
        indices = torch.nonzero(mask).squeeze(1)

        if batch is not None:
            indices = indices[batch[indices] == batch[i]]
            non_center_mask = indices != i
            indices = indices[non_center_mask]

        if indices.size(0) > max_num_neighbors:
            _, topk_indices = torch.topk(distances, max_num_neighbors, largest=False)
            indices = indices[topk_indices]

        indices = indices[indices != i]

        row_list.append(torch.full_like(indices, i))
        col_list.append(indices)

    row = torch.cat(row_list, dim=0)
    col = torch.cat(col_list, dim=0)

    return torch.stack([row, col], dim=0)
