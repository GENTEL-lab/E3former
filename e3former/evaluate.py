"""Global residue-weighted inverse-folding and binding-site metrics."""

import csv
import math

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


@torch.no_grad()
def evaluate(model, dataloader, device, predictions_path=None):
    inverse = model.config.task.task == "inverse_folding"
    output = "residue_type" if inverse else "node_label"
    loss_sum = correct = residues = proteins = 0
    scores, targets, records = [], [], []
    model.eval()
    for batch in dataloader:
        batch = model.featurise(batch.to(device))
        logits = model(batch)[output]
        target = batch.residue_type if inverse else batch.node_y.long()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite predictions for {batch.id}")
        if inverse:
            per_residue = F.cross_entropy(logits, target, reduction="none")
            loss_sum += per_residue.double().sum().item()
        else:
            scores.append(logits[:, 1].cpu())
            targets.append(target.cpu())
        correct += (logits.argmax(-1) == target).sum().item()
        residues += target.numel()
        proteins += batch.num_graphs
        if predictions_path is not None:
            cpu_logits, cpu_targets = logits.cpu(), target.cpu()
            cpu_loss = per_residue.cpu() if inverse else None
            for i, name in enumerate(batch.id):
                start, end = batch.ptr[i : i + 2].tolist()
                if inverse:
                    records.append(
                        {
                            "protein": name,
                            "residues": end - start,
                            "ppl": math.exp(cpu_loss[start:end].double().mean().item()),
                        }
                    )
                else:
                    for j in range(start, end):
                        records.append(
                            {
                                "protein": name,
                                "residue_index": j - start,
                                "target": cpu_targets[j].item(),
                                "score": cpu_logits[j, 1].item(),
                            }
                        )
    result = {"proteins": proteins, "residues": residues, "accuracy": correct / residues}
    if inverse:
        result.update(loss=loss_sum / residues, ppl=math.exp(loss_sum / residues))
    else:
        result["auprc"] = float(
            average_precision_score(torch.cat(targets).numpy(), torch.cat(scores).numpy())
        )
    if predictions_path is not None:
        with predictions_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return result
