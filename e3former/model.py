"""Training hooks with residue-weighted loss and explicit failures."""

import torch

from proteinworkshop.models.base import BenchMarkModel


class Model(BenchMarkModel):
    def _step(self, batch, stage):
        labels = self.get_labels(batch)
        predictions = self(batch)
        loss = self.compute_loss(predictions, labels)["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss for {batch.id}")
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.num_nodes,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs)
