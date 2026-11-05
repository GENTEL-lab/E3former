"""Implementation of the AUPRC metric in ``torchmetrics``."""

import torch
from torchmetrics import Metric
from sklearn.metrics import precision_recall_curve, average_precision_score


class AUPRC(Metric):
    """Class for AUPRC metric.

    :param compute_on_cpu: Whether to compute the metric on CPU,
            defaults to ``True``.
    :type compute_on_cpu: bool, optional
    """

    def __init__(self, compute_on_cpu: bool = True) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.compute_on_cpu = compute_on_cpu

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the current state of the metric.

        :param preds: Tensor of predictions
        :type preds: torch.Tensor
        :param target: Tensor of targets
        :type target: torch.Tensor
        """
        preds = preds[:, 1]
        target = target[:, 1]

        self.preds.append(preds.detach())
        self.targets.append(target.detach())

    def compute(self) -> torch.Tensor:
        """
        Computes the AUPRC metric.
        :return: AUPRC metric value
        :rtype: torch.Tensor
        """
        return self.auprc(torch.cat(self.preds).flatten(), torch.cat(self.targets).flatten())

    @staticmethod
    def auprc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Area under precision-recall curve.

        Parameters:
        :param pred: predictions. Shape ``(N,)``
        :param target: binary targets of shape ``(N,)``
        """
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        auprc_value = average_precision_score(target_np, pred_np)

        return torch.tensor(auprc_value)
