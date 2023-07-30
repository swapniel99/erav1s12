import torch
from torch import Tensor
from torchmetrics import Metric


class RunningAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        preds = preds.argmax(dim=1)
        total = target.numel()
        self.correct += preds.eq(target).sum()
        self.total += total

    def compute(self):
        return 100 * self.correct.float() / self.total
