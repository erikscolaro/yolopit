"""The cost term added to the loss during the search.

- StandardRegularizer: sum of lambda_m * (PIT cost_m / its initial value), the "classic" PLiNIO
  use: it keeps pruning as long as the lambda outweighs the loss of accuracy.
- DuccioRegularizer: PLiNIO's DUCCIO (arXiv:2206.00302): pushes each cost down only while it is
  above its target, with a strength that starts small and grows over the first half of the
  search. Targets are given for the whole model and converted here to the PIT part.
"""
from __future__ import annotations

import torch
from plinio.regularizers import DUCCIO


class StandardRegularizer:
    mode = "standard"

    def __init__(self, lambdas: dict, cost0: dict):
        self.lambdas = dict(lambdas)
        self.cost0 = dict(cost0)

    def __call__(self, net, epoch: int, epochs: int) -> torch.Tensor:
        return sum(lam * net.get_cost(m) / self.cost0[m] for m, lam in self.lambdas.items())

    def describe(self) -> str:
        return " + ".join(f"{lam:g}*{m}" for m, lam in self.lambdas.items()) + \
            " (each as a fraction of its initial value)"


class DuccioRegularizer:
    mode = "duccio"

    def __init__(self, targets_pit: dict):
        self.targets_pit = dict(targets_pit)
        self._duccio = None

    @property
    def ready(self) -> bool:
        return self._duccio is not None

    def start(self, task_loss: float):
        """DUCCIO scales its strengths on the task loss at the beginning of the search."""
        self._duccio = DUCCIO(self.targets_pit, task_loss=float(task_loss))

    def __call__(self, net, epoch: int, epochs: int) -> torch.Tensor:
        if self._duccio is None:
            raise RuntimeError("DuccioRegularizer.start(task_loss) was not called")
        return self._duccio(net, epoch=epoch, n_epochs=epochs)

    def strengths(self) -> dict:
        if self._duccio is None or self._duccio.final_strengths is None:
            return {}
        return {k: float(v) for k, v in self._duccio.final_strengths.items()}

    def describe(self) -> str:
        if not self.targets_pit:
            return "DUCCIO, no target below the initial cost: nothing to prune"
        return "DUCCIO, targets on the PIT part: " + ", ".join(
            f"{m} <= {v:,.0f}" for m, v in self.targets_pit.items())
