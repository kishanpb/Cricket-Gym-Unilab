"""A local residual actor that initially leaves the external controller unchanged."""

import torch
from rsl_rl.models import MLPModel


class ZeroResidualModel(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)
