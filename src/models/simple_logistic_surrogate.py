import torch
import torch.nn as nn


class SimpleLogisticSurrogate(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(SimpleLogisticSurrogate, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, dtype=torch.float64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)