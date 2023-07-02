import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLinearLayers(nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear1 = nn.Linear(in_features=in_features, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=1024)
        self.fc = nn.Linear(in_features=1024, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)
        y = F.tanh(y)
        y = self.linear2(y)
        y = F.tanh(y)
        y = self.fc(y)

        return y