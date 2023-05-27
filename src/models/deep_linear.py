import torch
import torch.nn as nn


class DeepLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DeepLinear, self).__init__()

        self.linear1 = nn.Linear(in_features=in_features, out_features=1024, dtype=torch.float64)
        self.linear2 = nn.Linear(in_features=1024, out_features=1024, dtype=torch.float64)
        self.linear3 = nn.Linear(in_features=1024, out_features=512, dtype=torch.float64)
        self.fc = nn.Linear(in_features=512, out_features=out_features, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)
        y = F.relu(y)
        return self.fc(y)