from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ClassicDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        super(ClassicDataset, self).__init__()
        self.X = X
        self.y = y
        
        self.n_classes = np.unique(y).shape[0]
        self.n_features = X.shape[1]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]