import torch
from torch.utils.data import Dataset


class ClassicDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        super(ClassicDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]