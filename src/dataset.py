import torch.utils.data.Dataset as Dataset

class myDataset(Dataset):
    def __init__(X, y):
        self.X = X
        self.y = y

    def __len

