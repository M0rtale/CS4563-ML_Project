from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, X:torch.tensor, y:torch.tensor)->None:
        self.X = X
        self.y = y

    def __len__(self)->int:
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return (self.X[index, :], self.y[index, :])
    
    def getXY(self):
        return (self.X, self.y)

