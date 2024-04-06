from ml import get_data, split
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

USE_PRUNE = True
TARGET = "MM256"
DEVICE = 'cpu'

def extrapolate_time(X:torch.tensor, name:tuple)->torch.tensor:
    year_index = name.index("year")
    month_index = name.index("month")
    X_first = X[:,:year_index]
    X_second = X[:, year_index+1:]
    X = torch.hstack([X_first, X_second])
    return X

def main():
    # Create a figure containing the plot of each feaure against the target
    dataset, meta = get_data(USE_PRUNE)
    data = np.array(dataset.tolist(), dtype=np.float64)
    data = torch.from_numpy(data).to(DEVICE)
    target_index = meta.names().index(TARGET)
    X, y = split(data, target_index)
    y = y.squeeze()
    X = extrapolate_time(X)
    print(X.shape)
    return
    rows = math.ceil(X.shape[1]/2)
    for i in range(X.shape[1]):
        if i >= target_index:
            index = i + 1
        else:
            index = i
        x = X[:, i]
        plt.plot(x, y, "o")
        title = f"{meta.names()[index]} vs {TARGET}"
        plt.title(title)
        plt.xlabel(meta.names()[index])
        plt.ylabel(TARGET)
        #plt.show() 
        plt.savefig(f"../plots/{meta.names()[index]}.png")
        plt.clf()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Cuda is available, switching to cuda")
        DEVICE = "cuda"
    else:
        print("Cuda is not available, using CPU")
    main()

