from ml import get_data, split
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

USE_PRUNE = False
TARGET = "MM256"
DEVICE = 'cpu'

def extrapolate_time(X:torch.tensor, name:list)->torch.tensor:
    # Extrapolate time related attribute to one attribute that is in seconds.
    year_index = name.index("year")
    month_index = name.index("month")
    day_index = name.index("day")
    hour_index = name.index("hour")
    minute_index = name.index("minute")
    second_index = name.index("second")
    # Assume every month is 30 days for ease of calculation.
    timestamp = X[:,second_index] + 60 * (X[:,minute_index] + 60*(X[:,hour_index]+24*(X[:,day_index]+30*(X[:, month_index]+12*X[:,year_index]))))
    timestamp = timestamp.reshape((-1,1))
    X = torch.hstack([X, timestamp])
    name.append("timestamp")
    return X

def main():
    # Create a figure containing the plot of each feaure against the target
    data, meta = get_data(USE_PRUNE)
    names = meta.names()
    target_index = names.index(TARGET)
    X, y = split(data, target_index)
    y = y.squeeze()
    X = extrapolate_time(X, names)
    #return
    for i in range(X.shape[1]):
        if i >= target_index:
            index = i + 1
        else:
            index = i
        x = X[:, i]
        plt.plot(x, y, "o")
        title = f"{names[index]} vs {TARGET}"
        plt.title(title)
        plt.xlabel(names[index])
        plt.ylabel(TARGET)
        #plt.show() 
        plt.savefig(f"../plots/{names[index]}.png")
        plt.clf()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Cuda is available, switching to cuda")
        DEVICE = "cuda"
    else:
        print("Cuda is not available, using CPU")
    main()

