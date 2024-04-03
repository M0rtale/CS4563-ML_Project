from scipy.io import arff
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

TARGET = "MM256"
DEBUG = True
USE_PRUNE = False
DEVICE = 'cpu'
NUM_ITER = 10000
LEARNING_RATE = 0.001
EXP_NAME = "test"

def LOG(*args, **kwargs) -> None:
    '''simple debug print wrapper'''
    if DEBUG:
        print(*args)


def get_data() -> (object, object):
    '''gets the dataset from ../dataset/dataset.arff'''
    LOG("Beginning to get data")
    if USE_PRUNE:
        dataset, meta = arff.loadarff('../dataset/pruned.arff')
    else:
        dataset, meta = arff.loadarff('../dataset/dataset.arff')
    LOG("Stopped getting data")
    return dataset, meta

def MSE(predicted:torch.tensor, actual:torch.tensor) -> torch.tensor:
    '''Returns the Mean Squared Error between the predicted tensor value and actual'''
    diff_squared = torch.pow((predicted - actual), 2)
    loss = torch.sum(diff_squared) / predicted.shape[0]
    return loss

def train(data: torch.tensor, targetIndex: int) -> torch.tensor:
    '''Kickstarts the traninig process of the dataset, assumes the data is normalized'''
    y = data[:, targetIndex].reshape((data.shape[0], 1))
    X_first = data[:,:targetIndex]
    X_second = data[:, targetIndex+1:]
    X = torch.hstack([X_first, X_second])
    start = time.time()
    w_global = torch.linalg.pinv(X).matmul(y)
    end = time.time()
    LOG("Time for global optimization:", end-start)
    
    LOG("y shape:", y.shape)
    LOG("X shape:", X.shape)
    
    pred = torch.matmul(X, w_global)
    loss = MSE(pred, y)
    LOG("Global loss:", loss)
    return w_global

def main() -> None:
    '''this is the entry of the program.
    {r}'''
    start = time.time()
    dataset, meta = get_data()
    end = time.time()
    LOG("Time for global optimization:", end-start)
    data = np.array(dataset.tolist(), dtype=np.float64)
    data = torch.from_numpy(data).to(DEVICE)
    data = torch.nn.functional.normalize(data)
    LOG("Data shape:", data.shape)

    #send to train
    w = train(data, meta.names().index(TARGET))
    LOG(w)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--exp", default="test", type=str) 
    args = parser.parse_args()
    EXP_NAME = args.exp
    if torch.cuda.is_available():
        LOG("Cuda is available, switching to cuda")
        DEVICE = "cuda"
    else:
        LOG("Cuda is not available, using CPU")
    main()
