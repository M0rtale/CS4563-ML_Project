from scipy.io import arff
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse


TARGET = "MM256"
DEBUG = True
USE_PRUNE = True
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

def train(data: torch.tensor, targetIndex: int, num_iter:int, lr:float) -> torch.tensor:
    '''Kickstarts the traninig process of the dataset, assumes the data is normalized'''
    y = data[:, targetIndex].reshape((data.shape[0], 1))
    X_first = data[:,:targetIndex]
    X_second = data[:, targetIndex+1:]
    X = torch.hstack([X_first, X_second])
    w = torch.zeros((X.shape[1], 1), dtype = torch.float64, requires_grad=True).to(DEVICE)
    w_global = torch.linalg.pinv(X).matmul(y)
    
    LOG("y shape:", y.shape)
    LOG("w shape:", w.shape)
    LOG("X shape:", X.shape)
    optimizer = torch.optim.SGD([w], lr=lr)

    writer = SummaryWriter(log_dir=f"../log/{EXP_NAME}")


    for i in range(num_iter):
        pred = torch.matmul(X, w)
        loss = MSE(pred, y)
        writer.add_scalar("Loss", loss, global_step=i)
        loss.backward()
        optimizer.step()
        #LOG(w)
    
    LOG(w_global)
    pred = torch.matmul(X, w_global)
    loss = MSE(pred, y)
    LOG("Global loss:", loss)
    pred = torch.matmul(X, w)
    loss = MSE(pred, y)
    LOG("Local loss:", loss)
    return w

def main() -> None:
    '''this is the entry of the program.
    {r}'''
    dataset, meta = get_data()
    data = np.array(dataset.tolist(), dtype=np.float64)
    data = torch.from_numpy(data).to(DEVICE)
    data = torch.nn.functional.normalize(data)
    LOG("Data shape:", data.shape)

    #send to train
    w = train(data, meta.names().index(TARGET), NUM_ITER, LEARNING_RATE)
    LOG(w)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--exp", default="test", type=str) 
    parser.add_argument("--lr", default=0.01, type=float) 
    parser.add_argument("--iter", default=1000, type=int) 
    args = parser.parse_args()
    EXP_NAME = args.exp
    LEARNING_RATE = args.lr
    NUM_ITER = args.iter
    if torch.cuda.is_available():
        LOG("Cuda is available, switching to cuda")
        DEVICE = "cuda"
    else:
        LOG("Cuda is not available, using CPU")
    main()
