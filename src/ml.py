from scipy.io import arff
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from dataset import myDataset
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from random import shuffle
from math import floor
from sklearn.preprocessing import PolynomialFeatures

TARGET = "MM256"
DEBUG = True
USE_PRUNE = False
USE_SHARED = False
DEVICE = 'cpu'
EXP_NAME = "test"
PRUNED_SHAPE = (1000,34)
FULL_SHAPE = (9199930,34)

def LOG(*args, **kwargs) -> None:
    '''simple debug print wrapper'''
    if DEBUG:
        print(*args)


def get_data(prune:bool, shared:bool) -> tuple[object, object]:
    '''gets the dataset from ../dataset/dataset.arff'''
    if shared:
        LOG("Beginning to get data from namedpipe")
        # Load Pruned to help with the metadata
        dataset, meta = arff.loadarff("../dataset/pruned.arff")
        if prune:
            shm = shared_memory.SharedMemory(name='nppruned')
            np_array = np.ndarray(shape=PRUNED_SHAPE, dtype=np.float64, buffer=shm.buf)
            ret = np.ndarray(shape=PRUNED_SHAPE, dtype=np.float64)
            ret[:] = np_array[:]
            #cleanup after ourselves to ensure resource is persistent
            unregister(shm._name, 'shared_memory')
            shm.close()
        else:
            shm = shared_memory.SharedMemory(name='npfull')
            np_array = np.ndarray(shape=FULL_SHAPE, dtype=np.float64, buffer=shm.buf)
            ret = np.ndarray(shape=FULL_SHAPE, dtype=np.float64)
            ret[:] = np_array[:]
            #cleanup after ourselves to ensure resource is persistent
            unregister(shm._name, 'shared_memory')
            shm.close()
        LOG("Stopped getting data")
        data = torch.tensor(ret, dtype=torch.float16).to(DEVICE)
        print(data.dtype)
        return data, meta

    else:
        LOG("Beginning to get data from file")
        if prune:
            dataset, meta = arff.loadarff('../dataset/pruned.arff')
        else:
            dataset, meta = arff.loadarff('../dataset/dataset.arff')
        LOG("Stopped getting data")
        data = np.array(dataset.tolist(), dtype=np.float16)
        data = torch.tensor(data, dtype=torch.float16).to(DEVICE)
        return data, meta

def RSS(predicted:torch.tensor, actual:torch.tensor) -> torch.tensor:
    diff_squared = torch.pow((predicted - actual), 2)
    cost = torch.sum(diff_squared)
    return cost

def TSS(actual:torch.tensor) -> torch.tensor:
    mean = torch.mean(actual).expand(actual.shape[0], 1)
    return RSS(actual, mean)

def R_squared(predicted:torch.tensor, actual:torch.tensor) -> torch.tensor:
    return 1 - RSS(predicted, actual) / TSS(actual)

def splitXY(data:torch.tensor, targetIndex: int) -> tuple[torch.tensor, torch.tensor]:
    y = data[:, targetIndex].reshape((data.shape[0], 1))
    X_first = data[:,:targetIndex]
    X_second = data[:, targetIndex+1:]
    X = torch.hstack([X_first, X_second])
    return X, y

def train(X:torch.tensor, y:torch.tensor) -> torch.tensor:
    '''Kickstarts the traninig process of the dataset, assumes the data is normalized'''
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    start = time.time()
    w_global = torch.linalg.pinv(X).matmul(y)
    # w_global = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(torch.transpose(X, 0, 1), X)), torch.transpose(X, 0, 1)), y)
    end = time.time()
    LOG("Time for global optimization:", end-start)
    
    LOG("y:", y)
    LOG("X:", X)
    
    pred = torch.matmul(X, w_global)
    loss = torch.nn.functional.mse_loss(pred, y)
    LOG("Train loss:", loss)
    return w_global

def splitData(X: torch.tensor, y:torch.tensor)\
    ->tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    length = X.shape[0]
    random_indices = list(range(0, length))
    shuffle(random_indices)
    train_end = floor(length * 0.15)
    train_indices = random_indices[0:train_end]
    test_end = floor(length*0.1)
    test_indices = random_indices[train_end: train_end+test_end]
    val_indices = random_indices[train_end+test_end:]
    X_train = X[train_indices, :]
    y_train = y[train_indices, :]
    X_test = X[test_indices, :]
    y_test = y[test_indices, :]
    X_val = X[val_indices, :]
    y_val = y[val_indices, :]
    return X_train, y_train, X_test, y_test, X_val, y_val


def train_eval(X: torch.tensor, y:torch.tensor)->torch.tensor:
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y)
    del X, y
    X_test.cpu()
    y_test.cpu()
    #send to train
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_train.cpu())
    X_poly = torch.tensor(X_poly,dtype=torch.float16).to(DEVICE)
    X_poly = torch.nn.functional.normalize(X_poly)
    print("X poly shape", X_poly.shape)
    print("X_poly dtype: ", X_poly.dtype)
    del X_train
    #del y_train
    w = train(X_poly, y_train)
    LOG('output weights:',w)
    LOG("weight shape: ", w.shape)
    del X_poly
    

    X_test.to(DEVICE)
    y_test.to(DEVICE)
    X_poly = poly.fit_transform(X_test.cpu())
    X_poly = torch.from_numpy(X_poly).to(DEVICE)
    test_pred = torch.matmul(X_poly, w)
    test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
    LOG('test loss:',test_loss)
    LOG("R^2: ", R_squared(test_pred, y_test))
    LOG("RSS: ", RSS(test_pred, y_test))
    LOG("TSS: ", TSS(y_test))
    return w


def main() -> None:
    '''this is the entry of the program.
    {r}'''
    start = time.time()
    data, meta = get_data(USE_PRUNE, USE_SHARED)
    end = time.time()
    LOG("Time for data getting:", end-start)
    #data = torch.nn.functional.normalize(data)
    LOG("Data shape:", data.shape)
    X, y = splitXY(data, meta.names().index(TARGET))
    del data
    #X = torch.nn.functional.normalize(X)
    #X_poly = X
    # LOG("Data shape after transform:", X_poly.shape)
    train_eval(X, y)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--exp", default="test", type=str) 
    parser.add_argument("--full", action="store_true", default=False) 
    parser.add_argument("--shared", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()
    EXP_NAME = args.exp
    USE_PRUNE = not args.full
    USE_SHARED = args.shared
    if not args.cpu:
        if torch.cuda.is_available():
            LOG("Cuda is available, switching to cuda")
            DEVICE = "cuda"
        else:
            LOG("Cuda is not available, using CPU")
    main()
