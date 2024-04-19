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
        data = torch.from_numpy(ret).to('cpu')
        return data, meta

    else:
        LOG("Beginning to get data from file")
        if prune:
            dataset, meta = arff.loadarff('../dataset/pruned.arff')
        else:
            dataset, meta = arff.loadarff('../dataset/dataset.arff')
        LOG("Stopped getting data")
        data = np.array(dataset.tolist(), dtype=np.float64)
        data = torch.from_numpy(data).to('cpu')
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
    start = time.time()
    X_inv = torch.linalg.pinv(X)
    X.cpu()
    w_global = X_inv.matmul(y)
    del X_inv
    X.to(DEVICE)
    # w_global = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(torch.transpose(X, 0, 1), X)), torch.transpose(X, 0, 1)), y)
    end = time.time()
    LOG("Time for global optimization:", end-start)
    pred = torch.matmul(X, w_global)
    loss = torch.nn.functional.mse_loss(pred, y)
    LOG('Training MSE:',loss)
    LOG("Training R^2: ", R_squared(pred, y))
    LOG("Training RSS: ", RSS(pred, y))
    LOG("Training TSS: ", TSS(y))
    return w_global

def train_reg(X:torch.tensor, y:torch.tensor, lamb:float) -> torch.tensor:
    '''Kickstarts the traninig process of the dataset, assumes the data is normalized'''
    start = time.time()
    X_squared = torch.matmul(torch.transpose(X, 0, 1), X)
    I_prime = torch.eye(X.shape[1], X.shape[1])
    I_prime[0][0] = 0
    inside_inv = X_squared + X.shape[0] * lamb * I_prime
    del X_squared
    inv = torch.linalg.inv(inside_inv)
    first_part = torch.matmul(inv, torch.transpose(X, 0, 1))
    del inv
    w_global = torch.matmul(first_part, y)
    del first_part
    end = time.time()
    LOG("Time for global optimization:", end-start)
    pred = torch.matmul(X, w_global)
    loss = torch.nn.functional.mse_loss(pred, y)
    LOG('Training MSE:',loss)
    LOG("Training R^2: ", R_squared(pred, y))
    LOG("Training RSS: ", RSS(pred, y))
    LOG("Training TSS: ", TSS(y))
    return w_global

def splitData(X: torch.tensor, y:torch.tensor, train: float, test: float,)->tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    length = X.shape[0]
    random_indices = list(range(0, length))
    shuffle(random_indices)
    train_end = floor(length * train)
    train_indices = random_indices[0:train_end]
    test_end = floor(length * test)
    test_indices = random_indices[train_end: train_end+test_end]
    val_indices = random_indices[train_end+test_end:]
    X_train = X[train_indices, :]
    y_train = y[train_indices, :]
    X_test = X[test_indices, :]
    y_test = y[test_indices, :]
    X_val = X[val_indices, :]
    y_val = y[val_indices, :]
    return X_train, y_train, X_test, y_test, X_val, y_val


def train_eval(X: torch.tensor, y:torch.tensor, lamb = 0) ->torch.tensor:
    # Train and evalute linear regression model
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.8, 0.2)
    del X, y
    X_test.cpu()
    y_test.cpu()
    X_train = torch.nn.functional.normalize(X_train)
    #send to train
    #del y_train
    #send to cuda
    X_test.to(DEVICE)
    y_train.to(DEVICE)
    if lamb > 0:
        w = train_reg(X_train, y_train, lamb)
    else:
        w = train(X_train, y_train)
    LOG('output weights:',w)
    LOG("weight shape: ", w.shape)
    

    X_test.to(DEVICE)
    y_test.to(DEVICE)
    X_test = torch.nn.functional.normalize(X_test)
    test_pred = torch.matmul(X_test, w)
    test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
    LOG('MSE:',test_loss)
    LOG("R^2: ", R_squared(test_pred, y_test))
    LOG("RSS: ", RSS(test_pred, y_test))
    LOG("TSS: ", TSS(y_test))
    return w

def train_eval_poly(X: torch.tensor, y:torch.tensor, lamb=0)->torch.tensor:
    # Train and evaluate linear regression model with polynomial transformation of degree 2
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.1, 0.1)
    del X, y

    #send to train
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_train.cpu())
    X_poly = torch.from_numpy(X_poly).to(DEVICE)
    X_poly = torch.nn.functional.normalize(X_poly)
    
    print("X poly shape", X_poly.shape)
    del X_train
    #del y_train
    y_train = y_train.to(DEVICE)
    if lamb > 0:
        w = train_reg(X_poly, y_train, lamb)
    else:
        w = train(X_poly, y_train)
    LOG('output weights:',w)
    LOG("weight shape: ", w.shape)
    del X_poly
    

    y_test = y_test.to(DEVICE)
    X_poly = poly.fit_transform(X_test.cpu())
    X_poly = torch.from_numpy(X_poly).to(DEVICE)
    X_poly = torch.nn.functional.normalize(X_poly)
    test_pred = torch.matmul(X_poly, w)
    test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
    LOG('MSE:',test_loss)
    LOG("R^2: ", R_squared(test_pred, y_test))
    LOG("RSS: ", RSS(test_pred, y_test))
    LOG("TSS: ", TSS(y_test))
    return w


def main(poly:bool, reg:float) -> None:
    '''this is the entry of the program.
    {r}'''
    start = time.time()
    data, meta = get_data(USE_PRUNE, USE_SHARED)
    end = time.time()
    LOG("Time for getting data:", end-start)
    #data = torch.nn.functional.normalize(data)
    LOG("Data shape:", data.shape)
    X, y = splitXY(data, meta.names().index(TARGET))
    del data
    del meta
    if poly:
        train_eval_poly(X, y, reg)
    else:
        train_eval(X, y, reg)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--full", action="store_true", default=False) 
    parser.add_argument("--shared", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--poly", action="store_true", default=False)
    parser.add_argument("--reg", type=float, default=0)
    args = parser.parse_args()
    USE_PRUNE = not args.full
    USE_SHARED = args.shared
    if not args.cpu:
        if torch.cuda.is_available():
            LOG("Cuda is available, switching to cuda")
            DEVICE = "cuda"
        else:
            LOG("Cuda is not available, using CPU")
    main(args.poly, args.reg)
