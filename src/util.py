import torch
from scipy.io import arff
from random import shuffle
from math import floor
import numpy as np

DEBUG = True

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

