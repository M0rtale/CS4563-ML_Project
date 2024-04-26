import torch
from scipy.io import arff
from random import shuffle
from math import floor, ceil
import numpy as np
from dataset import myDataset
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister

PRUNED_SHAPE = (1000,34)
FULL_SHAPE = (9199930,34)

DEBUG = True

def LOG(*args, **kwargs) -> None:
    '''simple debug print wrapper'''
    if DEBUG:
        print(*args)
        
def f_sigmoid(z: torch.tensor) -> torch.tensor:
    return torch.sigmoid(z)

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


def onehot_encoding(y:torch.tensor, device:str)->torch.tensor:
    # Use binning to turn y from a continuous value into 16 discrete bins. 
    new_y = torch.zeros((y.shape[0], 16), dtype=torch.float64).to(device)
    y = y.squeeze(1)
    # First 15 bins takes value in interval of 0.1
    for i in range(0, 15):
        condition = torch.logical_and(y>=i/10, y < (i+1)/10)
        new_y[:, i] = torch.where(condition, torch.ones_like(condition), torch.zeros_like(condition))
    # # Next 28 bins take internval of 1
    # for i in range(3,31):
    #     condition = torch.logical_and(y>=i, y < (i+1))
    #     new_y[:, i+27] = torch.where(condition, torch.ones_like(condition), torch.zeros_like(condition))
    # Last bin for every other value
    condition = y >= 1.5
    new_y[:, 15] = torch.where(condition, torch.ones_like(condition), torch.zeros_like(condition))
    return new_y

def onehot_decoding(y:torch.tensor):
    y_new = torch.argmax(y, dim=1).reshape(-1, 1).float()
    y_new[y_new < 15] = y_new[y_new < 15] * 0.1
    # condition = torch.logical_and(y_new>=30, y_new < 58)
    # y_new[condition] -= 27
    y_new[y_new == 15] = 1.5
    return y_new

def classify(w:torch.tensor, X:torch.tensor)->torch.tensor:
    pred = f_sigmoid(torch.matmul(X, w))
    # LOG(pred[:10])
    y = torch.zeros_like(pred).scatter_(1, torch.argmax(pred, dim=1).unsqueeze(1), 1.)
    return y

def accuracy(pred:torch.tensor, y:torch.tensor)->float:
    y = torch.argmax(y, dim=1).reshape(-1,1)
    pred = torch.argmax(pred, dim=1).reshape(-1,1)
    return float(torch.sum(pred==y)/y.shape[0])

def precision(confu: torch.tensor) -> torch.tensor:
    '''input is a confusion matrix, output is a list of precisions for each class'''
    column_sums = torch.sum(confu, dim=0) + 0.00000001
    diagnol = torch.diag(confu)
    avg = diagnol / column_sums
    avg = avg.reshape(-1, 1)
    return avg

def recall(confu: torch.tensor) -> torch.tensor:
    '''input is a confusion matrix, output is a list of recall for each class'''
    row_sums = torch.sum(confu, dim=1) + 0.00000001
    diagnol = torch.diag(confu)
    avg = diagnol / row_sums
    avg = avg.reshape(-1, 1)
    return avg

def confusion(pred: torch.tensor, y:torch.tensor) -> torch.tensor:
    """Create a confusion matrix, each column is predicted class, each row is actual class"""
    matrix = torch.zeros((pred.shape[1], pred.shape[1]))
    for i in range(pred.shape[1]):
        actual = y[:, i] == 1
        predicted = pred[actual, :]
        matrix[i, :] = torch.sum(predicted, dim=0)
    return matrix

def f1_score(prec: torch.tensor, rec: torch.tensor) -> torch.tensor:
    'returns f1 of the prec rec'
    return 2 * ((prec * rec) / (prec + rec + 0.00000001))
            


def up_and_down(X:torch.tensor, y:torch.tensor, target:int) -> tuple[torch.tensor, torch.tensor]:
    y_pos = y[y==1, None]
    y_neg = y[y==0, None]
    X_pos = X[y.squeeze()==1, :]
    X_neg = X[y.squeeze()==0, :]
    if y_pos.shape[0] > 0:
        if y_neg.shape[0] > target:
            ratio = target / y_neg.shape[0]
            X_neg, y_neg, _, _, _, _ = splitData(X_neg, y_neg, ratio, 0)
        ratio = y_neg.shape[0] / y_pos.shape[0]
        if ratio > 1:
            y_pos = y_pos.repeat(ceil(ratio), 1)
            X_pos = X_pos.repeat(ceil(ratio), 1)
        elif ratio < 1:
            X_pos, y_pos, _, _, _, _ = splitData(X_pos, y_pos, ratio, 0)
        if y_pos.shape[0] > y_neg.shape[0]:
            y_pos = y_pos[:y_neg.shape[0], :]
            X_pos = X_pos[:y_neg.shape[0], :]
        elif y_pos.shape[0] < y_neg.shape[0]:
            X_neg = X_neg[:y_pos.shape[0], :]
            y_neg = y_neg[:y_pos.shape[0], :]
        LOG("y_pos: ", y_pos.shape)
        LOG("y_neg after: ", y_neg.shape)
        y = torch.vstack((y_pos, y_neg))
        X = torch.vstack((X_pos, X_neg))
    return X, y