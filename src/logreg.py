import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

import argparse
import time

from dataset import myDataset
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister

from util import *

TARGET = "MM256"
USE_PRUNE = False
USE_SHARED = False
DEVICE = 'cpu'
PRUNED_SHAPE = (1000,34)
FULL_SHAPE = (9199930,34)

def train_one_vs_all(X:torch.tensor, y:torch.tensor, iter: int, lr: float) -> torch.tensor:
    '''Kickstarts the traninig process of the dataset, assumes the data is normalized'''
    y_pos = y[y==1, None]
    y_neg = y[y==0, None]
    X_pos = X[y.squeeze()==1, :]
    X_neg = X[y.squeeze()==0, :]
    w = torch.zeros((X.shape[1], 1), dtype=torch.float64).to(DEVICE)
    if y_pos.shape[0] > 0:
        ratio = y_pos.shape[0] / y_neg.shape[0]
        if ratio < 1 and ratio > 0:
            # LOG("Ratio: ", ratio)
            # LOG("y_neg before: ", y_neg.shape)
            X_neg, y_neg, _, _, _, _ = splitData(X_neg, y_neg, ratio, 0)
        if ratio > 1:
            X_pos, y_pos, _, _, _, _ = splitData(X_pos, y_pos, 1/ratio, 0)
        difference = y_neg.shape[0] - y_pos.shape[0]
        if difference > 0:
            y_neg = y_neg[:-difference, :]
            X_neg = X_neg[:-difference, :]
        elif difference < 0:
            y_pos = y_pos[:difference, :]
            X_pos = X_pos[:difference, :]
        LOG("y_pos: ", y_pos.shape)
        LOG("y_neg after: ", y_neg.shape)
        y = torch.vstack((y_pos, y_neg))
        X = torch.vstack((X_pos, X_neg))
    for _ in range(iter):
        #w = w + a * (XT (y - sigmoid(Xw)))
        w = w + lr * ( torch.matmul( torch.transpose(X, 0, 1), (y - f_sigmoid(torch.matmul(X, w))) )) / X.shape[0]
    return w

def train_one_vs_all_reg(X:torch.tensor, y:torch.tensor, iter: int, lr: float, lamb:float) -> torch.tensor:
    '''Kickstarts the traninig process of the dataset, assumes the data is normalized'''
    y_pos = y[y==1, None]
    y_neg = y[y==0, None]
    X_pos = X[y.squeeze()==1, :]
    X_neg = X[y.squeeze()==0, :]
    w = torch.zeros((X.shape[1], 1), dtype=torch.float64).to(DEVICE)
    if y_pos.shape[0] > 0 and ratio > 0:
        ratio = y_pos.shape[0] / y_neg.shape[0]
        if ratio < 1:
            # LOG("Ratio: ", ratio)
            # LOG("y_neg before: ", y_neg.shape)
            X_neg, y_neg, _, _, _, _ = splitData(X_neg, y_neg, ratio, 0)
        # LOG("y_pos: ", y_pos.shape)
        # LOG("y_neg after: ", y_neg.shape)
        y = torch.vstack((y_pos, y_neg))
        X = torch.vstack((X_pos, X_neg))
    for _ in range(iter):
        #w = w + a * (XT (y - sigmoid(Xw)))
        w = w + lr * ((torch.matmul( torch.transpose(X, 0, 1), (y - f_sigmoid(torch.matmul(X, w))) )) / X.shape[0] - lamb * w)
    return w

def train_eval(X: torch.tensor, y:torch.tensor, iter: int, lr: float, lamb = 0) ->torch.tensor:
    # Train and evalute linear regression model
    X = X.cuda()
    X = torch.nn.functional.normalize(X)
    X = torch.hstack((torch.ones(X.shape[0], 1).to(DEVICE), X))
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.8, 0.2)
    w = torch.zeros(X.shape[1], y.shape[1], dtype=torch.float64).to(DEVICE)
    del X, y
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    #send to train
    #del y_train
    #send to cuda
    for i in range(y_train.shape[1]):
        LOG("Start training model ", i)
        start = time.time()
        if lamb > 0:
            w[:, i] = train_one_vs_all_reg(X_train, y_train[:, i, None], iter, lr, lamb).squeeze(1)
        else:
            w[:, i] = train_one_vs_all(X_train, y_train[:, i, None], iter, lr).squeeze(1)
        
        end = time.time()
        LOG("Time for training:", end-start)
    pred = classify(w, X_train)
    print("Training accuracy:", accuracy(pred, y_train))

    decoded_pred = onehot_decoding(pred)
    LOG("Decoded pred: ", decoded_pred[:10])
    LOG("Decoded y: ", onehot_decoding(y_train)[:10])
    # LOG('output weights:',w)
    # LOG("weight shape: ", w.shape)

    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    
    test_pred = classify(w, X_test)
    print("Testing accuracy:", accuracy(test_pred, y_test))
    return w

def train_eval_poly(X: torch.tensor, y:torch.tensor, iter: int, lr: float, lamb=0)->torch.tensor:
    # Train and evaluate linear regression model with polynomial transformation of degree 2
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.4, 0.1)
    del X, y

    #send to train
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_train.cpu())
    X_poly = torch.from_numpy(X_poly).to(DEVICE)
    X_poly = torch.nn.functional.normalize(X_poly)
    X_poly[:, 0] = 1
    
    print("X poly shape", X_poly.shape)
    del X_train
    #del y_train
    y_train = y_train.to(DEVICE)
    if lamb > 0:
        w = train_one_vs_all_reg(X_poly, y_train, iter, lr, lamb)
    else:
        w = train_one_vs_all(X_poly, y_train, iter, lr)
    # LOG('output weights:',w)
    # LOG("weight shape: ", w.shape)

    pred = classify(w, X_poly)
    print("Training accuracy:", accuracy(pred, y_train))
    del X_poly
    
    y_test = y_test.to(DEVICE)
    X_poly = poly.fit_transform(X_test.cpu())
    X_poly = torch.from_numpy(X_poly).to(DEVICE)
    X_poly = torch.nn.functional.normalize(X_poly)
    X_poly[:, 0] = 1
    test_pred = classify(w, X_poly)
    print("Training accuracy:", accuracy(pred, y_test))
    return w


def main(poly:bool, reg:float, iter: int, lr: float) -> None:
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

    #encode y
    y = onehot_encoding(y, DEVICE)

    if poly:
        train_eval_poly(X, y, iter, lr, reg)
    else:
        train_eval(X, y, iter, lr, reg)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--full", action="store_true", default=False) 
    parser.add_argument("--shared", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--poly", action="store_true", default=False)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
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
    main(args.poly, args.reg, args.iter, args.lr)
