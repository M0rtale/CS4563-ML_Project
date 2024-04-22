from util import *
import argparse

TARGET = "MM256"
USE_PRUNE = False
USE_SHARED = False
DEVICE = 'cpu'


def dummy_encoding(y:torch.tensor)->torch.tensor:
    new_y = torch.zeros((y.shape[0], 58))
    y = y.squeeze(1)
    for i in range(0, 30):
        condition = torch.logical_and(y>=i/10, y < (i+1)/10)
        new_y[:, i] = torch.where(condition, torch.ones_like(condition), torch.zeros_like(condition))
    for i in range(3,31):
        condition = torch.logical_and(y>=i, y < (i+1))
        new_y[:, i+27] = torch.where(condition, torch.ones_like(condition), torch.zeros_like(condition))
    return new_y


def main(iter:int, lr:float, lamb:float):
    data, meta = get_data(USE_PRUNE, USE_SHARED)
    data = data.to(DEVICE)
    X, y = splitXY(data, meta.names().index(TARGET))
    X = torch.nn.functional.normalize(X)

    y = dummy_encoding(y)

    del data
    del meta
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.8, 0.2)
    del X, y
    X_test = X_test.to("cpu")
    y_test = y_test.to("cpu")
    # train the softmax
    # W_relu, b_relu = train_nn(nn_structure, X_train, y_train, iter, lr)
    # pred = predict_y(W_relu, b_relu, X_train, len(nn_structure))
    # loss = torch.nn.functional.mse_loss(pred, y_train)
    # LOG('Training MSE:',loss)
    # LOG("Training R^2: ", R_squared(pred, y_train))
    # LOG("Training RSS: ", RSS(pred, y_train))
    # LOG("Training TSS: ", TSS(y_train))

    # del X_train, y_train
    # X_test = X_test.to(DEVICE)
    # y_test = y_test.to(DEVICE)
    # test_pred = predict_y(W_relu, b_relu, X_test, len(nn_structure))
    # test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
    # LOG('MSE:',test_loss)
    # LOG("R^2: ", R_squared(test_pred, y_test))
    # LOG("RSS: ", RSS(test_pred, y_test))
    # LOG("TSS: ", TSS(y_test))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--full", action="store_true", default=False) 
    parser.add_argument("--shared", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
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
    main(args.iter, args.lr, args.reg)