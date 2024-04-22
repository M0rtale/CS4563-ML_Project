from util import *
import argparse

TARGET = "MM256"
USE_PRUNE = False
USE_SHARED = False
DEVICE = 'cpu'


def f_relu(z):
    z[z<0]=0
    return z


def f_deriv_relu(z):
    f_d = torch.zeros_like(z).to(DEVICE)
    f_d[z>=0] = 1
    return f_d

def calculate_out_layer_delta(y, a_out, z_out):
    # delta^(nl) = -(y_i - a_i^(nl)) * f'(z_i^(nl))
    return -(y-a_out) * f_deriv_relu(z_out) 

def calculate_hidden_delta_relu(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return torch.matmul(delta_plus_1, torch.transpose(w_l, 0,1)) * f_deriv_relu(z_l)

def feed_forward_relu(x, W, b):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        # z^(l+1) = W^(l)*a^(l) + b^(l)
        z[l+1] = torch.matmul(node_in, W[l]) 
        z[l+1] += b[l]  
        if l ==len(W):
            a[l+1] = f_relu(z[l+1]) # a^(l+1) = f(z^(l+1))
        else:
            a[l+1] = f_relu(z[l+1])

    return a, z

def setup_and_init_weights(nn_structure):
    W = {} #creating a dictionary i.e. a set of key: value pairs
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = torch.rand((nn_structure[l-1], nn_structure[l]), dtype=torch.float64).to(DEVICE)
        b[l] = torch.rand((nn_structure[l],), dtype=torch.float64).to(DEVICE)
    return W, b

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25, lamb=0):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    N = len(y)
    LOG('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        LOG('Iteration {} of {}'.format(cnt, iter_num))
        delta = {}
        # perform the feed forward pass and return the stored a and z values, to be used in the
        # gradient descent step
        a, z = feed_forward_relu(X, W, b)
        # loop from nl-1 to 1 backpropagating the errors
        for l in range(len(nn_structure), 0, -1):
            if l == len(nn_structure):
                delta[l] = calculate_out_layer_delta(y, a[l], z[l])
            else:
                if l > 1:
                    delta[l] = calculate_hidden_delta_relu(delta[l+1], W[l], z[l])
                # triW^(l) = triW^(l) + delta^(l+1) * transpose(a^(l))
                W[l] += -alpha * (1.0/N * torch.matmul(torch.transpose(a[l], 0, 1), delta[l+1]) + lamb*W[l])
                b[l] += -alpha * (1.0/N * torch.sum(delta[l+1], dim=0))
                # trib^(l) = trib^(l) + delta^(l+1)
        # perform the gradient descent step for the weights in each layer
        cnt += 1
    return W, b

def predict_y(W, b, X, n_layers):
    a, _ = feed_forward_relu(X, W, b)
    y = a[n_layers]
    return y


def main(iter:int, lr:float, lamb:float):
    data, meta = get_data(USE_PRUNE, USE_SHARED)
    data = data.to(DEVICE)
    X, y = splitXY(data, meta.names().index(TARGET))
    X = torch.nn.functional.normalize(X)
    del data
    del meta
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.8, 0.2)
    del X, y
    X_test = X_test.to("cpu")
    y_test = y_test.to("cpu")
    nn_structure = [33, 20, 10, 1]
    # train the NN
    W_relu, b_relu = train_nn(nn_structure, X_train, y_train, iter, lr)
    pred = predict_y(W_relu, b_relu, X_train, len(nn_structure))
    loss = torch.nn.functional.mse_loss(pred, y_train)
    LOG('Training MSE:',loss)
    LOG("Training R^2: ", R_squared(pred, y_train))
    LOG("Training RSS: ", RSS(pred, y_train))
    LOG("Training TSS: ", TSS(y_train))

    del X_train, y_train
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    test_pred = predict_y(W_relu, b_relu, X_test, len(nn_structure))
    test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
    LOG('MSE:',test_loss)
    LOG("R^2: ", R_squared(test_pred, y_test))
    LOG("RSS: ", RSS(test_pred, y_test))
    LOG("TSS: ", TSS(y_test))
    


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
