from util import *
import numpy.random as r
from sklearn.metrics import accuracy_score
import argparse

TARGET = "MM256"
USE_PRUNE = False
USE_SHARED = False
DEVICE = 'cpu'
PRUNED_SHAPE = (1000,34)
FULL_SHAPE = (9199930,34)

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
    return torch.matmul(torch.transpose(w_l, 0,1), delta_plus_1) * f_deriv_relu(z_l)

def feed_forward_relu(x, W, b):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*a^(l) + b^(l)
        if l ==len(W):
            a[l+1] = f_relu(z[l+1]) # a^(l+1) = f(z^(l+1))
        else:
            a[l+1] = f_relu(z[l+1])

    return a, z

def setup_and_init_weights(nn_structure):
    W = {} #creating a dictionary i.e. a set of key: value pairs
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1])) #Return “continuous uniform” random floats in the half-open interval [0.0, 1.0). 
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = torch.zeros((nn_structure[l], nn_structure[l-1])).to(DEVICE)
        tri_b[l] = torch.zeros((nn_structure[l],)).to(DEVICE)
    return tri_W, tri_b

def train_nn_relu(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    N = len(y)
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        for i in range(N):
            delta = {}
            # perform the feed forward pass and return the stored a and z values, to be used in the
            # gradient descent step
            a, z = feed_forward_relu(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], a[l], z[l])
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta_relu(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(a^(l))
                    tri_W[l] += torch.matmul(delta[l+1][:,None], torch.transpose(a[l][:,None], 0, 1))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/N * tri_W[l])
            b[l] += -alpha * (1.0/N * tri_b[l])
        cnt += 1
    return W, b

def predict_y(W, b, X, n_layers):
    N = X.shape[0]
    y = torch.zeros((N,)).to(DEVICE)
    for i in range(N):
        a, z = feed_forward_relu(X[i, :], W, b)
        y[i] = a
    return y


def main(iter:int, lr:float):
    data, meta = get_data(USE_PRUNE, USE_SHARED)
    data.to(DEVICE)
    X, y = splitXY(data, meta.names().index(TARGET))
    del data
    del meta
    X_train, y_train, X_test, y_test, _, _ = splitData(X, y, 0.8, 0.2)
    X_test = X_test.to("cpu")
    y_test = y_test.to("cpu")
    X_train = torch.nn.functional.normalize(X_train)
    nn_structure = [33, 10, 1]
    # train the NN
    W_relu, b_relu = train_nn_relu(nn_structure, X_train, y_train, iter, lr)

    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    y_pred = predict_y(W_relu, b_relu, X_test, 3)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--full", action="store_true", default=False) 
    parser.add_argument("--shared", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    USE_PRUNE = not args.full
    USE_SHARED = args.shared
    if not args.cpu:
        if torch.cuda.is_available():
            LOG("Cuda is available, switching to cuda")
            DEVICE = "cuda"
        else:
            LOG("Cuda is not available, using CPU")
    main(args.iter, args.lr)
