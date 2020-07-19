import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

###############################################
### Generate 10 datasets for linear example ###
###############################################
np.random.seed(1)
def linear_generator(p, N, s, K=10):
    np.random.seed(1)
    directory = './data/linear/p_' + str(p) + '_N_' + str(N) + '_s_' + str(s)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for k in range(K):
        X_orginal = np.random.normal(0, 1, (N, p))
        rho = 0.2
        X = np.zeros((N, p))
        X[:, 0] = X_orginal[:, 0]
        X[:, p-1] = X_orginal[:, p-1]
        for i in range(1, p-1):
            X[:, i] = X_orginal[:, i] + rho * (X_orginal[:, i-1] + X_orginal[:, i+1])
        m = np.sqrt(2 * np.log(p)/N)
        M = 100 * m
        e = np.random.normal(0, 1, N)
        beta = np.zeros(p)
        #non_zero = np.random.choice(p, s, replace=False)
        non_zero = np.arange(s)
        beta[non_zero] = np.random.uniform(m, M, s)
        y = X.dot(beta) + e
        X_all = X
        fn_X = directory + '/X_' + str(k) + '.txt'
        fn_y = directory + '/y_' + str(k) + '.txt'
        fn_beta = directory + '/beta_' + str(k) + '.txt'
        np.savetxt(fn_X, X_all)
        np.savetxt(fn_y, y)
        np.savetxt(fn_beta, beta)

###################################################
### Generate 10 datasets for non-linear example ###
###################################################
np.random.seed(1)
def nonlinear_generator(N=500, s=4, p=500, K=10):
    np.random.seed(1)
    directory = './data/nonlinear/p_' + str(p) + '_N_' + str(N) + '_s_' + str(s)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for k in range(K):
        ss = 0
        npos = 0
        nneg = 0
        X = []
        y = []
        steps = 0
        while(ss < N):
            _X = np.random.normal(0, 1, (1, p))
            _e = np.random.normal(0, 1, (1, 1))
            _X = (_X+_e)/2
            _Z = np.exp(_X[:, 0]) + _X[:, 1]**2 + 5*np.sin(_X[:, 2]*_X[:, 3]) - 3
            _y = (_Z>0)*1.0
            if _y > 0 and npos < 300:
                X.append(_X)
                y.append(_y)
                npos += 1
                ss += 1
            if _y <= 0 and nneg < 300:
                X.append(_X)
                y.append(_y)
                nneg += 1
                ss += 1
            steps += 1
        print(steps)
        X_all = np.concatenate(X)
        y = np.array(y)
        fn_X = directory + '/X_' + str(k) + '.txt'
        fn_y = directory + '/y_' + str(k) + '.txt'
        np.savetxt(fn_X, X_all)
        np.savetxt(fn_y, y)

##############################
### Data Loading functions ###
##############################
def data_load_n(k, normalization=False, directory="./data/nonlinear/p_500_N_600_s_4/"):
    # Directory for the datasets
    x = np.loadtxt(directory+'X_'+str(k)+'.txt')
    y = np.loadtxt(directory+'y_'+str(k)+'.txt')
    n = x.shape[0]
    n_pos = len(np.where(y == 1)[0])
    n_neg = len(np.where(y == 0)[0])
    # Take first 300 samples as training set
    train_pos_idx = np.where(y == 1)[0][:int(n_pos/2)]
    train_neg_idx = np.where(y == 0)[0][:int(n_neg/2)]
    # Take last 300 samples as testing set
    test_pos_idx = np.where(y == 1)[0][int(n_pos/2):]
    test_neg_idx = np.where(y == 0)[0][int(n_neg/2):]
    train_idx = np.sort(np.append(train_pos_idx, train_neg_idx))
    test_idx = np.sort(np.append(test_pos_idx, test_neg_idx))
    x_test = x[test_idx]
    y_test = y[test_idx]
    x = x[train_idx]
    y = y[train_idx]
    N, p = x.shape
    # normalize if needed
    if normalization:
        for j in range(p):
            x_test[:, j] = x_test[:, j]/np.sqrt(np.sum(x[:, j]**2)/float(N))
            x[:, j] = x[:, j]/np.sqrt(np.sum(x[:, j]**2)/float(N))
    X, Y = torch.Tensor(x), torch.Tensor(y)
    X, Y = X.type(torch.FloatTensor), Y.type(torch.LongTensor)
    X, Y = Variable(X), Variable(Y)
    X_test, Y_test = torch.Tensor(x_test), torch.Tensor(y_test)
    X_test, Y_test = X_test.type(torch.FloatTensor), Y_test.type(torch.LongTensor)
    X_test, Y_test = Variable(X_test), Variable(Y_test)
    return X, Y, X_test, Y_test

def data_load_l(k, normalization=True, directory = './data/linear/p_1000_N_1000_s_100/'):
    # Directory for the datasets
    x = np.loadtxt(directory+'X_'+str(k)+'.txt')
    y = np.loadtxt(directory+'y_'+str(k)+'.txt')
    beta = np.loadtxt(directory+'beta_'+str(k)+'.txt')
    n = x.shape[0]
    # Take last 500 samples as testing set
    supp = np.where(beta != 0)[0]
    x_test = x[int(n/2):]
    y_test = y[int(n/2):]
    # Take first 500 samples as training set
    x = x[:int(n)/2]
    y = y[:int(n)/2]
    N, p = x.shape
    # normalize if needed
    if normalization:
        for j in range(p):
            x_test[:, j] = x_test[:, j]/np.sqrt(np.sum(x[:, j]**2)/float(N))
            x[:, j] = x[:, j]/np.sqrt(np.sum(x[:, j]**2)/float(N))
    X, Y = torch.Tensor(x), torch.Tensor(y)
    X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
    X, Y = Variable(X), Variable(Y)
    X_test, Y_test = torch.Tensor(x_test), torch.Tensor(y_test)
    X_test, Y_test = X_test.type(torch.FloatTensor), Y_test.type(torch.FloatTensor)
    X_test, Y_test = Variable(X_test), Variable(Y_test)
    return X, Y, X_test, Y_test, supp

##############################
### Measurement of support ###
##############################
# Function for calculating fsr and nsr
def measure(TRUEs, ESTs):
    TRUEs = [set(true) for true in TRUEs]
    ESTs = [set(est) for est in ESTs]
    fs = 0.
    fs_d = 0.
    ns = 0.
    ns_d = 0.
    for i in range(len(TRUEs)):
        fs += len(ESTs[i].difference(TRUEs[i]))
        fs_d += len(ESTs[i])
        ns += len(TRUEs[i].difference(ESTs[i]))
        ns_d += len(TRUEs[i])
    
    fsr = fs/fs_d
    nsr = ns/ns_d
    return fsr, nsr

# Function for calculating accuracy with given model and data
def accuracy(model, x, y):
    out = model(x)
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    return sum(pred_y == y.data.numpy())/float(x.shape[0])

# Function for calculating mse with given model and data
def mse(model, x, y):
    lf = torch.nn.MSELoss()
    out = model(x)
    loss = lf(out, y)
    return loss.data.numpy()[0]


############################
### Weight Normalization ###
############################
# Weight Normalization for Neural Network models
class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'
    
    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()
    
    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            
            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            
            # remove w from parameter list
            del self.module._parameters[name_w]
            
            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)
    
    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)
    
    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

#########################################
### Hadamard product, for first layer ###
#########################################
class DotProduct(torch.nn.Module):
    def __init__(self, in_features):
        super(DotProduct, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data.normal_(0, stdv)
    def forward(self, input):
        output_np = input * self.weight.expand_as(input)
        return output_np
    def __ref__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'

