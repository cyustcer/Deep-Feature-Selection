import numpy as np
import torch
from models import Net_nonlinear, Net_linear
from utils import accuracy, mse

### Function for one DFS iteration
def DFS_epoch(model, s, supp_x, data, label, loss_func, optimizer0, optimizer, Ts, step=1):
    p = data.shape[1]
    ### Find the directions with 2s largest absolute value of gradients
    out = model(data)
    LOSS = []
    loss = loss_func(out, label)
    loss.backward(retain_graph=True)
    z = model.hidden0.weight.grad.data
    z_sort, z_indices = torch.sort(-torch.abs(z))
    Z = z_indices[:2*s].numpy()
    T = set(Z)
    T = set(T).union(supp_x) # merge with previous support
    Tc = np.setdiff1d(np.arange(p), list(T)) # indices for not updating variables
    ### Training over candidate support
    for j in range(Ts):
        tmp = list(model.hidden0.weight.data.numpy()) # Save the model parameters
        # updating W
        for _ in range(step):
            out = model(data)
            loss = loss_func(out, label)
            #print(loss)
            LOSS.append(loss.data.numpy().tolist())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # set non-updating w to be the original values
            if len(Tc) > 0:
                model.hidden0.weight.data.numpy()[Tc] = np.array(tmp)[Tc]
        # updating w
        for _ in range(step):
            out = model(data)
            loss = loss_func(out, label)
            #print(loss)
            LOSS.append(loss.data.numpy().tolist())
            optimizer0.zero_grad()
            loss.backward(retain_graph=True)
            optimizer0.step()
            # set non-updating w to be the original values
            if len(Tc) > 0:
                model.hidden0.weight.data.numpy()[Tc] = np.array(tmp)[Tc]
    
    if len(Tc) > 0:
        model.hidden0.weight.data.numpy()[Tc] = 0
    ### Find the w's with the largest magnitude
    w = model.hidden0.weight.data
    w_sort, w_indices = torch.sort(-torch.abs(w))
    supp_x = w_indices[:s].numpy()
    #print("Final Support: ", supp_x)
    supp_x_c = np.setdiff1d(range(p), list(supp_x)) # un-selected variables
    model.hidden0.weight.data.numpy()[supp_x_c] = 0 # pruning
    return model, supp_x, LOSS


def training_n(X, Y, X_test, Y_test, c, s, 
               epochs=10, n_hidden1=50, n_hidden2=10, learning_rate=0.05, Ts=250, step=5):
    N, p = X.shape
    torch.manual_seed(1) # set seed 
    # Define neural network model
    model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=2)
    # reserve model for the save of best model
    best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=2)
    # Define optimization algorithm for two sets of parameters, where weight_decay is the lambdas
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=0.0025*c)
    optimizer0 = torch.optim.Adam(model.hidden0.parameters(), lr=learning_rate, weight_decay=0.0005*c)
    # Define loss function
    lf = torch.nn.CrossEntropyLoss()
    # Take track of loss function values and supports over iteration
    hist = []
    SUPP = []
    LOSSES = []
    supp_x = list(range(p)) # initial support
    SUPP.append(supp_x)
    
    ### DFS algorithm
    for i in range(epochs):
        # One DFS epoch
        model, supp_x, LOSS = DFS_epoch(model, s, supp_x, X, Y, lf, optimizer0, optimizer, Ts, step)
        LOSSES = LOSSES + LOSS
        supp_x.sort()
        # Save current loss function value and support
        hist.append(lf(model(X), Y).data.numpy().tolist())
        SUPP.append(supp_x)
        # Prevent divergence of optimization over support, save the current best model
        if hist[-1] == min(hist):
            best_model.load_state_dict(model.state_dict())
            best_supp = supp_x
        # Early stop criteria
        if len(SUPP[-1]) == len(SUPP[-2]) and (SUPP[-1] == SUPP[-2]).all():
            break
    
    # metrics calculation
    _err_train = 1-accuracy(best_model, X, Y) # training error for one-step
    _err_test = 1-accuracy(best_model, X_test, Y_test) # testing error for one-step
    
    ### Second step training (for two-step procedure)
    _optimizer = torch.optim.Adam(list(best_model.parameters())[1:], lr=0.01, weight_decay=0.0025)
    for _ in range(1000):
        out = best_model(X)
        loss = lf(out, Y)
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
    
    bic = (loss.data.numpy().tolist())*N*2. + s*np.log(N) # bic based on final model
    err_train = 1-accuracy(best_model, X, Y)
    err_test = 1-accuracy(best_model, X_test, Y_test)
    return best_model, best_supp, bic, [_err_train, _err_test], [err_train, err_test]


def training_l(X, Y, X_test, Y_test, supp, c, s,
               epochs=10, n_hidden1=1, learning_rate=0.01, Ts=1000, step=1, C=3.):
    N, p = X.shape
    torch.manual_seed(1) # set seed 
    # Define neural network model
    model = Net_linear(n_feature=p, n_hidden1=n_hidden1, n_output=1)
    # reserve model for the save of best model
    best_model = Net_linear(n_feature=p, n_hidden1=n_hidden1, n_output=1)
    # Define optimization algorithm for two sets of parameters, where weight_decay is the lambdas
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.001, weight_decay=0.0025*c)
    optimizer0 = torch.optim.SGD(model.hidden0.parameters(), lr=0.001, weight_decay=0.0005*c)
    # Define loss function
    lf = torch.nn.MSELoss()
    # Take track of loss function values and supports over iteration
    hist = []
    SUPP = []
    supp_x = range(p) # initial support
    SUPP.append(supp_x)
    
    ### DFS algorithm
    for i in range(epochs):
        # One DFS epoch
        model, supp_x, _ = DFS_epoch(model, s, supp_x, X, Y, lf, optimizer0, optimizer, Ts, step)
        supp_x.sort()
        # Save current loss function value and support
        hist.append(lf(model(X), Y).data.numpy().tolist())
        SUPP.append(supp_x)
        # Prevent divergence of optimization over support, save the current best model
        if hist[-1] == min(hist):
            best_model.load_state_dict(model.state_dict())
            best_supp = supp_x
        # Early stop criteria
        if len(SUPP[-1]) == len(SUPP[-2]) and len(set(SUPP[-1]).difference(SUPP[-2])) == 0:
            break
    # metrics calculation
    fs = set(best_supp).difference(supp) # false selection number
    ns = set(supp).difference(best_supp) # negative selection number
    _err_train = mse(best_model, X, Y) # training error
    _err_test = mse(best_model, X_test, Y_test) # testing error
    _bic = N*np.log(_err_train) + C*s*np.log(N) # bic
    
    ### Second step training (for two-step procedure)
    _optimizer = torch.optim.Adam(list(best_model.parameters())[1:], lr=0.5)
    for _ in range(5000):
        out = best_model(X)
        loss = lf(out, Y)
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
        hist.append(loss.data.numpy().tolist())
    
    bic = N*np.log(loss.data.numpy().tolist()) + C*s*np.log(N) # bic based on final model
    err_train = mse(best_model, X, Y)
    err_test = mse(best_model, X_test, Y_test)
    return best_model, best_supp, bic, [_err_train, _err_test], [err_train, err_test]
