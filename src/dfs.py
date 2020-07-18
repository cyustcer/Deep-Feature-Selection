import numpy as np
import torch

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
    #print(Z)
    T = set(Z)
    T = set(T).union(supp_x) # merge with previous support
    #print("T:")
    #print(T)
    Tc = np.setdiff1d(np.arange(p), list(T)) # indices for not updating variables
    ### Training over candidate support
    for j in range(Ts):
        tmp = list(model.hidden0.weight.data.numpy()) # Save the model parameters
        # updating W
        for _ in range(step):
            out = model(data)
            loss = loss_func(out, label)
            LOSS.append(loss.data.numpy().tolist())
            #print(j)
            #print(loss)
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
            LOSS.append(loss.data.numpy().tolist())
            #print(j)
            #print(loss)
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
    print("Final support:")
    print(supp_x)
    supp_x_c = np.setdiff1d(range(p), list(supp_x)) # un-selected variables
    model.hidden0.weight.data.numpy()[supp_x_c] = 0 # pruning
    return model, supp_x, LOSS

