import numpy as np
import torch
from torch.autograd import Variable

def DFS_epoch(model, s, supp_x, data, label, loss_func, optimizer1, optimizer2, step1, step2):
    p = data.shape[1]
    _w_before = model.hidden0.weight.data.clone()
    non_nan = (label == label)
    for i in range(step1):
        out = model(data)
        loss = loss_func(out[non_nan], label[non_nan])
        optimizer1.zero_grad()
        loss.backward(retain_graph=True)
        optimizer1.step()
    
    out = model(data)
    loss = loss_func(out[non_nan], label[non_nan])
    loss.backward()
    _w_after = model.hidden0.weight.data.clone()
    z = _w_after - _w_before
    
    z_sort, z_indices = torch.sort(-torch.abs(z))
    Z = z_indices[:2*s].numpy()
    T = set(Z)
    T = set(T).union(supp_x)
    
    Tc = np.setdiff1d(np.arange(p), list(T))
    model.hidden0.weight.data[Tc] = 0
    
    for j in range(step2):
        out = model(data)
        loss = loss_func(out[non_nan], label[non_nan])
        optimizer2.zero_grad()
        loss.backward()
        if len(Tc) > 0:
            optimizer2.param_groups[0]['params'][0].grad[Tc] = 0
        optimizer2.step()
    
    w = model.hidden0.weight.data
    w_sort, w_indices = torch.sort(-torch.abs(w))
    supp_x = w_indices[:s].numpy()
    supp_x_c = np.setdiff1d(np.arange(p), supp_x)
    model.hidden0.weight.data[supp_x_c] = 0
    
    return model, supp_x

