import autograd.numpy as np

def sgd(grd, x0, learning_rate, num_iters, callback=None):
    x = x0.copy()
    for i in range(num_iters):
        g = grd(x, i)
        if callback: callback(x, i, g)
        x = x - learning_rate(i)*g
    return x
    
    
    
    
 
