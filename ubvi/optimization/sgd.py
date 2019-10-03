import autograd.numpy as np

def sgd(grd, x, learning_rate, num_iters, callback=None):
    for i in range(num_iters):
        g = grd(x, i)
        if callback: callback(x, i, g)
        x = x - learning_rate(i)*g
    return x
    
    
    
    
 
