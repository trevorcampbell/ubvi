import autograd.numpy as np
import time

def adam(x0, obj, grd, learning_rate, num_iters, callback = None):
    b1=0.9
    b2=0.999
    eps=10**-8
    x = x0.copy()
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    t0 = time.perf_counter()
    for i in range(num_iters):
        g = grd(x, i)
        if callback and (i == 0 or i == num_iters - 1 or (time.perf_counter() - t0 > 0.5)): 
            callback(i, x, obj(x, i), g)
            t0 = time.perf_counter()
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x
    
    
    
    
    
