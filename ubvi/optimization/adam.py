import autograd.numpy as np

def adam(x0, obj, grd, learning_rate, num_iters, x_to_str=None, print_every = None):
    b1=0.9
    b2=0.999
    eps=10**-8
    x = x0.copy()
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grd(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x
    
    
    
    
    
