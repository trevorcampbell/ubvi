import autograd.numpy as np



class Optimization(object):
    
    def optimize(self):
        raise NotImplementedError



class Adam(Optimization):
    
    def __init__(self, learning_rate, num_iters, print_every):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.print_every = print_every
        
    def optimize(self, grd, x):
        b1=0.9
        b2=0.999
        eps=10**-8
        callback=None
        m = np.zeros(len(x))
        v = np.zeros(len(x))
        for i in range(self.num_iters):
            g = grd(x)
            if callback: callback(x, i, g)
            m = (1 - b1) * g      + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1**(i + 1))    # Bias correction.
            vhat = v / (1 - b2**(i + 1))
            x = x - self.learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
        return x
    
    