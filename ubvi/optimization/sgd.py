import autograd.numpy as np

#TODO code sgd
def sgd(self, grd, x, learning_rate, num_iters, print_every, callback=None):
    b1=0.9
    b2=0.999
    eps=10**-8
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(self.num_iters):
        g = grd(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - self.learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x
    
    
    
    
 
