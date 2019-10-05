
def sgd(x0, obj, grd, learning_rate, num_iters, x_to_str=None, print_every = None):
    x = x0.copy()
    for i in range(num_iters):
        g = grd(x, i)
        if callback: callback(x, i, g)
        x = x - learning_rate(i)*g
    return x
    
    
    
    
 
