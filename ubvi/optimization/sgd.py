import time

def sgd(x0, obj, grd, learning_rate, num_iters, callback = None):
    t0 = time.perf_counter()
    x = x0.copy()
    for i in range(num_iters):
        g = grd(x, i)
        if callback and time.perf_counter() - t0 > 0.5: 
            callback(i, x, obj(x, i), g)
            t0 = time.perf_counter()
        x = x - learning_rate(i)*g
    return x
    
    
    
    
 
