import autograd.numpy as np

def simplex_sgd(x0, obj, grd, learning_rate, num_iters, callback = None):
    x = x0.copy()
    for i in range(num_iters):
        g = grd(x, i)
        #project gradient onto simplex
        g -= g.dot(np.ones(g.shape[0]))*np.ones(g.shape[0])/g.shape[0]
        if callback: callback(i, x, obj(x, i), g)
        #take the step
        x -= learning_rate(i)*g
        #account for numerical precision stuff
        x /= x.sum()
        #if left simplex, project
        if np.any(x<0):
            x = _simplex_projection(x)
    return x

def _simplex_projection(x):
    u = np.sort(x)[::-1]
    idcs = np.arange(1, u.shape[0]+1)
    rho_nz = u + 1./idcs*(1.-np.cumsum(u)) > 0
    rho = idcs[rho_nz].max()
    lmb = 1./rho*(1. - u[:rho].sum())
    out = np.maximum(x+lmb, 0.)
    return out/out.sum()

