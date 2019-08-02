import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad
import time



def adam(grad, x, num_iters, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        m = (1-b1)*g + b1*m
        v = (1-b2)*g**2 + b2*v
        mhat = m / (1-b1**(i+1))
        vhat = v / (1-b2**(i+1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat)+eps)
    return x



def simplex_projection(x):
  u = np.sort(x)[::-1]
  idcs = np.arange(1, u.shape[0]+1)
  rho_nz = u + 1./idcs*(1.-np.cumsum(u)) > 0
  rho = idcs[rho_nz].max()
  lmb = 1./rho*(1. - u[:rho].sum())
  out = np.maximum(x+lmb, 0.)
  return out/out.sum()



def simplex_sgd(grad, x, num_iters=100, step_size=0.001):
    for i in range(num_iters):
        g = grad(x)
        g -= g.dot(np.ones(g.shape[0]))*np.ones(g.shape[0])/g.shape[0]
        step = step_size/(1+i)
        x -= step * g
        x /= x.sum()
        if np.any(x<0):
            x = simplex_projection(x)
    return x



def mvnlogpdf(x, mu, Sig, Siginv, Diag):
    if Diag:
        return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.sum(Sig, axis=1) - 0.5*np.sum((x[:,np.newaxis,:]-mu)**2*np.exp(-Sig), axis=2)
    if len(Sig.shape)>2:
        return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x[:,np.newaxis,:]-mu)*((Siginv*((x[:,np.newaxis,:]-mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)
    else:
        return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x-mu)*np.linalg.solve(Sig, (x-mu).T).T).sum(axis=1)



def objective(logp, mu, L, g_mu, g_Sig, g_Siginv, Diag, g_w, lmb, n_samples):
    d = g_mu.shape[1]
    std_samples = np.random.randn(n_samples, d)
    if Diag:
        Sig = L
        h_samples = mu+np.exp(0.5*Sig)*std_samples
    else:
        Sig = np.dot(L, np.atleast_2d(L).T)
        h_samples = mu + np.dot(std_samples, np.atleast_2d(L).T)
    lf = logp(h_samples).mean()
    if g_w.shape[0] > 0:
        lg = mvnlogpdf(h_samples[:,np.newaxis] if len(h_samples.shape)==1 else h_samples, g_mu, g_Sig, g_Siginv, Diag)
        lg = logsumexp(lg + np.log(np.maximum(g_w, 1e-64)), axis=1).mean()
    else:
        lg = 0
    lh = mvnlogpdf(h_samples[:,np.newaxis] if len(h_samples.shape)==1 else h_samples, np.atleast_2d(mu), np.atleast_2d(Sig), None, Diag).mean()
    return lg + lmb*lh - lf



def kl_estimate(logp, g_mu, g_Sig, g_Siginv, Diag, g_w, n_samples, d):
    out = 0
    for k in range(g_w.shape[0]):
        samples = g_mu[k] + np.random.randn(n_samples, d)*np.exp(0.5*g_Sig[k]) if Diag else np.random.multivariate_normal(g_mu[k], g_Sig[k], n_samples)
        lg = mvnlogpdf(samples, g_mu, g_Sig, g_Siginv, Diag)
        lg = logsumexp(lg+np.log(np.maximum(g_w, 1e-64)), axis=1)
        lf = logp(samples)
        out += g_w[k]*(lg.mean() - lf.mean())
    return out



def new_weights(logp, g_mu, g_Sig, g_Siginv, Diag, n_samples, d, num_opt_itrs, n_init=10):
    obj = lambda z: kl_estimate(logp, np.atleast_2d(g_mu), np.atleast_2d(g_Sig), np.atleast_2d(g_Siginv), Diag, z, n_samples, d)
    grd = grad(obj)
    x = np.ones(g_Sig.shape[0]) / float(g_Sig.shape[0])
    return simplex_sgd(grd, x, step_size=0.1, num_iters = num_opt_itrs)



def bbvi(logp, N, d, Diag, n_samples, lmb, adam_learning_rate=lambda itr: 1/(1+itr), adam_num_iters=1000, n_init=1):
    g_mu = np.zeros((N,d))
    #g_Sig saves the logrithm of variances when assuming Diagonal structure of covariance matrix
    #g_Sig saves the covariance matrix when assuming general covariance matrix
    g_Sig = np.zeros((N,d)) if Diag else np.zeros((N,d,d))
    g_Siginv = np.zeros((N,d)) if Diag else np.zeros((N,d,d))
    g_w = np.zeros(N)
    G_w = np.zeros((N,N))
    kl = np.zeros(N)
    cput = np.zeros(N)
    
    for i in range(N):
        t0 = time.process_time()
        
        #optimize the next component
        obj = lambda x: objective(logp, x[:d], x[d:] if Diag else x[d:].reshape((d,d)), np.atleast_2d(g_mu[:i]), 
                                  np.atleast_2d(g_Sig[:i]), np.atleast_2d(g_Siginv[:i]), Diag, g_w[:i], lmb[i], n_samples)
        grd = grad(obj)
        x0 = None
        obj0 = np.inf
        for n in range(n_init):
            if Diag:
                lSig = np.random.normal(0, 1)
                if i==0:
                    mu0 = np.random.randn(d)*np.exp(lSig)
                else:
                    k = n%i
                    mu0 = np.random.randn(d)*np.exp(lSig) + g_mu[k-1]
                xtmp = np.hstack((mu0, np.zeros(d)))
            else:
                Sig = np.exp(np.random.normal(0,3))*np.eye(d)
                Sig = np.dot(Sig, Sig.T)
                if i==0:
                    mu0 = np.random.multivariate_normal(np.zeros(d), Sig)
                else:
                    k = n%i
                    mu0 = np.random.multivariate_noraml(g_mu[k-1], Sig)
                L0 = np.eye(d)
                xtmp = np.hstack((mu0, L0.reshape(d*d)))
            objtmp = obj(xtmp)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                
        optimized_params = adam(grd, x0, num_iters=adam_num_iters, learning_rate=adam_learning_rate)
        g_mu[i] = optimized_params[:d]
        if Diag:
            g_Sig[i] = optimized_params[d:]
        else:
            L = optimized_params[d:].reshape((d,d))
            g_Sig[i] = np.dot(L, L.T)
            g_Siginv[i] = np.linalg.inv(g_Sig[i])
        
        #optimize the weights
        g_w[:i+1]=1 if i==0 else new_weights(logp, g_mu[:i+1], g_Sig[:i+1], g_Siginv[:i+1], Diag, n_samples, d, adam_num_iters, n_init)
        G_w[i, :i+1] = g_w[:i+1]
        #compute current KL estimate
        kl[i] = kl_estimate(logp, g_mu[:i+1], g_Sig[:i+1], g_Siginv[:i+1], Diag, g_w[:i+1], n_samples, d)
        cput[i] = time.process_time() - t0
        
    return g_mu, g_Sig, g_w, G_w, cput, kl


    




        