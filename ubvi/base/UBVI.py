import autograd.numpy as np
from scipy.optimize import nnls
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats
from autograd import grad
import time 

np.set_printoptions(precision=2, linewidth=1000)

def adam(grad, x, num_iters, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        m = (1-b1)*g + b1*m
        v = (1-b2)*g**2 + b2*v
        mhat = m / (1 - b1**(i+1))
        vhat = v / (1 - b2**(i+1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat)+eps)
    return x



def mvnlogpdf(x, mu, Sig, Siginv, Diag):
    if Diag:
        return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.sum(Sig, axis=1) - 0.5*np.sum((x[:,np.newaxis,:]-mu)**2*np.exp(-Sig), axis=2)
    if len(Sig.shape) > 2:
        return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x[:, np.newaxis, :]-mu)*((Siginv*(x[:,np.newaxis,:]-mu)[:,:,np.newaxis,:]).sum(axis=3))).sum(axis=2)
    else:
        return -0.5*mu.shape[0]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x-mu)*np.linalg.solve(Sig, (x-mu).T).T).sum(axis=1)



def log_sqrt_pair_integral(mu, Sig, mui, Sigi, Diag):
    if Diag:
        Sig2 = np.log(0.5)+np.logaddexp(Sig,Sigi)
        return -0.125*np.sum(np.exp(-Sig2)*(mu-mui)**2, axis=1) - 0.5*np.sum(Sig2, axis=1) + 0.25*np.sum(Sig) + 0.25*np.sum(Sigi, axis=1)
    else:
        Sig2 = 0.5*(Sig+Sigi)
        return -0.125*((mu-mui) * np.linalg.solve(Sig2, mu-mui)).sum(axis=1) - 0.5*np.linalg.slogdet(Sig2)[1] + 0.25*np.linalg.slogdet(Sig)[1] + 0.25*np.linalg.slogdet(Sigi)[1]
        


def logg(x, g_mu, g_Sig, g_Siginv, Diag, g_lmb):
    if g_lmb.shape[0] > 0:
        logg_x = 0.5*mvnlogpdf(x[:,np.newaxis] if len(x.shape)==1 else x, g_mu, g_Sig, g_Siginv, Diag)
        return logsumexp(logg_x + np.log(np.maximum(g_lmb, 1e-64)), axis=1)
    else:
        return -np.inf*np.ones(x.shape[0])



def logfg_est(logf, mu, Sig, Diag, n_samples):
    if Diag:
        samples = mu + np.exp(0.5*Sig)*np.random.randn(n_samples, mu.shape[0])
        lg = 0.5*stats.norm.logpdf(samples, mu, np.exp(0.5*Sig)).sum(axis=1)
    else:
        samples = np.random.multivariate_normal(mu, Sig, n_samples)
        lg = 0.5*stats.multivariate_normal.logpdf(samples, mu, Sig)
    lf = logf(samples)
    ln = np.log(n_samples)
    return logsumexp(lf - lg - ln)
    


def sample_g(g_mu, g_Sig, g_Siginv, Diag, g_lmb, Z, n_samples):
    g_samples = np.zeros((n_samples, g_mu.shape[1]))
    g_ps = (g_lmb[:, np.newaxis]*Z*g_lmb).flatten()
    g_ps /= g_ps.sum()
    pair_samples = np.random.multinomial(n_samples, g_ps).reshape((g_lmb.shape[0], g_lmb.shape[0]))
    pair_samples = pair_samples + pair_samples.T
    for i in range(pair_samples.shape[0]):
        pair_samples[i,i] /= 2
    cur_idx = 0
    for j in range(g_lmb.shape[0]):
        for k in range(j+1):
            n_samps = pair_samples[j,k]
            if Diag:
                Sigp = 2./np.exp(np.logaddexp(-g_Sig[j], -g_Sig[k]))
                mup = 0.5*Sigp*(np.exp(-g_Sig[j])*g_mu[j] + np.exp(-g_Sig[k])*g_mu[k])
                g_samples[cur_idx:cur_idx+n_samps] = mup + np.sqrt(Sigp)*np.random.randn(n_samps, mup.shape[0])
            else:
                Sigp = 2.0*np.linalg.inv(g_Siginv[j]+g_Siginv[k])
                mup = 0.5*np.dot(Sigp, np.dot(g_Siginv[j], g_mu[j]) + np.dot(g_Siginv[k], g_mu[k]))
                g_samples[cur_idx:cur_idx+n_samps] = np.random.multivariate_normal(mup, Sigp, n_samps)
            cur_idx += n_samps
    return g_samples



def hellsq_est(logf, g_mu, g_Sig, g_Siginv, Diag, g_lmb, Z, n_samples):
    samples = sample_g(g_mu, g_Sig, g_Siginv, Diag, g_lmb, Z, n_samples)
    lf = logf(samples)
    lg = logg(samples, g_mu, g_Sig, g_Siginv, Diag, g_lmb)
    ln = np.log(n_samples)
    return 1 - np.exp(logsumexp(lf-lg-ln) - 0.5*logsumexp(2*lf-2*lg-ln))

    

def new_weights(Z, logfg):
    Linv = np.linalg.inv(np.linalg.cholesky(Z))
    d = np.exp(logfg - logfg.max())
    b = nnls(Linv, -np.dot(Linv, d))[0]
    lbd = np.dot(Linv, b+d)
    return np.maximum(0, np.dot(Linv.T, lbd/np.sqrt((lbd**2).sum())))



def objective(logf, mu, L, Diag, g_mu, g_Sig, g_Siginv, g_lmb, logfg, n_samples):
    d = g_mu.shape[1]
    std_samples = np.random.randn(n_samples, d)
    if Diag:
        lgh = -np.inf if g_lmb.shape[0]==0 else logsumexp(np.log(np.maximum(g_lmb, 1e-64)) + log_sqrt_pair_integral(mu, L, g_mu, g_Sig, Diag))
        h_samples = mu+np.exp(0.5*L)*std_samples
        lh = (0.5*mvnlogpdf(h_samples, mu[np.newaxis, :], L[np.newaxis, :], None, Diag)).flatten()
    else:
        Sig = np.dot(L, np.atleast_2d(L).T)
        lgh = -np.inf if g_lmb.shape[0]==0 else logsumexp(np.log(np.maximum(g_lmb, 1e-64)) + log_sqrt_pair_integral(mu, Sig, g_mu, g_Sig, Diag))
        h_samples = mu + np.dot(std_samples, np.atleast_2d(L).T)
        lh = 0.5 * mvnlogpdf(h_samples, mu, Sig, None, Diag)
    lf = logf(h_samples)
    ln = np.log(n_samples)
    lf_num = logsumexp(lf - lh - ln)
    lg_num = logfg + lgh
    log_denom = 0.5*np.log(1-np.exp(2*lgh))
    if lf_num > lg_num:
        logobj = lf_num - log_denom + np.log(1 - np.exp(lg_num - lf_num)) 
        return logobj
    else:
        logobj = lg_num - log_denom + np.log(1 - np.exp(lf_num - lg_num))
        return -logobj
    
    

def ubvi(logf, N, d, Diag, n_samples, n_logfg_samples, adam_learning_rate=lambda itr: 1/(1+itr), adam_num_iters=1000, n_init=1):
    g_mu = np.zeros((N,d))
    # g_Sig saves the logarithm of the variance when using Diagonal covariance structure!!!
    # g_Sig saves the covariance matrix of the component distribution when using general covariance structure!!!
    g_Sig = np.zeros((N,d)) if Diag else np.zeros((N,d,d))
    g_lmb = np.zeros(N)
    G_lmb = np.zeros((N,N))
    Z = np.zeros((N,N))
    cput = np.zeros(N)
    hellsq = np.zeros(N)
    g_Siginv = np.zeros((N,d)) if Diag else np.zeros((N,d,d))
    logfg = -np.inf * np.ones(N)
    logfgsum = -np.inf
    
    for i in range(N):
        print("Adding component " + str(i+1) + "...")
        t0 = time.process_time()
        # optimize the next component
        obj = lambda x: -objective(logf, x[:d], x[d:] if Diag else x[d:].reshape((d,d)), Diag, np.atleast_2d(g_mu[:i]), np.atleast_2d(g_Sig[:i]), 
                                        np.atleast_2d(g_Siginv[:i]), g_lmb[:i], logfgsum, n_samples)
        grd = grad(obj)
        #initializat component parameters 
        x0 = None
        obj0 = np.inf
        for n in range(n_init):
            if i==0:
                mu0 = mu0 = np.random.randn(d) if Diag else np.random.multivariate_normal(np.zeros(d), np.eye(d))
                # L0 represents the logrithm of elements' variances when using Diagonal covariance structure!!!
                # L0 represents the Cholesky decomposition of covariance matrix when using general covariance structure!!!
                L0 = np.zeros(d) if Diag else np.eye(d)
            else:
                k = np.random.choice(np.arange(i), p=(g_lmb[:i]**2)/(g_lmb[:i]**2).sum())
                if Diag:
                    mu0 = g_mu[k] + np.random.randn(d)*4*np.exp(g_Sig[k])
                    L0 = np.random.randn(d) + g_Sig[k] 
                else:
                    Cov = 16 * g_Sig[k]
                    mu0 = np.random.multivariate_normal(g_mu[k], Cov)
                    L0 = np.exp(np.random.randn())*g_Sig[k]
            xtmp = np.hstack((mu0, L0)) if Diag else np.hstack((mu0, L0.reshape(d*d)))
            objtmp = obj(xtmp)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
            
        #component optimization
        optimized_params = adam(grd, x0, num_iters=adam_num_iters, learning_rate=adam_learning_rate)
        g_mu[i] = optimized_params[:d]
        if Diag:
            g_Sig[i] = optimized_params[d:]
        else:
            L = optimized_params[d:].reshape((d,d))
            g_Sig[i] = np.dot(L, L.T)
            g_Siginv[i] = np.linalg.inv(g_Sig[i])
        
        #update Z
        Znew = np.exp(log_sqrt_pair_integral(g_mu[i], g_Sig[i], g_mu[:i+1], g_Sig[:i+1], Diag))
        Z[i, :i+1] = Znew
        Z[:i+1, i] = Znew
        
        #add a new log<f, g>
        logfg[i] = logfg_est(logf, g_mu[i], g_Sig[i], Diag, n_logfg_samples)
        
        #optimize the weights
        g_lmb[:i+1] = 1 if i==0 else new_weights(Z[:i+1, :i+1], logfg[:i+1])
        G_lmb[i, :i+1] = g_lmb[:i+1]
        
        #compute the current log<f, g> estimate
        logfgsum = logsumexp(np.hstack((-np.inf, logfg[:i+1]+np.log(np.maximum(g_lmb[:i+1], 1e-64)))))  
        
        #new hellinger squared estimate
        hellsq[i] = hellsq_est(logf, g_mu[:i+1], g_Sig[:i+1], g_Siginv[:i+1], Diag, g_lmb[:i+1], Z[:i+1, :i+1], n_logfg_samples)
        
        #cpu time of this iteration
        cput[i] = time.process_time() - t0 
        
    return g_mu, g_Sig, g_lmb, G_lmb, Z, cput, hellsq