import autograd.numpy as np
from scipy.optimize import nnls
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats
#from autograd.misc.optimizers import adam
from autograd import grad
import time

__DEBUG_MODE__ = False
np.set_printoptions(precision=2, linewidth=1000)

def adam(grad, x, num_iters, learning_rate, 
        b1=0.9, b2=0.999, eps=10**-8,callback=None):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x


##assumes x is 2d (each row is one datum), mu is 2d (each row is a mean), and sig is 3d (each row is a covar). Returns a N x k matrix, for N data, k means.
#def mvnlogpdf(x, mu, Sig, Siginv):
#  #(x[:,np.newaxis,:]-mu) is nxkxd; sig=kxdxd 
#  return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x[:,np.newaxis,:]-mu)*((Siginv*((x[:,np.newaxis,:]-mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)

def mvnlogpdf(x, mu, Sig, Siginv):
  if len(Sig.shape) > 2:
    #(x[:,np.newaxis,:]-mu) is nxkxd; sig=kxdxd 
    return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x[:,np.newaxis,:]-mu)*((Siginv*((x[:,np.newaxis,:]-mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)
  else:
    return -0.5*mu.shape[0]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x-mu)*np.linalg.solve(Sig, (x-mu).T).T).sum(axis=1)

#returns array of [log < N(mu, Sig)^{1/2},  N(mu_i, Sig_i)^{1/2} >] over i
def log_sqrt_pair_integral(mu, Sig, mui, Sigi):
  Sig2 = 0.5*(Sig+Sigi)
  return -0.125*((mu-mui) * np.linalg.solve(Sig2, mu-mui)).sum(axis=1) - 0.5*np.linalg.slogdet(Sig2)[1] + 0.25*np.linalg.slogdet(Sig)[1] + 0.25*np.linalg.slogdet(Sigi)[1]
  
#returns the log objective function for the component opt in hellinger boosting
def objective(logf, mu, L, g_mu, g_Sig, g_Siginv, g_lmb, logfg, n_samples, allow_negative=False):

  d = g_mu.shape[1]

  #compute the covariance matrix from cholesky
  Sig = np.dot(L, np.atleast_2d(L).T)

  lgh = -np.inf if g_lmb.shape[0] == 0 else logsumexp(np.log(np.maximum(g_lmb, 1e-64)) + log_sqrt_pair_integral(mu, Sig, g_mu, g_Sig))
  #get samples from h
  std_samples = np.random.randn(n_samples, d)
  h_samples = mu+np.dot(std_samples, np.atleast_2d(L).T)
  lf = logf(h_samples)
  lh = 0.5*mvnlogpdf(h_samples, mu, Sig, None) #stats.multivariate_normal.logpdf(h_samples, mu, Sig)
  ln = np.log(n_samples)
  lf_num = logsumexp(lf - lh - ln)
  lg_num = logfg + lgh 
  log_denom = 0.5*np.log(1.-np.exp(2*lgh))

  if lf_num > lg_num:
    logobj = lf_num - log_denom + np.log(1.-np.exp(lg_num-lf_num))
    return logobj
  else:
    logobj = lg_num - log_denom + np.log(1.-np.exp(lf_num-lg_num))
    return -logobj

  #if g_lmb.shape[0] > 0:
  #  log_denom = 0.5*logsumexp( np.array( [0., 2.*logsumexp(np.log(np.maximum(g_lmb, 1e-64)) + log_sqrt_pair_integral(mu, Sig, g_mu, g_Sig))] ), b=np.array([1., -1.]))
  #else:
  #  log_denom = 0.

  ##get samples from h
  #std_samples = np.random.randn(n_samples, d)
  #h_samples = mu+np.dot(std_samples, np.atleast_2d(L).T)

  ##compute empirical vectors for logh, logf, and logg
  #lh = 0.5*mvnlogpdf(h_samples, mu, Sig, None) #stats.multivariate_normal.logpdf(h_samples, mu, Sig)
  #lf = logf(h_samples)
  #lg = logg(h_samples, g_mu, g_Sig, g_Siginv, g_lmb) + logfg
  #ln = np.log(n_samples)

  #lf_num = logsumexp(lf - lh - ln)
  #lg_num = -np.inf if log_denom == 0. else logsumexp(lg - lh - ln)

  #if lf_num > lg_num:
  #  logobj = lf_num - log_denom + np.log(1.-np.exp(lg_num-lf_num))
  #  return logobj
  #else:
  #  logobj = lg_num - log_denom + np.log(1.-np.exp(lf_num-lg_num))
  #  return -logobj

#returns the log of g = sum_i lmb_i * g_i, g_i = N(mu_i, Sig_i)^{1/2}
def logg(x, g_mu, g_Sig, g_Siginv, g_lmb):
  if g_lmb.shape[0] > 0:
    logg_x = 0.5*mvnlogpdf(x[:,np.newaxis] if len(x.shape)==1 else x, g_mu, g_Sig, g_Siginv)
    return logsumexp(logg_x + np.log(np.maximum(g_lmb, 1e-64)), axis=1)
  else:
    return -np.inf*np.ones(x.shape[0])

#estimates log<f, g_i> using samples from g_i^2
def logfg_est(logf, mu, Sig, n_samples):
  samples = np.random.multivariate_normal(mu, Sig, n_samples)
  lf = logf(samples)
  lg = 0.5*stats.multivariate_normal.logpdf(samples, mu, Sig)
  ln = np.log(n_samples)
  return logsumexp(lf - lg - ln)

#samples from g^2
def sample_g(g_mu, g_Sig, g_Siginv, g_lmb, Z, n_samples):
  g_samples = np.zeros((n_samples, g_mu.shape[1]))
  #compute # samples in each mixture pair
  g_ps = (g_lmb[:, np.newaxis]*Z*g_lmb).flatten()
  g_ps /= g_ps.sum()
  pair_samples = np.random.multinomial(n_samples, g_ps).reshape((g_lmb.shape[0], g_lmb.shape[0]))
  #symmetrize (will just use lower triangular below for efficiency)
  pair_samples = pair_samples + pair_samples.T
  for i in range(pair_samples.shape[0]):
    pair_samples[i,i] /= 2

  #invert sigs
  #fill in samples
  cur_idx = 0
  for j in range(g_lmb.shape[0]):
    for k in range(j+1):
      n_samps = pair_samples[j,k]
      Sigp = 2.0*np.linalg.inv(g_Siginv[j,:,:]+g_Siginv[k,:,:])
      mup = 0.5*np.dot(Sigp, np.dot(g_Siginv[j,:,:], g_mu[j,:]) + np.dot(g_Siginv[k,:,:], g_mu[k,:]))
      g_samples[cur_idx:cur_idx+n_samps, :] = np.random.multivariate_normal(mup, Sigp, n_samps)
      cur_idx += n_samps
  return g_samples

def hellsq_est(logf, g_mu, g_Sig, g_Siginv, g_lmb, Z, n_samples):
  samples = sample_g(g_mu, g_Sig, g_Siginv, g_lmb, Z, n_samples)
  lf = logf(samples)
  lg = logg(samples, g_mu, g_Sig, g_Siginv, g_lmb)
  ln = np.log(n_samples)
  return 1. - np.exp(logsumexp(lf-lg-ln) - 0.5*logsumexp(2*lf-2*lg-ln))

def new_weights(Z, logfg):
    Linv = np.linalg.inv(np.linalg.cholesky(Z))
    d = np.exp(logfg-logfg.max()) #the opt is invariant to d scale, so normalize to have max 1
    b = nnls(Linv, -np.dot(Linv, d))[0]
    lbd = np.dot(Linv, b+d)
    return np.maximum(0., np.dot(Linv.T, lbd/np.sqrt(((lbd**2).sum()))))
  
def print_perf(x, itr, gradient, print_every, d, obj):
  if itr == 0:
    print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'Mu', 'Eigvals(Sig)', 'GradNorm', 'Alignment'))
  if itr % print_every == 0:
    L = x[d:].reshape((d,d))
    print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(d,4)]), str(np.linalg.eigvalsh(np.dot(L,L.T))[:min(d,4)]), np.sqrt((gradient**2).sum()), -obj(x, itr)))
  
def ubvi(logf, N, d, n_samples, n_logfg_samples, adam_learning_rate= lambda itr : 1./(1.+itr), adam_num_iters=1000, print_every=10, n_init=1, init_inflation=16):
  #create storage for mixture
  g_mu = np.zeros((N, d))
  g_Sig = np.zeros((N, d, d))
  g_Siginv = np.zeros((N, d, d))
  g_lmb = np.zeros(N)
  G_lmb = np.zeros((N,N))
  Z = np.zeros((N, N))
  logfg = -np.inf*np.ones(N)
  cput = np.zeros(N)
  logfgsum = -np.inf
  

  for i in range(N):
    
    t0 = time.process_time() 

    #optimize the next component
    obj = lambda x, itr : -objective(logf, x[:d], x[d:].reshape((d,d)), np.atleast_2d(g_mu[:i, :]), np.atleast_2d(g_Sig[:i, :, :]), np.atleast_2d(g_Siginv[:i,:,:]), g_lmb[:i], logfgsum, n_samples, allow_negative = False if itr < 0 else True)
    grd = grad(obj)

    try:
      print('Initialization')
      x0 = None
      obj0 = np.inf
      for n in range(n_init):
        if i==0:
          mu0 = np.random.multivariate_normal(np.zeros(d), np.eye(d))
          L0 = np.eye(d)
        else:
          k = np.random.choice(np.arange(i), p=(g_lmb[:i]**2)/(g_lmb[:i]**2).sum())
          mu0 = np.random.multivariate_normal(g_mu[k,:], init_inflation*g_Sig[k,:,:])
          L0 = np.exp(np.random.randn())*g_Sig[k,:,:]
        xtmp = np.hstack((mu0, L0.reshape(d*d)))
        objtmp = obj(xtmp, -1)
        if objtmp < obj0:
          x0 = xtmp
          obj0 = objtmp
          print('improved x0: ' + str(x0) + ' with obj0 = ' + str(obj0))
        print('improved x0: ' + str(x0) + ' with obj0 = ' + str(obj0))
      if x0 is None:
        raise ValueError
      print('Optimization of component ' + str(i+1))
      optimized_params = adam(grd, x0, num_iters=adam_num_iters, learning_rate=adam_learning_rate,  
                   callback=lambda prms, itr, grd : print_perf(prms, itr, grd, print_every, d, obj) )
    except KeyboardInterrupt:
      raise
    except:
      if __DEBUG_MODE__:
        raise
      print('Optimization of component ' + str(i+1) + ' terminated unsuccessfully. Breaking out early...')
      g_mu = np.atleast_2d(g_mu[:i, :])
      g_Sig = np.atleast_3d(g_Sig[:i,:,:])
      g_lmb = g_lmb[:i]
      G_lmb = np.atleast_2d(G_lmb[:i,:i])
      Z = np.atleast_2d(Z[:i,:i])
      cput = cput[:i]
      return g_mu, g_Sig, g_lmb, G_lmb, Z, cput
      

    print('Component optimization complete')
    
    g_mu[i, :] = optimized_params[:d]
    L = optimized_params[d:].reshape((d,d))
    g_Sig[i, :, :] = np.dot(L, L.T)
    g_Siginv[i,:,:] = np.linalg.inv(g_Sig[i,:,:])

    

    print('Updating Z...')
    #update Z
    Znew = np.exp(log_sqrt_pair_integral(g_mu[i, :], g_Sig[i, :, :], g_mu[:i+1, :], g_Sig[:i+1, :, :]))
    Z[i, :i+1] = Znew
    Z[:i+1, i] = Znew

    print('Updating log<f,g>...')
    #add a new logfg
    logfg[i] = logfg_est(logf, g_mu[i,:], g_Sig[i,:,:], n_logfg_samples)


    print('Updating weights...')
    #optimize the weights
    g_lmb[:i+1] = 1 if i == 0 else new_weights(Z[:i+1,:i+1], logfg[:i+1])
    G_lmb[i,:i+1] = g_lmb[:i+1] 

    #compute current log<f, g> estimate
    logfgsum = logsumexp(np.hstack((-np.inf, logfg[:i+1] + np.log(np.maximum(g_lmb[:i+1], 1e-64)))))

    print('Optimal mean: ' + str(g_mu[i,:]))
    print('Optimal cov: ' + str(g_Sig[i,:,:]))
    print('New weights: ' + str(g_lmb[:i+1]))
    print('New Z: ' + str(Z[:i+1,:i+1]))
    print('New log<f,g>: ' + str(logfg[:i+1]))

    hellsq = hellsq_est(logf, g_mu[:i+1,:], g_Sig[:i+1,:,:], g_Siginv[:i+1,:,:], g_lmb[:i+1], Z[:i+1,:i+1], n_logfg_samples)
    print('New Hellinger-Squared estimate: ' + str(hellsq))
  
    cput[i] = time.process_time() - t0



  return g_mu, g_Sig, g_lmb, G_lmb, Z, cput




##
##    #plot the contours
##    import matplotlib.pyplot as plt
##    x = np.linspace(1.5, 3.5, 1000)
##    y = np.linspace(1.0, 3.0, 1000)
##    xx, yy = np.meshgrid(x, y)
##    x = xx.reshape(-1,1)
##    y = yy.reshape(-1,1)
##    X = np.hstack((x,y, np.zeros((x.shape[0], 1))))
##    #plot the truth
##    Y = 2*logf(X, True).reshape(1000,1000)
##    Y -= Y.max()
##    Yf = np.exp(Y)/(np.exp(Y).sum())
##    #Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
##    #Levels = np.array([0.001, .005, 0.015, 0.025])
##    #plt.contour(xx, yy, Y, levels=Levels, colors='black', linewidths=2) #cmap="Blues_r")
##    plt.contour(xx, yy, Yf, colors='black', linewidths=2) #cmap="Blues_r")
##    
##    #plot UBVI
##    print('computing lgoggagsdgahsdfgkjsdhfg')
##    Y = 2*logg(X, g_mu[:i+1,:], g_Sig[:i+1,:,:], g_Siginv[:i+1,:,:], g_lmb[:i+1]).reshape(1000,1000)
##    Y -= Y.max()
##    Yg = np.exp(Y)/(np.exp(Y).sum())
##    #Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
##    #Levels = np.array([0.001, .005, 0.015, 0.025])
##    #plt.contour(xx, yy, Y, levels=Levels, colors=pal[0], linewidths=2) #cmap="Dark2")
##    plt.contour(xx, yy, Yg, colors='blue', linewidths=2) #cmap="Dark2")
##
##    plt.scatter(samples[:,0], samples[:,1])
##
##    print('discretized hellinger')
##    print(str(1. - np.sqrt(Yg*Yf).sum()))
##    print('2norm')
##    print(str(np.sqrt(Yg*Yg).sum()))
##    print(str(np.sqrt(Yf*Yf).sum()))
##
##    plt.show()
##


