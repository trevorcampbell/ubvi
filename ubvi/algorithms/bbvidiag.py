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
        #x = x - step_size/(1.+max(0, i-num_iters/2.))*mhat/(np.sqrt(vhat) + eps)
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x

def simplex_projection(x):
  u = np.sort(x)[::-1]
  idcs = np.arange(1, u.shape[0]+1)
  rho_nz = u + 1./idcs*(1.-np.cumsum(u)) > 0
  rho = idcs[rho_nz].max()
  lmb = 1./rho*(1. - u[:rho].sum())
  out = np.maximum(x+lmb, 0.)
  return out/out.sum()

def simplex_sgd(grad, x, callback=None, num_iters=100, step_size=0.001):
    for i in range(num_iters):
        g = grad(x, i)
        #project gradient onto simplex
        g -= g.dot(np.ones(g.shape[0]))*np.ones(g.shape[0])/g.shape[0]
        if callback: callback(x, i, g)
        #compute the step size
        step = step_size/(1.+i) #max(0, i-num_iters/2.))
        #take the step
        x -= step*g
        #account for numerical precision stuff
        x /= x.sum()
        #if left simplex, project
        if np.any(x<0):
          x = simplex_projection(x)
    return x

##assumes x is 2d (each row is one datum), mu is 2d (each row is a mean), and sig is 2d (each row is a vector of variances). 
##Returns a N x k matrix, for N data, k means.
def mvnlogpdf(x, mu, lSig):
  #(x[:,np.newaxis,:]-mu) is nxkxd; sig=kxd 
  return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.sum(lSig, axis=1) - 0.5*np.sum((x[:,np.newaxis,:]-mu)**2*np.exp(-lSig), axis=2)


#returns the log objective function for the component opt in hellinger boosting
def objective(logp, mu, lSig, g_mu, g_lSig, g_w, lmb, n_samples):

  d = g_mu.shape[1]

  #get samples from h
  std_samples = np.random.randn(n_samples, d)
  h_samples = mu+np.exp(0.5*lSig)*std_samples

  #compute log densities
  lf = logp(h_samples).mean()
  if g_w.shape[0] > 0:
    lg = mvnlogpdf(h_samples[:,np.newaxis] if len(h_samples.shape)==1 else h_samples, g_mu, g_lSig)
    lg = logsumexp(lg+np.log(np.maximum(g_w, 1e-64)), axis=1).mean()
  else:
    lg = 0.
  #lh = stats.multivariate_normal.logpdf(h_samples, mu, Sig).mean()
  lh = mvnlogpdf(h_samples[:, np.newaxis] if len(h_samples.shape)==1 else h_samples, np.atleast_2d(mu), np.atleast_2d(lSig)).mean()
  #lh_exact = 0.5*np.linalg.slogdet(2.*np.pi*np.exp(1)*Sig)[1]

  return lg+lmb*lh-lf

def kl_estimate(logp, g_mu, g_lSig, g_w, n_samples, d):
  out = 0.
  for k in range(g_w.shape[0]):
    samples = g_mu[k, :] + np.random.randn(n_samples, d)*np.exp(0.5*g_lSig[k,:])
    lg = mvnlogpdf(samples, g_mu, g_lSig)
    lg = logsumexp(lg+np.log(np.maximum(g_w, 1e-64)), axis=1)
    lf = logp(samples)
    out += g_w[k]*(lg.mean()-lf.mean())
  return out #lg.mean()-lf.mean()


def print_perf_w(x, itr, gradient, print_every, d, obj):
  if itr == 0:
    print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
  if itr % print_every == 0:
    print("{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x), np.sqrt((gradient**2).sum()), obj(x, itr)))

  
def print_perf(x, itr, gradient, print_every, d, obj):
  if itr == 0:
    print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'Mu', 'Log(Sig)', 'GradNorm', 'Boosting Obj'))
  if itr % print_every == 0:
    print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(d,4)]), str(x[d:d+min(d,4)]), np.sqrt((gradient**2).sum()), obj(x, itr)))

def new_weights(logp, g_mu, g_lSig, n_samples, print_every, d, num_opt_itrs, n_init=10):
  obj = lambda z, itr : kl_estimate(logp, np.atleast_2d(g_mu), np.atleast_2d(g_lSig), z, n_samples, d)
  grd = grad(obj)
  x = np.ones(g_lSig.shape[0])/float(g_lSig.shape[0])
  return simplex_sgd(grd, x, step_size=0.1, num_iters=num_opt_itrs, 
           callback=lambda prms, itr, grd : print_perf_w(prms, itr, grd, print_every, d, obj))

  
def bbvi(logp, N, d, n_samples, lmb, adam_learning_rate=lambda itr : 1./(itr+1.), adam_num_iters=1000, print_every=10, n_init=1):
  #create storage for mixture
  g_mu = np.zeros((N, d))
  g_lSig = np.zeros((N, d))
  g_w = np.zeros(N)
  G_w = np.zeros((N,N))
  cput = np.zeros(N)
  
  for i in range(N):
    t0 = time.process_time() 
      
    #optimize the next component
    obj = lambda x, itr : objective(logp, x[:d], x[d:], np.atleast_2d(g_mu[:i, :]), np.atleast_2d(g_lSig[:i, :]), g_w[:i], lmb(i), n_samples)
    grd = grad(obj)

    try:
      print('Initialization')
      x0 = None
      obj0 = np.inf
      for n in range(n_init):
        lSig = np.random.normal(0, 1)
        if i==0:
          mu0 = np.random.randn(d)*np.exp(lSig)
        else:
          k = n%i
          mu0 = np.random.randn(d)*np.exp(lSig) + g_mu[k-1,:]
        xtmp = np.hstack((mu0, np.zeros(d)))
        objtmp = obj(xtmp, -1)
        if objtmp < obj0:
          x0 = xtmp
          obj0 = objtmp
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
      g_lSig = np.atleast_2d(g_lSig[:i,:])
      g_w = g_w[:i]
      G_w = np.atleast_2d(G_w[:i,:])
      cput = cput[:i]
      return g_mu, np.exp(g_lSig), g_w, G_w, cput

    print('Component optimization complete')
    
    g_mu[i, :] = optimized_params[:d]
    g_lSig[i, :] = optimized_params[d:]

    print('Updating weights...')
    #optimize the weights
    try:
      g_w[:i+1] = 1. if i == 0 else new_weights(logp, g_mu[:i+1,:], g_lSig[:i+1,:], n_samples, print_every, d, adam_num_iters, n_init)
    except KeyboardInterrupt:
      raise
    except:
      if __DEBUG_MODE__:
        raise
      print('Optimization of component ' + str(i+1) + ' terminated unsuccessfully. Breaking out early...')
      g_mu = np.atleast_2d(g_mu[:i, :])
      g_lSig = np.atleast_2d(g_lSig[:i,:])
      g_w = g_w[:i]
      G_w = np.atleast_2d(G_w[:i,:i])
      cput = cput[:i]
      return g_mu, np.exp(g_lSig), g_w, G_w, cput

    G_w[i,:i+1] = g_w[:i+1] 
    print('Optimal mean: ' + str(g_mu[i,:]))
    print('Optimal cov: ' + str(np.exp(g_lSig[i,:])))
    print('New weights: ' + str(g_w[:i+1]))

    kl = kl_estimate(logp, g_mu[:i+1,:], g_lSig[:i+1,:], g_w[:i+1], n_samples, d)
    print('New KL estimate: ' + str(kl))
  
    cput[i] = time.process_time() - t0

  return g_mu, np.exp(g_lSig), g_w, G_w, cput


