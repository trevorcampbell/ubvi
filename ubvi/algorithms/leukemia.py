import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats
from autograd.scipy.special import gammaln
from autograd import grad
import time

from ubvi import ubvi
from bbvi import bbvi

import matplotlib.pyplot as plt
import os

from hmc import hmc, nuts

import pystan

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



np.seterr(invalid='raise', over='raise')

#theta = (beta, log lambda, log tau, log c^2)
def log_horseshoelogistic(theta, Z):
    N = Z.shape[0]
    theta2 = np.atleast_2d(theta)
    d = int((theta2.shape[1] - 2)/2)
    logtau0 = np.log(1e-3) #from "yes but did it work" appendix

    #logistic likelihood
    dots = -np.dot(Z, theta2[:, :d].T)
    log_lik = -np.sum(np.maximum(dots, 0) + np.log1p(np.exp(-np.abs(dots))), axis=0)
  
    #inverse gamma(2,8) c^2 prior
    log_c_prior = 2.*np.log(8.) - gammaln(2.) - (2.+1.)*theta2[:,-1] - 8./np.exp(theta2[:, -1])
    #half cauchy tau prior
    log_tau_prior = np.log(2.)-np.log(np.pi) + logtau0 - logsumexp( np.vstack( (2*theta2[:, -2], 2*logtau0*np.ones(theta2.shape[0]))), axis=0)
    #half cauchy lmb prior
    log_lmb_prior = np.sum(np.log(2.)-np.log(np.pi) - np.maximum(2*theta2[:,d:-2], 0) - np.log1p(np.exp(-np.abs(2*theta2[:,d:-2]))), axis=1)

    #normal beta prior
    log_prm_num = 2*theta2[:,-2][:,np.newaxis]  + 2*theta2[:, d:-2]
    tlc = 2*theta2[:,-2][:,np.newaxis] + 2*theta2[:,d:-2] - theta2[:,-1][:,np.newaxis]
    log_prm_denom = np.maximum(tlc, 0) + np.log1p(np.exp(-np.abs(tlc)))
    logprm = log_prm_num - log_prm_denom
    log_beta_prior = -0.5*d*np.log(2*np.pi) - 0.5*np.sum(logprm, axis=1) - 0.5*np.sum((theta2[:, :d])**2/np.exp(logprm), axis=1)
    #prm = theta2[:, -2][:,np.newaxis]**2*theta2[:, -1][:,np.newaxis]*theta2[:,d:-2]**2/(theta2[:,-1][:,np.newaxis]+theta2[:,-2][:,np.newaxis]**2*theta2[:,d:-2]**2)
    #log_beta_prior = -0.5*d*np.log(2*np.pi) - 0.5*np.sum(np.log(prm), axis=1) - 0.5*np.sum((theta2[:, :d])**2/prm, axis=1)

    return log_lik + log_c_prior + log_tau_prior + log_lmb_prior + log_beta_prior

def logf(theta, Z):
    return 0.5*log_horseshoelogistic(theta, Z)


def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']

  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  v = X[:, :-1].std(axis=0)
  X[:, :-1] = (X[:, :-1] - m)/v

  Z = data['y'][:, np.newaxis]*X
  data.close()
  return Z, X[:, :-1], Y


print('loading leukemia data')
Z, X, Y = load_data('./data/leukemia.npz')
Y[Y == -1] = 0

horse = lambda theta : log_horseshoelogistic(theta, Z)
horsef = lambda theta : logf(theta, Z)
ghorse = grad(horse)

print('initialization')
adam_learning_rate = lambda itr : 0.01/(1.+itr)
num_opt_itrs = 30000
if not os.path.exists('leukx0.npy'):
  print('no initialization found, optimizing')
  gahorse = grad(lambda x, i : -horse(x))
  x0 = np.zeros(2*Z.shape[1]+2)
  def cbk(x, i, g):
    if i % 100 == 0:
      print('i: ' + str(i)+' obj: ' + str(horse(x)) + ' tau: ' + str(x[-2]) + ' c^2: ' + str(x[-1]))
  x0 = adam(gahorse, x0, learning_rate = adam_learning_rate, num_iters=num_opt_itrs, callback = cbk)
  print('final x:')
  print(x0)
  np.save('leukx0.npy', x0)
else:
  print('initialization found, loading')
  x0 = np.load('leukx0.npy')

beta_init = x0[:Z.shape[1]-1]
beta0_init = x0[Z.shape[1]-1]
lmb_init = np.exp(x0[Z.shape[1]:2*Z.shape[1]])[:-1]
tau_init = np.exp(x0[-2])
csq_init = np.exp(x0[-1])

print(beta_init)
print(beta0_init)
print(lmb_init)
print(tau_init)
print(csq_init)


############################## NUTS via STAN ######################################

leukemia_code = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0,upper=1> y[n]; // outputs
  matrix[n,d] x; // inputs
  real<lower=0> scale_icept; // prior std for the intercept
  real<lower=0> scale_global; // scale for the half-t prior for tau
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
}
parameters {
  real beta0; // intercept
  vector[d] z; // auxiliary parameter
  real<lower=0> tau; // global shrinkage parameter
  vector<lower=0>[d] lmb; // local shrinkage parameter
  real<lower=0> caux; // auxiliary
}
transformed parameters {
  real<lower=0> c;
  vector[d] beta; // regression coefficients
  vector[n] f; // latent values
  vector<lower=0>[d] lmb_tilde;
  c = slab_scale * sqrt(caux);
  lmb_tilde = sqrt( c^2 * square(lmb) ./ (c^2 + tau^2* square(lmb)) );
  beta = z .* lmb_tilde*tau;
  f = beta0 + x*beta;
}
model {
  z ~ normal(0,1);
  lmb ~ cauchy(0,1);
  tau ~ cauchy(0, scale_global);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  beta0 ~ normal(0,scale_icept);
  y ~ bernoulli_logit(f);
}
"""


N_samples = 10000
N_per = 2000
leukemia_data = {'x': X, 'y':Y.astype(int), 'scale_icept': 10., 'scale_global':1/(X.shape[1]-1)*2/np.sqrt(X.shape[0]), 'd': X.shape[1], 'n': X.shape[0], 'slab_scale': 5., 'slab_df':4.}
if not os.path.exists('leukemia_model.pk'):
  sm = pystan.StanModel(model_code=leukemia_code)
  f = open('leukemia_model.pk', 'wb')
  pk.dump(sm, f)
  f.close()
else:
  f = open('leukemia_model.pk', 'rb')
  sm = pk.load(f)
  f.close()
t0 = time.process_time()
for i in range(int(N_samples/N_per)):
  if not os.path.exists('leukemia_samples_'+str(i)+'.npy'):
    fit = sm.sampling(data=leukemia_data, iter=N_per*2, chains=1, init=[dict(beta0=beta0_init, tau=tau_init,lmb=lmb_init, c=np.sqrt(csq_init))]*4, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True, check_hmc_diagnostics=True)
    #fit.extract has 3 dims: iterations, chains, parametrs
    f = open('leukemia_params_'+str(i)+'.log', 'w')
    f.write(str(fit.model_pars)+'\n')
    f.write(str(fit.par_dims)+'\n')
    f.close()
    np.save('leukemia_samples_'+str(i)+'.npy', fit.extract(permuted=False))
tf = time.process_time()
np.save('leukemia_mcmc_time.npy', tf-t0)


res = []
N = 4
d = 2*Z.shape[1]+2
n_samples = 1000
adam_num_iters = 10000
print_every=100
n_init=50
n_subsample = 100
adam_learning_rate = lambda itr : 0.1/(1.+itr)
lmb = lambda itr : 10./(itr+1)

print('Running ID ' + str(int(sys.argv[1]))) #just to make sure user input it before running things

############################## UBVI ######################################


print('running ubvi')
ubvi_ = ubvi(horsef, N, d, n_samples, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)
res.append(ubvi_)
f = open('leukemia'+str(sys.argv[1])+'.pk', 'wb')
pk.dump(res, f)
f.close()

############################## BBVI ######################################

print('running bbvi')
bbvi_ = bbvi(horse, N, d, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init)
res.append(bbvi_)
f = open('leukemia'+str(sys.argv[1])+'.pk', 'wb')
pk.dump(res, f)
f.close()

############################## ADVI ######################################

print('running advi')
lmb = lambda itr : 1.
advi_ = bbvi(horse, 1, d, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init)
res.append(advi_)
f = open('leukemia_'+str(sys.argv[1])+'.pk', 'wb')
pk.dump(res, f)
f.close()

