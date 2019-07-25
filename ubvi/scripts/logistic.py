import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats
from autograd.scipy.special import gammaln
from autograd import grad
import os
import time
import pickle as pk
import sys
import pystan
from ubvidiag import ubvi
from bbvidiag import bbvi

def log_cauchy(X):
  return -np.log(np.pi) - np.log(1.+X**2)

def log_multivariate_t(X, mu, Sigma, df):    
    p = Sigma.shape[0]
    dots = np.sum((X-mu)*(np.dot(np.linalg.inv(Sigma), (X-mu).T).T), axis=1)
    return -0.5*(df+p)*np.log(1.+dots/df) + gammaln(0.5*(df+p)) - gammaln(0.5*df) - 0.5*p*np.log(df*np.pi) - 0.5*np.linalg.slogdet(Sigma)[1]

def log_logist(theta, Z, nu, mu, Sig):
    dots = -np.dot(Z, theta.T)
    log_lik = -np.sum(np.maximum(dots, 0) + np.log1p(np.exp(-np.abs(dots))), axis=0)
    log_pri = log_multivariate_t(theta[:, :-1], mu, Sig, nu) + log_cauchy(theta[:, -1])
    return log_pri + log_lik

def log_f(theta, Z, nu, mu, Sig):
    return 0.5*log_logist(theta, Z, nu, mu, Sig)

def load_data(dnm, subset_sz=None):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  Xt = data['Xt']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Xt[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (Xt[:, :-1] - m).T).T
  Z = data['y'][:, np.newaxis]*X
  Zt = data['yt'][:, np.newaxis]*Xt
  data.close()
  if subset_sz is None:
    subset_sz = Z.shape[0]
  return Z[:subset_sz, :], X[:subset_sz, :-1], Y[:subset_sz]


print('loading datasets')
subset_sz = 20
Z_synth, X_synth, Y_synth = load_data('./data/synth.npz', subset_sz)
Y_synth[Y_synth == -1] = 0
Z_ds1, X_ds1, Y_ds1 = load_data('./data/ds1.npz', subset_sz)
Y_ds1[Y_ds1 == -1] = 0
Z_phish, X_phish, Y_phish = load_data('./data/phishing.npz', subset_sz)
Y_phish[Y_phish == -1] = 0


#ensure we use the same Sig each time
np.random.seed(1)
nu = 2.
Sig_synth = np.array([[1., -0.9], [-0.9, 1.]])
mu_synth = np.zeros(2)
Sig_ds1 = np.random.randn(X_ds1.shape[1], X_ds1.shape[1])
Sig_ds1 = Sig_ds1.T.dot(Sig_ds1)
mu_ds1 = np.zeros(Sig_ds1.shape[0])
Sig_phish = np.random.randn(X_phish.shape[1], X_phish.shape[1])
Sig_phish = Sig_phish.T.dot(Sig_phish)
mu_phish = np.zeros(Sig_phish.shape[0])
np.random.seed()


d_synth = Z_synth.shape[1]
d_ds1 = Z_ds1.shape[1]
d_phish = Z_phish.shape[1]



############################## NUTS via STAN ######################################

logistic_code = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0,upper=1> y[n]; // outputs
  matrix[n,d] x; // inputs
  real<lower=0> nu; // multivariate t prior scale
  vector[d] mu; // multivariate t mean
  matrix[d,d] Sig; // multivariate t Sig
}
parameters {
  real theta0; // intercept
  vector[d] theta; // auxiliary parameter
}
transformed parameters {
  vector[n] f;
  f = theta0 + x*theta;
}
model {
  theta0 ~ cauchy(0, 1);
  theta ~ multi_student_t(nu, mu, Sig);
  y ~ bernoulli_logit(f);
}
"""

logp_synth = lambda theta : log_logist(theta, Z_synth, nu, mu_synth, Sig_synth)
logf_synth = lambda theta : log_f(theta, Z_synth, nu, mu_synth, Sig_synth)

logp_ds1 = lambda theta: log_logist(theta, Z_ds1, nu, mu_ds1, Sig_ds1)
logf_ds1 = lambda theta: log_f(theta, Z_ds1, nu, mu_ds1, Sig_ds1) 

logp_phish = lambda theta: log_logist(theta, Z_phish, nu, mu_phish, Sig_phish)
logf_phish = lambda theta: log_f(theta, Z_phish, nu, mu_phish, Sig_phish)

#try to make results directory if it doesn't already exist
if not os.path.exists('results/'):
  os.mkdir('results')
#load stan models if possible
if not os.path.exists('results/logistic_model.pk'):
  sm = pystan.StanModel(model_code=logistic_code)
  f = open('results/logistic_model.pk', 'wb')
  pk.dump(sm, f)
  f.close()
else:
  f = open('results/logistic_model.pk', 'rb')
  sm = pk.load(f)
  f.close()

logistic_data_synth = {'x': X_synth, 'y':Y_synth.astype(int), 'd': X_synth.shape[1], 'n': X_synth.shape[0], 'mu':mu_synth, 'nu':nu, 'Sig':Sig_synth}
logistic_data_ds1 = {'x': X_ds1, 'y':Y_ds1.astype(int), 'd': X_ds1.shape[1], 'n': X_ds1.shape[0], 'mu':mu_ds1, 'nu':nu, 'Sig':Sig_ds1}
logistic_data_phish = {'x': X_phish, 'y':Y_phish.astype(int), 'd': X_phish.shape[1], 'n': X_phish.shape[0], 'mu':mu_phish, 'nu':nu, 'Sig':Sig_phish}
N_samples = 40000
N_per = 2000
for d, nm in [(logistic_data_synth, 'synth'), (logistic_data_ds1, 'ds1'), (logistic_data_phish, 'phish')]:
  if not os.path.exists('results/logistic_samples_'+nm+'.npy'):
    t0 = time.process_time()
    fit = sm.sampling(data=d, iter=N_samples*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    #fit.extract has 3 dims: iterations, chains, parametrs
    f = open('results/logistic_params_' + nm+'.log', 'w')
    f.write(str(pystan.check_hmc_diagnostics(fit))+'\n')
    f.write(str(fit.model_pars)+'\n')
    f.write(str(fit.par_dims)+'\n')
    f.close()
    np.save('results/logistic_samples_'+nm+'.npy', fit.extract(permuted=False))
    tf = time.process_time()
    np.save('results/'+nm+'_mcmc_time.npy', tf-t0)
    #if samples are too large for memory, use this instead
    #t0 = time.process_time()
    #for i in range(int(N_samples/N_per)):
    #  if not os.path.exists('logistic_samples_'+nm+'_'+str(i)+'.npy'):
    #    fit = sm.sampling(data=d, iter=N_per*2, chains=4, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    #    #fit.extract has 3 dims: iterations, chains, parametrs
    #    f = open('logistic_params_' + nm+'_'+str(i)+'.log', 'w')
    #    f.write(str(pystan.check_hmc_diagnostics(fit))+'\n')
    #    f.write(str(fit.model_pars)+'\n')
    #    f.write(str(fit.par_dims)+'\n')
    #    f.close()
    #    np.save('logistic_samples_'+nm+'_'+str(i)+'.npy', fit.extract(permuted=False))
    #tf = time.process_time()
    #np.save(nm+'_mcmc_time.npy', tf-t0)

N = 10
adam_num_iters = 10000
print_every=100
n_init=1000
n_samples = 1000
n_logfg_samples = 1000000
adam_learning_rate= lambda itr : 1./np.sqrt(1+itr)

############################## synthetic ######################################

if sys.argv[1] == 'synth':
  lmb = lambda itr : 10.0/(1.+itr)
  lmb1 = lambda itr : 1.
  
  print('Synth UBVI')
  ubvi_synth = ubvi(logf_synth, N, d_synth, n_samples, n_logfg_samples, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)
  print('Synth BBVI')
  bbvi_synth = bbvi(logp_synth, N, d_synth, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init)
  print('Synth ADVI')
  advi_synth = bbvi(logp_synth, 1, d_synth, n_samples, lmb1, adam_learning_rate, adam_num_iters, print_every, n_init)
  
  f = open('results/logistic_synth_results_'+sys.argv[2]+'.pk', 'wb')
  pk.dump([ubvi_synth, bbvi_synth, advi_synth], f)
  f.close()

################################## DS1 ########################################
if sys.argv[1] == 'ds1':
  lmb = lambda itr : 10./(1.+itr)
  lmb1 = lambda itr : 1.
  
  print('DS1 UBVI')
  ubvi_ds1 = ubvi(logf_ds1, N, d_ds1, n_samples, n_logfg_samples, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)
  print('DS1 BBVI')
  bbvi_ds1 = bbvi(logp_ds1, N, d_ds1, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init)
  print('DS1 ADVI')
  advi_ds1 = bbvi(logp_ds1, 1, d_ds1, n_samples, lmb1, adam_learning_rate, adam_num_iters, print_every, n_init)
  
  f = open('results/logistic_ds1_results_'+sys.argv[2]+'.pk', 'wb')
  pk.dump([ubvi_ds1, bbvi_ds1, advi_ds1], f)
  f.close()



################################# Phishing ####################################

if sys.argv[1] == 'phish':
  lmb = lambda itr : 10./(1.+itr)
  lmb1 = lambda itr : 1.
  
  print('PHISH UBVI')
  ubvi_phish = ubvi(logf_phish, N, d_phish, n_samples, n_logfg_samples, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)
  print('PHISH BBVI')
  bbvi_phish = bbvi(logp_phish, N, d_phish, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init)
  print('PHISH ADVI')
  advi_phish = bbvi(logp_phish, 1, d_phish, n_samples, lmb1, adam_learning_rate, adam_num_iters, print_every, n_init)
  
  
  f = open('results/logistic_phish_results_'+sys.argv[2]+'.pk', 'wb')
  pk.dump([ubvi_phish, bbvi_phish, advi_phish], f)
  f.close()


#if data too big for memory chunk it via this code
#def log_logist(theta, Z, n_subsample, nu, mu, Sig):
#    N = Z.shape[0]
#    M, d = theta.shape
#
#    #if we are out of observations in this epoch or Z changed
#    if log_logist.itr >= Z.shape[0] or log_logist.idcs.shape[0] != Z.shape[0]:
#      log_logist.itr = 0
#      log_logist.idcs = np.arange(Z.shape[0])
#      np.random.shuffle(log_logist.idcs)
#    N_taken = min(log_logist.itr+n_subsample, Z.shape[0]) - log_logist.itr
#    idcs = log_logist.idcs[log_logist.itr:log_logist.itr+N_taken]
#
#    NM_max = 100000000
#    if idcs.shape[0]*M < NM_max:
#      dots = -np.dot(Z[idcs, :], theta.T)
#      log_lik = -Z.shape[0]/N_taken*np.sum(np.maximum(dots, 0) + np.log1p(np.exp(-np.abs(dots))), axis=0)
#    else:
#      N_so_far = 0
#      N_each = int(NM_max/M)
#      log_lik = np.zeros(M)
#      while N_so_far < idcs.shape[0]:
#        N_taken_internal = min(N_so_far+N_each, idcs.shape[0]) - N_so_far
#        dots = -np.dot(Z[idcs[N_so_far:N_so_far+N_taken_internal],:], theta.T)
#        log_lik += -Z.shape[0]/N_taken*np.sum(np.maximum(dots, 0) + np.log1p(np.exp(-np.abs(dots))), axis=0)
#        N_so_far += N_taken_internal
#        print(str(N_so_far) + ' / ' + str(idcs.shape[0]))
#    log_logist.itr += N_taken
#
#    #log_pri = stats.multivariate_normal.logpdf(theta, np.zeros(d), np.eye(d))
#    log_pri = log_multivariate_t(theta[:, :-1], mu, Sig, nu) + log_cauchy(theta[:, -1])
#    #log_lik = - np.sum(np.log1p(np.exp(-Z.dot(theta.T))), axis=0)
#    return log_pri + log_lik
#log_logist.itr = np.inf
#log_logist.idcs = None
#
#def log_f(theta, Z, n_subsample, nu, mu, Sig):
#    return 0.5*log_logist(theta, Z, n_subsample, nu, mu, Sig)


