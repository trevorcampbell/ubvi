import autograd.numpy as np
from autograd.scipy.special import gammaln
import os
import time
import pickle as pk
import pystan

from ubvi.components import Gaussian
from ubvi.optimization import adam as ubvi_adam
from ubvi.inference import UBVI, BBVI
from ubvi.autograd import logsumexp

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
    data.close()
    if subset_sz is None:
        subset_sz = Z.shape[0]
    return Z[:subset_sz, :], X[:subset_sz, :-1], Y[:subset_sz]

np.random.seed(1)

print('loading datasets')
subset_sz = 20
Z_synth, X_synth, Y_synth = load_data('../data/synth.npz', subset_sz)
Y_synth[Y_synth == -1] = 0
Z_ds1, X_ds1, Y_ds1 = load_data('../data/ds1.npz', subset_sz)
Y_ds1[Y_ds1 == -1] = 0
Z_phish, X_phish, Y_phish = load_data('../data/phishing.npz', subset_sz)
Y_phish[Y_phish == -1] = 0

#ensure we use the same Sig each time
nu = 2.
Sig_synth = np.array([[1., -0.9], [-0.9, 1.]])
mu_synth = np.zeros(2)
Sig_ds1 = np.random.randn(X_ds1.shape[1], X_ds1.shape[1])
Sig_ds1 = Sig_ds1.T.dot(Sig_ds1)
mu_ds1 = np.zeros(Sig_ds1.shape[0])
Sig_phish = np.random.randn(X_phish.shape[1], X_phish.shape[1])
Sig_phish = Sig_phish.T.dot(Sig_phish)
mu_phish = np.zeros(Sig_phish.shape[0])

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
###############################################################################



logp_synth = lambda theta : log_logist(theta, Z_synth, nu, mu_synth, Sig_synth)
logp_ds1 = lambda theta: log_logist(theta, Z_ds1, nu, mu_ds1, Sig_ds1)
logp_phish = lambda theta: log_logist(theta, Z_phish, nu, mu_phish, Sig_phish)


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
   

############################## (U)BVI / ADVI Params ######################################


N = 10
diag = True
n_samples = 2000
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 10./np.sqrt(itr+1)
adam_num_iters = 10000
n_init = 10000
init_inflation = 16


adam = lambda x0, obj, grd : ubvi_adam(x0, obj, grd, adam_learning_rate, adam_num_iters, callback = gauss.print_perf)

############################## synthetic ######################################
if sys.argv[1] == 'synth':
    gauss = Gaussian(d_synth, diag)
    lmb_bbvi = lambda itr : 10.0/(1.+itr)
    lmb_advi = lambda itr : 1.
        
    print('Synth UBVI')
    ubvi = UBVI(logp_synth, gauss, adam, n_init = n_init, n_samples = n_samples, n_logfg_samples = n_logfg_samples, init_inflation = init_inflation)
    ubvi_synth = ubvi.build(N)
    
    print('Synth BVI')
    bbvi = BBVI(logp_synth, gauss, adam, lmb = lmb_bbvi, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    bbvi_synth = bbvi.build(N)
    
    print('Synth ADVI')
    advi = BBVI(logp_synth, gauss, adam, lmb = lmb_advi, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    advi_synth = advi.build(1)
    
    f = open('results/logistic_synth_results_'+sys.argv[2]+'.pk', 'wb')
    pk.dump([ubvi_synth, bbvi_synth, advi_synth], f)
    f.close()

################################## DS1 ########################################

if sys.argv[1] == 'ds1':
    gauss = Gaussian(d_ds1, diag)
    lmb_bbvi = lambda itr : 10./(1.+itr)
    lmb_advi = lambda itr : 1.
    
    print('DS1 UBVI')
    ubvi = UBVI(logp_ds1, gauss, adam, n_init = n_init, n_samples = n_samples, n_logfg_samples = n_logfg_samples, init_inflation = init_inflation)
    ubvi_ds1 = ubvi.build(N)
    
    print('DS1 BVI')
    bbvi = BBVI(logp_ds1, gauss, adam, lmb = lmb_bbvi, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    bbvi_ds1 = bbvi.build(N)
    
    print('DS1 ADVI')
    advi = BBVI(logp_ds1, gauss, adam, lmb = lmb_advi, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    advi_ds1 = advi.build(1)
       
    f = open('results/logistic_ds1_results_'+sys.argv[2]+'.pk', 'wb')
    pk.dump([ubvi_ds1, bbvi_ds1, advi_ds1], f)
    f.close()
################################# Phishing ####################################

if sys.argv[1] == 'phish':
    gauss = Gaussian(d_phish, diag)
    lmb_bbvi = lambda itr : 10./(1.+itr)
    lmb_advi = lambda itr : 1.
    
    print('DS1 UBVI')
    ubvi = UBVI(logp_phish, gauss, adam, n_init = n_init, n_samples = n_samples, n_logfg_samples = n_logfg_samples, init_inflation = init_inflation)
    ubvi_phish = ubvi.build(N)
    
    print('DS1 BVI')
    bbvi = BBVI(logp_phish, gauss, adam, lmb = lmb_bbvi, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    bbvi_phish = bbvi.build(N)
    
    print('DS1 ADVI')
    advi = BBVI(logp_phish, gauss, adam, lmb = lmb_advi, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    advi_phish = advi.build(1)
     
    f = open('results/logistic_phish_results_'+sys.argv[2]+'.pk', 'wb')
    pk.dump([ubvi_phish, bbvi_phish, advi_phish], f)
    f.close()
