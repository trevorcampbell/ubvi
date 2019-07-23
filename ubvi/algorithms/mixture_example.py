import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats

from ubvi import ubvi, logg
from bbvi import bbvi, mvnlogpdf
import pickle as pk
import os

np.seterr(invalid='raise')

def logp(x):
   #lw = np.log(np.array([0.4, 0.3, 0.3]))
   lw = np.log(np.array([0.5, 0.5]))
   #lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 10, np.atleast_2d(10))[:,np.newaxis]))
   lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis]))
   #lf = np.hstack(( stats.multivariate_normal.logpdf(x, np.zeros(2), np.eye(2))[:,np.newaxis], stats.multivariate_normal.logpdf(x, np.ones(2), np.eye(2))[:,np.newaxis]))
   return logsumexp(lf + lw, axis=1)

def logf(x):
   return 0.5*logp(x)



N = 2
d = 1
n_samples = 500
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 0.1/np.sqrt(itr+1)
adam_num_iters = 3000
print_every=100
n_init=1000



ubvi_ = ubvi(logf, N, d, n_samples, n_logfg_samples, adam_learning_rate, adam_num_iters, print_every, n_init=n_init, init_inflation=100)

lmb = lambda itr : 10
bbvi_10 = bbvi(logp, N, d, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)

#import bokeh.plotting as bkp
#g_mu, g_Sig, g_w = bbvi_17
#X = np.linspace(-20,60,5000)
#lg = mvnlogpdf(X[:,np.newaxis], g_mu, g_Sig, np.linalg.inv(g_Sig))
#lg = logsumexp(lg+np.log(g_w), axis=1)
#fig = bkp.figure()
#fig.line(X, np.exp(0.5*lg), line_width=6.5, color='blue', legend='BBVI('+str(lmb)+')')
#fig.line(X, np.exp(0.5*logp(X)), line_width=6.5, line_color='black', legend='p(x)') 
#bkp.show(fig)


lmb = lambda itr : 1.
bbvi_1 = bbvi(logp, N, d, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)

lmb = lambda itr : 30.
bbvi_30 = bbvi(logp, N, d, n_samples, lmb, adam_learning_rate, adam_num_iters, print_every, n_init=n_init)


if not os.path.exists('results/'):
  os.mkdir('results')

f = open('results/mixture_results.pk', 'wb')
pk.dump(([bbvi_1, bbvi_10, bbvi_30], [1,10, 30], ubvi_), f)
f.close()

   
