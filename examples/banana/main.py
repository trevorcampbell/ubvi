import autograd.numpy as np
import pickle as pk
import os

from ubvi.components import Gaussian
from ubvi.optimization import adam as ubvi_adam
from ubvi.inference import UBVI, BBVI
from ubvi.autograd import logsumexp

def logp(X):
    b = 0.1
    x = X[:,0]
    y = X[:,1]
    return -x**2/200 - (y+b*x**2-100*b)**2/2 - np.log(2*np.pi*10)

np.random.seed(1)

N_runs = 1000
N = 30
d = 2
diag = True
n_samples = 5000
n_logfg_samples = 100000
adam_learning_rate= lambda itr : 0.1/np.sqrt(itr+1)
adam_num_iters = 5000
n_init = 10000
init_inflation = 100
lmb_good = lambda itr : 1./(1+itr)
lmb_bad = lambda itr : 70./(1+itr)


gauss = Gaussian(d, diag)
adam = lambda x0, obj, grd : ubvi_adam(x0, obj, grd, adam_learning_rate, adam_num_iters, callback = gauss.print_perf)


if not os.path.exists('results/'):
  os.mkdir('results')

for i in range(N_runs):

    #UBVI
    ubvi = UBVI(logp, gauss, adam, n_init = n_init, n_samples = n_samples, n_logfg_samples = n_logfg_samples, init_inflation = init_inflation)
    banana_ubvi = ubvi.build(N)

    #BBVI Good
    bbvi_good = BBVI(logp, gauss, adam, lmb = lmb_good, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    banana_bbvi_good = bbvi_good.build(N)

    #BBVI Bad
    bbvi_bad = BBVI(logp, gauss, adam, lmb = lmb_bad, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    banana_bbvi_bad = bbvi_bad.build(N)

    if os.path.exists('results/banana.pk'):
        f = open('results/banana.pk', 'rb')
        res = pk.load(f)
        f.close()
        res[0].append(banana_ubvi)
        res[1].append(banana_bbvi_good)
        res[2].append(banana_bbvi_bad)
        f = open('results/banana.pk', 'wb')
        pk.dump(res, f)
        f.close()
    else:
        f = open('results/banana.pk', 'wb')
        pk.dump(([banana_ubvi], [banana_bbvi_good], [banana_bbvi_bad]), f)
        f.close()

