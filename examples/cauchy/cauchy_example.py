import autograd.numpy as np
import pickle as pk
import os

from distributions import Gaussian
from optimizations import Adam
from ubvi import UBVI
from bbvi import BBVI


def cauchy(x):
    return (- np.log(1 + x**2) - np.log(np.pi)).flatten()

def logf(x):
   return 0.5*cauchy(x)


N_runs = 1
N = 3
d = 1
diag=False
n_samples = 100
n_logfg_samples = 10000
adam_num_iters = 5000
adam_learning_rate = lambda itr : 0.1/np.sqrt(1.+itr)
n_init=1000
lmb_good = lambda itr : 1./(1+itr)
lmb_bad = lambda itr : 70./(1+itr)
print_every = 100



if not os.path.exists('results/'):
    os.mkdir('results')

gauss = Gaussian(d, diag)
adam = Adam(adam_learning_rate, adam_num_iters, print_every)

for i in range(N_runs):
    print('RUN ' + str(i+1)+'/'+str(N_runs))
    
    ubvi = UBVI(logf, N, gauss, adam, n_samples, n_init, n_logfg_samples)
    cauchy_ubvi = ubvi.build()
    
    bbvi = BBVI(cauchy, N, gauss, adam, n_samples, n_init, lmb_good)
    cauchy_bbvi = bbvi.build()
    
    bbvi2 = BBVI(cauchy, N, gauss, adam, n_samples, n_init, lmb_bad)
    cauchy_bbvi2 = bbvi2.build()
    
    if os.path.exists('results/cauchy.pk'):
        f = open('results/cauchy.pk', 'rb')
        res = pk.load(f)
        f.close()
        res[0].append(cauchy_ubvi)
        res[1].append(cauchy_bbvi)
        res[2].append(cauchy_bbvi2)
        f = open('results/cauchy.pk', 'wb')
        pk.dump(res, f)
        f.close()
    else:
        f = open('results/cauchy.pk', 'wb')
        pk.dump((cauchy_ubvi, cauchy_bbvi, cauchy_bbvi2), f)
        f.close()