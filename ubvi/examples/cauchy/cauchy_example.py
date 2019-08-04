import autograd.numpy as np
import pickle as pk
import os

from UBVI import ubvi
from BBVI import bbvi



def cauchy(x):
    return (- np.log(1 + x**2) - np.log(np.pi)).flatten()

def logf(x):
   return 0.5*cauchy(x)



N_runs = 10
N = 30
d = 1
Diag=False
n_samples = 100
n_logfg_samples = 10000
adam_num_iters = 5000
adam_learning_rate = lambda itr : 0.1/np.sqrt(1.+itr)
n_init=1000
lmb_good = lambda itr : 1./(1+itr)
lmb_bad = lambda itr : 70./(1+itr)



if not os.path.exists('results/'):
    os.mkdir('results')



for i in range(N_runs):
    print('RUN ' + str(i+1)+'/'+str(N_runs))
    
    ubvi_ = ubvi(logf, N, d, Diag, n_samples, n_logfg_samples, adam_learning_rate, adam_num_iters, n_init=n_init)
    
    bbvi_ = bbvi(cauchy, N, d, Diag, n_samples, lmb_good, adam_learning_rate, adam_num_iters, n_init)
    
    bbvi2_ = bbvi(cauchy, N, d, Diag, n_samples, lmb_bad, adam_learning_rate, adam_num_iters, n_init)
    
    if os.path.exists('results/cauchy.pk'):
        f = open('results/cauchy.pk', 'rb')
        res = pk.load(f)
        f.close()
        res[0].append(ubvi_)
        res[1].append(bbvi_)
        res[2].append(bbvi2_)
        f = open('results/cauchy.pk', 'wb')
        pk.dump(res, f)
        f.close()
    else:
        f = open('results/cauchy.pk', 'wb')
        pk.dump(([ubvi_], [bbvi_], [bbvi2_]), f)
        f.close()



