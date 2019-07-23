import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
import os
import sys


def kl_gaussian(mu1, Sig1, mu2, Sig2, ridge=1e-9):
  r = ridge*np.eye(mu1.shape[0])
  #print(np.linalg.eigvalsh(Sig1+r).min())
  #print(np.linalg.eigvalsh(Sig2+r).min())
  k1 = np.trace(np.linalg.solve(Sig2+r, Sig1+r))
  k2 = (mu1-mu2).T.dot( np.linalg.solve(Sig2+r, mu1-mu2   ))
  k3 = np.linalg.slogdet(Sig2+r)[1] - np.linalg.slogdet(Sig1+r)[1]
  return 0.5*( k1 + k2 + k3 - mu1.shape[0] )

ds = np.array([2, 3, 5, 7, 9, 10, 20, 30, 50, 70, 85, 100, 150, 200, 250, 300, 350, 400, 450, 500])
n_trials = 100
n_imptc = 100000
nms = ['Hellinger(p, q)', 'ChiSq(q || p)', 'KL(q || p)', 'KL(p || q)']

#h_kls = np.zeros(ds.shape[0]*n_trials)
#kl_kls = np.zeros(ds.shape[0]*n_trials)
true2dsigs = np.zeros((n_trials, 2, 2))
sigsqs = np.zeros((4, ds.shape[0]*n_trials))
logkls = np.zeros((4, ds.shape[0]*n_trials))
logimptcerrs = np.zeros((4, ds.shape[0]*n_trials))
logconds = np.zeros(ds.shape[0]*n_trials)
for i in range(ds.shape[0]):
  for j in range(n_trials):
    sys.stdout.write('Dimension: ' + str(i+1)+'/'+str(ds.shape[0])+' Trial: ' + str(j+1) +'/' + str(n_trials)+'               \r')
    sys.stdout.flush()
    d = ds[i]
    A = np.random.randn(d, d)
    A = np.dot(A,A.T)
    lmb, V = np.linalg.eigh(A)
    logcond = np.log10(lmb.max()) - np.log10(lmb.min())

    if i == 0:
      true2dsigs[j, :, :] = A

    #res = minimize(fun=lambda x : 0.5*np.linalg.slogdet(A + x*np.eye(d))[1] - 0.25*np.linalg.slogdet(x*np.eye(d))[1], 
    #               jac=lambda x : 0.5*np.trace( np.linalg.inv(x*np.eye(d) + A) ) -0.25*d/x,
    #               x0=1., method='BFGS', options={'disp':True})

    #minimize reverse KL 
    kl_sigsq = float(d)/( (1./np.linalg.eigvalsh(A)).sum())

    #minimize forward KL
    fkl_sigsq = np.linalg.eigvalsh(A).sum()/float(d)
    
    #minimize hellinger
    res = minimize(fun=lambda x : 0.5* np.log(lmb+x).sum() - 0.25*d*np.log(x),
                   jac=lambda x : 0.5* (1.0/(lmb+x)).sum() -0.25*d/x,
                   x0=1., method='Nelder-Mead')#, options={'disp':True})
    hellinger_sigsq = res.x

    #minimize chisq
    res = minimize(fun=lambda x : 0.5*d*np.log(x) - 0.5*(np.log(2./lmb - 1./x)).sum() if 2*x > lmb.max() else np.inf,
                   jac=lambda x : 0.5*d/x - 0.5*(1./(2./lmb - 1./x)).sum()/x**2,
                   x0 = lmb.max(), method='Nelder-Mead')#, options={'disp':True})
    chisq_sigsq = res.x

    #save sigsqs and compute forward KLs
    ss = np.array([hellinger_sigsq, chisq_sigsq, kl_sigsq, fkl_sigsq])
    sigsqs[:, i*n_trials+j] = ss
    logkls[:, i*n_trials+j] = np.array([np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), ss[i]*np.eye(d))) for i in range(4)])
 
    #sigsqs[0, i*n_trials+j] = hellinger_sigsq
    #sigsqs[1, i*n_trials+j] = chisq_sigsq
    #sigsqs[2, i*n_trials+j] = kl_sigsq
    #sigsqs[3, i*n_trials+j] = fkl_sigsq
    #
    ##compute forward KLs
    #logkls[0, i*n_trials+j] = np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), hellinger_sigsq*np.eye(d)))
    #logkls[1, i*n_trials+j] = np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), chisq_sigsq*np.eye(d)))
    #logkls[2, i*n_trials+j] = np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), kl_sigsq*np.eye(d)))
    #logkls[3, i*n_trials+j] = np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), chisq_sigsq*np.eye(d)))
    logconds[i*n_trials+j] = logcond

    #compute importance sampling errors
    for k in range(4):
      samps = np.random.multivariate_normal(np.zeros(d), ss[k]*np.eye(d), n_imptc)
      sTV = samps.dot(V)
      logratios = -0.5*np.log(lmb).sum() + 0.5*d*np.log(ss[k]) - 0.5*((samps.dot(V)**2)/lmb).sum(axis=1) + 0.5*(samps**2).sum(axis=1)/ss[k]
      maxlr = logratios.max()
      logratios -= maxlr
      est = np.exp(maxlr)*((np.exp(logratios)*(samps.T)).dot(samps))/n_imptc
      logimptcerrs[k, i*n_trials+j] = np.log10(np.linalg.norm(est - A))

    ##h_kls[i, j] = kl_gaussian(np.zeros(d), A, np.zeros(d), hellinger_var*np.eye(d))
    ##kl_kls[i, j] = kl_gaussian(np.zeros(d), A, np.zeros(d), kl_var*np.eye(d))
    #kls.append(['Hellinger', ds[i], j, np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), hellinger_sigsq*np.eye(d))), logcond])
    #kls.append(['KL', ds[i], j, np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), kl_sigsq*np.eye(d))), logcond])
    #kls.append(['ChiSq', ds[i], j, np.log10(kl_gaussian(np.zeros(d), A, np.zeros(d), chisq_sigsq*np.eye(d))), logcond])
    ##print('Hellinger KL(p || q)')
    ##print(kl_gaussian(np.zeros(d), A, np.zeros(d), hellinger_var*np.eye(d)))
    ##
    ##print('KL KL(p || q)')
    ##print(kl_gaussian(np.zeros(d), A, np.zeros(d), kl_var*np.eye(d)))

sys.stdout.write('\n')
sys.stdout.flush()

np.savez('results.npz', logkls=logkls, logconds=logconds, ds=ds, n_trials=n_trials, sigsqs=sigsqs, true2dsigs=true2dsigs, logimptcerrs=logimptcerrs, n_imptc=n_imptc, nms=nms)

#data = pd.DataFrame(columns=['Divergence', 'Dimension', 'Trial', 'Log KL Divergence', 'Log Condition Number'], data=kls)
#data.to_pickle('results.pkl')

   
