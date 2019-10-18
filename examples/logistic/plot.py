import numpy as np
import pickle as pk
import bokeh.plotting as bkp
import bokeh.layouts as bkl
from scipy.stats import cauchy as sp_cauchy

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from common import mixture_logpdf, preprocess_plot, postprocess_plot, pal, logFmtr, kl_estimate, mixture_sample

def expected_dist(X, Y):
  dists = 0.
  for i in range(X.shape[0]):
    sys.stdout.write(str(i+1)+'/'+str(X.shape[0])+'       \r')
    sys.stdout.flush()
    dists += np.sqrt(((X[i,:] - Y)**2).sum(axis=1)).sum()
  dists /= X.shape[0]*Y.shape[0]
  sys.stdout.write('\n')
  sys.stdout.flush()
  return dists

def mixture_energy_dist(true_samples, g_mu, g_Sig, g_w, n_samples):
  #take samples from the mixture
  g_samples = np.zeros((n_samples, true_samples.shape[1]))
  #compute # samples in each mixture component
  n_samples_k = np.random.multinomial(n_samples, g_w/g_w.sum()) #normalize g_w just in case numerical precision loss earlier
  c_samples = np.cumsum(n_samples_k)
  #sample the mixture components
  for j in range(g_mu.shape[0]):
    if n_samples_k[j] > 0:
      g_samples[(0 if j == 0 else c_samples[j-1]):c_samples[j], :] = np.random.multivariate_normal(g_mu[j,:], g_Sig[j,:,:], n_samples_k[j])

  #f = bkp.figure()
  #f.circle(true_samples[:, 0], true_samples[:, 1], color='black')
  #f.circle(g_samples[:, 0], g_samples[:, 1], color='blue')
  #bkp.show(f)

  print('computing EYY')
  EYY = expected_dist(true_samples, true_samples)
  print('computing EXX')
  EXX = expected_dist(g_samples, g_samples)
  print('computing EXY')
  EXY = expected_dist(g_samples, true_samples)
  assert 2*EXY-EXX-EYY > 0
  return 2*EXY - EXX - EYY

#iter, chain, params
nms = ['synth', 'ds1', 'phish']
ds = [2, 10, 10]

figs = []
n_energy_dist_samples = 1000
n_theta_subsample = 10000
n_energy_dist_samples = 100
n_theta_subsample = 100

fig = bkp.figure(width=1000, height=1000, y_axis_type='log', x_axis_label='# Components', y_axis_label='Energy Dist.')
preprocess_plot(fig, '42pt', True, False)

fig2 = bkp.figure(width=1000, height=1000, y_axis_type='log', x_axis_type='log', x_axis_label='CPU Time (s)', y_axis_label='Energy Dist.')
preprocess_plot(fig2, '42pt', True, True)

figs = [[fig, fig2]]

for nm, d in zip(nms, ds):
  samps = np.load('results/logistic_samples_'+nm+'.npy')
  theta0s = samps[:, :, 0]
  theta0s=theta0s.reshape(theta0s.shape[0]*theta0s.shape[1])
  thetas = samps[:, :, 1:d+1]
  thetas=thetas.reshape((thetas.shape[0]*thetas.shape[1], thetas.shape[2]))
  logprbs = samps[:,:,-1]

  ##plot diagnostic log probs for mixing
  #figs.append([])
  #for i in range(logprbs.shape[1]):
  #  fig = bkp.figure()
  #  preprocess_plot(fig, '24pt', False)
  #  fig.line(np.arange(logprbs.shape[0]), logprbs[:, i])
  #  postprocess_plot(fig, '24pt')
  #  figs[-1].append(fig)

  #print means/stds for samples 
  print(nm + ' intercept')
  print(theta0s.mean(axis=0))
  print(theta0s.std(axis=0))


  print(nm + ' parameter')
  print(thetas.mean(axis=0))
  print(thetas.std(axis=0))

  #collect prm and intercept into single matrix for energy distance computation
  thetas = np.hstack((thetas, theta0s[:, np.newaxis]))
  thetas = thetas[:n_theta_subsample,:]

  #load all results, checking to find max # components found
  fnames = [fn for fn in os.listdir('results/') if 'logistic_'+nm+'_results_' in fn]
  n_comp_max = 0
  for fn in fnames:
    print(fn)
    f = open(os.path.join('results', fn), 'rb')
    ubvi, bbvi, advi = pk.load(f)
    f.close()
    n_comp_max = max(n_comp_max, ubvi[-1]['weights'].shape[0], bbvi[-1]['weights'].shape[0])

  #allocate memory for results
  energy_ubvi = np.zeros((len(fnames), n_comp_max))
  energy_bbvi = np.zeros((len(fnames), n_comp_max))
  energy_advi = np.zeros(len(fnames))
  cput_ubvi = np.zeros((len(fnames), n_comp_max))
  cput_bbvi = np.zeros((len(fnames), n_comp_max))
  cput_advi = np.zeros(len(fnames))
  
  for n, fn in enumerate(fnames):
    #compute energy distance as a function of cput
    print(fn)
    f = open(os.path.join('results',fn), 'rb')
    ubvi, bbvi, advi = pk.load(f)
    f.close()


    #UBVI
    #compute energy distances
    for i in range(ubvi['weights'].shape[0]):
      print('computing energy dist: ' + nm + ' UBVI ' + str(n+1) +'/'+str(len(fnames))+ ' results files, iter ' + str(i+1) + '/' + str(ubvi[-1].shape[0]))
      energy_ubvi[n, i] = mixture_energy_dist(thetas, ubvi[i]['mus'], ubvi[i]['Sigs'], ubvi[i]['weights'], n_energy_dist_samples)
      cput_ubvi[n, i] = ubvi[i]['cput']

    #for any remaining components, just use fixed approx (since terminated early)
    cput_ubvi[n, ubvi[-1]['weights'].shape[0]:] = cput_ubvi[n, ubvi[-1]['weights'].shape[0]-1] 
    for i in range(ubvi[-1].shape[0], n_comp_max):
      energy_ubvi[n, i] = energy_ubvi[n, i-1]

    #BBVI
    #compute energy distances
    for i in range(bbvi['weights'].shape[0]):
      print('computing energy dist: ' + nm + ' BBVI ' + str(n+1) +'/'+str(len(fnames))+ ' results files, iter ' + str(i+1) + '/' + str(bbvi[-1].shape[0]))
      energy_bbvi[n, i] = mixture_energy_dist(thetas, bbvi[i]['mus'], bbvi[i]['Sigs'], bbvi[i]['weights'], n_energy_dist_samples)
      cput_bbvi[n, i] = bbvi[i]['cput']

    #for any remaining components, just use fixed approx (since terminated early)
    cput_bbvi[n, bbvi[-1]['weights'].shape[0]:] = cput_bbvi[n, bbvi[-1]['weights'].shape[0]-1] 
    for i in range(bbvi[-1].shape[0], n_comp_max):
      energy_bbvi[n, i] = energy_bbvi[n, i-1]

 
    print('computing energy dist: ' + nm + ' ADVI ' + str(n+1) +'/'+str(len(fnames))+ ' results files')
    energy_advi[n] = mixture_energy_dist(thetas, advi[0]['mus'], advi[0]['Sigs'], advi[0]['weights'], n_energy_dist_samples)
    cput_advi[n] = advi[0]['cput']

  cadvi50 = np.percentile(cput_advi, 50, axis=0)
  eadvi50 = np.percentile(energy_advi, 50, axis=0)

  eubvi50 = np.percentile(energy_ubvi, 50, axis=0)/eadvi50
  eubvi25 = np.percentile(energy_ubvi, 25, axis=0)/eadvi50
  eubvi75 = np.percentile(energy_ubvi, 75, axis=0)/eadvi50

  ebbvi50 = np.percentile(energy_bbvi, 50, axis=0)/eadvi50
  ebbvi25 = np.percentile(energy_bbvi, 25, axis=0)/eadvi50
  ebbvi75 = np.percentile(energy_bbvi, 75, axis=0)/eadvi50


  cubvi50 = np.percentile(cput_ubvi, 50, axis=0)/cadvi50
  cubvi25 = np.percentile(cput_ubvi, 25, axis=0)/cadvi50
  cubvi75 = np.percentile(cput_ubvi, 75, axis=0)/cadvi50

  cbbvi50 = np.percentile(cput_bbvi, 50, axis=0)/cadvi50
  cbbvi25 = np.percentile(cput_bbvi, 25, axis=0)/cadvi50
  cbbvi75 = np.percentile(cput_bbvi, 75, axis=0)/cadvi50



  fig.line(np.arange(energy_ubvi.shape[1])+1, eubvi50, color=pal[0], legend='UBVI',line_width=10)
  fig.line(np.arange(energy_ubvi.shape[1])+1, eubvi25, color=pal[0], line_dash='dashed', legend='UBVI',line_width=3)
  fig.line(np.arange(energy_ubvi.shape[1])+1, eubvi75, color=pal[0], line_dash='dashed', legend='UBVI',line_width=3)

  fig.line(np.arange(energy_bbvi.shape[1])+1, ebbvi50, color=pal[1], legend='BVI',line_width=10)
  fig.line(np.arange(energy_bbvi.shape[1])+1, ebbvi25, color=pal[1], line_dash='dashed', legend='BVI',line_width=3)
  fig.line(np.arange(energy_bbvi.shape[1])+1, ebbvi75, color=pal[1], line_dash='dashed', legend='BVI',line_width=3)

  fig.line(np.arange(energy_ubvi.shape[1])+1, np.ones(energy_ubvi.shape[1]),   color=pal[2], legend='ADVI',line_width=10)
  #3fig.line(np.arange(energy_ubvi.shape[1])+1, np.ones(energy_ubvi.shape[1])*(eadvi25), color=pal[2], line_dash='dashed', legend='ADVI',line_width=3)
  #3fig.line(np.arange(energy_ubvi.shape[1])+1, np.ones(energy_ubvi.shape[1])*(eadvi75), color=pal[2], line_dash='dashed', legend='ADVI', line_width=3)

  #fig = bkp.figure(width=1000, height=1000, y_axis_type='log', x_axis_label='CPU Time (s)', y_axis_label='Energy Dist.')
  #preprocess_plot(fig, '42pt', True)
  fig2.circle(cubvi50, eubvi50, color=pal[0], legend='UBVI', size=20)
  fig2.segment(x0=cubvi25, x1 = cubvi75, y0 = eubvi50, y1 = eubvi50, color=pal[0], legend='UBVI', line_width=4)
  fig2.segment(x0=cubvi50, x1 = cubvi50, y0 = eubvi25, y1 = eubvi75, color=pal[0], legend='UBVI', line_width=4)

  fig2.circle(cbbvi50, ebbvi50, color=pal[1], legend='BVI', size=20)
  fig2.segment(x0=cbbvi25, x1 = cbbvi75, y0 = ebbvi50, y1 = ebbvi50, color=pal[1], legend='BVI', line_width=4)
  fig2.segment(x0=cbbvi50, x1 = cbbvi50, y0 = ebbvi25, y1 = ebbvi75, color=pal[1], legend='BVI', line_width=4)

  fig2.circle(1., 1., color=pal[2], legend='ADVI', size=20)
  #fig.segment(x0=cadvi25, x1 = cadvi75, y0 = eadvi50, y1 = eadvi50, color=pal[2], legend='ADVI', line_width=4)
  #fig.segment(x0=cadvi50, x1 = cadvi50, y0 = eadvi25, y1 = eadvi75, color=pal[2], legend='ADVI', line_width=4)

postprocess_plot(fig, '36pt')
postprocess_plot(fig2, '36pt')
bkp.show(bkl.gridplot(figs))
  

##plot the contours of the first 2 components of each
#for nm, logp, d in [('synth', logp_synth, X_synth.shape[1]), ('ds1', logp_ds1, X_ds1.shape[1]), ('phish', logp_phish, X_phish.shape[1])]:
#  #get some samples
#  samps = np.load('logistic_samples_'+nm+'.npy')
#  theta0 = samps[:,:,0].reshape(samps.shape[0]*samps.shape[1])
#  theta = samps[:,:,1:d+1].reshape(samps.shape[0]*samps.shape[1], d)
#  samples = np.hstack((theta, theta0[:, np.newaxis]))
#
#  mean = samples.mean(axis=0)
#  std = samples.std(axis=0)
#  coordinates = (0, 1)
#  without_coordinates = list(range(samples.shape[1]))
#  without_coordinates.remove(coordinates[0])
#  without_coordinates.remove(coordinates[1])
#
#  xmin = samples[:,coordinates[0]].min()
#  xmax = samples[:,coordinates[0]].max()
#  ymin = samples[:,coordinates[1]].min()
#  ymax = samples[:,coordinates[1]].max()
#
#  #stretch a bit beyond min/max
#  xmin, xmax = xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05
#  ymin, ymax = ymin - (ymax-ymin)*0.05, ymax + (ymax-ymin)*0.05
# 
#  N_grid = 1000
#  x = np.linspace(xmin, xmax, N_grid)
#  y = np.linspace(ymin, ymax, N_grid)
#  xx, yy = np.meshgrid(x, y)
#  x = xx.reshape(-1,1)
#  y = yy.reshape(-1,1)
#  X = np.hstack((x,y, np.tile(mean[without_coordinates], (x.shape[0], 1))))
#  #plot the truth
#  Y = logp(X).reshape(N_grid, N_grid)
#  Y -= Y.max()
#  #Yf = np.exp(Y)/(np.exp(Y).sum())
#  Yf = Y
#
#  #Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
#  #Levels = np.array([0.001, .005, 0.015, 0.025])
#  #plt.contour(xx, yy, Y, levels=Levels, colors='black', linewidths=2) #cmap="Blues_r")
#  plt.contour(xx, yy, Yf, colors='black', linewidths=2, levels=np.linspace(Yf.min(), Yf.max(), 20)) #cmap="Blues_r")
#  plt.scatter(samples[:,coordinates[0]], samples[:,coordinates[1]])
#  plt.show()


