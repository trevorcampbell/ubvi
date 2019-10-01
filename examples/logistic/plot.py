import numpy as np
from bokeh.models import FuncTickFormatter
import bokeh.palettes as bkpl
from bokeh.models import Span
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import pickle as pk
import sys
import os

logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079'];
tens = Math.floor(Math.abs(Math.log10(tick))/10.);
ones = Math.floor(Math.abs(Math.log10(tick))) - tens*10;
ret = '';
if (Math.log10(tick) < 0){
  ret = ret + '10\u207B';
} else {
  ret = ret+'10';
}
if (tens == 0){
  ret = ret + trns[ones];
} else {
  ret = ret + trns[tens] + trns[ones];
}
return ret;
""")



np.set_printoptions(linewidth=1000)

pal = bkpl.colorblind['Colorblind'][8]
pl = [pal[1], pal[0], pal[3]]
pl.extend(pal[4:8])
pal = pl



  #max_mem = 100000000
  #X_chunk_sz = int(X.shape[0]/2)
  ##X_chunk_sz = int(max_mem/(Y.shape[0]*Y.shape[1]))
  ##print(X_chunk_sz)
  #itr = 0
  #while itr < X.shape[0]:
  #  sys.stdout.write('On idx ' + str(itr) + ' / ' + str(X.shape[0])+'             \r')
  #  sys.stdout.flush()
  #  dists += np.sqrt(((X[itr:itr+X_chunk_sz, np.newaxis, :] - Y)**2).sum(axis=2)).sum()
  #  itr += X_chunk_sz
  #dists /= X.shape[0]*Y.shape[0]
  #sys.stdout.write('\n')


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


def preprocess_plot(fig, axis_font_size, y_log_scale, x_log_scale):
  fig.xaxis.axis_label_text_font_size= axis_font_size
  fig.xaxis.major_label_text_font_size= axis_font_size
  fig.yaxis.axis_label_text_font_size= axis_font_size
  fig.yaxis.major_label_text_font_size= axis_font_size
  if y_log_scale:
    fig.yaxis.formatter = logFmtr
  if x_log_scale:
    fig.xaxis.formatter = logFmtr

    #fig.xaxis.formatter = logFmtr
  #fig.toolbar.logo = None
  #fig.toolbar_location = None

def postprocess_plot(fig, legend_font_size, orientation='vertical', location='top_right', glyph_width=80, glyph_height=40):
  fig.legend.label_text_font_size= legend_font_size
  fig.legend.orientation=orientation
  fig.legend.location=location
  fig.legend.glyph_width=glyph_width
  fig.legend.glyph_height=glyph_height
  fig.legend.spacing=5
  fig.xgrid.grid_line_color=None
  fig.ygrid.grid_line_color=None


#['beta0', 'z', 'tau', 'lambda', 'caux', 'c', 'beta', 'f', 'lambda_tilde']
#[[], [7128], [], [7128], [], [], [7128], [72], [7128]]
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
    #hack to get results from python2 code on supercloud
    if int(fn.split('_')[3][:-3]) > 39:
      ubvi, bbvi, advi = pk.load(f, encoding='latin1')
    else:
      ubvi, bbvi, advi = pk.load(f)
    f.close()
    n_comp_max = max(n_comp_max, ubvi[-1].shape[0], bbvi[-1].shape[0])
  #allocate memory for results
  energy_ubvi = np.zeros((len(fnames), n_comp_max))
  energy_bbvi = np.zeros((len(fnames), n_comp_max))
  energy_advi = np.zeros(len(fnames))
  cput_ubvi = np.zeros((len(fnames), n_comp_max))
  cput_bbvi = np.zeros((len(fnames), n_comp_max))
  cput_advi = np.zeros(len(fnames))
  
  for n, fn in enumerate(fnames):
    #compute energy distance as a function of cput
    #u_mu, u_Sig, u_lmb, U_LMB, Z, u_cput = ubvi
    #a_mu, a_lSig, a_w, A_W, a_cput = ubvi
    print(fn)
    f = open(os.path.join('results',fn), 'rb')
    #hack to get results from python2 code on supercloud
    if int(fn.split('_')[3][:-3]) > 39:
      ubvi, bbvi, advi = pk.load(f, encoding='latin1')
    else:
      ubvi, bbvi, advi = pk.load(f)
    f.close()

    di1, di2 = np.diag_indices(ubvi[0].shape[1]) 
    #compute energy distances
    cput_ubvi[n, :] = np.cumsum(ubvi[5])
    for i in range(ubvi[-1].shape[0]):
      print('computing energy dist: ' + nm + ' UBVI ' + str(n+1) +'/'+str(len(fnames))+ ' results files, iter ' + str(i+1) + '/' + str(ubvi[-1].shape[0]))
      #for ubvi, need to create the mixture model from the sqrt components
      u_mu = np.zeros((ubvi[0].shape[0]**2, ubvi[0].shape[1]))
      u_Sig = np.zeros((ubvi[0].shape[0]**2, ubvi[0].shape[1], ubvi[0].shape[1]))
      u_w = np.zeros(ubvi[0].shape[0]**2)
      comp_itr = 0
      for j in range(ubvi[0].shape[0]):
        for k in range(ubvi[0].shape[0]):
          u_w[comp_itr]  = ubvi[3][i, j]*ubvi[4][j, k]*ubvi[3][i,k]
          Sigp = 2./(1./ubvi[1][j,:] + 1./ubvi[1][k,:])
          di = ([comp_itr]*ubvi[0].shape[1], di1, di2)
          u_Sig[di] = Sigp
          u_mu[comp_itr, :] = 0.5*Sigp*(ubvi[0][j,:]/ubvi[1][j,:] + ubvi[0][k,:]/ubvi[1][k,:])
          comp_itr += 1
      energy_ubvi[n, i] = mixture_energy_dist(thetas, u_mu, u_Sig, u_w, n_energy_dist_samples)

    #for any remaining components, just use fixed approx (since terminated early)
    cput_ubvi[n, ubvi[-1].shape[0]:] = cput_ubvi[n, ubvi[-1].shape[0]-1] 
    for i in range(ubvi[-1].shape[0], n_comp_max):
      energy_ubvi[n, i] = energy_ubvi[n, i-1]

    cput_bbvi[n, :bbvi[-1].shape[0]] = np.cumsum(bbvi[4])
    for i in range(bbvi[-1].shape[0]):
      print('computing energy dist: ' + nm + ' BVI ' + str(n+1) +'/'+str(len(fnames))+ ' results files, iter ' + str(i+1) + '/' + str(bbvi[-1].shape[0]))
      b_Sig = np.zeros((bbvi[0].shape[0], bbvi[0].shape[1], bbvi[0].shape[1]))
      for j in range(bbvi[0].shape[0]):
        di = ([j]*ubvi[0].shape[1], di1, di2)
        b_Sig[di] = bbvi[1][j, :]
      energy_bbvi[n, i] = mixture_energy_dist(thetas, bbvi[0], b_Sig, bbvi[3][i,:], n_energy_dist_samples)

    #for any remaining components, just use fixed approx (since terminated early)
    cput_bbvi[n, bbvi[-1].shape[0]:] = cput_bbvi[n, bbvi[-1].shape[0]-1] 
    for i in range(bbvi[-1].shape[0], n_comp_max):
      energy_bbvi[n, i] = energy_bbvi[n, i-1]
 
    print('computing energy dist: ' + nm + ' ADVI ' + str(n+1) +'/'+str(len(fnames))+ ' results files')
    a_Sig = np.zeros((1, advi[0].shape[1], advi[0].shape[1])) 
    di = ([0]*ubvi[0].shape[1], di1, di2)
    a_Sig[di] = advi[1][0,:]
    energy_advi[n] = mixture_energy_dist(thetas, advi[0], a_Sig, advi[2], n_energy_dist_samples)
    cput_advi[n] = advi[4]

    #the computers on supercloud are much slower than my other ones; 
    #across the board the factor of time is nearly 2
    if int(fn.split('_')[3][:-3]) > 39:
      cput_advi/=2
      cput_bbvi/=2
      cput_ubvi/=2


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


