import numpy as np
import bokeh.palettes as bkpl
from bokeh.models import Span
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import pickle as pk

np.set_printoptions(linewidth=1000)

pal = bkpl.colorblind['Colorblind'][8]
pl = [pal[1], pal[0], pal[3]]
pl.extend(pal[4:8])
pal = pl


def preprocess_plot(fig, axis_font_size, log_scale):
  fig.xaxis.axis_label_text_font_size= axis_font_size
  fig.xaxis.major_label_text_font_size= axis_font_size
  fig.yaxis.axis_label_text_font_size= axis_font_size
  fig.yaxis.major_label_text_font_size= axis_font_size
  if log_scale:
    fig.yaxis.formatter = logFmtr
    fig.xaxis.formatter = logFmtr
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
beta0s = np.empty(0)
taus = np.empty(0)
betas = np.empty(0)
lmbs = np.empty(0)
logprbs = np.empty(0)
for i in range(2):
  samps = np.load('leukemia_samples_'+str(i)+'.npy')


  idx = 0
  beta0 = samps[:, 0, idx]
  idx += 1
  z = samps[:, 0, idx:idx+7128]
  idx += 7128
  tau = samps[:, 0, idx]
  idx += 1
  lmb = samps[:, 0, idx:idx+7128]
  idx += 7128
  caux = samps[:, 0, idx]
  idx += 1
  c = samps[:, 0, idx]
  idx += 1
  beta = samps[:, 0, idx:idx+7128]
  idx += 7128
  f = samps[:, 0, idx:idx+72]
  idx += 72
  lmb_tilde = samps[:, 0, idx:idx+7128]
  idx += 7128
  lp = samps[:, 0, idx]

  beta0s = np.hstack((beta0s, beta0))
  taus = np.hstack((taus, np.log(tau)))
  if lmbs.size == 0:
    lmbs = lmb
    betas = beta
  else:
    lmbs = np.vstack((lmbs, np.log(lmb)))
    betas = np.vstack((betas, beta))
  logprbs = np.hstack((logprbs, lp))

idcs = np.argsort(np.fabs(betas).mean(axis=0))[::-1]
betas = betas[:, idcs[0]]
lmbs = lmbs[:, idcs[0]]

#remove bottom/top percentile outliers
#beta0s = beta0s[np.logical_and(beta0s > np.percentile(beta0s, 5), beta0s < np.percentile(beta0s, 95))]
#betas = betas[np.logical_and(betas > np.percentile(betas, 5), betas < np.percentile(betas, 95))]
lmbs = lmbs[lmbs < np.percentile(lmbs, 81)]
#taus = taus[np.logical_and(taus > np.percentile(taus, 5), taus < np.percentile(taus, 95))]


figs = []
for data in [taus,beta0s,betas,lmbs,logprbs]:
  hist,edges = np.histogram(data, density=True, bins=50)
  fig = bkp.figure()
  preprocess_plot(fig, '24pt', False)
  fig.quad(top=hist,bottom=0,left=edges[:-1], right=edges[1:])
  postprocess_plot(fig, '24pt')
  figs.append(fig)

bkp.show(bkl.gridplot([figs]))
 
