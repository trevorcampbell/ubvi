import numpy as np
import bokeh.palettes as bkpl
from bokeh.models import Span
import bokeh.plotting as bkp
import bokeh.layouts as bkl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import pickle as pk
import autograd.scipy.stats as stats
from autograd.scipy.misc import logsumexp
from ubvi import logg
from bbvi import mvnlogpdf

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



def logp(x):
   #lw = np.log(np.array([0.4, 0.3, 0.3]))
   lw = np.log(np.array([0.5, 0.5]))
   #lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 10, np.atleast_2d(10))[:,np.newaxis]))
   lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis]))
   #lf = np.hstack(( stats.multivariate_normal.logpdf(x, np.zeros(2), np.eye(2))[:,np.newaxis], stats.multivariate_normal.logpdf(x, np.ones(2), np.eye(2))[:,np.newaxis]))
   return logsumexp(lf + lw, axis=1)

f = open('results/mixture_results.pk', 'rb')
res = pk.load(f)
f.close()

bbvis = res[0]
bbvi_lmbs = res[1]
ubvi = res[2]

X = np.linspace(-20,60,5000)
fig = bkp.figure()
preprocess_plot(fig, '24pt', False)

for i in range(len(bbvis)):
  g_mu, g_Sig, g_w, G_w, cput = bbvis[i]
  lmb = bbvi_lmbs[i]
  lg = mvnlogpdf(X[:,np.newaxis], g_mu, g_Sig, np.linalg.inv(g_Sig))
  lg = logsumexp(lg+np.log(g_w), axis=1)
  fig.line(X, np.exp(0.5*lg), line_width=6.5, color=pal[i+1], legend='BVI'+str(lmb))
fig.line(X, np.exp(0.5*logp(X)), line_width=6.5, line_color='black', legend='p(x)') 
g_mu, g_Sig, g_lmb, G_lmb, Z, cput = ubvi
fig.line(X, np.exp(logg(X, g_mu, g_Sig, np.linalg.inv(g_Sig), g_lmb)), line_width=6.5, line_dash=[20, 20], color=pal[0], legend='UBVI')

postprocess_plot(fig, '24pt')

bkp.show(fig)
