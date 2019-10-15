import numpy as np
from ubvi.autograd import logsumexp
import bokeh.palettes as bkpl
from bokeh.models import FuncTickFormatter

pal = bkpl.colorblind['Colorblind'][8]
pl = [pal[1], pal[0], pal[3]]
pl.extend(pal[4:8])
pal = pl

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
'\u2079']
if (Math.log10(tick) < 0){
  return '10\u207B'+trns[Math.round(Math.abs(Math.log10(tick)))];
} else {
  return '10'+trns[Math.round(Math.abs(Math.log10(tick)))];
}
""")




def preprocess_plot(fig, axis_font_size, log_scale_x = False, log_scale_y = False):
    fig.xaxis.axis_label_text_font_size= axis_font_size
    fig.xaxis.major_label_text_font_size= axis_font_size
    fig.yaxis.axis_label_text_font_size= axis_font_size
    fig.yaxis.major_label_text_font_size= axis_font_size
    if log_scale_x:
      fig.xaxis.formatter = logFmtr
    if log_scale_y:
      fig.yaxis.formatter = logFmtr
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

def mixture_logpdf(X, mu, Sig, wt):
    if len(Sig.shape) < 3:
        Sig = Sig[:, :, np.newaxis]
    Siginv = np.linalg.inv(Sig)
    inner_prods = (X[:,np.newaxis,:]-mu)[:,:,:,np.newaxis] * Siginv *  (X[:,np.newaxis,:]-mu)[:,:,np.newaxis,:]
    lg = -0.5*inner_prods.sum(axis=3).sum(axis=2)
    lg -= 0.5*mu.shape[1]*np.log(2*np.pi) + 0.5*np.linalg.slogdet(Sig)[1] 
    return logsumexp(lg[:, wt>0]+np.log(wt[wt>0]), axis=1)

def mixture_sample(mu, Sig, wt, n_samples):
    if len(Sig.shape) < 3:
        Sig = Sig[:, :, np.newaxis]
    cts = np.random.multinomial(n_samples, wt)
    X = np.zeros((n_samples, mu.shape[1]))
    c = 0
    for k in range(wt.shape[0]):
        X[c:c+cts[k], :] = np.random.multivariate_normal(mu[k, :], Sig[k, :, :], cts[k])
        c += cts[k]
    return X
    
def kl_estimate(mus, Sigs, wts, logp, p_samps, direction='forward'):
    lp = logp(p_samps)
    if direction == 'forward':
        lq = mixture_logpdf(p_samps, mus, Sigs, wts)
        kl = (lp - lq).mean()
    else:
        lq = mixture_logpdf(p_samps, mus, Sigs, wts)
        ratio_max = (lq - lp).max()
        kl = np.exp(ratio_max)*((lq - lp)*np.exp( (lq-lp) - ratio_max)).mean()
    return kl
