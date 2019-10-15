import numpy as np
from ubvi.autograd import logsumexp
import bokeh.palettes as bkpl

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

def mixture_logpdf(X, mu, Sig, wt):
    if len(Sig.shape) < 3:
        Sig = Sig[:, :, np.newaxis]
    Siginv = np.linalg.inv(Sig)
    inner_prods = (X[:,np.newaxis,:]-mu)[:,:,:,np.newaxis] * Siginv *  (X[:,np.newaxis,:]-mu)[:,:,np.newaxis,:]
    lg = -0.5*inner_prods.sum(axis=3).sum(axis=2)
    lg -= 0.5*mu.shape[1]*np.log(2*np.pi) + 0.5*np.linalg.slogdet(Sig)[1] 
    return logsumexp(lg+np.log(wt), axis=1)
    

