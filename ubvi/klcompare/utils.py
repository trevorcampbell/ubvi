from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import numpy as np

def plot_gaussian(plot, mu, Sig, color, linewidth, linealpha, line_dash, name):
  t = np.linspace(0., 2*np.pi, 100)
  t = np.array([np.cos(t), np.sin(t)])
  t = np.linalg.cholesky(Sig).dot(t) + mu[:, np.newaxis]
  plot.line(t[0, :], t[1, :], color=color, line_width=linewidth, alpha=linealpha, line_dash=line_dash, legend=name)



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

def postprocess_plot(fig, legend_font_size, orientation='vertical', location='top_right', glyph_width=80, glyph_height=20):
  fig.legend.label_text_font_size= legend_font_size
  fig.legend.orientation=orientation
  fig.legend.location=location
  fig.legend.glyph_width=glyph_width
  fig.legend.glyph_height=glyph_height
  fig.legend.spacing=5
  fig.xgrid.grid_line_color=None
  fig.ygrid.grid_line_color=None


pal = bokeh.palettes.colorblind['Colorblind'][8]
pl = [pal[0], pal[1], pal[3]]
pl.extend(pal[4:8])
pal = pl

