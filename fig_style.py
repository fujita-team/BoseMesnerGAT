"""Common matplotlib style for all figures of Section 7.
Titles 14 pt, axis labels 12 pt, legends 10 pt (bold, as in the original figures)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TITLE, LABEL, LEGEND, TICK = 14, 12, 10, 10


def apply_style():
    plt.rcParams.update({
        'axes.titlesize': TITLE, 'axes.titleweight': 'bold',
        'axes.labelsize': LABEL, 'axes.labelweight': 'bold',
        'legend.fontsize': LEGEND,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'axes.grid': True, 'grid.alpha': 0.35, 'grid.linestyle': '--',
        'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    })


def legend(ax, **kw):
    """Legend with bold 10-pt text."""
    kw.setdefault('fontsize', LEGEND)
    leg = ax.legend(**kw)
    for t in leg.get_texts():
        t.set_fontweight('bold')
    return leg


def title(ax, text):
    ax.set_title(text, fontsize=TITLE, fontweight='bold')


def labels(ax, x, y):
    ax.set_xlabel(x, fontsize=LABEL, fontweight='bold')
    ax.set_ylabel(y, fontsize=LABEL, fontweight='bold')


COLORS = dict(cheb='#2ca02c', real='#1f77b4', soft='#d62728', pos='#ff7f0e',
              attn='#9467bd', mis='#7f7f7f')
