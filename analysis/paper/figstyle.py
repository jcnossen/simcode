"""Shared style of the main paper figures (Fig 2 = analysis/papersim/fig_sim_frcopt.py, Figs 3 / 4 = analysis/paper/fig_exp_new.py).

The figures are drawn at their print size, 180 mm wide (FIG_W inches), so every font size below is the printed size in pt:
labels / titles 7 pt, ticks / legends 6 pt (small annotations >= 5.5 pt), panel letters 8 pt lowercase bold (a, b, c ... as the
SI figures). Font: Arial, or a metric-compatible substitute (Liberation Sans, Helvetica) when Arial is not installed; TrueType
fonts embedded in PDF / PS (fonttype 42), SVG text kept as text.

  apply()                          rcParams (call once, before the figure is made)
  COL / LABEL / METHODS            one method palette: SMLM tab:blue, SIMFLUX tab:green, DECODE tab:orange, SIMCODE crimson
                                   (= analysis/papersim/fig_density.COL); internal-only preprint SIMCODE 'v1' tab:brown
  CLIP_PCT                         the one render clip percentile of every super-resolution render (pixels above it saturate)
  letter(ax, s, ...)               panel letter
  len_label(nm)                    '500 nm', '2 µm' (µm from 1000 nm on)
  scalebar(ax, x0, y0, L, ...)     bar + label on a dark backing (data coordinates of an image axes)
  frc_axes(ax)                     the one FRC axes style: x 0-0.1 nm^-1, 'Spatial frequency (nm^-1)', y 'FRC' 0 / 0.5 / 1,
                                   dashed threshold line labelled '1/7'
  save(fig, out)                   <out>.pdf / .svg (300 dpi rasters) and <out>.png (300 dpi preview)"""
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe

FIG_W = 180 / 25.4            # inches: the print width of a full-width figure
FS_LABEL = 7.0                # axis labels, titles
FS_TICK = 6.0                 # tick labels, legends
FS_SMALL = 5.5                # annotations (minimum)
FS_LETTER = 8.0               # panel letters
LW = 1.0                      # data lines
LW_AX = 0.6                   # axes / ticks
CLIP_PCT = 99.0               # render clip percentile (all renders of Figs 2-4)
FRC_XMAX = 0.1                # 1 / nm

COL = dict(smlm='tab:blue', sf='tab:green', decode='tab:orange', v2='crimson', v1='tab:brown')
LABEL = dict(smlm='SMLM', sf='SIMFLUX', decode='DECODE', v2='SIMCODE', v1='SIMCODE (preprint)')
METHODS = ['smlm', 'sf', 'decode', 'v2']


def font_family():
    """Arial if installed, else the first metric-compatible substitute found (Liberation Sans, Helvetica), else DejaVu Sans"""
    for f in ('Arial', 'Liberation Sans', 'Helvetica', 'Nimbus Sans'):
        try:
            fm.findfont(fm.FontProperties(family=f), fallback_to_default=False)
            return f
        except ValueError:
            continue
    return 'DejaVu Sans'


FONT = font_family()


def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'], 'font.size': FS_LABEL,
        'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold', 'mathtext.cal': FONT, 'mathtext.sf': FONT, 'mathtext.tt': FONT,
        'mathtext.default': 'regular',
        'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_LABEL, 'axes.linewidth': LW_AX, 'axes.titlepad': 3,
        'axes.labelpad': 2,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'xtick.major.width': LW_AX, 'ytick.major.width': LW_AX,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
        'legend.fontsize': FS_TICK, 'legend.title_fontsize': FS_TICK, 'legend.frameon': False, 'legend.handlelength': 1.5,
        'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.25,
        'lines.linewidth': LW, 'lines.markersize': 3, 'patch.linewidth': 0.5, 'hatch.linewidth': 0.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'figure.dpi': 100,
    })


def letter(ax, s, dx=-0.02, dy=1.02, transform=None, ha='right'):
    return ax.text(dx, dy, s.lower(), transform=transform or ax.transAxes, fontsize=FS_LETTER, fontweight='bold',
                   ha=ha, va='bottom')


def len_label(nm):
    return f'{nm / 1000:g} µm' if nm >= 1000 else f'{nm:g} nm'


BACKING = dict(boxstyle='square,pad=0.15', facecolor='black', alpha=0.55, edgecolor='none')


def scalebar(ax, x0, y0, L, nm, up=True, color='w', lw=1.5, fs=FS_TICK):
    """bar from (x0, y0) to (x0 + L, y0) in data coordinates of an image axes (y down for imshow), the label above it
    (up=True: towards smaller y on an image) on a dark backing; nm = the length shown in the label"""
    ax.plot([x0, x0 + L], [y0, y0], '-', color=color, lw=lw, solid_capstyle='butt',
            path_effects=[pe.Stroke(linewidth=lw + 1.2, foreground='k', alpha=0.6), pe.Normal()])
    t = ax.annotate(len_label(nm), (x0 + L / 2, y0), xytext=(0, 2 if up else -2), textcoords='offset points', color=color,
                    fontsize=fs, ha='center', va='bottom' if up else 'top', bbox=BACKING)
    return t


def scalebar_axes(ax, frac, nm, x=0.05, y=0.06, **kw):
    """scale bar in axes-fraction coordinates: frac = bar length / axes width"""
    lw = kw.pop('lw', 1.5); color = kw.pop('color', 'w'); fs = kw.pop('fs', FS_TICK)
    ax.plot([x, x + frac], [y, y], '-', color=color, lw=lw, solid_capstyle='butt', transform=ax.transAxes,
            path_effects=[pe.Stroke(linewidth=lw + 1.2, foreground='k', alpha=0.6), pe.Normal()])
    return ax.annotate(len_label(nm), (x + frac / 2, y), xycoords='axes fraction', xytext=(0, 2), textcoords='offset points',
                       color=color, fontsize=fs, ha='center', va='bottom', bbox=BACKING)


def frc_axes(ax, ylabel='FRC', xmax=FRC_XMAX, ymin=-0.03):
    ax.axhline(1 / 7, color='0.45', ls='--', lw=0.7, zorder=0)
    ax.text(xmax * 0.985, 1 / 7 + 0.015, '1/7', color='0.35', fontsize=FS_SMALL, ha='right', va='bottom')
    ax.set_xlim(0, xmax); ax.set_ylim(ymin, 1.02)
    if xmax == 0.1:
        ax.set_xticks([0, 0.025, 0.05, 0.075, 0.1]); ax.set_xticklabels(['0', '0.025', '0.05', '0.075', '0.1'])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_xlabel('Spatial frequency (nm$^{-1}$)'); ax.set_ylabel(ylabel)


def save(fig, out, exts=('pdf', 'svg', 'png'), **kw):
    for ext in exts:
        fig.savefig(f'{out}.{ext}', dpi=300, **kw)
        print(f'{out}.{ext}', flush=True)
