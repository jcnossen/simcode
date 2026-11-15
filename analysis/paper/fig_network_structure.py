"""
SI figure of the two network architectures (si_network_structure_v3): (a) the detector (pass 1 and pass 2), (b) the emitter
network EmitterNet (pass 1 and pass 2). Layer sizes are those of the code (simcode/detector/model.py, simcode/emitter/model.py,
simcode/layers/blocks.py); the widths and parameter counts are read from the shipped m1809 checkpoints (CPU only) and
checked against the numbers drawn in the figure.

    python analysis/paper/fig_network_structure.py [--out /home/jc/simcode2/out/paper_figs_v2] [--models /home/jc/simcode2/models]

-> <out>/si_network_structure_v3.{pdf,svg,png} + si_network_structure_v3_caption.txt
"""
import argparse, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 6.2})

W, H = 180.0, 158.0
MM = 1 / 25.4
FS, FT, FL = 6.0, 6.8, 10
C = dict(inp='#f0f0f0', inpe='#666666', conv='#e6eef9', conve='#2f6db5', res='#c9dbf2', out='#e3f3e9', oute='#2b8a50',
         p2='#fcefe1', p2e='#d0701a', k='#222222')
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------------------------------------ numbers from the checkpoints
def checkpoint_facts(models):
    """widths, out-layer size and parameter counts of the m1809 networks (and the same counts of m25 / g13 if present)"""
    import torch
    sys.path.insert(0, REPO)
    from simcode.detector.model import build, n_params
    from simcode.emitter.model import EmitterNet
    F = {}
    for s in ('m1809', 'm25', 'g13'):
        for f, kind in (('det', 'd'), ('hd', 'e'), ('p2det', 'd'), ('p2headL', 'e')):
            p = f'{models}/{s}/{f}_{s}.pt'
            if not os.path.exists(p): continue
            c = torch.load(p, map_location='cpu', weights_only=False)
            a, sd = c.get('args') or {}, c['model']
            il = 'cin_il.weight' in sd
            net = build(a, illum=il) if kind == 'd' else EmitterNet(6, a.get('ch', 96), illum=il, bf16=False)
            net.load_state_dict(sd)
            F[(s, f)] = dict(n=n_params(net), ch=a.get('ch'), stem_ch=a.get('stem_ch'), step=c.get('step'),
                             shapes={k: tuple(v.shape) for k, v in sd.items()}, calib=os.path.basename(a.get('calib', '')))
    return F


def check(F):
    d, h = F[('m1809', 'det')], F[('m1809', 'hd')]
    p2d, p2h = F[('m1809', 'p2det')], F[('m1809', 'p2headL')]
    assert list(d['ch']) == [48, 96, 160] and d['stem_ch'] == 16
    assert d['shapes']['stem.0.weight'] == (16, 1, 3, 3) and p2d['shapes']['stem.0.weight'] == (16, 3, 3, 3)
    assert d['shapes']['cin.weight'] == (48, 97, 3, 3) and p2d['shapes']['cin_il.weight'] == (48, 12, 3, 3)
    assert d['shapes']['head.2.weight'] == (4, 48, 1, 1)
    assert h['ch'] == 96 and h['shapes']['cin.weight'] == (96, 18, 3, 3) and p2h['shapes']['cin_il.weight'] == (96, 12, 3, 3)
    assert h['shapes']['mlp.0.weight'] == (256, 1550) and h['shapes']['out.weight'] == (41, 256)
    assert h['shapes']['wmap.0.weight'] == (32, 96, 3, 3) and h['shapes']['dmap.2.weight'] == (6, 32, 1, 1)
    assert len({k.split('.')[1] for k in h['shapes'] if k.startswith('blk.')}) == 4
    for s in ('m25', 'g13'):
        for f in ('det', 'hd', 'p2det', 'p2headL'):
            if (s, f) in F: assert F[(s, f)]['n'] == F[('m1809', f)]['n'], (s, f)
    return dict(det=d['n'], p2det=p2d['n'], hd=h['n'], p2hd=p2h['n'])


# ------------------------------------------------------------------------------------------------ drawing
class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W * MM, H * MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis('off')

    def box(self, x, y, w, h, fc='white', ec='k', lw=0.6, r=1.0, ls='-', z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                         ls=ls, zorder=z))

    def text(self, x, y, s, size=FS, ha='left', va='top', color='k', weight='normal', z=5, **kw):
        return self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, weight=weight, zorder=z, **kw)

    def lines(self, x, y, rows, size=FS, dy=None, **kw):
        dy = dy or size * 0.3528 * 1.36
        for i, s in enumerate(rows):
            self.text(x, y + i * dy, s, size=size, **kw)
        return y + len(rows) * dy

    def layer(self, x, y, w, h, title, rows=(), fc=None, ec=None, size=FS, ls='-', tcol=None):
        self.box(x, y, w, h, fc=fc or C['conv'], ec=ec or C['conve'], ls=ls)
        self.text(x + w / 2, y + 1.0, title, size=size + 0.3, ha='center', weight='bold', color=tcol or 'k')
        self.lines(x + w / 2, y + 1.0 + (size + 0.3) * 0.3528 * 1.45, list(rows), size=size, ha='center')

    def arrow(self, x0, y0, x1, y1, color='k', lw=0.8, ls='-', ms=6, cs='arc3'):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                          ls=ls, zorder=4, shrinkA=0, shrinkB=0, connectionstyle=cs))

    def seg(self, xs, ys, color='k', lw=0.8, ls='-'):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=4, solid_capstyle='butt')

    def img(self, a, x, y, w, h, cmap='gray', vmin=None, vmax=None, z=3):
        self.ax.imshow(a, extent=(x, x + w, y + h, y), cmap=cmap, vmin=vmin, vmax=vmax, zorder=z, interpolation='nearest',
                       aspect='auto')
        self.ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='#444444', lw=0.3, zorder=z + 0.1))


def spot(n=11, A=1.0, seed=0):
    yy, xx = np.mgrid[:n, :n]
    g = A * np.exp(-((xx - 5.2) ** 2 + (yy - 4.8) ** 2) / (2 * 1.3 ** 2)) + 0.1
    return g + 0.05 * np.random.default_rng(seed).standard_normal((n, n))


def panel_a(cv, N):
    y0 = 2.0
    cv.text(1, y0 - 0.5, 'a', size=FL, weight='bold')
    cv.text(6, y0, 'Detector (pass 1 and pass 2: one network class)', size=FT + 0.6, weight='bold')
    # input
    for t in range(4):
        cv.img(spot(A=0.6 + 0.4 * np.cos(t), seed=t), 3 + t * 1.6, 11 + t * 1.6, 9, 9, vmin=0, vmax=1.2, z=3 + 0.1 * t)
    cv.lines(2, 36.6, ['window 6 × H × W (photons),', r'$\mathrm{sign}(y)\,\log(1+|y|/30)$'], size=FS - 0.4)
    # stem + mean
    cv.layer(20, 9, 25, 17, 'per-frame stem', ['weights shared over', 'the 6 frames:', 'conv3×3 1→16, GELU',
                                                'conv3×3 16→16, GELU'], size=FS - 0.4)
    cv.layer(20, 28, 25, 7.5, 'window mean', ['normalised, 1 ch'], fc=C['inp'], ec=C['inpe'])
    cv.arrow(16.8, 15.5, 20, 15.5); cv.seg([10, 10], [24.8, 31.5]); cv.arrow(10, 31.5, 20, 31.5)
    # cin
    cv.layer(48.5, 9, 16, 26.5, 'input conv', ['concat', '6×16 + 1', '= 97 ch', 'conv3×3', '97→48', 'GELU'])
    cv.arrow(45, 17.5, 48.5, 17.5); cv.arrow(45, 31.5, 48.5, 31.5)
    # pass-2 additions
    cv.box(2, 43, 62.5, 13.5, fc=C['p2'], ec=C['p2e'], ls=(0, (3, 1.5)))
    cv.text(3.2, 43.8, 'pass 2 only (illumination of the estimated pattern)', size=FS, weight='bold', color=C['p2e'])
    cv.lines(3.2, 46.8, ['maps per frame: cos part (6) + sin part (6) = 12 × H × W',
                         'stem input (y_t, cos_t, sin_t): first conv 3→16 instead of 1→16',
                         '+ conv3×3 12→48 (no bias) added before the input GELU',
                         'new weights zero-initialised = the pass-1 detector at start'], size=FS - 0.5, dy=2.35)
    cv.arrow(56.5, 43, 56.5, 35.6, color=C['p2e'], ls='--')
    # U-Net levels
    L0, L1, L2, lh = 9.0, 22.5, 36.0, 11.0
    sz = FS - 0.5
    cv.layer(67.5, L0, 19.5, lh, 'level 0', ['2 × ResBlock 48', 'H × W'], fc=C['res'], size=sz)
    cv.layer(89, L1, 19.5, lh, 'level 1', ['2 × ResBlock 96', 'H/2 × W/2'], fc=C['res'], size=sz)
    cv.layer(110.5, L2, 19.5, lh, 'level 2', ['2 × ResBlock 160', 'dilation 1, 2; H/4'], fc=C['res'], size=sz)
    cv.layer(132, L1, 19.5, lh, 'up 1', ['1×1 160→96, ×2', '+ skip, ResBlock'], fc=C['res'], size=sz)
    cv.layer(153, L0, 18, lh, 'up 0', ['1×1 96→48, ×2', '+ skip, ResBlock'], fc=C['res'], size=sz)
    cv.arrow(64.5, 14.5, 67.5, 14.5)
    cv.arrow(84, 20, 89, 25); cv.text(79.0, 21.2, 'conv3×3 /2\n48→96', size=FS - 1.0, ha='center', va='top')
    cv.arrow(105.5, 33.5, 110.5, 38.5); cv.text(100.5, 34.7, 'conv3×3 /2\n96→160', size=FS - 1.0, ha='center', va='top')
    cv.arrow(130, 38.5, 135, 33.5); cv.arrow(151.5, 25, 156, 20)
    cv.arrow(87, 11.0, 153, 11.0, color='#777777', ls='--', lw=0.6)
    cv.text(120, 10.5, 'skip (added)', size=FS - 1.0, ha='center', va='bottom', color='#555555')
    cv.arrow(108.5, 24.5, 132, 24.5, color='#777777', ls='--', lw=0.6)
    # head + outputs
    cv.layer(172.3, L0, 7.5, lh + 13.5, 'head', ['conv', '3×3', '48→48', 'GELU', 'conv', '1×1', '48→4'], fc=C['out'],
             ec=C['oute'], size=FS - 1.0)
    cv.arrow(171, 14.5, 172.3, 14.5)
    cv.box(132.5, 36.5, 47, 19.5, fc=C['out'], ec=C['oute'])
    cv.text(133.5, 37.4, 'per pixel (native 108 nm grid):', size=FS, weight='bold')
    cv.lines(133.5, 40.6, [r'logit ($p=\sigma(\cdot)$); offset $d_{x,y} = 1.5\tanh(\cdot)$ px;',
                           r'$\log\sigma_\mathrm{off} = -5 + 6\,\sigma(\cdot)$',
                           'candidates: greedy NMS on the sub-pixel points',
                           '(radius 0.7 px), top-k = 640 per 64 × 64 px,',
                           r'stored if $p\geq 0.3$ (neighbour splat: $p\geq 0.2$)'], size=FS - 0.6)
    cv.arrow(176.05, 33.5, 176.05, 36.5, color=C['oute'])
    # legend + counts
    cv.lines(68, 50.0, [r'ResBlock($c$, $d$): $x + \mathrm{conv}_{3\times3}(\mathrm{GELU}(\mathrm{GN}_8(\mathrm{conv}_{3\times3,\,d}(\mathrm{GELU}(x)))))$',
                        f'parameters: pass 1 {N["det"]:,}; pass 2 {N["p2det"]:,} (+{N["p2det"] - N["det"]:,})'],
             size=FS - 0.3)


def panel_b(cv, N):
    y0 = 62.0
    cv.seg([1, W - 1], [y0 - 2.5, y0 - 2.5], color='#bbbbbb', lw=0.5)
    cv.text(1, y0 - 0.5, 'b', size=FL, weight='bold')
    cv.text(6, y0, 'EmitterNet (pass 1 and pass 2: one network class), per candidate', size=FT + 0.6, weight='bold')
    # inputs
    yi = y0 + 7
    cv.box(2, yi, 44, 41, fc=C['inp'], ec=C['inpe'])
    cv.text(3.2, yi + 1.0, 'input: 21 × 21 patch per candidate', size=FS, weight='bold')
    rows = [('6', r'frames, $\mathrm{sign}(y)\log(1+|y|/30)$'),
            ('1', 'window mean, same normalisation'),
            ('6', r'frames, linear: $(y - \bar b)/100$'),
            ('1', 'window mean, linear'),
            ('2', 'coordinate maps (x, y relative to the'),
            ('', 'detector point, / 10 px)'),
            ('1', 'neighbour splat: other candidates'),
            ('', r'($p\geq0.2$), $p\,\exp(-d^2/2\cdot0.5^2)$, own'),
            ('', 'stamp removed'),
            ('1', 'detector-probability map'),
            ('18', 'channels')]
    for i, (n, s) in enumerate(rows):
        yy = yi + 4.6 + i * 3.1
        cv.text(6.0, yy, n, size=FS - 0.3, ha='right', weight='bold' if n == '18' else 'normal')
        cv.text(7.2, yy, s, size=FS - 0.3, weight='bold' if n == '18' else 'normal')
    cv.text(3.2, yi + 4.6 + 11 * 3.1 + 0.3, r'$\bar b$: mean over frames of the border median', size=FS - 0.8, color='#555555')
    # pass-2 inputs
    cv.box(2, yi + 43.5, 44, 12, fc=C['p2'], ec=C['p2e'], ls=(0, (3, 1.5)))
    cv.text(3.2, yi + 44.4, 'pass 2 only', size=FS, weight='bold', color=C['p2e'])
    cv.lines(3.2, yi + 47.5, ['+ 12 illumination patches (cos, sin)',
                              '→ conv3×3 12→96 (no bias), added to the',
                              'input conv; zero-init. (= pass 1 at start)'], size=FS - 0.5)
    # trunk
    cv.layer(49, yi, 12, 41, 'input', ['conv3×3', '18→96', 'GELU'])
    cv.arrow(46, yi + 20, 49, yi + 20)
    cv.seg([46, 55], [yi + 49.5, yi + 49.5], color=C['p2e'], ls='--')
    cv.arrow(55, yi + 49.5, 55, yi + 41.2, color=C['p2e'], ls='--')
    cv.layer(64, yi, 19, 41, 'trunk', ['4 × ResBlock', '96 channels', 'dilation', '1, 2, 1, 2', 'on 21 × 21', '', 'bf16'],
             fc=C['res'])
    cv.arrow(61, yi + 20, 64, yi + 20)
    # branches
    bx, bw = 87, 36
    cv.layer(bx, yi, bw, 11, 'weight map', ['conv3×3 96→32, GELU', '1×1 32→1 → m'])
    cv.layer(bx, yi + 12.5, bw, 11, 'per-frame maps', ['conv3×3 96→32, GELU', r'1×1 32→6 → $D_t$'])
    cv.layer(bx, yi + 25, bw, 16, 'global', ['conv3×3 /2 96→96, GELU', 'conv3×3 /2 96→96, GELU', 'avg-pool 4 × 4 → 1536'])
    for yy in (yi + 5.5, yi + 18, yi + 33):
        cv.arrow(83, yy, bx, yy)
    # scalars
    cv.box(bx, yi + 43.5, bw, 12, fc=C['inp'], ec=C['inpe'])
    cv.lines(bx + 1.2, yi + 44.4, ['14 scalars: border-median bg', r'$b_{0,t}$ (6, normalised), detector', 'offset (2), zeros (6)'],
             size=FS - 0.4)
    # MLP + out
    mx = 127
    cv.layer(mx, yi + 25, 22, 30.5, 'MLP', ['concat 1550', 'linear 1550→256', 'GELU', 'linear 256→256', 'GELU',
                                            'linear 256→41', '(fp32)'], fc=C['res'])
    cv.arrow(bx + bw, yi + 33, mx, yi + 33)
    cv.arrow(bx + bw, yi + 49.5, mx, yi + 49.5)
    # read-out
    rx = 127
    cv.box(rx, yi, 52, 22.5, fc=C['out'], ec=C['oute'])
    cv.text(rx + 1.2, yi + 0.9, 'linear read-out (fp32)', size=FS + 0.3, weight='bold')
    cv.lines(rx + 1.2, yi + 4.2, [
        r'$g$: PSF template at the detector offset (calibrated',
        r'in-situ PSF); $w_0 = g/(g\cdot g)$',
        r'$w = w_0\,e^{m}$, scaled so that $w\cdot g = 1$',
        r'$w_t = w + D_t$, $D_t$ with its $g$-component removed',
        r'$I_{\mathrm{lin},t}=\mathbf{w}_t\cdot(\mathbf{y}_t-bg_t)$ (sum over the patch pixels)'], size=FS - 0.3)
    cv.arrow(bx + bw, yi + 5.5, rx, yi + 5.5); cv.arrow(bx + bw, yi + 18, rx, yi + 18)
    # out split bar
    ox, oy, ow = 152, yi + 25, 27
    segs = [('dI', 6, '#4c78a8'), ('log diag L', 6, '#72a0cf'), ('off-diag L', 15, '#a9c6e8'), ('dx, dy', 2, '#e45756'),
            (r'log $\sigma_{x,y}$', 2, '#f19e9c'), ('bg', 6, '#59a14f'), (r'z, log $s_{z|xy}$, $b_x$, $b_y$', 4, '#b07aa1')]
    cv.text(ox, oy, '41 outputs', size=FS, weight='bold')
    x = ox; u = ow / 41
    for i, (nm, n, col) in enumerate(segs):
        cv.ax.add_patch(Rectangle((x, oy + 3.4), n * u, 2.6, fc=col, ec='white', lw=0.3, zorder=3))
        cv.ax.add_patch(Rectangle((ox, oy + 7.6 + i * 3.0), 2.0, 2.0, fc=col, ec='none', zorder=3))
        cv.text(ox + 2.8, oy + 7.4 + i * 3.0, f'{n}  {nm}', size=FS - 0.6)
        x += n * u
    cv.arrow(mx + 22, yi + 46, ox - 0.2, yi + 46)
    # outputs
    yo = yi + 59
    cv.box(2, yo, 177, 27.5, fc='white', ec=C['oute'], lw=0.8)
    cv.text(3.2, yo + 0.9, 'outputs (the solve uses I, L and the x, y prior; z of pass-2 rows comes from pass 1)', size=FS + 0.3,
            weight='bold')
    cv.lines(3.2, yo + 4.6, [
        r'$bg_t = b_{0,t} + q_{bg,t}\sqrt{|b_{0,t}|+1}$',
        r'$I_t = I_{\mathrm{lin},t} + c\,dI_t$,   $c = \max(3,\sqrt{\langle I_\mathrm{lin}\rangle_+ + 9\langle bg\rangle_+})$',
        r'6 × 6 Cholesky $L$: diag $1.5\,c\,e^{q}$ ($q$ clamped to ±4), off-diagonal $c\,q$',
        r'(intensity covariance $LL^\mathsf{T}$; the solve scales it per photon bin)'], size=FS - 0.2, dy=3.6)
    cv.lines(98, yo + 4.6, [
        r'prior $x, y$ = detector point $+\,0.5\tanh(\cdot)$ px',
        r'$\log\sigma_{x,y} = -4.5 + 5\,\sigma(\cdot)$ ($x, y$ uncorrelated)',
        r'$z$ (px, relative to the PSF focal plane),',
        r'$\log s_{z|xy} = -4 + 5.5\,\sigma(\cdot)$, $\mathbf{b}$ clamped to ±5:',
        r'3 × 3 covariance $L_p = [[\sigma_x,0,0],[0,\sigma_y,0],$',
        r'$[b_x\sigma_x, b_y\sigma_y, s_{z|xy}]]$'], size=FS - 0.2, dy=2.75)
    cv.text(3.2, yo + 22.6, f'parameters: pass 1 {N["hd"]:,}; pass 2 {N["p2hd"]:,} (+{N["p2hd"] - N["hd"]:,}).  '
                             'bf16 autocast for the CNN trunk, branches and MLP; fp32 for the output layers, read-out and Cholesky.',
            size=FS - 0.4)
    cv.arrow(164.5, yi + 55.5, 164.5, yo, color=C['oute'])
    cv.arrow(152.5, yi + 22.5, 152.5, yi + 25, color=C['oute'])


def build(out, models):
    F = checkpoint_facts(models)
    N = check(F)
    cv = Canvas()
    panel_a(cv, N); panel_b(cv, N)
    os.makedirs(out, exist_ok=True)
    base = f'{out}/si_network_structure_v3'
    for ext in ('pdf', 'svg'):
        cv.fig.savefig(f'{base}.{ext}')
    cv.fig.savefig(f'{base}.png', dpi=300)
    plt.close(cv.fig)
    cnt = '\n'.join(f'     {s} {f}: {v["n"]:,} parameters, step {v["step"]}, trained on {v["calib"]}' for (s, f), v in F.items())
    with open(f'{base}_caption.txt', 'w') as f:
        f.write(CAPTION.replace('@COUNTS@', cnt).format(**N, dd=N['p2det'] - N['det'], dh=N['p2hd'] - N['hd']))
    return base


CAPTION = r"""si_network_structure_v3: caption material (generated by analysis/paper/fig_network_structure.py)

DRAFT CAPTION
Figure S?. Network architectures. Pass 1 and pass 2 use the same two network classes. The pass-2 networks add input
channels for the illumination maps of the estimated pattern (orange). These new weights are zero-initialised, so training
starts exactly from the pass-1 network. (a) Detector: a fully convolutional U-Net with 3 levels (48, 96, 160 channels),
residual blocks with GroupNorm, and bilinear upsampling with additive skips. It runs on a 6-frame window. A per-frame
stem with weights shared over the frames is followed by an input convolution on the 6x16 stem channels and the window
mean. Per pixel of the native grid it outputs a detection logit, a sub-pixel offset and the offset's log-sigma.
Candidates are selected by greedy non-maximum suppression on the sub-pixel points (radius 0.7 px) followed by top-k. The
detector has {det:,} parameters in pass 1 and {p2det:,} in pass 2. (b) EmitterNet runs per candidate on a 21x21 patch with
18 input channels: normalised and linear frames, their window means, coordinate maps, and two neighbour channels (a splat
of the other detected candidates and the detector-probability map). A trunk of 4 residual blocks (96 channels) feeds three
branches: a weight-map modulation m, 6 signed per-frame maps D_t, and a pooled global feature that goes through an MLP to
41 scalar outputs. The intensities come from a learned linear read-out of the pixels, I_lin,t = sum_q w_tq (y_tq - bg_t),
whose weights are built around the PSF template g at the detector offset and satisfy w_t . g = 1. A learned correction is
added: I_t = I_lin,t + c dI_t. The network further outputs the Cholesky factor of a full 6x6 intensity covariance, the
per-frame background, and a 3D position prior (x, y, z) with a 3x3 covariance (x, y uncorrelated; z | x, y linear in x, y).
EmitterNet has {hd:,} parameters in pass 1 and {p2hd:,} in pass 2.

FACTS (code = /home/jc/simcode2-paperfigs2, branch paperfigs2; checkpoints /home/jc/simcode2/models/<set>/, read on CPU)

Parameter counts (simcode.detector.model.n_params on the network built from each checkpoint and loaded with its state
dict; the script asserts that m25 and g13 have the same counts as m1809):
@COUNTS@
   pass-2 extras: detector +{dd:,} = stem first conv 16x(3-1)x3x3 = 288 + cin_il 48x12x3x3 = 5,184;
                  EmitterNet +{dh:,} = cin_il 96x12x3x3 (no bias).

(a) Detector (simcode/detector/model.py: Detector, build; simcode/layers/blocks.py: ResBlock, norm_photons;
    simcode/detector/select.py: select_subpixel; simcode/emitter/model.py: detect_candidates; processing/front_fast.py)
   - args of the checkpoints: ch = [48, 96, 160], stem_ch = 16, up = 1 (native output grid), arch 'detector'.
   - norm_photons: sign(y) log1p(|y| / 30) on the photons; the window mean uses the same scale.
   - stem: Conv2d(1, 16, 3) GELU Conv2d(16, 16, 3) GELU per frame (shared weights); pass 2: Conv2d(3, 16, 3) on
     (frame_t, cos_t, sin_t) (the pass-1 weights go into channel 0, the rest zero: Detector.load_pass1).
   - cin: Conv2d(6*16 + 1 = 97, 48, 3); pass 2 + cin_il: Conv2d(12, 48, 3, bias=False), zero-init, added before the GELU.
   - e0: 2 ResBlock(48); d1: Conv2d(48, 96, 3, stride 2); e1: 2 ResBlock(96); d2: Conv2d(96, 160, 3, stride 2);
     e2: ResBlock(160, dil 1), ResBlock(160, dil 2); u1: Conv2d(160, 96, 1) -> bilinear x2 + s1 -> r1 ResBlock(96);
     u0: Conv2d(96, 48, 1) -> bilinear x2 + s0 -> r0 ResBlock(48); head: Conv2d(48, 48, 3) GELU Conv2d(48, 4, 1)
     (logit bias init -4). Input padded (replicate) to a multiple of 4.
   - ResBlock(c, d): x + c2(GELU(GroupNorm(8, c)(c1(GELU(x))))), c1 = 3x3 dilation d, c2 = 3x3.
   - outputs: logit; dxy = 1.5 tanh (px relative to the pixel centre); log_sig = -5 + 6 sigmoid.
   - selection: select_subpixel (greedy NMS on the points pixel + offset, radius 0.7 px, 7x7 neighbourhood, iterated to the
     fixed point; top-k per image); front_fast: k = 640 S^2 / 64^2, detect p_min 0.2 (THR0 / NBR_PMIN, the neighbour
     splat), stored candidates p >= CAND_PMIN = 0.3 (processing/common.py). Detector inference under bf16 autocast.

(b) EmitterNet (simcode/emitter/model.py: EmitterNet, nbr_map, nbr_patches, PSFCol, apply_to_candidates)
   - args: ch = 96, bf16 = 1, arch 'emitter_net'; patch PP = 21; T = 6.
   - input channels (forward): norm_photons(patch) [6], norm_photons(mean over frames) [1], lin = (patch - lvl) / 100 [6]
     with lvl = mean over frames of the per-frame border median b0_t, lin mean [1], cx, cy = (pixel - detector point) /
     10 px [2], nbrp [2] = (splat of the other candidates p >= 0.2: p exp(-d^2 / (2 0.5^2)) on 7x7 px, own stamp removed;
     sigmoid(detector logit)). Total 2T + 4 + 2 = 18 (cin.weight (96, 18, 3, 3)).
   - The PSF template g is NOT an input channel: it enters only the read-out (w0 = g / (g.g); D_t projected orthogonal to
     g). docs/METHOD.md stage 1 lists "the PSF template" among the inputs; the figure follows the code. g = PSFCol: the
     calibrated in-situ PSF (spline LUT, 21x21) at the detector offset, z fixed at 50 nm (PSFCol default z = 0.05 um).
   - cin: Conv2d(18, 96, 3); pass 2 + cin_il Conv2d(12, 96, 3, bias=False) zero-init on the illumination patches.
   - blk: ResBlock(96, 1 + i % 2), i = 0..3 -> dilations 1, 2, 1, 2 (on the full 21x21).
   - wmap: Conv2d(96, 32, 3) GELU Conv2d(32, 1, 1) (zero-init) -> m; dmap: Conv2d(96, 32, 3) GELU Conv2d(32, 6, 1)
     (zero-init) -> d; D_t = d * max(w0), minus its g-component.
   - down: Conv2d(96, 96, 3, s2) GELU Conv2d(96, 96, 3, s2) GELU AdaptiveAvgPool2d(4) -> 96 x 16 = 1536.
   - scalars: norm_photons(b0) [6], detector offset off [2], zeros [6] (nsc = 2T + 2 = 14; the zero slots are a fixed
     placeholder). MLP: Linear(1550, 256) GELU Linear(256, 256) GELU; out: Linear(256, 41) (zero-init).
   - out layout q = [dI 6 | log diag L 6 | off-diag L 15 | dx, dy 2 | log sx, sy 2 | bg 6 | z, lsz, b_x, b_y 4] = 41.
   - read-out: w = w0 exp(clamp(m, -5, 5)), w / (w.g); w_t = w + D_t; I_lin,t = sum_q w_tq (y_tq - bg_t) (y = photons);
     bg_t = b0_t + q_bg,t sqrt(|b0_t| + 1); c = clamp(sqrt(sum(I_lin+)/6 + 9 mean(bg+)), min 3); I_t = I_lin,t + c dI_t;
     L diag = 1.5 c exp(clamp(ld, -4, 4)), L off-diag = c od.
   - prior: x, y = detector point + 0.5 tanh(q); log sx, sy = -4.5 + 5 sigmoid(q); z = q (px of the lateral pixel,
     relative to the calibration's focal plane); lsz = -4 + 5.5 sigmoid(q); b = clamp(q, -5, 5);
     L_p = [[sx, 0, 0], [0, sy, 0], [b_x sx, b_y sy, s_z|xy]] (prior_cov).
   - precision: torch.autocast bf16 for cin / blk / branch first convs / down / MLP (trunk()); fp32 for the 1x1 read-out
     convs of the branches, the out layer, the read-out and the Cholesky.
   - The solve scales LL^T per photon bin (kap_I2) and the prior sigma (kap_p) (solve/solve.solve_inputs).
   - Pass 2: the z output of the pass-2 network is not used (z from pass 1, solve/z_conditioning.py).

Differences from the preprint's network figure (si_network_structure_v2): the preprint network had per-frame 2-layer
U-Nets, three intensity-estimator U-Nets and a combiner U-Net with 64/128/256 features; v2 has a separate detector
U-Net and one per-candidate EmitterNet with a linear read-out and full covariance.

Code vs docs notes
   - docs/METHOD.md stage 1 lists the PSF template as an input of the head; in the code it only shapes the read-out
     weights (see above).
   - models/README.md table lists the m1809 PSF as insitu/calib_1809_2_psf3d_avg.json; the shipped checkpoints' args
     say calib_1809_2_psf_v2.json (model.json 'psf' agrees with the checkpoints).
"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/jc/simcode2/out/paper_figs_v2')
    ap.add_argument('--models', default='/home/jc/simcode2/models')
    a = ap.parse_args()
    print(build(a.out, a.models))
