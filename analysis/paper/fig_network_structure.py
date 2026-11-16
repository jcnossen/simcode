"""
SI figure of the two network architectures (si_network_structure_v3): (a) the detection network, (b) the emitter network
(per detected emitter); pass 1 and pass 2 share each architecture, the pass-2 additions are drawn in orange. Layer sizes
are those of the code (simcode/detector/model.py, simcode/emitter/model.py, simcode/layers/blocks.py); the widths and
parameter counts are read from the shipped m1809 checkpoints (CPU only) and checked against the numbers drawn in the
figure. Colours: grey = input, blue = learned layers, white (outlined) = fixed computation, green = output, orange = pass 2
only. After drawing, layout_problems() checks every text for font size >= 5.5 pt (deepest mathtext sub/superscript
>= 4.4 pt at 0.7 per level), text-text overlaps, and text crossing a box edge or closer than 0.8 mm (horizontal) /
0.45 mm (vertical) to it from inside (the script fails if it finds one; --draft saves anyway).

    python analysis/paper/fig_network_structure.py [--out /home/jc/simcode2/out/paper_figs_v2] [--models /home/jc/simcode2/models]

-> <out>/si_network_structure_v3.{pdf,svg,png} + si_network_structure_v3_caption.txt
"""
import argparse, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patheffects as pe

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 6.0})

W, H = 180.0, 145.5
MM = 1 / 25.4
FS, FSS, FT, FL, FM = 6.0, 5.8, 7.0, 9.0, 9.0        # rows, dense rows, panel titles, panel letters, output symbols
MIN_PT = 5.5
MIN_SCRIPT_PT = 4.4                                    # deepest sub/superscript (0.7 per level)
# colour scheme (stated in the caption): grey = input, one blue = learned layers, white / outlined = fixed computation,
# green = output, orange = pass 2 only
C = dict(inp='#eeeeee', inpe='#6b6b6b', net='#d6e4f5', nete='#2f6db5', fix='#ffffff', fixe='#333333', out='#e3f3e9',
         oute='#2b8a50', p2='#fcefe1', p2e='#d0701a', skip='#777777', k='#222222')
P2LS = (0, (3, 1.5))
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
        for f, kind, fn in (('det', 'd', 'detector_pass1'), ('hd', 'e', 'emitter_pass1'), ('p2det', 'd', 'detector_pass2'), ('p2headL', 'e', 'emitter_pass2')):
            p = f'{models}/{s}/{fn}.pt'
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
def lh(size):
    """line spacing (mm) of a text row of the given size (pt)"""
    return size * 0.3528 * 1.36


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W * MM, H * MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis('off')
        self.boxes = []                       # (patch, name) of the drawn boxes, for layout_problems()

    def box(self, x, y, w, h, fc='white', ec='k', lw=0.6, r=1.0, ls='-', z=1, name=''):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
        self.ax.add_patch(p); self.boxes.append((p, name))

    def text(self, x, y, s, size=FS, ha='left', va='top', color='k', weight='normal', z=5, **kw):
        return self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, weight=weight, zorder=z, **kw)

    def lines(self, x, y, rows, size=FS, dy=None, **kw):
        dy = dy or lh(size)
        for i, s in enumerate(rows):
            self.text(x, y + i * dy, s, size=size, **kw)
        return y + len(rows) * dy

    def layer(self, x, y, w, h, title, rows=(), kind='net', size=FSS, ls='-', name=''):
        """a box with a bold title and centred rows; kind = inp / net / fix / out / p2 (colour scheme)"""
        self.box(x, y, w, h, fc=C[kind], ec=C[kind + 'e'], ls=ls, name=name or title)
        self.text(x + w / 2, y + 0.9, title, size=size + 0.3, ha='center', weight='bold')
        self.lines(x + w / 2, y + 0.9 + lh(size + 0.3) * 1.06, list(rows), size=size, ha='center')

    def arrow(self, x0, y0, x1, y1, color='k', lw=0.7, ls='-', ms=5.5, cs='arc3', halo=False):
        """halo: white outline, so that a line it crosses shows a gap (bridge)"""
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls,
                            zorder=4.5 if halo else 4, shrinkA=0, shrinkB=0, connectionstyle=cs)
        if halo: a.set_path_effects([pe.Stroke(linewidth=lw + 3.0, foreground='white'), pe.Normal()])
        self.ax.add_patch(a)

    def rich(self, xc, y, parts, **kw):
        """one line of (text, size) parts, centred at xc, common baseline y"""
        r = self.fig.canvas.get_renderer()
        ts = [self.text(0, y, s, size=sz, va='baseline', **kw) for s, sz in parts]
        ws = [t.get_window_extent(r).width / self.fig.dpi * 25.4 for t in ts]
        x = xc - sum(ws) / 2
        for t, w in zip(ts, ws):
            t.set_x(x); x += w

    def seg(self, xs, ys, color='k', lw=0.7, ls='-'):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=4, solid_capstyle='butt')

    def path(self, pts, color='k', lw=0.7, ls='-'):
        """polyline through pts with an arrow head on the last segment"""
        xs, ys = zip(*pts)
        if len(pts) > 2: self.seg(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls)
        self.arrow(xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw, ls=ls)

    def plus(self, x, y, r=1.15, ec='k'):
        """additive node (circle with +), centred at (x, y)"""
        self.ax.add_patch(Circle((x, y), r, fc='white', ec=ec, lw=0.6, zorder=6))
        self.ax.plot([x - 0.62 * r, x + 0.62 * r], [y, y], color=ec, lw=0.6, zorder=7, solid_capstyle='butt')
        self.ax.plot([x, x], [y - 0.62 * r, y + 0.62 * r], color=ec, lw=0.6, zorder=7, solid_capstyle='butt')

    def img(self, a, x, y, w, h, cmap='gray', vmin=None, vmax=None, z=3):
        self.ax.imshow(a, extent=(x, x + w, y + h, y), cmap=cmap, vmin=vmin, vmax=vmax, zorder=z, interpolation='nearest',
                       aspect='auto')
        self.ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='#444444', lw=0.3, zorder=z + 0.1))


def spot(n=11, A=1.0, seed=0):
    yy, xx = np.mgrid[:n, :n]
    g = A * np.exp(-((xx - 5.2) ** 2 + (yy - 4.8) ** 2) / (2 * 1.3 ** 2)) + 0.1
    return g + 0.04 * np.random.default_rng(seed).standard_normal((n, n))


def legend(cv, x, y):
    """one-row colour legend (both panels), baseline y"""
    items = [('inp', '-', 'input'), ('net', '-', 'learned layers'), ('fix', '-', 'fixed computation'),
             ('out', '-', 'output'), ('p2', P2LS, 'pass 2 only')]
    r = cv.fig.canvas.get_renderer()
    for k, ls, s in items:
        cv.ax.add_patch(Rectangle((x, y - 2.2), 3.4, 2.3, fc=C[k], ec=C[k + 'e'], lw=0.6, ls=ls, zorder=3))
        t = cv.text(x + 4.3, y - 1.05, s, size=FSS, va='center')
        x += 4.3 + t.get_window_extent(r).width / cv.fig.dpi * 25.4 + 3.2
    return x


def panel_a(cv):
    cv.text(1, 4.6, 'a', size=FL, weight='bold', va='baseline')
    cv.text(6, 4.6, 'Detection network', size=FT, weight='bold', va='baseline')
    # input window: 6 frames (label under the stack); window mean above the stem
    for t in range(6):
        cv.img(spot(A=0.75 + 0.45 * np.cos(2 * np.pi * t / 3 + 0.4), seed=t), 3 + t * 1.2, 13.5 + t * 1.2, 9, 9,
               vmin=0.05, vmax=0.9, z=3 + 0.1 * t)
    cv.lines(1.5, 30.6, ['6-frame window', '(H × W), photons'], size=FSS)
    cv.layer(20.5, 7.0, 25.5, 7.6, 'window mean', ['log-scaled, 1 channel'], kind='inp')
    cv.seg([7.5, 7.5], [13.5, 10.8]); cv.arrow(7.5, 10.8, 20.5, 10.8)
    cv.layer(20.5, 17.0, 25.5, 16.0, 'per-frame stem', ['log-scaled frames,', 'shared weights', 'conv3×3 1→16, GELU',
                                                          'conv3×3 16→16, GELU'])
    cv.arrow(18.0, 24.5, 20.5, 24.5)
    # input conv
    cv.layer(49, 7.0, 12.5, 26.0, 'input', ['conv3×3', '97→48', 'GELU'])
    cv.arrow(46, 10.8, 49, 10.8); cv.arrow(46, 24.5, 49, 24.5)
    # pass-2 additions: to the stem and to the input conv
    cv.box(2, 40.0, 60, 18.7, fc=C['p2'], ec=C['p2e'], ls=P2LS, name='pass 2 (a)')
    cv.text(3.2, 40.9, 'pass 2 only', size=FS + 0.2, weight='bold', color=C['p2e'])
    cv.lines(3.2, 44.2, ['per frame: expected relative illumination − 1',
                         'and its quadrature (90°-shifted) modulation',
                         '(12 maps of H × W)',
                         '→ per-frame stem: first conv 3→16 on (frame, 2 maps)',
                         '→ input conv: + conv3×3 12→48'], size=FSS)
    cv.arrow(33, 40.0, 33, 33.0, color=C['p2e'], ls='--')
    cv.arrow(55, 40.0, 55, 33.0, color=C['p2e'], ls='--')
    # U-Net
    L0, L1, L2, hl, hu, wl = 7.0, 24.0, 41.0, 12.6, 16.0, 21.0
    cv.layer(64.5, L0, wl, hl, 'level 0', ['2 × ResBlock 48', 'H × W'])
    cv.layer(87, L1, wl, hl, 'level 1', ['2 × ResBlock 96', 'H/2 × W/2'])
    cv.layer(109.5, L2, wl, hl, 'level 2', ['2 × ResBlock 160', 'dilation 1, 2', 'H/4 × W/4'])
    cv.layer(132, L1, 20.0, hu, 'up 1', ['conv1×1 160→96', 'upsample ×2', '   skip', 'ResBlock 96'])
    cv.layer(153.5, L0, 25.5, hu, 'up 0', ['conv1×1 96→48', 'upsample ×2', '   skip', 'ResBlock 48'])
    cv.arrow(61.5, 12.0, 64.5, 12.0)
    cv.arrow(82.5, L0 + hl, 87, L1 + 4.6)
    cv.lines(76, L1 + 0.6, ['conv3×3 /2', '48→96'], size=FSS, ha='center')
    cv.arrow(104.5, L1 + hl, 109.5, L2 + 4.6)
    cv.lines(97, L2 - 2.0, ['conv3×3 /2', '96→160'], size=FSS, ha='center')
    cv.arrow(130.5, L2 + 3.0, 136, L1 + hu)
    cv.arrow(149, L1, 153.5, L0 + hu - 2.5)
    # additive skips: after the upsampling, before the ResBlock (the '+ skip' row of the up blocks)
    ys = 0.9 + lh(FSS + 0.3) * 1.06 + 2 * lh(FSS) + 1.05       # centre of the third row below the box top
    for (xa, xb, xw, yt) in ((85.5, 153.5, 25.5, L0), (108, 132, 20.0, L1)):
        xp = xb + xw / 2 - 4.6
        cv.seg([xa, xb], [yt + ys, yt + ys], color=C['skip'], ls='--', lw=0.6)
        cv.arrow(xb, yt + ys, xp - 1.15, yt + ys, color=C['skip'], lw=0.6)
        cv.plus(xp, yt + ys, r=1.05, ec='k')
    # head (learned), per-pixel outputs, candidate selection (fixed computation)
    cv.layer(153.5, 26.0, 25.5, 10.2, 'head', ['conv3×3 48→48, GELU', 'conv1×1 48→4'])
    cv.arrow(166.25, L0 + hu, 166.25, 26.0)
    cv.box(138, 43.0, 41, 12.6, fc=C['out'], ec=C['oute'], name='per-pixel outputs')
    cv.text(139.5, 43.9, 'per pixel (native grid)', size=FS + 0.2, weight='bold')
    cv.lines(139.5, 47.2, [r'detection probability $p$', 'sub-pixel offset (≤ 1.5 px)', '± its uncertainty'], size=FS)
    cv.arrow(166.25, 36.2, 166.25, 43.0)
    cv.layer(138, 58.0, 25.5, 10.0, 'candidate selection', ['NMS, radius 0.7 px', r'$p \geq 0.3$ kept'], kind='fix')
    cv.arrow(150.75, 55.6, 150.75, 58.0)
    cv.arrow(163.5, 63.0, 170, 63.0)
    cv.text(171, 63.0, 'to', size=FS, va='center')
    cv.text(174.2, 63.0, 'b', size=FS + 1.5, va='center', weight='bold')
    legend(cv, 79.0, 4.6)
    return 68.0


def panel_b(cv, b0):
    cv.seg([1, W - 1], [b0 - 1.2, b0 - 1.2], color='#bbbbbb', lw=0.5)
    cv.text(1, b0 + 4.2, 'b', size=FL, weight='bold', va='baseline')
    cv.text(6, b0 + 4.2, 'Emitter network (per detected emitter)', size=FT, weight='bold', va='baseline')
    # input patch (from a)
    yi = b0 + 7.5
    cv.box(2, yi, 36, 41.2, fc=C['inp'], ec=C['inpe'], name='input patch')
    cv.lines(3.2, yi + 0.9, ['21 × 21 px patch at each', 'candidate from a'], size=FS + 0.2, weight='bold')
    rows = [('6', 'frames, log-scaled'),
            ('1', 'window mean, log-scaled'),
            ('6', 'frames, linear (minus the'),
            ('', 'window background level)'),
            ('1', 'window mean, linear'),
            ('2', 'coordinate maps (relative to'),
            ('', 'the sub-pixel position)'),
            ('1', 'neighbour map (the other'),
            ('', 'detected emitters)'),
            ('1', 'detection-probability map'),
            ('18', 'channels')]
    for i, (n, s) in enumerate(rows):
        yy = yi + 7.3 + i * 3.0
        cv.text(6.4, yy, n, size=FSS, ha='right', weight='bold' if n == '18' else 'normal')
        cv.text(7.4, yy, s, size=FSS, weight='bold' if n == '18' else 'normal')
    # pass-2 input
    yp = yi + 43.6
    cv.box(2, yp, 36, 21.6, fc=C['p2'], ec=C['p2e'], ls=P2LS, name='pass 2 (b)')
    cv.text(3.2, yp + 0.9, 'pass 2 only', size=FS + 0.2, weight='bold', color=C['p2e'])
    cv.lines(3.2, yp + 4.2, ['12 illumination patches: per', 'frame, expected relative',
                             'illumination − 1 and its', 'quadrature (90°-shifted)', 'modulation',
                             '→ input conv: + conv3×3 12→96'], size=FSS)
    # input conv, trunk
    yc = yi + 20.6                                  # centre line of the input box
    cv.layer(40.2, yc - 6.5, 11.5, 13.0, 'input', ['conv3×3', '18→96', 'GELU'])
    cv.arrow(38, yc, 40.2, yc)
    cv.path([(38, yp + 9.4), (45.95, yp + 9.4), (45.95, yc + 6.5)], color=C['p2e'], ls='--')
    cv.layer(53.5, yc - 8.0, 14.0, 16.0, 'trunk', ['ResBlock 96', '× 4', 'dilation', '1, 2, 1, 2'])
    cv.arrow(51.7, yc, 53.5, yc)
    # branches (fan out from the trunk) and the scalars
    bx, bw = 70.5, 29.5
    yw, yf, yg, ys = yi, yi + 12.0, yi + 24.0, yi + 42.5
    cv.layer(bx, yw, bw, 10.0, 'weight modulation', ['conv3×3 96→32, GELU', 'conv1×1 32→1'])
    cv.layer(bx, yf, bw, 10.0, 'per-frame changes', ['conv3×3 96→32, GELU', 'conv1×1 32→6'])
    cv.layer(bx, yg, bw, 16.0, 'global', ['conv3×3 /2 96→96, GELU', 'conv3×3 /2 96→96, GELU', 'avg-pool 4 × 4',
                                          '→ 96 × 4 × 4 = 1536'])
    cv.seg([67.5, 68.25], [yc, yc]); cv.seg([68.25, 68.25], [yw + 5, yg + 8.0])
    for yy in (yw + 5, yf + 5, yg + 8.0):
        cv.arrow(68.25, yy, bx, yy)
    cv.layer(bx, ys, bw, 16.0, '8 scalars', ['log-scaled border-median', 'background per frame (6),',
                                             'sub-pixel offset (2)', '+ 6 constant zeros'], kind='inp')
    # read-out: photons and PSF in (grey), weights from the branches, backgrounds from the + node below
    rx, rw, ry, rh = 102.5, 54.5, yi + 3.0, 14.5
    cv.layer(101.0, b0 + 0.6, 19.3, 7.4, 'patch photons', ['6 frames'], kind='inp')
    cv.layer(121.0, b0 + 0.6, 44.5, 7.4, 'in-situ PSF (calibrated, z = 50 nm)', ['at the detected sub-pixel position'],
             kind='inp')
    cv.arrow(110.65, b0 + 8.0, 110.65, ry); cv.arrow(140.0, b0 + 8.0, 140.0, ry)
    cv.box(rx, ry, rw, rh, fc=C['fix'], ec=C['fixe'], name='linear read-out')
    cv.text(rx + rw / 2, ry + 0.9, 'linear read-out', size=FSS + 0.3, ha='center', weight='bold')
    cv.text(rx + rw / 2, ry + 4.3, 'weights: PSF × modulation + per-frame changes', size=FSS, ha='center')
    cv.rich(rx + rw / 2, ry + 11.3, [('per frame: Σ weights × (photons − ', FSS), (r'$\hat{\mu}_{b_{i,j}}$', FM),
                                       (')', FSS)])
    cv.arrow(bx + bw, yw + 5.0, rx, yw + 5.0); cv.arrow(bx + bw, ry + rh - 3.0, rx, ry + rh - 3.0)
    # MLP
    mw = 21.5
    cv.layer(rx, yg + 1.5, mw, 13.0, 'MLP', ['1550→256, GELU', '256→256, GELU', '256→41'])
    cv.arrow(bx + bw, yg + 8.0, rx, yg + 8.0)
    cv.path([(bx + bw, ys + 8.0), (rx + 10, ys + 8.0), (rx + 10, yg + 14.5)])
    # the 41 outputs (thin box, outputs of the learned layers)
    xc = rx + mw + 1.75                             # the backgrounds' connector up into the read-out
    lx, lw_ = xc + 1.75, 30.5
    ly = ry + rh + 1.2
    out = [('6', ['intensity corrections'], 0.0), ('2', ['position offset'], 2.6), ('2', ['position uncertainties'], 0.3),
           ('1', [r'$z$'], 0.3), ('1', [r'$z$ uncertainty', r'given $x, y$'], 0.3), ('2', [r'$z$ slopes'], 0.3),
           ('21', ['Cholesky factor:', '6 diagonal,', '15 off-diagonal'], 0.6), ('6', ['background corrections'], 0.9)]
    yy, ymid = ly + 4.2, []
    for n, ss, gap in out:
        yy += gap
        cv.text(lx + 5.0, yy, n, size=FSS, ha='right')
        cv.lines(lx + 6.0, yy, ss, size=FSS)
        ymid.append((yy, yy + len(ss) * lh(FSS)))
        yy += len(ss) * lh(FSS)
    lh_ = yy + 1.0 - ly
    cv.box(lx, ly, lw_, lh_, fc='white', ec=C['nete'], lw=0.4, r=0.6, name='41 outputs')
    cv.text(lx + 1.2, ly + 0.8, '41 outputs', size=FS + 0.2, weight='bold')
    cv.arrow(rx + mw, yg + 8.0, lx, yg + 8.0, halo=True)
    mid = [(a + b) / 2 - 0.35 for a, b in ymid]
    # combination nodes and output tags at the right edge
    xe, xn, tx, tw, th = lx + lw_, 160.7, 164.2, 14.8, 9.8
    def tag(yc_, sym, word, kind='out'):
        cv.box(tx, yc_ - th / 2, tw, th, fc=C[kind], ec=C[kind + 'e'], name='tag ' + word)
        cv.text(tx + tw / 2, yc_ - 0.4, sym, size=FM, ha='center', va='baseline')
        cv.text(tx + tw / 2, yc_ + 1.2, word, size=FSS, ha='center', va='top')
    # intensities = read-out + correction
    yI = ry + rh / 2
    cv.plus(xn, yI); cv.arrow(rx + rw, yI, xn - 1.15, yI)
    cv.seg([xe, xn], [mid[0], mid[0]]); cv.arrow(xn, mid[0], xn, yI + 1.15)
    cv.arrow(xn + 1.15, yI, tx, yI); tag(yI, r'$\hat{\mu}_{N_{i,j}}$', 'intensities')
    # position = detected sub-pixel position + offset; uncertainties, z (bracket)
    y2, y3 = ymid[1][0] + 0.1, ymid[5][1] - 0.5
    yP = (y2 + y3) / 2
    cv.seg([xe - 1.2, xe - 0.4, xe - 0.4, xe - 1.2], [y2, y2, y3, y3])
    cv.seg([xe - 0.4, xe], [yP, yP])
    cv.plus(xn, yP); cv.arrow(xe, yP, xn - 1.15, yP)
    cv.arrow(xn + 1.15, yP, tx, yP); tag(yP, r'$\hat{\theta}_{\mathrm{dl}}\pm\sigma_{\mathrm{dl}}$', 'position')
    yD = mid[0] + 2.0                             # the detected position (grey input) above the position tag
    cv.box(tx, yD - 6.2, tw, 7.6, fc=C['inp'], ec=C['inpe'], name='detected position')
    cv.lines(tx + tw / 2, yD - 5.5, ['detected', 'position (a)'], size=FSS, ha='center')
    cv.path([(tx, yD), (xn, yD), (xn, yP - 1.15)])
    # uncertainties of the intensities
    cv.arrow(xe, mid[6], tx, mid[6]); tag(mid[6], r'$\hat{\sigma}_{N_{i,j}}$', 'uncertainties')
    # backgrounds = border median + correction; also into the read-out (connector on the left of the output list)
    yB = mid[7] + 4.6
    cv.plus(xn, yB)
    cv.seg([xe, xn], [mid[7], mid[7]]); cv.arrow(xn, mid[7], xn, yB - 1.15)
    yl = max(ys + 16.0, yB) + 3.0
    cv.seg([bx + bw / 2, bx + bw / 2, xn], [ys + 16.0, yl, yl]); cv.arrow(xn, yl, xn, yB + 1.15)
    cv.text((bx + bw / 2 + xc) / 2, yl - 0.4, 'border-median background', size=FSS, ha='center', va='bottom')
    cv.arrow(xn + 1.15, yB, tx, yB); tag(yB, r'$\hat{\mu}_{b_{i,j}}$', 'backgrounds')
    cv.seg([xn - 1.15, xc, xc], [yB, yB, ry + rh + 3.0]); cv.arrow(xc, ry + rh + 3.0, xc, ry + rh)
    cv.text(xc - 0.8, ry + rh + 0.6, r'$\hat{\mu}_{b_{i,j}}$', size=FM, ha='right', va='top')
    return max(yl, yp + 21.6, yB + th / 2)


def script_depth(s):
    """deepest sub/superscript level of a mathtext string (0 for plain text)"""
    d = best = 0; stack = []
    i = 0
    while i < len(s):
        if s[i] in '_^':
            if i + 1 < len(s) and s[i + 1] == '{':
                stack.append(True); d += 1; best = max(best, d); i += 2; continue
            best = max(best, d + 1)
        elif s[i] == '{':
            stack.append(False)
        elif s[i] == '}' and stack:
            if stack.pop(): d -= 1
        i += 1
    return best


def layout_problems(cv, tol_mm=0.8):
    """font sizes < MIN_PT (sub/superscripts: shrunk 0.7 per level, >= MIN_SCRIPT_PT), overlapping texts, texts that
    cross a box edge or come closer than tol_mm to it from inside"""
    r = cv.fig.canvas.get_renderer()
    px = cv.fig.dpi * MM
    T = [(t, t.get_window_extent(r)) for t in cv.ax.texts if t.get_text().strip()]
    B = [(n, p.get_window_extent(r)) for p, n in cv.boxes]
    out = []
    for t, _ in T:
        s, fs = t.get_text(), t.get_fontsize()
        if fs < MIN_PT - 1e-6: out.append(f'font {fs} pt: {s!r}')
        if '$' in s and fs * 0.7 ** script_depth(s) < MIN_SCRIPT_PT - 1e-6:
            out.append(f'script {fs * 0.7 ** script_depth(s):.2f} pt (depth {script_depth(s)}): {s!r}')
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T[i][1], T[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.3 * px and oy > 0.3 * px:
                out.append(f'texts overlap: {T[i][0].get_text()!r} / {T[j][0].get_text()!r}')
    for t, a in T:
        for n, b in B:
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox <= 0 or oy <= 0: continue
            # horizontal clearance tol_mm; vertical clearance 0.45 mm (the row pitch is set by the font)
            mx, my = tol_mm * px, 0.45 * px
            inside = a.x0 >= b.x0 + mx and a.x1 <= b.x1 - mx and a.y0 >= b.y0 + my and a.y1 <= b.y1 - my
            if not inside: out.append(f'text crosses or touches box {n!r}: {t.get_text()!r}')
    for t, a in T:
        if a.x0 < 0 or a.x1 > cv.fig.bbox.x1 or a.y0 < 0 or a.y1 > cv.fig.bbox.y1:
            out.append(f'text outside the figure: {t.get_text()!r}')
    return out


def build(out, models, draft=False):
    F = checkpoint_facts(models)
    N = check(F)
    cv = Canvas()
    ya = panel_a(cv); yb = panel_b(cv, ya + 2.6)
    print(f'panel b bottom at {yb:.1f} mm (figure height {H} mm)')
    probs = layout_problems(cv)
    if probs:
        msg = 'layout problems:\n  ' + '\n  '.join(probs)
        if not draft: raise SystemExit(msg)
        print(msg)
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

DRAFT CAPTION (symbols as in the preprint / main-text Methods; see out/paper_figs_v2/NOTATION.md)
Figure S?. Network architectures. Colours (legend at the top): grey, inputs; blue, learned layers; white, fixed
computations; green, outputs; orange, pass 2 only; + nodes add their inputs. Pass 1 and pass 2 use the same two architectures. The pass-2 networks add input channels for the
illumination of the pattern estimated after pass 1 (orange); these new weights are initialized at zero, so that training
starts exactly from the pass-1 network. ResBlock(c, d): h + conv3x3(GELU(GN(conv3x3,d(GELU(h))))), with c channels,
dilation d of the first convolution and group normalization GN with 8 groups. GELU activations are used for all
non-output activations.
(a) Detection network: a fully convolutional U-Net with 3 levels (48, 96 and 160 channels). Downsampling is by 3x3
convolutions with stride 2, upsampling by a 1x1 convolution and bilinear interpolation, and the skip connections are added
(+). The input is a moving window of 6 frames in photons, log-scaled (sign(y) log(1 + |y| / 30)). Each frame passes
through a stem whose weights are shared over the frames; the input convolution combines the 6 x 16 stem channels and the
window mean (97 channels). For each pixel of the native camera grid the network outputs a detection logit (p = its
sigmoid), a sub-pixel offset (at most 1.5 px) and the logarithm of the offset's uncertainty. Candidates are selected by
greedy non-maximum suppression (NMS) on the detected sub-pixel positions (radius 0.7 px, up to 4 iterations), followed by
a top-k (at most 640 candidates per 64 x 64 px); candidates with p >= 0.2 enter the neighbour channel of (b), and those with
p >= 0.3 are kept. Pass 2: per frame, the expected relative illumination minus 1 and its quadrature (90-degree-shifted)
modulation (12 maps);
the stem's first convolution takes each frame with its 2 maps (3 -> 16 channels instead of 1 -> 16), and a 3x3
convolution of the 12 maps (12 -> 48, no bias) is added to the input convolution before its GELU. The detection network
has {det:,} parameters ({p2det:,} in pass 2).
(b) Emitter network, run on a 21 x 21 px patch around each detected emitter. Inputs: the frames and their window mean,
log-scaled as in (a) and linear (photons minus the window background level, the mean over the frames of the border-median
background); coordinate maps (x and y relative to the detected sub-pixel position, divided by 10 px); and two neighbour
channels: Gaussian stamps (sigma 0.5 px, height p) at the other detected emitters (own stamp removed) and the
detection-probability map. A trunk of 4 residual blocks (96 channels, dilations 1, 2, 1, 2, full 21 x 21 resolution)
feeds three branches: a weight modulation, per-frame weight changes and a pooled global feature (96 x 4 x 4 = 1536). The
MLP takes the global feature and 8 scalars, the log-scaled border-median background of each frame and the sub-pixel
offset (plus 6 constant zero inputs; 1536 + 14 = 1550 inputs), and gives 41 outputs (thin box). The intensity of each
frame is a linear read-out: a weighted sum of the patch photons minus the background mu_b_ij (the network's background
output, so the read-out follows the background correction), with weights given by the in-situ PSF at the detected
sub-pixel position (the calibrated PSF evaluated at z = 50 nm) times the weight modulation, scaled to unit response to the
PSF, plus the per-frame changes (which have zero response to the PSF). The network outputs, per detected emitter: the 6
intensities mu_N_ij (angle i, phase step j) = read-out + learned correction (+), with their uncertainties sigma_N_ij
(the network also predicts their correlations, as a Cholesky factor, which the fit uses); the 6 backgrounds mu_b_ij =
border median + learned correction (+; the correction is scaled with sqrt(|border median| + 1)); and the position
theta_dl = (x, y, z) +- sigma_dl, with x, y the detected sub-pixel position from (a) plus a learned offset of at most
0.5 px (+), and sigma_dl the position uncertainties of x and y and the
uncertainty of z given x and y (z depends on x and y through the two z slopes). The intensity corrections and
uncertainties are scaled with the signal and background of the emitter. The fit uses the intensities, their
uncertainties and the position; the z of the final localizations is that of pass 1. The CNN and the MLP run in bfloat16,
the output layers, the read-out and the Cholesky factor in 32-bit floating point. The emitter network has {hd:,}
parameters ({p2hd:,} in pass 2; the 12 illumination patches enter through a 3x3 convolution, 12 -> 96 channels, no bias,
added to the input convolution).

SYMBOLS (paper symbol <- code name; NOTATION.md): mu_N_ij <- I_t = I_lin,t + c dI_t (t = 3 (i - 1) + (j - 1)); sigma_N_ij <-
sqrt(diag(L L^T)), L = the 6x6 Cholesky factor (diag 1.5 c exp(q), off-diagonal c q; c = max(3, sqrt(<I_lin>_+ +
9 <bg>_+)), the 'signal and background' scale of the caption); mu_b_ij <- bg_t = b0_t + q_bg,t sqrt(|b0_t| + 1);
theta_dl, sigma_dl <- the prior x, y, z and sigma_x, sigma_y, s_z|xy ('position uncertainties', 'z uncertainty given x, y');
'z slopes' <- b = (dz/dx, dz/dy). The read-out weights: g = PSF template, w0 = g / (g.g), w = w0 exp(m) / (w.g) (m = the
weight modulation), w_t = w + D_t (D_t = the per-frame weight changes minus their g-component), I_lin,t = sum_q w_tq (n_tq
- bg_t) with n = the patch photons (the read-out uses the corrected backgrounds bg_t). The correlations (off-diagonal of
L) are used by the fit; the covariance ablation (out/ablation_cov/REPORT.md) found no measurable effect of them in pass 2
(NOTATION.md section 6.1: describe the output as intensities with uncertainties, at most one sentence on correlations).

FOR THE SI TEXT (paper session; staged SI_after_1st_review.tex, caption of fig:network_structure, lines 181-182)
   - 'the 2D position (x, y)': the emitter network also gives z (theta_dl = (x, y, z) +- sigma_dl, NOTATION.md 6.5).
   - 'a background fluorescence at the emitter position (bg)': it is 6 backgrounds mu_b_ij, one per frame (border median
     + learned correction), and 'bg' is not a preprint symbol.
   - 'mean and uncertainty values (mu, sigma) ... 6 intensities (N_i,j) with their full 6 x 6 covariance matrix': write
     intensities mu_N_ij with uncertainties sigma_N_ij (NOTATION.md 6.1); at most one sentence on the correlations.
   - 'weights normalized to the PSF of the emitter': unit response to the in-situ PSF at the detected sub-pixel position.
   - 'modified version of the DECODE architecture': the current networks (separate detection U-Net, per-emitter network
     with a linear read-out) differ from DECODE more than the earlier network did; consider 'inspired by'.
   - the staged figure (stage/figures/si_network_structure_v3.pdf) is an older render: copy the current PDF.

FACTS (code = /home/jc/simcode2, branch main; checkpoints /home/jc/simcode2/models/<set>/, read on CPU)

Parameter counts (simcode.detector.model.n_params on the network built from each checkpoint and loaded with its state
dict; the script asserts that m25 and g13 have the same counts as m1809):
@COUNTS@
   pass-2 extras: detector +{dd:,} = stem first conv 16x(3-1)x3x3 = 288 + cin_il 48x12x3x3 = 5,184;
                  EmitterNet +{dh:,} = cin_il 96x12x3x3 (no bias).

(a) Detector (simcode/detector/model.py: Detector, build; simcode/layers/blocks.py: ResBlock, norm_photons;
    simcode/detector/select.py: select_subpixel; simcode/emitter/model.py: detect_candidates; processing/inference_fast.py)
   - args of the checkpoints: ch = [48, 96, 160], stem_ch = 16, up = 1 (native output grid), arch 'detector'.
   - norm_photons: sign(y) log1p(|y| / 30) on the photons; the window mean uses the same scale.
   - stem: Conv2d(1, 16, 3) GELU Conv2d(16, 16, 3) GELU per frame (shared weights); pass 2: Conv2d(3, 16, 3) on
     (frame_t, cos_t, sin_t) (the pass-1 weights go into channel 0, the rest zero: Detector.load_pass1).
   - illumination maps (pattern/illumination.illum_maps): cos part rel (1 + m cos) - 1, sin part rel m sin, per frame.
   - cin: Conv2d(6*16 + 1 = 97, 48, 3); pass 2 + cin_il: Conv2d(12, 48, 3, bias=False), zero-init, added before the GELU.
   - e0: 2 ResBlock(48); d1: Conv2d(48, 96, 3, stride 2); e1: 2 ResBlock(96); d2: Conv2d(96, 160, 3, stride 2);
     e2: ResBlock(160, dil 1), ResBlock(160, dil 2); u1: Conv2d(160, 96, 1) -> bilinear x2 + s1 -> r1 ResBlock(96);
     u0: Conv2d(96, 48, 1) -> bilinear x2 + s0 -> r0 ResBlock(48); head: Conv2d(48, 48, 3) GELU Conv2d(48, 4, 1)
     (logit bias init -4). Input padded (replicate) to a multiple of 4.
   - ResBlock(c, d): x + c2(GELU(GroupNorm(8, c)(c1(GELU(x))))), c1 = 3x3 dilation d, c2 = 3x3.
   - outputs: logit; dxy = 1.5 tanh (px relative to the pixel centre); log_sig = -5 + 6 sigmoid.
   - selection: select_subpixel (greedy NMS on the points pixel + offset, radius 0.7 px, 7x7 neighbourhood, iters = 4:
     up to 4 suppression rounds, stopping early at the fixed point; top-k per image); inference_fast: k = 640 S^2 / 64^2,
     detect p_min 0.2 (THR0 / NBR_PMIN, the neighbour map), stored candidates p >= CAND_PMIN = 0.3
     (processing/common.py). Detector inference under bf16 autocast.

(b) EmitterNet (simcode/emitter/model.py: EmitterNet, nbr_map, nbr_patches, PSFCol, apply_to_candidates)
   - args: ch = 96, bf16 = 1, arch 'emitter_net'; patch PP = 21; T = 6.
   - input channels (forward): norm_photons(patch) [6], norm_photons(mean over frames) [1], lin = (patch - lvl) / 100 [6]
     with lvl = mean over frames of the per-frame border median b0_t, lin mean [1], cx, cy = (pixel - detector point) /
     10 px [2], nbrp [2] = (map of the other candidates p >= 0.2: p exp(-d^2 / (2 0.5^2)) on 7x7 px, own stamp removed;
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
   - out layout q = [dI 6 | log diag L 6 | off-diag L 15 | dx, dy 2 | log sx, sy 2 | bg 6 | z, lsz, b_x, b_y 4] = 41
     (the figure lists them in the order of the outputs at the right edge: dI, dx dy, log sx sy, z, lsz, b, L, bg).
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

Differences from the earlier network figure (si_network_structure_v2): the earlier network had per-frame 2-layer
U-Nets, three intensity-estimator U-Nets and a combiner U-Net with 64/128/256 features; the current one has a separate detector
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
    ap.add_argument('--draft', action='store_true', help='save even if layout_problems() finds problems (printed)')
    a = ap.parse_args()
    print(build(a.out, a.models, a.draft))
