"""
SI figure of the two network architectures (si_network_structure_v3): (a) the detection network, (b) the emitter network
(per detected emitter); pass 1 and pass 2 share each architecture, the pass-2 additions are drawn in orange. Layer sizes
are those of the code (simcode/detector/model.py, simcode/emitter/model.py, simcode/layers/blocks.py); the widths and
parameter counts are read from the shipped m1809 checkpoints (CPU only) and checked against the numbers drawn in the
figure. Colours: grey = input, blue = learned layers, white (outlined) = fixed computation, green = output, orange = pass 2
only. After drawing, layout_problems() checks every text for font size >= 5.5 pt, text-text overlaps and text crossing a
box edge (the script fails if it finds one).

    python analysis/paper/fig_network_structure.py [--out /home/jc/simcode2/out/paper_figs_v2] [--models /home/jc/simcode2/models]

-> <out>/si_network_structure_v3.{pdf,svg,png} + si_network_structure_v3_caption.txt
"""
import argparse, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 6.0})

W, H = 180.0, 128.0
MM = 1 / 25.4
FS, FSS, FT, FL, FM = 6.0, 5.8, 7.0, 9.0, 8.0        # rows, dense rows, panel titles, panel letters, output symbols
MIN_PT = 5.5
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

    def arrow(self, x0, y0, x1, y1, color='k', lw=0.7, ls='-', ms=5.5, cs='arc3'):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                          ls=ls, zorder=4, shrinkA=0, shrinkB=0, connectionstyle=cs))

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
    return g + 0.05 * np.random.default_rng(seed).standard_normal((n, n))


def panel_a(cv):
    cv.text(1, 4.6, 'a', size=FL, weight='bold', va='baseline')
    cv.text(6, 4.6, 'Detection network', size=FT, weight='bold', va='baseline')
    # input window: frame stack, label under it; window mean above the stem
    for t in range(4):
        cv.img(spot(A=0.6 + 0.4 * np.cos(t), seed=t), 3 + t * 1.6, 14 + t * 1.6, 9, 9, vmin=0, vmax=1.2, z=3 + 0.1 * t)
    cv.lines(1.5, 30.6, ['6-frame window (H × W),', 'photons, log-scaled'], size=FSS)
    cv.layer(20.5, 7.0, 25.5, 7.6, 'window mean', ['log-scaled, 1 channel'], kind='inp')
    cv.seg([7.5, 7.5], [14, 10.8]); cv.arrow(7.5, 10.8, 20.5, 10.8)
    cv.layer(20.5, 17.0, 25.5, 13.0, 'per-frame stem', ['shared over the frames', 'conv3×3 1→16, GELU', 'conv3×3 16→16, GELU'])
    cv.arrow(16.8, 23.5, 20.5, 23.5)
    # input conv
    cv.layer(49, 7.0, 12.5, 23.0, 'input', ['conv3×3', '97→48', 'GELU'])
    cv.arrow(46, 10.8, 49, 10.8); cv.arrow(46, 23.5, 49, 23.5)
    # pass-2 additions: to the stem and to the input conv
    cv.box(2, 38.8, 60, 18.7, fc=C['p2'], ec=C['p2e'], ls=P2LS, name='pass 2 (a)')
    cv.text(3.2, 39.7, 'pass 2 only', size=FS + 0.2, weight='bold', color=C['p2e'])
    cv.lines(3.2, 43.0, ['per frame: expected relative illumination − 1',
                         'and its 90°-shifted part (12 maps of H × W)',
                         '→ per-frame stem: first conv 3→16 on',
                         '    (frame, its 2 maps)',
                         '→ input conv: + conv3×3 12→48'], size=FSS)
    cv.arrow(33, 38.8, 33, 30.0, color=C['p2e'], ls='--')
    cv.arrow(55, 38.8, 55, 30.0, color=C['p2e'], ls='--')
    # U-Net
    L0, L1, L2, hl, wl = 7.0, 21.0, 35.0, 12.6, 20.0
    cv.layer(65, L0, wl, hl, 'level 0', ['2 × ResBlock 48', 'H × W'])
    cv.layer(87, L1, wl, hl, 'level 1', ['2 × ResBlock 96', 'H/2 × W/2'])
    cv.layer(109, L2, wl, hl, 'level 2', ['2 × ResBlock 160', 'dilation 1, 2', 'H/4 × W/4'])
    cv.layer(131, L1, wl, hl, 'up 1', ['conv1×1 160→96', 'upsample ×2', 'ResBlock 96'])
    cv.layer(153.5, L0, 25.5, hl, 'up 0', ['conv1×1 96→48', 'upsample ×2', 'ResBlock 48'])
    cv.arrow(61.5, 12.0, 65, 12.0)
    cv.arrow(82.5, L0 + hl, 87, L1 + 4.6)
    cv.lines(76, L1 + 0.6, ['conv3×3 /2', '48→96'], size=FSS, ha='center')
    cv.arrow(104.5, L1 + hl, 109, L2 + 4.6)
    cv.lines(98, L2 + 0.6, ['conv3×3 /2', '96→160'], size=FSS, ha='center')
    cv.arrow(129, L2 + 3.0, 135, L1 + hl)
    cv.arrow(149.5, L1, 153.5, L0 + hl - 3.5)
    # additive skips: dashed into a + node on the edge of the up block
    for (xa, xb, yy) in ((85, 153.5, L0 + 2.6), (107, 131, L1 + 2.6)):
        cv.arrow(xa, yy, xb - 1.15, yy, color=C['skip'], ls='--', lw=0.6)
        cv.plus(xb, yy, ec=C['skip'])
    # head (learned), per-pixel outputs, candidate selection (fixed computation)
    cv.layer(153.5, 24.0, 25.5, 10.2, 'head', ['conv3×3 48→48, GELU', 'conv1×1 48→4'])
    cv.arrow(166.25, L0 + hl, 166.25, 24.0)
    cv.box(138, 37.4, 41, 12.6, fc=C['out'], ec=C['oute'], name='per-pixel outputs')
    cv.text(139.2, 38.3, 'per pixel (native grid)', size=FS + 0.2, weight='bold')
    cv.lines(139.2, 41.6, [r'detection probability $p$', 'sub-pixel offset (≤ 1.5 px)', '± its uncertainty'], size=FS)
    cv.arrow(166.25, 34.2, 166.25, 37.4, color=C['oute'])
    cv.layer(138, 52.0, 25.5, 7.4, 'candidate selection', ['NMS, r = 0.7 px'], kind='fix')
    cv.arrow(150.75, 50.0, 150.75, 52.0)
    cv.arrow(163.5, 55.7, 170, 55.7)
    cv.text(171, 55.7, 'to', size=FS, va='center')
    cv.text(174.2, 55.7, 'b', size=FS + 1.5, va='center', weight='bold')
    # colour legend (meaning stated in the caption too)
    items = [('inp', '-', 'input'), ('net', '-', 'learned layers'), ('fix', '-', 'fixed computation'),
             ('out', '-', 'output'), ('p2', P2LS, 'pass 2 only')]
    for i, (k, ls, s) in enumerate(items):
        x, y = (66 if i < 3 else 92), 46.5 + (i % 3) * 3.6
        cv.ax.add_patch(Rectangle((x, y), 3.4, 2.3, fc=C[k], ec=C[k + 'e'], lw=0.6, ls=ls, zorder=3))
        cv.text(x + 4.4, y + 1.15, s, size=FSS, va='center')


def panel_b(cv):
    b0 = 61.0
    cv.seg([1, W - 1], [b0 - 1.2, b0 - 1.2], color='#bbbbbb', lw=0.5)
    cv.text(1, b0 + 4.2, 'b', size=FL, weight='bold', va='baseline')
    cv.text(6, b0 + 4.2, 'Emitter network (per detected emitter)', size=FT, weight='bold', va='baseline')
    # input patch (from a)
    yi = b0 + 6.5
    cv.box(2, yi, 36, 38.0, fc=C['inp'], ec=C['inpe'], name='input patch')
    cv.text(3.2, yi + 0.9, 'from a: 21 × 21 patch', size=FS + 0.2, weight='bold')
    rows = [('6', 'frames, log-scaled'),
            ('1', 'window mean, log-scaled'),
            ('6', 'frames, linear (minus the'),
            ('', 'window background level)'),
            ('1', 'window mean, linear'),
            ('2', 'coordinate maps (relative to'),
            ('', 'the sub-pixel position)'),
            ('1', 'neighbour splat (the other'),
            ('', 'detected emitters)'),
            ('1', 'detection-probability map'),
            ('18', 'channels')]
    for i, (n, s) in enumerate(rows):
        yy = yi + 4.4 + i * 3.0
        cv.text(6.4, yy, n, size=FSS, ha='right', weight='bold' if n == '18' else 'normal')
        cv.text(7.4, yy, s, size=FSS, weight='bold' if n == '18' else 'normal')
    # pass-2 input
    yp = yi + 40.4
    cv.box(2, yp, 36, 18.8, fc=C['p2'], ec=C['p2e'], ls=P2LS, name='pass 2 (b)')
    cv.text(3.2, yp + 0.9, 'pass 2 only', size=FS + 0.2, weight='bold', color=C['p2e'])
    cv.lines(3.2, yp + 4.2, ['12 illumination patches: per', 'frame, expected relative',
                             'illumination − 1 and its', '90°-shifted part', '→ input conv: + conv3×3 12→96'], size=FSS)
    # input conv, trunk
    yc = yi + 19.0                                  # centre line of the input box
    cv.layer(40.5, yc - 6.5, 10.5, 13.0, 'input', ['conv3×3', '18→96', 'GELU'])
    cv.arrow(38, yc, 40.5, yc)
    cv.path([(38, yp + 9.4), (45.75, yp + 9.4), (45.75, yc + 6.5)], color=C['p2e'], ls='--')
    cv.layer(53.5, yc - 8.0, 14.0, 16.0, 'trunk', ['ResBlock 96', '× 4', 'dilation', '1, 2, 1, 2'])
    cv.arrow(51, yc, 53.5, yc)
    # branches (fan out from the trunk)
    bx, bw = 70.5, 28.5
    yw, yf, yg, ys = yi, yi + 12.0, yi + 24.0, yi + 40.0
    cv.layer(bx, yw, bw, 10.0, 'weight modulation', ['conv3×3 96→32, GELU', 'conv1×1 32→1'])
    cv.layer(bx, yf, bw, 10.0, 'per-frame changes', ['conv3×3 96→32, GELU', 'conv1×1 32→6'])
    cv.layer(bx, yg, bw, 13.0, 'global', ['conv3×3 /2 96→96, GELU', 'conv3×3 /2 96→96, GELU', 'avg-pool 4 × 4'])
    cv.seg([67.5, 68.25], [yc, yc]); cv.seg([68.25, 68.25], [yw + 5, yg + 6.5])
    for yy in (yw + 5, yf + 5, yg + 6.5):
        cv.arrow(68.25, yy, bx, yy)
    cv.layer(bx, ys, bw, 9.6, '8 scalars', ['border-median background', 'per frame, sub-pixel offset'], kind='inp')
    # inputs of the read-out: patch photons and the PSF (grey), read-out (fixed computation)
    rx, rw, ry = 101.5, 56, yi + 4.5
    cv.layer(rx, b0 + 0.6, 18.8, 7.4, 'patch photons', ['6 frames'], kind='inp')
    cv.layer(rx + 20, b0 + 0.6, rw - 20, 7.4, 'in-situ PSF', ['at the detected sub-pixel position'], kind='inp')
    cv.arrow(rx + 9.4, b0 + 8.0, rx + 9.4, ry); cv.arrow(rx + 20 + (rw - 20) / 2, b0 + 8.0, rx + 20 + (rw - 20) / 2, ry)
    cv.layer(rx, ry, rw, 10.0, 'linear read-out', ['weights: PSF × modulation + per-frame changes',
                                                   'per frame: Σ weights × (photons − background)'], kind='fix')
    cv.arrow(bx + bw, yw + 7.0, rx, yw + 7.0); cv.arrow(bx + bw, yf + 2.5, rx, yf + 2.5)
    # MLP and its 41 outputs
    mw = 20
    cv.layer(rx, yg, mw, 13.0, 'MLP', ['1550→256, GELU', '256→256, GELU', '256→41'])
    cv.arrow(bx + bw, yg + 6.5, rx, yg + 6.5)
    cv.path([(bx + bw, ys + 4.8), (rx + 10, ys + 4.8), (rx + 10, yg + 13.0)])
    lx, ly = rx + mw + 3.5, yg - 3.4
    cv.arrow(rx + mw, yg + 6.5, lx - 0.6, yg + 6.5)
    cv.text(lx, ly, '41 outputs', size=FS + 0.2, weight='bold')
    # (count, lines, gap before the row in mm): the gaps keep the output tags apart
    out = [('6', ['intensity corrections'], 0), ('6 + 15', ['Cholesky factor', '(diagonal, off-diagonal)'], 0.35),
           ('2', ['position offset'], 2.2), ('2', ['position uncertainties'], 0.35),
           ('4', [r'$z$ and its uncertainty', r'given $x, y$; 2 $z$ slopes'], 0.35), ('6', ['background corrections'], 4.0)]
    yy, ymid = ly + 3.3, []
    for n, ss, gap in out:
        yy += gap
        cv.text(lx + 6.2, yy, n, size=FSS, ha='right')
        cv.lines(lx + 7.0, yy, ss, size=FSS)
        ymid.append((yy, yy + len(ss) * lh(FSS)))
        yy += len(ss) * lh(FSS)
    # combination nodes and output tags at the right edge
    xe, xn, tx, tw, th = lx + 33.0, 161.4, 164.4, 14.6, 9.6
    def tag(yc_, sym, word):
        cv.box(tx, yc_ - th / 2, tw, th, fc=C['out'], ec=C['oute'], name='tag ' + word)
        cv.text(tx + tw / 2, yc_ - 0.7, sym, size=FM, ha='center', va='baseline')
        cv.text(tx + tw / 2, yc_ + 1.1, word, size=FSS, ha='center', va='top')
    mid = [(a + b) / 2 - 0.35 for a, b in ymid]
    # intensities = read-out + correction
    yI = ry + 5.0
    cv.plus(xn, yI); cv.arrow(rx + rw, yI, xn - 1.15, yI)
    cv.seg([xe, xn], [mid[0], mid[0]]); cv.arrow(xn, mid[0], xn, yI + 1.15)
    cv.arrow(xn + 1.15, yI, tx, yI); tag(yI, r'$\hat{\mu}_{N_{i,j}}$', 'intensities')
    # uncertainties of the intensities
    cv.arrow(xe, mid[1], tx, mid[1]); tag(mid[1], r'$\hat{\sigma}_{N_{i,j}}$', 'uncertainties')
    # position: offset, uncertainties, z (bracket)
    y2, y3 = ymid[2][0] + 0.2, ymid[4][1] - 0.6
    cv.seg([xe - 1.0, xe, xe, xe - 1.0], [y2, y2, y3, y3])
    yP = (y2 + y3) / 2
    cv.arrow(xe, yP, tx, yP); tag(yP, r'$\hat{\theta}_{dl}\pm\sigma_{dl}$', 'position')
    # backgrounds = border median + correction
    yB = mid[5]
    cv.plus(xn, yB); cv.arrow(xe, yB, xn - 1.15, yB)
    yl = yB + th / 2 + 2.2
    cv.seg([bx + bw / 2, bx + bw / 2, xn], [ys + 9.6, yl, yl]); cv.arrow(xn, yl, xn, yB + 1.15)
    cv.text(rx + 10, yl - 0.4, 'border-median background', size=FSS, ha='center', va='bottom')
    cv.arrow(xn + 1.15, yB, tx, yB); tag(yB, r'$\hat{\mu}_{b_{i,j}}$', 'backgrounds')
    return max(yl, yp + 18.8)


def layout_problems(cv, tol_mm=0.45):
    """font sizes < MIN_PT, overlapping texts, texts that cross a box edge (not fully inside or outside of it)"""
    r = cv.fig.canvas.get_renderer()
    px = cv.fig.dpi * MM
    T = [(t, t.get_window_extent(r)) for t in cv.ax.texts if t.get_text().strip()]
    B = [(n, p.get_window_extent(r)) for p, n in cv.boxes]
    out = []
    for t, _ in T:
        if t.get_fontsize() < MIN_PT - 1e-6: out.append(f'font {t.get_fontsize()} pt: {t.get_text()!r}')
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T[i][1], T[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.3 * px and oy > 0.3 * px:
                out.append(f'texts overlap: {T[i][0].get_text()!r} / {T[j][0].get_text()!r}')
    m = tol_mm * px
    for t, a in T:
        for n, b in B:
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox <= 0 or oy <= 0: continue
            inside = a.x0 >= b.x0 + m and a.x1 <= b.x1 - m and a.y0 >= b.y0 + m and a.y1 <= b.y1 - m
            if not inside: out.append(f'text crosses box {n!r}: {t.get_text()!r}')
    for t, a in T:
        if a.x0 < 0 or a.x1 > cv.fig.bbox.x1 or a.y0 < 0 or a.y1 > cv.fig.bbox.y1:
            out.append(f'text outside the figure: {t.get_text()!r}')
    return out


def build(out, models, draft=False):
    F = checkpoint_facts(models)
    N = check(F)
    cv = Canvas()
    panel_a(cv); yb = panel_b(cv)
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
Figure S?. Network architectures. Colours: grey, inputs; blue, learned layers; white, fixed computations; green, outputs;
orange, pass 2 only. Pass 1 and pass 2 use the same two architectures. The pass-2 networks add input channels for the
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
p >= 0.3 are kept. Pass 2: per frame, the expected relative illumination minus 1 and its 90-degree-shifted part (12 maps);
the stem's first convolution takes each frame with its 2 maps (3 -> 16 channels instead of 1 -> 16), and a 3x3
convolution of the 12 maps (12 -> 48, no bias) is added to the input convolution before its GELU. The detection network
has {det:,} parameters ({p2det:,} in pass 2).
(b) Emitter network, run on a 21 x 21 px patch around each detected emitter. Inputs: the frames and their window mean,
log-scaled as in (a) and linear (photons minus the window background level, the mean over the frames of the border-median
background); coordinate maps (x and y relative to the detected sub-pixel position, divided by 10 px); and two neighbour
channels: Gaussian stamps (sigma 0.5 px, height p) at the other detected emitters (own stamp removed) and the
detection-probability map. A trunk of 4 residual blocks (96 channels, dilations 1, 2, 1, 2, full 21 x 21 resolution)
feeds three branches: a weight modulation, per-frame weight changes and a pooled global feature (96 x 4 x 4 = 1536). The
MLP takes the global feature and 8 scalars, the border-median background of each frame and the sub-pixel offset (plus 6
constant zero inputs; 1550 inputs in total), and gives 41 outputs. The intensity of each frame is a linear read-out: a
weighted sum of the patch photons minus the background mu_b_ij, with weights given by the in-situ PSF at the detected
sub-pixel position (the calibrated PSF evaluated at z = 50 nm) times the weight modulation, scaled to unit response to the
PSF, plus the per-frame changes (which have zero response to the PSF). The network outputs, per detected emitter: the 6
intensities mu_N_ij (angle i, phase step j) = read-out + learned correction (+), with their uncertainties sigma_N_ij
(the network also predicts their correlations, as a Cholesky factor, which the fit uses); the 6 backgrounds mu_b_ij =
border median + learned correction (+); and the position theta_dl = (x, y, z) +- sigma_dl, with x, y the detected
sub-pixel position plus an offset of at most 0.5 px, and sigma_dl the position uncertainties of x and y and the
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
    simcode/detector/select.py: select_subpixel; simcode/emitter/model.py: detect_candidates; processing/front_fast.py)
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
     up to 4 suppression rounds, stopping early at the fixed point; top-k per image); front_fast: k = 640 S^2 / 64^2,
     detect p_min 0.2 (THR0 / NBR_PMIN, the neighbour splat), stored candidates p >= CAND_PMIN = 0.3
     (processing/common.py). Detector inference under bf16 autocast.

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
   - out layout q = [dI 6 | log diag L 6 | off-diag L 15 | dx, dy 2 | log sx, sy 2 | bg 6 | z, lsz, b_x, b_y 4] = 41
     (the figure's list puts the 6 background corrections last, next to their output).
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
