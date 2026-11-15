"""
Main-text schematic of the SIMCODE v2 pipeline (fig_pipeline_v2): pure vector (matplotlib patches / text; the tiny frame and
illumination glyphs are small rasters), text kept as text in the SVG (svg.fonttype none) and embedded TrueType in the PDF.

    python analysis/paper/fig_pipeline_v2.py [--out /home/jc/simcode2/out/paper_figs_v2]

-> <out>/fig_pipeline_v2.{pdf,svg,png} + fig_pipeline_v2_caption.txt (draft caption + the code facts behind each panel).
Content = the current code (docs/METHOD.md, simcode/{detector,emitter}/model.py, processing/pattern_estimation.py,
solve/solve.py + solve_fast.py, solve/z_conditioning.py, pattern/illumination.py, table.py, processing/front_fast.py).
Panel A is a labelled placeholder for the preprint's optical-system panel.
"""
import argparse, os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 6.5})

W, H = 180.0, 171.0                     # mm
MM = 1 / 25.4
# colours: pass 1 blue, pass 2 orange, solve green, pattern estimation purple, training grey
C = dict(p1='#2f6db5', p1l='#e6eef9', p2='#d0701a', p2l='#fcefe1', sol='#2b8a50', soll='#e3f3e9', pat='#7a4aa3',
         patl='#f1eaf7', tr='#555555', trl='#f2f2f2', out='#222222', a0='#2f6db5', a1='#c23b3b')
FS = 6.4          # body text (pt)
FT = 7.2          # box titles
FL = 10           # panel letters


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W * MM, H * MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis('off')

    def box(self, x, y, w, h, fc='white', ec='k', lw=0.7, r=1.2, ls='-', z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                         ls=ls, zorder=z))

    def text(self, x, y, s, size=FS, ha='left', va='top', color='k', weight='normal', z=5, **kw):
        return self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, weight=weight, zorder=z, **kw)

    def lines(self, x, y, rows, size=FS, dy=None, **kw):
        """rows of text, top-aligned from y; dy = line pitch (mm)"""
        dy = dy or size * 0.3528 * 1.38
        for i, s in enumerate(rows):
            self.text(x, y + i * dy, s, size=size, **kw)
        return y + len(rows) * dy

    def arrow(self, x0, y0, x1, y1, color='k', lw=0.9, ls='-', ms=7, z=4, cs='arc3'):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                          ls=ls, zorder=z, shrinkA=0, shrinkB=0, connectionstyle=cs))

    def seg(self, xs, ys, color='k', lw=0.9, ls='-', z=4):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='butt')

    def letter(self, x, y, s):
        self.text(x, y, s, size=FL, weight='bold', va='top')

    def img(self, a, x, y, w, h, cmap='gray', vmin=None, vmax=None, z=3):
        self.ax.imshow(a, extent=(x, x + w, y + h, y), cmap=cmap, vmin=vmin, vmax=vmax, zorder=z, interpolation='nearest',
                       aspect='auto')
        self.ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='#444444', lw=0.35, zorder=z + 0.1))


def spot(n=11, x0=5.3, y0=4.7, s=1.25, A=1.0, bg=0.08, seed=0):
    yy, xx = np.mgrid[:n, :n]
    rng = np.random.default_rng(seed)
    g = A * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s)) + bg
    return g + 0.04 * rng.standard_normal((n, n))


def illum(n=40, ang=0.0, pitch=7.0, ph=0.0, part='cos'):
    yy, xx = np.mgrid[:n, :n]
    k = 2 * np.pi / pitch
    a = k * (np.cos(ang) * xx + np.sin(ang) * yy) - ph
    return np.cos(a) if part == 'cos' else np.sin(a)


# per-frame modulation of the example emitter (angle a = t // 3, phase step s = t % 3)
MOD = np.array([1 + 0.85 * np.cos(0.6 - 2 * np.pi * s / 3) for s in range(3)] +
               [1 + 0.85 * np.cos(2.3 - 2 * np.pi * s / 3) for s in range(3)]) / 2


def panel_A(cv):
    cv.letter(1, 1.5, 'A')
    cv.box(6, 3, 38, 36.5, fc='white', ec='#999999', lw=0.8, ls=(0, (3, 2)))
    cv.text(25, 15, 'Optical system', size=FT, ha='center', weight='bold', color='#777777')
    cv.lines(25, 20, ['(placeholder: the preprint\'s', 'optical-system panel,', 'DMD-SIMFLUX TIRF)'], ha='center',
             color='#777777')


def panel_B(cv):
    x0, y0 = 49, 3
    cv.letter(x0 - 4, 1.5, 'B')
    cv.text(x0, 2.2, 'Training: simulation only, one PSF-matched model per movie', size=FT, weight='bold', color=C['tr'])
    bw, bh, gap = 30.0, 30.0, 3.6
    items = [
        ('In-situ calibration v2', ['of the target movie:', 'vectorial PSF fitted', 'on the movie\'s spots', '(Zernikes, focus);',
                                     'sCMOS gain, offset and', 'read-noise maps;', 'fitted emitter heights']),
        ('Matched simulator', ['this PSF + camera,', 'SIMFLUX pattern with', 'estimation errors,', 'DNA-PAINT kinetics,',
                               'filaments, close pairs,', 'brightness jitter,', 'structured background']),
        ('Network training', ['pass 1: detector,', 'then EmitterNet;', 'pass 2: same networks', '+ illumination inputs',
                              '(zero-initialised),', 'generic patterns in', 'half of the FOVs']),
        ('PSF-matched model', ['4 networks + solve', 'calibrations (sim', 'gate scenes):', 'm1809 for 1809/2;',
                               'm25 for 2.5 nM/2', 'and 20 nM/2']),
    ]
    for i, (t, rows) in enumerate(items):
        x = x0 + i * (bw + gap); y = y0 + 6.5
        cv.box(x, y, bw, bh, fc=C['trl'], ec=C['tr'], lw=0.7)
        cv.text(x + 1.5, y + 1.6, t, size=FS + 0.3, weight='bold', color=C['tr'])
        cv.lines(x + 1.5, y + 5.6, rows)
        if i < 3:
            cv.arrow(x + bw + 0.3, y + bh / 2, x + bw + gap - 0.3, y + bh / 2, color=C['tr'])


def panel_C(cv, y0):
    cv.letter(1, y0, 'C')
    cv.text(6, y0 + 0.7, 'Input', size=FT, weight='bold')
    # stack of 6 frames (spot brightness modulated by the pattern)
    fx, fy, fs = 6.0, y0 + 6.0, 9.0
    for t in range(6):
        a = spot(A=MOD[t] * 1.2, seed=t)
        cv.img(a, fx + t * 2.3, fy + t * 2.3, fs, fs, vmin=0, vmax=1.15, z=3 + t * 0.2)
    cv.text(fx + 2.3 * 5 + fs + 1.0, fy + 2.3 * 5 + fs / 2, r'$t$', size=FS + 1, va='center')
    cv.lines(6, fy + 2.3 * 5 + fs + 2.0, ['6-frame window =', '2 angles × 3 phase steps'])
    # timeline: frames, a window every 3 frames
    ty = fy + 2.3 * 5 + fs + 10.5
    sq = 2.05
    for f in range(12):
        col = C['a0'] if (f % 6) < 3 else C['a1']
        cv.ax.add_patch(Rectangle((6 + f * sq, ty), sq * 0.86, sq * 0.86, fc=col, ec='none', alpha=0.75, zorder=3))
    for j, s0 in enumerate((0, 3, 6)):
        yb = ty + 3.0 + j * 1.5
        cv.seg([6 + s0 * sq, 6 + (s0 + 6) * sq - 0.3], [yb, yb], color='k', lw=0.8)
    cv.lines(6, ty + 8.4, ['a window every 3 frames'])
    cv.text(6 + 1.5 * sq, ty - 0.8, 'angle 1', size=FS - 0.6, color=C['a0'], va='bottom', ha='center')
    cv.text(6 + 10.5 * sq, ty - 0.8, 'angle 2', size=FS - 0.6, color=C['a1'], va='bottom', ha='center')


def bars(cv, x, y, w, h, vals, errs=None, col=None):
    n = len(vals); bw = w / n
    cv.seg([x, x + w], [y + h, y + h], color='#444444', lw=0.4)
    for i, v in enumerate(vals):
        c = col[i] if col is not None else 'k'
        cv.ax.add_patch(Rectangle((x + i * bw + bw * 0.15, y + h - v * h), bw * 0.7, v * h, fc=c, ec='none', alpha=0.8, zorder=3))
        if errs is not None:
            cv.seg([x + i * bw + bw / 2] * 2, [y + h - (v + errs[i]) * h, y + h - (v - errs[i]) * h], color='k', lw=0.45)


def covmat(cv, x, y, s, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, 6)) * 0.35 + np.eye(6) * 1.2
    S = A @ A.T; d = np.sqrt(np.diag(S)); R = S / np.outer(d, d)
    cm = mpl.colormaps['RdBu_r']
    c = s / 6
    for i in range(6):
        for j in range(6):
            cv.ax.add_patch(Rectangle((x + j * c, y + i * c), c, c, fc=cm(0.5 + 0.5 * R[i, j]), ec='white', lw=0.15, zorder=3))


def panel_D(cv, y0):
    x0, x1 = 41.0, 121.0
    cv.letter(x0 - 4.5, y0, 'D')
    cv.box(x0, y0, x1 - x0, 56, fc=C['p1l'], ec=C['p1'], lw=0.9)
    cv.text(x0 + 1.5, y0 + 1.4, 'Pass 1: pattern-free networks', size=FT, weight='bold', color=C['p1'])
    # detector
    dx, dy, dw, dh = x0 + 1.8, y0 + 7.5, 26.0, 22.0
    cv.box(dx, dy, dw, dh, fc='white', ec=C['p1'], lw=0.7)
    cv.text(dx + 1.2, dy + 1.2, 'Detector', size=FS + 0.3, weight='bold', color=C['p1'])
    cv.lines(dx + 1.2, dy + 4.9, ['U-Net on the window', '→ logit + sub-pixel', 'offset per pixel;', 'sub-pixel NMS + top-k;',
                                  r'keep $p \geq 0.3$'])
    # candidate patch
    px_, py_ = dx, dy + dh + 3.2
    for t in range(3):
        cv.img(spot(n=11, A=0.6 + MOD[t], seed=10 + t), px_ + t * 1.6, py_ + t * 1.6, 9.0, 9.0, vmin=0, vmax=1.3, z=3 + t * 0.2)
    cv.lines(px_, py_ + 13.0, ['21×21×6 patch', 'per candidate', '+ neighbour maps'], size=FS - 0.3)
    cv.arrow(dx + 6, dy + dh, dx + 6, py_ - 0.3, color=C['p1'])
    # EmitterNet
    ex, ey, ew, eh = dx + dw + 3.4, dy, 20.8, 44.0
    cv.box(ex, ey, ew, eh, fc='white', ec=C['p1'], lw=0.7)
    cv.text(ex + 1.2, ey + 1.2, 'EmitterNet', size=FS + 0.3, weight='bold', color=C['p1'])
    cv.lines(ex + 1.2, ey + 4.9, ['per-candidate', 'CNN; learned', 'linear read-out', r'$I_t=\mathbf{w}_t\cdot(\mathbf{y}_t-bg_t)$',
                                  '+ correction'], size=FS - 0.2)
    cv.text(ex + 1.2, ey + eh - 1.2, 'no pattern\ninput', size=FS - 0.3, va='bottom', color=C['p1'], style='italic')
    cv.arrow(px_ + 12.6, py_ + 6.5, ex - 0.3, py_ + 6.5, color=C['p1'])
    # outputs
    ox = ex + ew + 2.5
    cv.arrow(ex + ew + 0.2, ey + 20, ox + 0.2, ey + 20, color=C['p1'])
    oy = ey - 0.5
    cv.text(ox + 1, oy, r'$I_1\ldots I_6$', size=FS)
    bars(cv, ox + 1, oy + 3.0, 11.5, 7.0, MOD * 0.85, errs=0.09 + 0.05 * MOD, col=[C['a0']] * 3 + [C['a1']] * 3)
    covmat(cv, ox + 14.0, oy + 3.0, 7.0)
    cv.text(ox + 1, oy + 11.6, r'full 6×6 covariance', size=FS - 0.3)
    cv.text(ox + 1, oy + 14.4, r'(Cholesky $L$)', size=FS - 0.3)
    cv.text(ox + 1, oy + 18.6, r'$bg_1\ldots bg_6$', size=FS)
    cv.text(ox + 1, oy + 23.2, r'3D prior $(x,y,z)$', size=FS)
    cv.ax.add_patch(Ellipse((ox + 6.0, oy + 32.0), 8.0, 4.4, angle=-25, fc=C['p1'], alpha=0.25, ec=C['p1'], lw=0.5, zorder=3))
    cv.ax.add_patch(Ellipse((ox + 6.0, oy + 32.0), 3.0, 1.6, angle=-25, fc=C['p1'], alpha=0.5, ec='none', zorder=3))
    cv.lines(ox + 12.0, oy + 29.0, ['+ 3×3', 'covariance'], size=FS - 0.3)
    cv.lines(ox + 1, oy + 37.5, [r'$z$ | $x,y$: $z + \mathbf{b}\cdot\Delta\mathbf{r}$'], size=FS - 0.3)
    return dict(out_x=ox, ex=ex, ey=ey)


def panel_E(cv, y0):
    x0, x1 = 129.0, 179.0
    cv.letter(x0 - 4.5, y0, 'E')
    cv.box(x0, y0, x1 - x0, 56, fc=C['patl'], ec=C['pat'], lw=0.9)
    cv.text(x0 + 1.5, y0 + 1.4, 'Pattern estimation', size=FT, weight='bold', color=C['pat'])
    # glyph: relative intensity vs pattern phase, cosine fit
    gx, gy, gw, gh = x0 + 4.5, y0 + 7.0, 22.0, 13.0
    rng = np.random.default_rng(3)
    th = rng.uniform(0, 2 * np.pi, 70)
    yv = np.clip(1 + 0.8 * np.cos(th) + 0.22 * rng.standard_normal(70), -0.25, 2.25)
    X = gx + th / (2 * np.pi) * gw; Y = gy + gh - (yv + 0.3) / 2.6 * gh
    cv.ax.scatter(X, Y, s=0.9, color=C['pat'], alpha=0.6, lw=0, zorder=3)
    tt = np.linspace(0, 2 * np.pi, 100)
    cv.ax.plot(gx + tt / (2 * np.pi) * gw, gy + gh - (1 + 0.8 * np.cos(tt) + 0.3) / 2.6 * gh, color='k', lw=0.8, zorder=4)
    cv.seg([gx, gx + gw], [gy + gh, gy + gh], color='#444444', lw=0.4)
    cv.seg([gx, gx], [gy, gy + gh], color='#444444', lw=0.4)
    cv.text(gx + gw / 2, gy + gh + 0.6, r'$\mathbf{k}_a\cdot\mathbf{r}-\psi_s-\varepsilon_a(t)$', size=FS - 0.6, ha='center')
    cv.text(gx - 0.8, gy + gh / 2, r'$I_t/B$', size=FS - 0.6, ha='right', va='center', rotation=90)
    cv.text(gx + gw + 1.5, gy + 1.0, 'clean,\nisolated\nemitter-\nwindows', size=FS - 0.8, color=C['pat'])
    yy = cv.lines(x0 + 1.5, gy + gh + 5.0, [
        r'$I_t = B\,\mathrm{rel}_s\,[1+m_a\cos(\mathbf{k}_a\cdot\mathbf{r}-\psi_s-\varepsilon_a(t))]$'], size=FS - 0.5)
    cv.lines(x0 + 1.5, yy + 0.6, [
        'from the pass-1 network intensities',
        'and prior positions only:',
        r'$\mathbf{k}_a$: FFT, then DFT refinement',
        r'per-step phases $\psi_s$, rel. intensities',
        r'modulation depth $m_a$',
        r'phase drift $\varepsilon_a(t)$ per angle (shared)'], size=FS - 0.2)


def panel_F(cv, y0):
    x0, x1 = 5.0, 63.0
    cv.letter(1, y0, 'F')
    cv.box(x0, y0, x1 - x0, 54, fc=C['p2l'], ec=C['p2'], lw=0.9)
    cv.text(x0 + 1.5, y0 + 1.4, 'Pass 2: pattern-aware networks', size=FT, weight='bold', color=C['p2'])
    gx, gy, gs = x0 + 2.0, y0 + 7.0, 9.0
    for i, (ang, part) in enumerate([(0.15, 'cos'), (0.15, 'sin'), (1.72, 'cos'), (1.72, 'sin')]):
        cv.img(illum(ang=ang, ph=0.4 * i, part=part), gx + i * (gs + 1.3), gy, gs, gs, cmap='RdBu_r', vmin=-1.2, vmax=1.2)
    cv.text(gx, gy + gs + 0.8, r'illumination maps of the ESTIMATED pattern:', size=FS - 0.3)
    cv.text(gx, gy + gs + 3.6, 'cos and sin part per frame (12 channels)', size=FS - 0.3)
    cv.lines(gx, gy + gs + 8.3, [
        'same Detector + EmitterNet, with extra',
        'illumination inputs (zero-initialised from',
        'the pass-1 networks, then fine-tuned);',
        'trained on GENERIC patterns (pitch 1.8–4 px,',
        'any orientation / phases / frame order) in',
        'half of the training FOVs, SIMCODE pattern',
        'in the others; outputs as in D'], size=FS - 0.1)


def panel_G(cv, y0):
    x0, x1 = 71.0, 131.0
    cv.letter(x0 - 4.5, y0, 'G')
    cv.box(x0, y0, x1 - x0, 54, fc=C['soll'], ec=C['sol'], lw=0.9)
    cv.text(x0 + 1.5, y0 + 1.4, 'Explicit pattern solve (per row)', size=FT, weight='bold', color=C['sol'])
    # glyph: NLL along one pattern direction with fringe minima; prior centre + fringe-shifted starts
    gx, gy, gw, gh = x0 + 4.0, y0 + 6.5, 30.0, 12.0
    u = np.linspace(-1.6, 1.6, 300)
    nll = 1 - 0.9 * np.cos(2 * np.pi * u) ** 1 * 0.5 - 0.5 * np.cos(2 * np.pi * u) + 0.55 * u ** 2
    nll = (nll - nll.min()) / (nll.max() - nll.min())
    cv.ax.plot(gx + (u + 1.6) / 3.2 * gw, gy + 1 + nll * (gh - 2), color=C['sol'], lw=0.9, zorder=4)
    cv.seg([gx, gx + gw], [gy + gh, gy + gh], color='#444444', lw=0.4)
    for k, uu in enumerate((-1, 0, 1)):
        xx = gx + (uu + 1.6) / 3.2 * gw
        cv.ax.plot([xx], [gy + gh + 1.4], marker='^', ms=2.6, color='k' if uu == 0 else '#777777', zorder=5, clip_on=False)
    cv.text(gx + gw / 2, gy + gh + 2.6, 'starts: prior centre + 8 fringe shifts', size=FS - 0.6, ha='center')
    cv.text(gx + gw / 2, gy + gh + 5.2, r'($\pm1$ period per angle)', size=FS - 0.6, ha='center')
    cv.text(gx + gw + 1.5, gy + 0.5, 'NLL along\n' + r'$\mathbf{k}_a$', size=FS - 0.6)
    cv.lines(x0 + 1.5, gy + gh + 9.3, [
        r'LM fit of $(x, y, B\,[, t_\mathrm{on}, t_\mathrm{off}])$ to $I_t$, with',
        r'$\Sigma(\mu)=s\,\Sigma_\mathrm{net}+c^2\,\mathrm{diag}(\mu^2)+c_a^2\,\mu\mu^\mathsf{T}$',
        r'(fixed $c$, $c_a$ of the model calibration)',
        r'+ Gaussian prior term on $(x, y)$',
        r'on/off kinetics kept if $\Delta(-2\log L)>9.21$',
        r'output: fringe posterior mean, calibrated $\sigma$',
        r'(pass-1 rows and pass-2 rows alike)'], size=FS - 0.1)


def panel_H(cv, y0):
    x0, x1 = 139.0, 179.0
    cv.letter(x0 - 4.5, y0, 'H')
    # z from pass 1
    cv.box(x0, y0, x1 - x0, 21.5, fc=C['p1l'], ec=C['p1'], lw=0.9)
    cv.text(x0 + 1.5, y0 + 1.4, r'$z$ from pass 1', size=FT, weight='bold', color=C['p1'])
    cv.lines(x0 + 1.5, y0 + 5.4, [
        'each pass-2 row is linked to its',
        'pass-1 candidate (≤ 1 px); that',
        'candidate\'s 3D prior (D) is',
        r'conditioned on the final $x, y$:',
        r'$z=z_\mathrm{p}+\mathbf{b}\cdot(\mathbf{r}-\mathbf{r}_\mathrm{p})$'], size=FS - 0.3)
    # selection
    sy = y0 + 25.5
    cv.box(x0, sy, x1 - x0, 11.5, fc='white', ec=C['out'], lw=0.8)
    cv.text(x0 + 1.5, sy + 1.4, 'Selection', size=FT, weight='bold')
    cv.lines(x0 + 1.5, sy + 5.0, [r'$p \geq 0.63$ and calibrated', r'$\max(\sigma_x,\sigma_y)\leq 12$ nm'], size=FS - 0.2)
    cv.arrow((x0 + x1) / 2, y0 + 21.5, (x0 + x1) / 2, sy - 0.2, color=C['out'])
    # table
    ty = sy + 15.0
    cv.box(x0, ty, x1 - x0, 14.0, fc=C['out'], ec=C['out'], lw=0.8)
    cv.text((x0 + x1) / 2, ty + 2.3, 'Localization table', size=FT, weight='bold', ha='center', color='white')
    cv.text((x0 + x1) / 2, ty + 6.6, r'$x, y, z$, $\sigma_x, \sigma_y, \sigma_z$, photons,', size=FS - 0.3, ha='center',
            color='white')
    cv.text((x0 + x1) / 2, ty + 9.6, 'one row per emitter-window', size=FS - 0.3, ha='center', color='white')
    cv.arrow((x0 + x1) / 2, sy + 11.5, (x0 + x1) / 2, ty - 0.2, color=C['out'])


def build(out):
    cv = Canvas()
    panel_A(cv); panel_B(cv)
    # inference header
    yi = 45.0
    cv.seg([1, W - 1], [yi - 1.6, yi - 1.6], color='#bbbbbb', lw=0.5)
    cv.text(W - 1, yi - 1.0, 'Inference on the measured movie', size=FS, color='#777777', ha='right')
    r2 = yi + 4.5
    panel_C(cv, r2)
    d = panel_D(cv, r2)
    panel_E(cv, r2)
    r3 = r2 + 63.5
    panel_F(cv, r3); panel_G(cv, r3); panel_H(cv, r3)
    # flow arrows
    cv.arrow(36.5, r2 + 16, 41.0, r2 + 16, color='k')                         # C -> D
    cv.arrow(121.2, r2 + 28, 129.0, r2 + 28, color=C['p1'], lw=1.1)          # D -> E (I_t + prior)
    cv.text(125.1, r2 + 26.6, r'$I_t$, $\mathbf{r}$', size=FS - 1.0, ha='center', va='bottom', color=C['p1'])
    # estimated pattern bus: E -> F (illumination inputs) and E -> G (solve)
    by = r3 - 4.6
    cv.seg([154.0, 154.0], [r2 + 56, by], color=C['pat'], lw=1.1)
    cv.seg([34.0, 154.0], [by, by], color=C['pat'], lw=1.1)
    cv.arrow(34.0, by, 34.0, r3 - 0.2, color=C['pat'], lw=1.1)
    cv.arrow(101.0, by, 101.0, r3 - 0.2, color=C['pat'], lw=1.1)
    cv.text(60.0, by + 0.4, 'estimated pattern', size=FS - 0.3, color=C['pat'], va='top', style='italic')
    # pass-1 rows -> solve (pass-1 table); pass-2 rows -> solve; solve -> z / selection
    cv.arrow(63.0, r3 + 27, 71.0, r3 + 27, color=C['p2'], lw=1.1)
    cv.text(67.0, r3 + 25.6, 'rows', size=FS - 1.0, ha='center', va='bottom', color=C['p2'])
    cv.arrow(131.0, r3 + 10.5, 139.0, r3 + 10.5, color=C['sol'], lw=1.1)
    cv.arrow(131.0, r3 + 31.0, 139.0, r3 + 31.0, color=C['sol'], lw=1.1)
    os.makedirs(out, exist_ok=True)
    base = f'{out}/fig_pipeline_v2'
    for ext in ('pdf', 'svg'):
        cv.fig.savefig(f'{base}.{ext}')
    cv.fig.savefig(f'{base}.png', dpi=300)
    plt.close(cv.fig)
    with open(f'{base}_caption.txt', 'w') as f:
        f.write(CAPTION)
    return base


CAPTION = r"""fig_pipeline_v2: caption material (generated by analysis/paper/fig_pipeline_v2.py)

DRAFT CAPTION
Figure 1. SIMCODE v2 pipeline. (A) Optical system (DMD-SIMFLUX TIRF microscope; panel from the preprint). (B) Training.
All networks are trained only on simulated data from an experiment-matched simulator that uses the in-situ calibration of
the target movie (vectorial PSF fitted on the movie's own spots, sCMOS camera maps, fitted emitter heights). The result is
one PSF-matched model per movie: a detector and an emitter network for each of the two passes, plus their solve
calibrations. (C) Input. The movie is processed in 6-frame windows (2 pattern orientations x 3 phase steps), with a new
window every 3 frames. (D) Pass 1. A pattern-free detector proposes candidates. For each candidate, an emitter network
(EmitterNet) outputs the 6 per-frame intensities I_t with a full 6x6 covariance (Cholesky factor), the per-frame
background and a 3D position prior (x, y, z) with a 3x3 covariance. The intensities come from a learned linear read-out of
the pixels plus a learned correction. The networks never see the illumination pattern in pass 1. (E) Pattern estimation.
The pattern is estimated from the pass-1 intensities and prior positions of clean, isolated emitter-windows: wave vectors
k_a, per-step phases psi_s and relative intensities, modulation depth m_a, and a shared per-angle phase drift eps_a(t).
(F) Pass 2. The same detector and emitter network, with extra input channels for the illumination maps (cos and sin
parts per frame) of the estimated pattern. They are trained on generic two-orientation patterns, so they accept any
measured pattern. (G) Explicit pattern solve, the same for both passes. For each row, a Levenberg-Marquardt fit of
(x, y, B [, t_on, t_off]) to the network intensities under the covariance Sigma(mu) = s Sigma_net + c^2 diag(mu^2) +
c_a^2 mu mu^T, with a Gaussian prior from the network position. The fit starts from the prior centre and from 8
fringe-shifted starts (+-1 period per angle). The on/off kinetics model is kept only if it lowers -2 log L by more than 9.21.
The output is the fringe posterior mean with a calibrated uncertainty. (H) z is taken from pass 1: each pass-2 row is
linked to its pass-1 candidate, and that candidate's 3D prior is conditioned on the solved x, y. The final table keeps the
rows with detection probability p >= 0.63 and calibrated max(sigma_x, sigma_y) <= 12 nm.

FACTS BEHIND EACH PANEL (code = /home/jc/simcode2-paperfigs2, branch paperfigs2, from the main commit 4c3af7f)

A  Placeholder (dashed box) for the preprint's optical-system panel (stage/figures/fig_pipeline.pdf panel A). Kept so the
   main figure stays self-contained; if the optical system is shown elsewhere (e.g. the preprint is cited), drop A and
   widen B. The user pastes the panel.

B  Training (docs/METHOD.md "Training", models/README.md "How a model is made", simcode/models.json).
   - In-situ calibration v2 = models/insitu/calib_<movie>_psf_v2.json (2026-11-15; out/psf_v2): vectorial PSF (torch_psf
     InSituEstimator, Zernike Noll 5-21, focus 0.20 um), corrected camera (gain 0.4256 e/ADU, offset 103.9 ADU, offset /
     read-noise maps), fitted spot heights (emitter_z_quantiles_um, clipped to [0, 0.40] um) = training height distribution
     (simulate/acquisition.height_quantiles).
   - One PSF for both pattern angles: the v2 PSF is a single in-situ fit on spots of both angles (json 'psf_v2' note: the
     per-pattern-angle photoselection of the psf3d calibration is not carried over; it replaces calib_1809_2_psf3d.json
     (+ _avg)). Per-axis PSFs are not used for training (no per-axis claim in the figure).
   - The shipped checkpoints' args (torch.load(...)['args']['calib']) point to calib_1809_2_psf_v2.json (m1809),
     calib_2p5nM_m2_psf_v2.json (m25), calib_gauss1.3.json (g13, preprint-condition simulations only, not shown).
   - Simulator (simulate/acquisition.py; METHOD.md): calibrated PSF with LUT / Zernike / focus jitter and a halo, the
     SIMFLUX pattern with estimation errors (pattern/illumination.perturb, error model 'real'), DNA-PAINT kinetics,
     filaments, close pairs, per-frame / per-step / per-angle brightness variation, structured and fluctuating background,
     sCMOS camera maps, paper-matched photon distributions.
   - Training order (models/README.md recipe): detector 40k steps -> EmitterNet fine-tuned 18k from hd_sc5nb ->
     pass-2 detector (illumination inputs zero-initialised from the pass-1 detector) -> pass-2 EmitterNet (3 stages,
     zero-initialised from the pass-1 EmitterNet). m25's pass-2 detector: dedicated (SIMCODE pattern only) 40k, then
     40k + 40k on the 50/50 generic mix.
   - Generic patterns (simulate/acquisition.AcquisitionSim._generic; --p_std 0.5 = the share of FOVs that keep the SIMCODE
     pattern in the final stages): pitch log-uniform 1.8-4.0 px, orientations -40..40 / 50..130 deg (separation 60-120
     deg), random k sign, independent per-frame phases, frame order cycle / permutation / interleaved, depth 0.5-1.0,
     power 0.8-1.25.
   - Model per movie: m1809 -> 1809/2; m25 -> 2.5 nM/2 and 20 nM/2 (models.json; 20 nM/2 uses m25 with the 20 nM camera
     calibration). Solve calibrations (calib/calib_sc_hd_<set>_real.json, calib_p2L_<set>_real.json) fitted on the gate
     scenes rendered with the model's PSF (analysis/psfmatch/fit_cal_set.py); selection-aware sigma rescale
     sigu_p{1,2}_<set>.json.

C  Input (simcode/run.py: window_frames=6, window_step=3; processing/front.py: stride 3, absolute start % 3 == 0). Frame
   t of a window: angle a = t // 3, phase step s = t % 3. The glyph frames show the per-frame brightness of one emitter
   modulated by the pattern (schematic).

D  Pass 1 (simcode/detector/model.py, simcode/detector/select.py, simcode/emitter/model.py, processing/front_fast.py).
   - Detector: U-Net, per-pixel logit, sub-pixel offset (dx, dy in +-1.5 px) and its log-sigma; select_subpixel = greedy
     NMS on the sub-pixel points (radius 0.7 px), top-k = 640 per 64x64 px (front_fast: k_cand = 640 S^2 / 64^2);
     stored candidates p >= 0.3 (processing/common.CAND_PMIN). p = sigmoid(detector logit).
   - EmitterNet on a 21x21x6 patch per candidate; neighbour channels = splat of the other candidates (p >= 0.2) and the
     detector-probability map (Fig. SI network structure).
   - Read-out (EmitterNet.forward): w_t = w0 exp(m) / (w . g) + D_t (D_t: per-frame signed maps with the PSF-template
     component projected out, so w_t . g = 1), I_lin,t = sum_q w_tq (y_tq - bg_t), I_t = I_lin,t + c dI_t (learned
     correction); full 6x6 Cholesky L of the intensity covariance; bg_t; x, y = detector point + 0.5 tanh(.) px with
     sigma_x, sigma_y; z (px of the lateral pixel, relative to the calibration's focal plane), s_z|xy and b = dz/dx, dz/dy
     -> 3x3 covariance L_p = [[sx,0,0],[0,sy,0],[b_x sx, b_y sy, s_z|xy]] (x, y uncorrelated).

E  Pattern estimation (simcode/processing/pattern_estimation.py, estimate_pattern):
   model I_t = B rel_s (1 + m_a cos(k_a.r - psi_s - eps_a(t))); s = t % 6, a = s // 3.
   - clean emitter-windows: same emitter (< 1 px) also detected in windows w-6 and w+6 (on through the window), all I_j > 0,
     isolated (no other detection within 3 px), photons >= 300; positions averaged over the linked windows.
   - k_a: FFT of an 8x super-resolved render inside the pitch annulus, then L-BFGS refinement of the direct DFT power
     (incoherent over 3000-frame blocks).
   - psi_s, rel_s: weighted regression per step; eps_a(t): per 150-frame block, linear interpolation between knots, 3-block
     running mean, mean removed (into psi); m_a: cross-cycle estimator on photon-disjoint window pairs (w, w+6);
     steps iterated n_iter = 3 times. The prior bias prior_off is removed first (METHOD.md).
   - The in-situ calibration's pattern entry is never used (METHOD.md).

F  Pass 2 (detector/model.py illum=True, emitter/model.py illum=True, pattern/illumination.illum_maps,
   processing/pass2_front.py / front_fast.py).
   - Illumination maps per frame: cos part relint (1 + m cos(.)) - 1 and quadrature relint m sin(.) -> [2T = 12, H, W];
     detector: stem gets (frame_t, cos_t, sin_t) per frame + a fusion conv on the 12 maps; EmitterNet: a conv on the 12
     cropped illumination patches added to the input conv. Weights zero-initialised -> starts exactly at pass 1.
   - Trained with a realistic pattern-error model ('real': phase ~0.17 rad rms, angle 0.04 deg, pitch 1.5e-3, depth
     U(-0.10, +0.04), relint 2 % per frame + 3 % per angle) and generic patterns in half the FOVs (p_std 0.5).
   - Outputs as pass 1 (I_t + 6x6 Cholesky, bg_t, prior); the pass-2 z is NOT used (biased upward at low photons).

G  Explicit pattern solve (simcode/solve/solve.py: solve_emitters, fringe_posterior, pattern_solve_all, solve_inputs;
   simcode/solve/solve_fast.py = the pipeline's exact fast version; simcode/solve/pattern.py fringe_shifts).
   - Model mu_t = B a_t(t_on, t_off) illum_t(x, y); free (x, y, B), + (t_on, t_off) for the kinetic model; LM 30 iters,
     per-start box +-1.5 px (cal 'box').
   - Sigma(mu) = s Sigma_net + c^2 diag(mu^2) + c_a^2 mu mu^T (the last term within each angle's frames; mask A2);
     Sigma_net = L L^T, s = kap_I2 per photon bin (~0.79-0.85) + 1e-3 I; c = c_a = 0.10 (cmult, cmult_ang of
     calib_sc_hd_<set>_real.json / calib_p2L_<set>_real.json for m1809 and m25). The exact Gaussian NLL incl. log det
     Sigma(mu) is the cost. c is fixed from the model calibration (the automatic-c stage is analysis only since 2026-11-14,
     solve/autoc.py).
   - Prior term: (r - r0) / s_p, r0 = network x, y minus the calibrated constant bias prior_off, s_p = sigma_prior x kap_p.
   - Starts: fringe_shifts(nf=1) = (2 nf + 1)^2 = 9 starts = the prior centre + 8 fringe-shifted starts with k_a.dr =
     2 pi n_a, n_a in {-1, 0, 1}.
   - Kinetics: kept if q0 - q1 > thr = 9.21, where q = the solve cost = chi^2 incl. the prior and log det = -2 log L
     (fringe_posterior docstring). So the threshold is on -2 log L (= the 99 % point of chi^2 with 2 dof), i.e. an NLL
     decrease of 4.6. The figure writes Delta(-2 log L) > 9.21, as the code (docs/METHOD.md corrected to match, branch
     figfix).
   - Posterior over the deduplicated starts (dedup 0.15 px), weights exp(-(cost - cost_min) / 2); output r = posterior mean
     r_pm; sigma_cal = sqrt(f s_pm^2 + spread_pm^2) x sig_ph(photons), f = sig_a max(q/dof, 1)^sig_g; then the table's
     selection-aware sigma rescale (sigu_p{1,2}_<set>.json).
   - Pass-1 rows are solved too (pass-1 table locs_p1); the main output is the pass-2 table.

H  z and selection (simcode/solve/z_conditioning.py: link_pass1, z_given_xy, pass1_z_columns; simcode/table.py: select).
   - link_pass1: one-to-one, same window, detector points within LINK_R = 1.0 px, iterated mutual nearest neighbours.
   - z = z_prior + b.(r - r_prior), var = s_z|xy^2 + sum_i b_i^2 sigma_i^2 (pass-1 3x3 covariance). Rows without a partner:
     z = NaN (METHOD.md: 2 % / 0.7 % of the paper's selection on real 2.5 nM / 0.5 nM chunks).
   - The pattern carries no z information; z is mostly the prior for dim emitters (METHOD.md).
   - select(): p >= P_GOOD = 0.63, max(sx, sy) * px_nm <= SIGMA_MAX_NM = 12 nm on the calibrated (rescaled) sigma, inside
     the crop. One row per emitter-window (6-frame window).

Possible caption decisions for the user
   - Keep or drop placeholder A (see A).
   - Docs vs code (resolved): models/README.md (5404c3e) and docs/METHOD.md (figfix) now name the in-situ PSF v2
     (calib_1809_2_psf_v2.json / calib_2p5nM_m2_psf_v2.json; models/m1809/model.json 'psf'), as the checkpoints; METHOD.md
     no longer lists the PSF template as an EmitterNet input channel (it only shapes the read-out weights).
"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/jc/simcode2/out/paper_figs_v2')
    a = ap.parse_args()
    print(build(a.out))
