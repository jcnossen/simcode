"""
Main-text Fig. 1 of the SIMCODE paper: the pipeline as a row of small icons and real mini-images, in the style of the
preprint's Fig. 1 (stage/figures/fig_pipeline.pdf), with few words; every explanation is in the caption file.

    PYTHONPATH=. python analysis/paper/fig_pipeline_v2.py [--out /home/jc/simcode2/out/paper_figs_v2]

-> <out>/fig_pipeline_v2.{pdf,svg,png} + fig_pipeline_v2_caption.txt.

Same page size as the preprint figure (644.8 x 474.5 pt + the extra height below), so panel A is the preprint's
optical-system drawing pasted unchanged (vector, clipped from fig_pipeline.pdf with pypdf) at scale 1, and the fonts have the
preprint's sizes (Liberation Sans = Arial metrics, mathtext DejaVu as the preprint's symbols).

Data of the mini-images (analysis/paper/fig_pipeline_v2_prep.py -> out/paper_figs_v2/cache_fig1/fig1_data.npz, CPU):
real 6-frame window of the 1809/2 movie (photo-electrons) and the m1809 pass-1 networks run on it (detection probability
map, one emitter's patch, intensities +- uncertainties, backgrounds, position); the estimated pattern of the 1809_2 run of
2026-11-15 (stripes, phase drift) and its clean emitter-windows (normalized intensity vs pattern phase); a simulated training
scene of the m1809 simulator with generic patterns; the SIMCODE zoom render of Fig. 3 (panel I).
"""
import argparse, os, subprocess, shutil
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse, Polygon, FancyBboxPatch

mpl.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8, 'axes.linewidth': 0.6})

PREPRINT = '/home/jc/paperreview/stage/figures/fig_pipeline.pdf'
CACHE = '/home/jc/simcode2/out/paper_figs_v2/cache_fig1/fig1_data.npz'
W, H = 644.83137, 462.0                   # pt (preprint page width; preprint height 474.5)
TOP = 5.0                                 # top margin (pt): the preprint's headings touch its page edge
A_BOX = (0.0, 0.0, 308.0, 250.0)          # panel A in the preprint page (x0, y0 from the top, w, h), pasted at the same place
FH, FL, FS, FX, FXS = 12.0, 17.0, 8.0, 7.5, 7.0      # heading, panel letter, subtitle, label, small label (pt)
COL = dict(det='#9ec5ef', em='#f2a65a', det2='#3d7fc4', em2='#d9822b', a1='#3b75af', a2='#c44e52', grey='#666666')
VIR = plt.get_cmap('viridis')


# ------------------------------------------------------------------------------------------------ canvas helpers
class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 72, H / 72))
        self.ov = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ov.set_xlim(0, W); self.ov.set_ylim(H - TOP, -TOP); self.ov.axis('off'); self.ov.patch.set_alpha(0)

    def axes(self, x, y, w, h, z=2):
        a = self.fig.add_axes([x / W, 1 - (y + TOP + h) / H, w / W, h / H], zorder=z)
        return a

    def img(self, im, x, y, w, h, cmap='viridis', vmin=None, vmax=None, ec='#888888', lw=0.6, z=2, interp='nearest'):
        a = self.axes(x, y, w, h, z)
        a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp, aspect='auto')
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(ec); s.set_linewidth(lw)
            if ec is None: s.set_visible(False)
        return a

    def stack(self, ims, x, y, s, dx, dy, **kw):
        """images drawn back to front: ims[0] at the back (top-left), ims[-1] in front (x + n dx, y + n dy)"""
        n = len(ims)
        for k, im in enumerate(ims):
            self.img(im, x + k * dx, y + k * dy, s, s, z=2 + k * 0.01, **kw)

    def text(self, x, y, s, size=FX, ha='left', va='top', color='k', **kw):
        return self.ov.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, **kw)

    def letter(self, x, y, s, title=None, title2=None):
        self.text(x, y, s, size=FL, va='top')
        if title:
            self.text(x + 21, y + 3.5, title, size=FS, va='top')
        if title2:
            self.text(x + 21, y + 13.5, title2, size=FS, va='top')

    def arrow(self, x0, y0, x1, y1, cs='arc3', lw=0.8, color='k', ms=7, ls='-'):
        self.ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                                          connectionstyle=cs, shrinkA=0, shrinkB=0, ls=ls, zorder=20))

    def cube(self, x, y, w, h, d, fc, ec=None, front_img=None):
        """box drawn as in the preprint (front face, top, right side; depth d to the upper right)"""
        c = mpl.colors.to_rgb(fc)
        top = tuple(min(1, v * 1.08 + 0.06) for v in c); side = tuple(v * 0.82 for v in c)
        ec = ec or tuple(v * 0.65 for v in c)
        o = self.ov
        o.add_patch(Polygon([(x, y), (x + d, y - d * 0.6), (x + w + d, y - d * 0.6), (x + w, y)], fc=top, ec=ec, lw=0.5, zorder=12))
        o.add_patch(Polygon([(x + w, y), (x + w + d, y - d * 0.6), (x + w + d, y + h - d * 0.6), (x + w, y + h)], fc=side, ec=ec,
                            lw=0.5, zorder=12))
        o.add_patch(Rectangle((x, y), w, h, fc=c, ec=ec, lw=0.5, zorder=12))
        if front_img is not None:
            a = self.axes(x + 2, y + 2, w - 4, h - 4, z=11)
            a.imshow(front_img, cmap='viridis', interpolation='bilinear', aspect='auto', alpha=0.85)
            a.axis('off')
            o.add_patch(Rectangle((x, y), w, h, fc='none', ec=ec, lw=0.5, zorder=13))


# ------------------------------------------------------------------------------------------------ data helpers
def illum_tile(kvec, psi, m, n=40, x0=0.0, y0=0.0, os_=4):
    """illumination 1 + m cos(k . r - psi) on an n x n px tile, sampled os_ times per pixel (no aliasing of the 2 px pitch)"""
    yy, xx = (np.mgrid[0:n * os_, 0:n * os_].astype(float) + 0.5) / os_ - 0.5
    return 1 + m * np.cos(kvec[0] * (xx + x0) + kvec[1] * (yy + y0) - psi)


def crop(a, cx, cy, r):
    cy0, cx0 = int(round(cy)) - r, int(round(cx)) - r
    cy0 = min(max(cy0, 0), a.shape[-2] - 2 * r); cx0 = min(max(cx0, 0), a.shape[-1] - 2 * r)
    return a[..., cy0:cy0 + 2 * r, cx0:cx0 + 2 * r], cx0, cy0


def drift_at(D, frames, a):
    kf = float(D['pat_knot']); dphi = D['pat_dphi']
    tt = frames / kf; i0 = np.clip(np.floor(tt).astype(int), 0, dphi.shape[1] - 2); w = np.clip(tt - i0, 0, 1)
    return dphi[a, i0] * (1 - w) + dphi[a, i0 + 1] * w


# ------------------------------------------------------------------------------------------------ panels
def training(cv, D):
    x0 = 317.0
    cv.text(x0, -2 + 1.0, 'Network training', size=FH)
    # B: simulated ground truth (3 FOVs), generic patterns, in-situ PSF
    cv.letter(319, 21, 'B', 'Simulate ground truth')
    gts = []
    for f in range(3):
        g = np.ones((40, 40, 3))
        gts.append(g)
    yB = 46
    for f in range(3):
        a = cv.axes(330 + f * 7, yB + f * 7, 46, 46, z=2 + f * 0.01)
        a.set_facecolor('white'); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_edgecolor('#444444'); s.set_linewidth(0.6)
        k = D['sim_fov'] == f
        a.scatter(D['sim_x'][k], D['sim_y'][k], s=1.2, c='k', lw=0)
        a.set_xlim(0, 40); a.set_ylim(40, 0)
    # generic patterns (the simulated FOVs' illumination, one frame each)
    # generic two-orientation patterns of the pass-2 training (ranges of simulate/acquisition.GENERIC: pitch 1.8-4 px,
    # orientation of angle 1 within +-40 deg of x; drawn analytically, oversampled)
    rng = np.random.default_rng(4)
    gen = []
    for f in range(3):
        pit = np.exp(rng.uniform(np.log(2.4), np.log(4.0))); th = np.radians(rng.uniform(-40, 40) + 90 * (f % 2))
        gen.append(illum_tile(2 * np.pi / pit * np.array([np.cos(th), np.sin(th)]), rng.uniform(0, 6.3), rng.uniform(0.5, 1.0), n=30))
    cv.stack(gen, 398, yB, 46, 7, 7, cmap='viridis', vmin=0, vmax=2, ec='#444444', interp='bilinear')
    cv.text(355, 113, 'emitters', size=FXS, ha='center', color=COL['grey'])
    cv.text(421, 113, 'generic patterns', size=FXS, ha='center', color=COL['grey'])
    cv.text(330, 124, 'Filaments and close emitter pairs', size=FX)
    cv.text(330, 133.5, 'In-situ PSF and camera of the movie', size=FX)
    # C: simulated 6-frame windows
    cv.letter(459, 21, 'C', 'Simulate 6-frame windows')
    fr = D['sim_frames'][1]
    cv.stack(list(fr), 488, 42, 54, 6, 6, cmap='viridis', vmin=np.median(fr), vmax=np.percentile(fr, 99.7), ec='#999999', lw=0.9)
    cv.arrow(452, 70, 484, 74, cs='arc3,rad=-0.25')
    # D: networks, outputs, loss
    cv.letter(429, 142, 'D', 'Pass through networks')
    cx, cy = 572, 172
    cv.cube(cx, cy, 24, 20, 9, COL['det'])
    cv.cube(cx, cy + 40, 24, 20, 9, COL['em'])
    cv.arrow(cx + 12, cy + 21, cx + 12, cy + 33, lw=0.7)
    # pass-2 inputs: illumination maps
    pt = illum_tile([1.1, 0.35], 0.0, 0.9, n=20)
    cv.img(pt, 618, 185, 17, 17, cmap='viridis', ec='#444444', interp='bilinear')
    cv.text(626.5, 205, 'pass 2', size=FXS, ha='center', color=COL['grey'])
    cv.arrow(617, 193, cx + 34, cy + 10, lw=0.6, color=COL['grey'], ms=5)
    cv.arrow(617, 196, cx + 34, cy + 47, lw=0.6, color=COL['grey'], ms=5, cs='arc3,rad=-0.2')
    # frames -> networks (curved from the stack)
    cv.arrow(600, 112, cx + 26, cy - 8, cs='arc3,rad=-0.35')
    # outputs
    ox = 500
    cv.text(ox, 167, r'$p$', size=9)
    cv.text(ox, 178, r'$\hat{\theta}_{dl}\pm\sigma_{dl}$', size=9)
    cv.text(ox, 191, r'$\hat{\mu}_{b}$', size=9)
    cv.text(ox, 217, r'$\hat{\mu}_{N}\pm\hat{\sigma}_{N}$', size=9)
    cv.arrow(cx - 4, cy + 10, 556, cy + 10, lw=0.7)
    cv.arrow(cx - 4, cy + 50, 556, cy + 50, lw=0.7)
    cv.ov.plot([557, 557], [cy + 10, cy + 10], lw=0)
    cv.arrow(496, 185, 452, 196, cs='arc3,rad=0.25')
    cv.arrow(496, 222, 452, 207, cs='arc3,rad=-0.25')
    cv.text(330, 194, 'Compute loss', size=FS)
    cv.text(330, 205, 'and backpropagate', size=FS)
    cv.arrow(375, 190, 375, 142)


def inference(cv, D):
    y0 = 254.0
    cv.text(-1, y0, 'Inference', size=FH)
    yl, yc = 276.0, 306.0                      # letters, top of the content
    k = int(D['pick']); ex, ey = float(D['c_x'][k]), float(D['c_y'][k])
    win = D['win']
    # E: input window (real frames, photons)
    cv.letter(0, yl, 'E', 'Input: 6-frame', 'window (photons)')
    wc, cx0, cy0 = crop(win, ex, ey, 12)
    lo, hi = np.median(wc), 0.8 * wc.max()          # background dark, spots bright
    cv.stack(list(wc), 2, yc - 2, 57, 4.6, 4.6, cmap='viridis', vmin=lo, vmax=hi, ec='#999999', lw=0.9)
    # angle / step labels under the stack
    for t in range(6):
        cv.ov.add_patch(Rectangle((5 + t * 9.5, 391), 8, 4, fc=COL['a1'] if t < 3 else COL['a2'], ec='none', zorder=12))
    cv.text(5, 398, r'angle $i$ = 1, 2', size=FXS, color=COL['grey'])
    cv.text(5, 406, r'step $j$ = 1, 2, 3', size=FXS, color=COL['grey'])
    cv.arrow(86, 340, 102, 340)
    # F: pass-1 networks
    cv.letter(98, yl, 'F', 'Pass 1:', 'no pattern')
    fx = 112
    cv.cube(fx, 318, 28, 24, 10, COL['det'])
    cv.text(fx + 14, 346, 'detect', size=FXS, ha='center', color=COL['grey'])
    patch = D['patch']
    cv.stack([patch[t] for t in range(0, 6, 2)], fx + 2, 357, 18, 3, 3, cmap='viridis', vmin=np.median(patch), vmax=0.8 * patch.max(),
             ec='#999999', lw=0.5)
    cv.arrow(fx + 14, 354, fx + 14, 356.5, lw=0.6, ms=4)
    cv.cube(fx, 390, 28, 24, 10, COL['em'])
    cv.text(fx + 14, 418, 'estimate', size=FXS, ha='center', color=COL['grey'])
    cv.arrow(fx + 14, 382, fx + 14, 388, lw=0.6, ms=4)
    # G: network outputs (box)
    gx0, gx1 = 168, 330
    cv.ov.add_patch(Rectangle((gx0, yc - 6), gx1 - gx0, 125, fc='none', ec='#9a9a9a', lw=1.2, zorder=1))
    cv.letter(gx0 - 4, yl, 'G', 'Network outputs')
    cv.arrow(fx + 40, 332, gx0 - 2, 332)
    cv.arrow(fx + 40, 402, gx0 - 2, 402)
    pm, _, _ = crop(D['pmap'], ex, ey, 12)
    cv.text(gx0 + 6, yc + 2, r'$p$', size=10)
    cv.img(pm, gx0 + 18, yc, 46, 46, cmap='viridis', vmin=0, vmax=1, ec='#666666')
    # the emitter's six intensities with uncertainties
    I, sI, bg = D['c_I'][k], D['c_sI'][k], D['c_bg'][k]
    a = cv.axes(gx0 + 82, yc + 10, 70, 38)
    xs = np.arange(6)
    a.bar(xs, I, width=0.7, color=[COL['a1']] * 3 + [COL['a2']] * 3)
    a.errorbar(xs, I, yerr=sI, fmt='none', ecolor='k', elinewidth=0.6, capsize=1.2, capthick=0.6)
    a.set_xlim(-0.6, 5.6); a.set_ylim(0, max(I + sI) * 1.08)
    a.set_xticks([]); a.set_yticks([]); [a.spines[s].set_visible(False) for s in ('top', 'right', 'left')]
    cv.text(gx0 + 117, yc - 2, r'$\hat{\mu}_{N}\pm\hat{\sigma}_{N}$', size=9, ha='center')
    # backgrounds
    a = cv.axes(gx0 + 82, yc + 72, 70, 20)
    a.bar(xs, bg, width=0.7, color='#9a9a9a')
    a.set_xlim(-0.6, 5.6); a.set_ylim(0, bg.max() * 1.15)
    a.set_xticks([]); a.set_yticks([]); [a.spines[s].set_visible(False) for s in ('top', 'right', 'left')]
    cv.text(gx0 + 117, yc + 58, r'$\hat{\mu}_{b}$', size=9, ha='center')
    # position (x, y) +- sigma and z +- sigma_z glyph
    cv.text(gx0 + 6, yc + 58, r'$\hat{\theta}_{dl}$', size=9)
    gxc, gyc = gx0 + 40, yc + 85
    cv.ov.add_patch(Ellipse((gxc, gyc), 26, 20, fc='#cfe0f3', ec='#3d7fc4', lw=0.6, zorder=12))
    cv.ov.add_patch(Ellipse((gxc, gyc), 2.6, 2.6, fc='#1f4e8c', ec='none', zorder=13))
    cv.ov.plot([gxc + 22, gxc + 22], [gyc - 12, gyc + 12], color='#3d7fc4', lw=0.9, zorder=12)
    cv.ov.plot([gxc + 19.5, gxc + 24.5], [gyc - 12, gyc - 12], color='#3d7fc4', lw=0.9, zorder=12)
    cv.ov.plot([gxc + 19.5, gxc + 24.5], [gyc + 12, gyc + 12], color='#3d7fc4', lw=0.9, zorder=12)
    cv.ov.plot([gxc + 22], [gyc], 'o', color='#1f4e8c', ms=2.2, zorder=13)
    cv.text(gxc, gyc + 14, r'$x, y$', size=FXS, ha='center', color=COL['grey'])
    cv.text(gxc + 26, gyc - 3, r'$z$', size=FXS, color=COL['grey'])
    # H: pattern estimation
    hx = 340
    cv.letter(hx - 2, yl, 'H', 'Estimate the', 'pattern')
    cv.arrow(gx1 + 1, 345, hx + 4, 345)
    kv, psi, m = D['pat_k'], D['pat_phi'], D['pat_m']
    tiles = [illum_tile(kv[s // 3], psi[s], m[s], n=14, os_=12) for s in range(6)]
    cv.stack(tiles, hx + 4, yc - 2, 44, 4.2, 4.2, cmap='viridis', vmin=0, vmax=2, ec='#555555', interp='bilinear')
    # normalized intensity vs pattern phase (clean emitter-windows of the run, angle 1)
    a = cv.axes(hx + 2, yc + 70, 76, 34)
    w, x, y, I6 = D['cl_w'], D['cl_x'], D['cl_y'], D['cl_I']
    ph, val = [], []
    for t in range(6):
        f = w + t; s = f % 6
        sel = s // 3 == 0
        eps = drift_at(D, f[sel].astype(float), 0)
        arg = kv[0, 0] * x[sel] + kv[0, 1] * y[sel] - psi[s[sel]] - eps + np.pi / 2      # sin convention
        ph.append(np.mod(arg, 2 * np.pi)); val.append(6 * I6[sel, t] / I6[sel].sum(1))
    ph, val = np.concatenate(ph), np.concatenate(val)
    rng = np.random.default_rng(1); sub = rng.choice(len(ph), min(700, len(ph)), replace=False)
    a.scatter(ph[sub], val[sub], s=0.6, c=COL['a1'], lw=0, alpha=0.55, rasterized=True)
    g = np.linspace(0, 2 * np.pi, 200)
    rel = np.exp(D['pat_logrel'] - D['pat_logrel'].mean())[:3].mean()
    a.plot(g, rel * (1 + m[:3].mean() * np.sin(g)), color='k', lw=0.9)
    a.set_xlim(0, 2 * np.pi); a.set_ylim(-0.1, 2.3)
    a.set_xticks([]); a.set_yticks([]); [a.spines[s].set_visible(False) for s in ('top', 'right')]
    # phase drift eps_i(t) over the movie
    a = cv.axes(hx + 2, yc + 109, 66, 18)
    dphi = D['pat_dphi']; tk = np.arange(dphi.shape[1]) * float(D['pat_knot'])
    for ai, c in ((0, COL['a1']), (1, COL['a2'])):
        a.plot(tk, np.degrees(dphi[ai]) - 40 * ai, color=c, lw=0.6)
    a.set_xlim(0, tk[-1]); a.axis('off')
    cv.text(hx + 70, yc + 113, r'$\varepsilon_i(t)$', size=FXS + 0.6, ha='left', color=COL['grey'])
    # I: pass 2 with the pattern
    ix = 440
    cv.letter(ix - 2, yl, 'I', 'Pass 2:', '+ pattern')
    cv.arrow(hx + 76, 345, ix + 6, 345)
    st = illum_tile(kv[0], psi[0], m[0], n=12, os_=12)
    st2 = illum_tile(kv[1], psi[3], m[3], n=12, os_=12)
    cv.cube(ix + 8, 318, 28, 24, 10, COL['det2'], front_img=st)
    cv.cube(ix + 8, 390, 28, 24, 10, COL['em2'], front_img=st2)
    cv.arrow(ix + 22, 346, ix + 22, 386, lw=0.6, ms=4)
    cv.text(ix + 22, 418, 'same outputs', size=FXS, ha='center', color=COL['grey'])
    # J: pattern fit with the network prior
    jx = 512
    cv.letter(jx - 2, yl, 'J', 'Pattern fit,', 'network prior')
    cv.arrow(ix + 48, 380, jx + 4, 360)
    a = cv.axes(jx + 4, yc + 6, 74, 54)
    g = np.linspace(-1.3 * np.pi, 1.3 * np.pi, 300)
    cols = ['#5ab4ac', '#d8b365', '#8c6bb1']
    x_e = 1.2
    for j in range(3):
        a.plot(g, 1 + 0.85 * np.sin(g - j * 2 * np.pi / 3), color=cols[j], lw=1.0)
        a.plot([x_e], [1 + 0.85 * np.sin(x_e - j * 2 * np.pi / 3) + (-0.06, 0.07, 0.05)[j]], 'o', color='k', ms=2.6, zorder=5)
    a.axvline(x_e, color='k', lw=0.6, ls='--')
    # prior on the position (the network's) under the curves
    pr = np.exp(-0.5 * ((g - (x_e - 0.3)) / 0.9) ** 2)
    a.fill_between(g, -0.95, -0.95 + 0.75 * pr, color='#cfe0f3', lw=0)
    a.plot(g, -0.95 + 0.75 * pr, color='#3d7fc4', lw=0.7)
    a.set_xlim(g[0], g[-1]); a.set_ylim(-1.0, 2.05); a.axis('off')
    cv.text(jx + 41, yc + 63, r'$\theta = (x, y, N)$', size=FX, ha='center')
    cv.text(jx + 4, yc + 47, 'prior', size=FXS, color='#3d7fc4', ha='left')
    # K: select and render (Fig. 3 SIMCODE zoom)
    kx = 594
    cv.letter(kx - 2, yl, 'K', 'Select', '& render')
    cv.arrow(jx + 79, 345, kx, 345)
    r = D['render_I']
    cv.img(r, kx + 1, yc + 10, 50, 50, cmap=None, ec='#444444', interp='bilinear')
    cv.text(kx - 1, yc + 66, r'$z$ from pass 1', size=FXS, color=COL['grey'])
    cv.text(kx - 1, yc + 76, r'$p$, $\sigma$ cut', size=FXS, color=COL['grey'])
    # pass-1 z -> output (long curve under the row, as the preprint's G -> K line)
    cv.ov.plot([gx0 + 80, gx0 + 80], [yc + 119, yc + 132], color=COL['grey'], lw=0.7, zorder=20)
    cv.ov.plot([gx0 + 80, kx + 14], [yc + 132, yc + 132], color=COL['grey'], lw=0.7, zorder=20)
    cv.arrow(kx + 14, yc + 132, kx + 14, yc + 88, lw=0.7, color=COL['grey'], ms=5)
    cv.text(gx0 + 84, yc + 133.5, r'$z$', size=FXS, color=COL['grey'])


def paste_panel_A(pdf_in, pdf_out):
    import pypdf
    from pypdf.generic import RectangleObject
    src = pypdf.PdfReader(PREPRINT).pages[0]
    Hs = float(src.mediabox.height)
    x0, yt, w, h = A_BOX
    box = RectangleObject((x0, Hs - yt - h, x0 + w, Hs - yt + TOP))
    src.mediabox = box; src.cropbox = box; src.trimbox = box
    wr = pypdf.PdfWriter(clone_from=pdf_in); page = wr.pages[0]
    # same distance from the top of our page as from the top of the preprint's page
    page.merge_transformed_page(src, pypdf.Transformation().translate(0, H - Hs - TOP), over=True)
    wr.write(pdf_out)


def build(out):
    D = dict(np.load(CACHE, allow_pickle=True))
    cv = Canvas()
    training(cv, D)
    inference(cv, D)
    os.makedirs(out, exist_ok=True)
    base = f'{out}/fig_pipeline_v2'
    tmp = f'{base}_noA.pdf'
    cv.fig.savefig(tmp, dpi=600)
    plt.close(cv.fig)
    paste_panel_A(tmp, f'{base}.pdf')
    os.remove(tmp)
    subprocess.run(['pdftocairo', '-svg', f'{base}.pdf', f'{base}.svg'], check=True)
    subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', f'{base}.pdf', base], check=True)
    k = int(D['pick']); f0 = int(D['win_f0'])
    fmt = lambda v: ', '.join(f'{x:.0f}' for x in v)
    head = (CAPTION_HEAD.replace('__F0__', str(f0)).replace('__F1__', str(f0 + 5)).replace('__GAIN__', f"{float(D['gain']):.4f}")
            .replace('__EX__', f"{float(D['c_x'][k]):.1f}").replace('__EY__', f"{float(D['c_y'][k]):.1f}")
            .replace('__P__', f"{float(D['c_p'][k]):.2f}").replace('__I__', fmt(D['c_I'][k])).replace('__SI__', fmt(D['c_sI'][k]))
            .replace('__BG__', ', '.join(f'{x:.1f}' for x in D['c_bg'][k])).replace('__M__', ', '.join(f'{x:.2f}' for x in D['pat_m'][::3])))
    old = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_pipeline_v2_textheavy.py')).read()
    facts = old[old.index('FACTS BEHIND EACH PANEL'):old.rindex('"""')]
    with open(f'{base}_caption.txt', 'w') as f:
        f.write(head + 'CODE FACTS (from the text-heavy version fig_pipeline_v2_textheavy; its panel letters -> this figure: '
                'B -> B-D, C -> E, D -> F-G, E -> H, F -> I, G -> J, H -> K; A was a placeholder, now the drawing)\n' + facts)
    return base


CAPTION_HEAD = r"""fig_pipeline_v2: caption material (generated by analysis/paper/fig_pipeline_v2.py)

DRAFT CAPTION (plain language; symbols only where the main text defines them, see out/paper_figs_v2/NOTATION.md)
Figure 1. SIMCODE pipeline.
(A) Optical system: DMD-based SIMFLUX TIRF microscope (as in the preprint). A digital micromirror device (DMD) projects a
sinusoidal illumination pattern into the sample; the pattern is shifted in three phase steps for each of two orientations.
Network training (B-D) uses simulated data only.
(B) Ground truth: emitters on filaments and as close pairs, with the DNA-PAINT blinking statistics of the experiment. The
simulator uses the movie's own in-situ calibration: the vectorial PSF fitted on the movie's spots, the camera gain, offset and
read-noise maps, and the fitted emitter heights. The illumination is the setup's two-orientation pattern (with realistic
estimation errors) or, for the pass-2 networks in half of the fields of view, a generic random pattern (pitch 1.8-4 pixels,
any orientation, phases and frame order), so that pass 2 accepts any measured pattern.
(C) Simulated 6-frame windows (photon counts).
(D) Two networks: a detector (blue) and an emitter network (orange). Their outputs (detection probability p, network
position theta_dl with uncertainty sigma_dl, backgrounds mu_b and the six intensities mu_N with uncertainties sigma_N) are
compared with the ground truth in the loss, and the networks are updated by backpropagation. The pass-2 versions get the
illumination pattern as extra input. One model is trained per PSF (here: the 1809/2 movie).
Inference (E-K) on the measured movie.
(E) The movie is processed in 6-frame windows (one frame per angle i = 1, 2 and phase step j = 1, 2, 3), a new window every
3 frames. Shown: one window of the 1809/2 movie (Fig. 3) in photons.
(F) Pass 1, without pattern information: the detector finds the emitters; for each candidate, the emitter network reads a
21 x 21 pixel x 6 frame patch (shown: frames 1, 3, 5 of one emitter).
(G) Network outputs of this window: detection probability p per pixel; for the marked emitter, the six intensities mu_N with
their uncertainties sigma_N (blue: angle 1, red: angle 2), the six backgrounds mu_b and the position theta_dl = (x, y, z)
with its uncertainty sigma_dl.
(H) The illumination pattern (six stripe images: 2 angles x 3 steps) is estimated from the pass-1 intensities and positions
of clean, isolated emitters, without any pixel fit. Shown: the normalized intensities of the 1809/2 run against the pattern
phase at their position, with the estimated modulation (black), and the slow phase drift eps_i(t) of both angles over the
movie (range about -18 to +13 degrees).
(I) Pass 2: detector and emitter network again, now with the estimated pattern as input (illumination maps per frame);
the outputs are the same as in G.
(J) Pattern fit for every localization: the expected intensities e_ij(theta) of the pattern model (curves: the three phase
steps of one angle) are fitted to the network intensities (dots), theta = (x, y, N), with the network position as a prior
(blue). Several starts one pattern period apart are tried and the posterior mean is kept, so the result does not lock on a
wrong fringe.
(K) The emitter height z is taken from pass 1. Localizations are kept if p >= 0.63 and the calibrated max(sigma_x, sigma_y) <=
12 nm, and rendered (shown: the SIMCODE zoom of Fig. 3I).

DATA OF THE MINI-IMAGES (analysis/paper/fig_pipeline_v2_prep.py -> out/paper_figs_v2/cache_fig1/fig1_data.npz; CPU)
   A  the preprint's drawing, pasted unchanged as vector graphics (stage/figures/fig_pipeline.pdf, clipped to x 0-308 pt,
      top 0-250 pt; same position and scale as in the preprint's figure, whose page is also 644.8 pt wide).
   B  emitters: ground truth of one simulated scene of the m1809 simulator (simulate/acquisition.AcquisitionSim, preset 0.5nM
      with the m1809 in-situ PSF v2, 3 fields of 40 x 40 px, struct_frac 0.7, pair_frac 0.3); generic patterns: drawn
      analytically in the ranges of acquisition.GENERIC (pitch 2.4-4 px shown for legibility, orientation +-40 deg
      around x / y, depth 0.5-1).
   C  the 6 frames of field 2 of that scene (photons; generic pattern draw, p_std 0).
   E  frames __F0__-__F1__ (0-based; window start % 6 == 0, so frame order = angle 1 steps 1-3, angle 2 steps 1-3) of the 1809/2 movie (cached first 12012 frames, out/archive/optclean/m1809_first12012.tif), (ADU -
      offset map) x gain __GAIN__ e/ADU of models/insitu/calib_1809_2_psf_v2.json, 24 x 24 px around the shown emitter.
   F, G  the m1809 pass-1 detector and emitter network (models.json: m1809/detector_pass1.pt, m1809/emitter_pass1.pt) run on this window on the CPU
      (fp32): detection probability map p (same 24 x 24 px), and for the emitter at (x, y) = (__EX__, __EY__) px (p = __P__,
      chosen as the brightest emitter with p >= 0.63 that is on in all 6 frames and has no other candidate within 8 px):
      intensities __I__ photons, uncertainties __SI__, backgrounds __BG__ photons/px.
      The position glyph is schematic (ellipse = sigma_x, sigma_y; bar = sigma_z).
   H  stripes: 1 + m_i cos(q_i . r - phi_ij) of the estimated pattern of the 1809_2 run of 2026-11-15
      (runs/simcode_v2_psfv2_20261115/pattern_est.pt; pitch 2.07 px, depth m = __M__), on 14 x 14 px. Scatter: 700 random
      points of 4000 clean emitter-windows of that run (si_phase_decouple_cleanrows.npz: pass-1 intensities, positions;
      angle 1, all 3 steps), 6 mu_N_ij / sum mu_N vs the pattern phase q_1 . (x, y) - phi_1j - eps_1(t) (sin convention);
      black: mean relative intensity x (1 + m_1 sin). Drift: eps_i(t) of pattern_est.pt (150-frame knots), both angles.
   I  cube faces: the estimated stripes (schematic for the illumination-map inputs).
   J  schematic (three phase-shifted sines, three measured intensities, prior on the position).
   K  the SIMCODE zoom render of Fig. 3 (panel I; embedded image of 1809_2_fig3_psfv2.pdf, 757 x 757 px).

SYMBOLS IN THE PANELS (all defined in the main text; NOTATION.md)
   p (detection probability, R), mu-hat_N +- sigma-hat_N (network intensities and uncertainties, P p.6), mu-hat_b (backgrounds,
   P p.2), theta-hat_dl +- sigma_dl (network position, P p.7; here with z, see NOTATION.md 6.5), i, j (angle, step), eps_i(t)
   (phase drift, R), theta = (x, y, N) (fit parameters, R), z, sigma. The figure does not show e_ij, q_i, phi_ij, m_i as
   symbols; the caption names them.

"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/jc/simcode2/out/paper_figs_v2')
    a = ap.parse_args()
    print(build(a.out))
