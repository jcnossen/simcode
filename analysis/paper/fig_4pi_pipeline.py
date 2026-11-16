"""
SI schematic of the 4Pi-SIMCODE pipeline and how it differs from the 2D SIMCODE pipeline (si_4pi_pipeline): (a) channel
mapping and patch cutting, (b) detector on the registered channel sum, (c) emitter network (inputs, where the channel offsets
enter, read-out windows, outputs), (d) the pattern-free 3D prior, (e) explicit solve with the in-situ SIM pattern and 4Pi phase
field, (f) SIMCODE vs 4Pi-SIMCODE. Same drawing style as fig_pipeline_v2.py / fig_network_structure.py (vector, text as text).

Content = the code (simcode/channels/{layout,front,registration}.py, simcode/emitter/fourpi.py, simcode/solve/fourpi.py,
simcode/processing/{fourpi,fourpi_pattern_phase}.py, simcode/setups/fourpi.py, simcode/pipeline/configs/fourpi.py) and the
checkpoints of the paper's 4Pi results (<4Pi work dir>/models/det4pi/final.pt, head4pi_v10.pt with its frozen prior =
head4pi_v3L.pt; read on CPU, layer sizes asserted). The small patch / window / read-out-vs-z images are rendered on the CPU
from the 4Pi PSF state stored in the emitter-network checkpoint (the PSF of all 4Pi results).

    CUDA_VISIBLE_DEVICES= python analysis/paper/fig_4pi_pipeline.py [--out /home/jc/simcode2/out/paper_figs_v2]
        [--models4pi /home/jc/simcode2/out/4pi/models] [--models /home/jc/simcode2/models]

-> <out>/si_4pi_pipeline.{pdf,svg,png} + si_4pi_pipeline_caption.txt
"""
import argparse, math, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle, Polygon

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 6.0})

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W, H = 180.0, 180.0
MM = 1 / 25.4
FS, FT, FL = 6.0, 6.9, 10
# colours as fig_pipeline_v2: networks blue, solve green, per-file estimation purple; channel offsets red; SIMCODE (2D) grey
C = dict(net='#2f6db5', netl='#e6eef9', res='#c9dbf2', sol='#2b8a50', soll='#e3f3e9', pat='#7a4aa3', patl='#f1eaf7',
         off='#c8102e', inp='#f0f0f0', inpe='#666666', two='#6b6b6b', k='#222222')
CH = [mpl.colormaps['viridis'](v) for v in (0.05, 0.38, 0.62, 0.85)]       # channel colours c0..c3
LAM = 0.257          # model fringe period (um), simcode/solve/fourpi.py FourPiSolve.solve(Lam=0.257)


# ------------------------------------------------------------------------------------------------ checkpoints (CPU)
def checkpoint_facts(m4, m2):
    import torch
    sys.path.insert(0, REPO)
    from simcode.detector.model import build, n_params
    from simcode.emitter.fourpi import FourPiEmitterNet, FourPiPrior
    from simcode.emitter.model import EmitterNet
    F = {}
    c = torch.load(f'{m4}/det4pi/final.pt', map_location='cpu', weights_only=False)
    det = build(c['args']); det.load_state_dict(c['model'])
    F['det'] = dict(n=n_params(det), args=c['args'], step=c.get('step'), sd={k: tuple(v.shape) for k, v in c['model'].items()})
    c = torch.load(f'{m4}/head4pi_v10.pt', map_location='cpu', weights_only=False)
    a = c['args']
    net = FourPiEmitterNet(ch=a['ch'], nblk=a['nblk'], wins=tuple(tuple(w) for w in a['wins']), bound=a.get('bound', 1.0),
                           nbr_in=a.get('nbr_in', False))
    net.load_state_dict(c['state'])
    npr = n_params(net.prior)
    F['emit'] = dict(n=n_params(net), n_prior=npr, n_own=n_params(net) - npr, args=a, it=c.get('it'), psf=c['psf'],
                     sd={k: tuple(v.shape) for k, v in c['state'].items()})
    c3 = torch.load(f'{m4}/head4pi_v3L.pt', map_location='cpu', weights_only=False)
    pr = FourPiPrior(ch=c3['args']['ch'], nblk=c3['args']['nblk']); pr.load_state_dict(c3['state'])
    same = max(float((c['state']['prior.' + k].float() - v.float()).abs().max()) for k, v in c3['state'].items())
    F['prior'] = dict(n=n_params(pr), args=c3['args'], it=c3.get('it'), same_as_in_emit=same)
    # 2D SIMCODE (m1809) for the comparison numbers
    c = torch.load(f'{m2}/m1809/detector_pass1.pt', map_location='cpu', weights_only=False)
    d2 = build(c['args']); d2.load_state_dict(c['model']); F['det2d'] = dict(n=n_params(d2))
    c = torch.load(f'{m2}/m1809/emitter_pass1.pt', map_location='cpu', weights_only=False)
    e2 = EmitterNet(6, c['args'].get('ch', 96), illum=False, bf16=False); e2.load_state_dict(c['model'])
    F['emit2d'] = dict(n=n_params(e2))
    return F


def check(F):
    d, e, p = F['det'], F['emit'], F['prior']
    assert list(d['args']['ch']) == [48, 96, 160] and d['args']['stem_ch'] == 16 and d['args']['up'] == 1
    assert d['sd']['stem.0.weight'] == (16, 1, 3, 3) and d['sd']['cin.weight'] == (48, 97, 3, 3)
    assert d['sd']['head.2.weight'] == (4, 48, 1, 1)
    assert d['n'] == F['det2d']['n']                                            # same detector architecture as SIMCODE
    a = e['args']
    assert a['arch'] == 'w' and a['ch'] == 64 and a['nblk'] == 4 and not a.get('nbr_in', False)
    assert [w[0] for w in a['wins']] == ['mf', 'cr', 'ci', 'flat']
    assert e['sd']['cin.weight'] == (64, 20, 3, 3)                            # 4 + 4 + 4 + 8 planes
    assert e['sd']['dmap.2.weight'] == (16, 64, 1, 1)                          # 4 channels x K = 4 windows
    assert e['sd']['mlp.0.weight'] == (128, 64 * 16 + 4) and e['sd']['mlp.2.weight'] == (5, 128)   # 4 bg + covariance scale
    assert e['sd']['prior.cin.weight'] == (96, 60, 3, 3)                      # 24 + 24 + 4 + 8 planes
    assert e['sd']['prior.mlp.0.weight'] == (384, 96 * 16 + 24 + 8)            # + 24 border bg + 8 channel offsets
    assert e['sd']['prior.out.weight'] == (150, 384)
    assert len({k.split('.')[1] for k in e['sd'] if k.startswith('blk.')}) == 4
    assert len({k.split('.')[2] for k in e['sd'] if k.startswith('prior.blk.')}) == 4
    assert p['same_as_in_emit'] == 0.0 and p['n'] == e['n_prior']
    return dict(det=d['n'], emit=e['n'], prior=e['n_prior'], own=e['n_own'], emit2d=F['emit2d']['n'])


# ------------------------------------------------------------------------------------------------ PSF renders (CPU)
class PSFImages:
    """components inc, cr, ci of the 4Pi PSF (the checkpoint's state) at (x, y, z): g(Phi) = inc + cos(Phi) cr - sin(Phi) ci"""
    def __init__(self, state, R=15):
        import torch
        from simcode.fourpi.coherent_psf import psf_from_state
        st = dict(state); st['meta'] = dict(st['meta']); st['meta']['R'] = R
        self.p = psf_from_state(st, device='cpu').eval(); self.torch = torch
        self.phi_c = np.degrees(np.asarray(st['phi']) - st['phi'][0]) % 360

    def comps(self, pos):
        torch = self.torch
        pos = torch.tensor(np.atleast_2d(pos), dtype=torch.float32)
        n = len(pos)
        with torch.no_grad():
            g0, gpi, g90 = (self.p(pos, phi_extra=torch.full((n,), ph)).numpy() for ph in (0.0, math.pi, 0.5 * math.pi))
        inc = 0.5 * (g0 + gpi); cr = 0.5 * (g0 - gpi); ci = inc - g90
        return inc, cr, ci                                                     # [n, 4, R, R]

    def image(self, pos, Phi):
        inc, cr, ci = self.comps(pos)
        return inc + np.cos(Phi) * cr - np.sin(Phi) * ci


def demo(psf, seed=3):
    """one example candidate: 6 frames x 4 channels, offsets, windows of channel 1, channel read-outs vs z, NLL(z)"""
    rng = np.random.default_rng(seed)
    x0, y0, z0, Phi0, dA = 0.22, -0.18, 0.06, 0.9, math.radians(-12.0)
    doff = np.array([[0, 0], [0.31, -0.22], [-0.38, 0.14], [0.17, 0.41]])
    N, bg = 1400.0, 0.6
    Pt = np.array([1 + 0.85 * np.cos(0.6 - 2 * np.pi * s / 3) for s in range(3)] + [1 + 0.85 * np.cos(2.3 - 2 * np.pi * s / 3) for s in range(3)])
    img = np.zeros((6, 4, 15, 15))
    for c in range(4):
        pos = [x0 + doff[c, 0], y0 + doff[c, 1], z0]
        for a in range(2):
            g = psf.image(pos, Phi0 + (dA if a else 0.0))[0, c]
            for s in range(3):
                t = 3 * a + s
                img[t, c] = rng.poisson(np.clip(N * Pt[t] * g + bg, 0, None))
    # the K = 4 canonical windows of channel 1 at its mapped prior position (simcode/emitter/fourpi.py window())
    c = 1
    inc, cr, ci = (v[0, c] for v in psf.comps([x0 + doff[c, 0], y0 + doff[c, 1], z0]))
    inc = np.clip(inc, 1e-7, None)
    u = inc / (inc + 0.003); mf = u / (u * inc).sum() * inc.sum()
    wins = dict(mf=mf, cr=cr / (inc + 0.003), ci=ci / (inc + 0.003), flat=np.ones_like(inc))
    # channel photons vs z (Phi0 fixed): the interference signal that carries z
    zs = np.linspace(-0.40, 0.40, 161)
    inc_z, cr_z, ci_z = psf.comps(np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], 1))
    gz = (inc_z + np.cos(Phi0) * cr_z - np.sin(Phi0) * ci_z).sum((-1, -2))  # [nz, 4]
    # NLL(z) of the 4 channel fractions of a 2700 e window (F profiled out) + the prior on z
    i0 = np.argmin(np.abs(zs - z0)); Ftot = 2700.0
    y = gz[i0] / gz[i0].sum() * Ftot
    mu = gz / gz.sum(1, keepdims=True) * Ftot
    chi2 = ((y[None] - mu) ** 2 / mu).sum(1)
    zp, sp = z0 + 0.02, 0.045
    prior = ((zs - zp) / sp) ** 2
    return dict(img=img, doff=doff, wins=wins, zs=zs, gz=gz, chi2=chi2, prior=prior, zp=zp, sp=sp, z0=z0, Pt=Pt,
                inc_stack=psf.comps(np.array([[0, 0, z] for z in (-0.3, 0.0, 0.3)]))[0].sum(1))


# ------------------------------------------------------------------------------------------------ drawing helpers
class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W * MM, H * MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis('off')

    def box(self, x, y, w, h, fc='white', ec='k', lw=0.6, r=1.0, ls='-', z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                         ls=ls, zorder=z))

    def text(self, x, y, s, size=FS, ha='left', va='top', color='k', weight='normal', z=6, **kw):
        return self.ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, weight=weight, zorder=z, **kw)

    def lines(self, x, y, rows, size=FS, dy=None, **kw):
        dy = dy or size * 0.3528 * 1.36
        for i, s in enumerate(rows):
            self.text(x, y + i * dy, s, size=size, **kw)
        return y + len(rows) * dy

    def arrow(self, x0, y0, x1, y1, color='k', lw=0.8, ls='-', ms=6, cs='arc3', z=4):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                          ls=ls, zorder=z, shrinkA=0, shrinkB=0, connectionstyle=cs))

    def seg(self, xs, ys, color='k', lw=0.8, ls='-', z=4, halo=False):
        pe = [patheffects.withStroke(linewidth=lw + 2.2, foreground='white')] if halo else None
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='butt', path_effects=pe)

    def img(self, a, x, y, w, h, cmap='gray', vmin=None, vmax=None, z=3, ec='#444444'):
        self.ax.imshow(a, extent=(x, x + w, y + h, y), cmap=cmap, vmin=vmin, vmax=vmax, zorder=z, interpolation='nearest',
                       aspect='auto')
        self.ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=ec, lw=0.3, zorder=z + 0.1))

    def letter(self, x, y, s, title=None, color='k'):
        self.text(x, y, s, size=FL, weight='bold')
        if title: self.text(x + 5.0, y + 0.7, title, size=FT, weight='bold', color=color)

    def tag(self, x, y, s='offset', size=FS - 0.6, ha='left', **kw):
        """channel-offset marker (red)"""
        return self.text(x, y, s, size=size, color=C['off'], ha=ha, **kw)


def flag(cv, x, y, s, mirror=False, color='k'):
    """small asymmetric glyph (three spots in an L + a bar) for the channel crops; mirror = left-right flipped"""
    pts = [(-0.35, -0.3), (-0.35, 0.05), (-0.35, 0.35), (0.0, 0.35), (0.3, 0.35)]
    for px, py in pts:
        cv.ax.add_patch(Circle((x + (-px if mirror else px) * s, y + py * s), 0.075 * s, fc=color, ec='none', zorder=5))
    cv.ax.add_patch(Circle((x + (-0.25 if mirror else 0.25) * s, y - 0.25 * s), 0.12 * s, fc=color, ec='none', zorder=5))


def patch_img(a, lo=None, hi=None):
    lo = np.percentile(a, 1) if lo is None else lo; hi = np.percentile(a, 99.7) if hi is None else hi
    return np.clip((a - lo) / (hi - lo), 0, 1)


# ------------------------------------------------------------------------------------------------ panels
def panel_a(cv, D):
    x0, y0 = 1.0, 1.0
    cv.letter(x0, y0, 'a', 'Channel mapping')
    # (i) camera frame with the 4 channel crops (raw columns 217, 652, 1420, 1855; width 360; channels 2, 3 mirrored)
    fx, fy, fw = 4.0, 12.0, 54.0
    s = fw / 2304.0; fh = 200 * s
    cv.ax.add_patch(Rectangle((fx, fy), fw, fh, fc='#d9d9d9', ec='#555555', lw=0.5, zorder=2))
    for c, (xc, mir) in enumerate(zip((217, 652, 1420, 1855), (False, False, True, True))):
        cx = fx + xc * s
        cv.ax.add_patch(Rectangle((cx, fy), 360 * s, fh, fc='#1e1e1e', ec=CH[c], lw=0.9, zorder=3))
        flag(cv, cx + 180 * s, fy + fh / 2, 3.1, mirror=mir, color='#f2f2f2')
        cv.text(cx + 180 * s, fy - 0.6, f'ch{c + 1}', size=FS - 0.3, ha='center', va='bottom', color=CH[c], weight='bold')
    cv.text(fx, fy + fh + 0.9, 'one camera frame, 4 interference channels', size=FS - 0.3)
    cv.text(fx, fy + fh + 3.5, '(ch3, ch4 mirrored)', size=FS - 0.3, color='#555555')
    # crops flipped back and overlaid on c0
    ay = fy + fh + 8.0
    cv.arrow(fx + 27, ay - 1.6, fx + 27, ay + 0.6)
    cv.text(fx + 28.2, ay - 1.4, 'crop, flip ch3 / ch4', size=FS - 0.6, color='#555555')
    by = ay + 1.6; bw_, bh_ = 11.8, 6.6
    for c in range(4):
        bx = fx + c * 14.0
        cv.ax.add_patch(Rectangle((bx, by), bw_, bh_, fc='#1e1e1e', ec=CH[c], lw=0.9, zorder=3))
        sh = [(0, 0), (0.35, -0.25), (-0.4, 0.2), (0.25, 0.35)][c]
        flag(cv, bx + bw_ / 2 + sh[0], by + bh_ / 2 + sh[1], 3.1, color='#f2f2f2')
    # (ii) order-2 polynomial map channel c <-> c0 (warped grid over the reference grid)
    mx, my, mw = 63.0, 9.0, 20.0
    u = np.linspace(-1, 1, 6); t = np.linspace(-1, 1, 40)
    warp = lambda a, b: (a + 0.10 * a * a + 0.07 * a * b + 0.06, b + 0.08 * b * b - 0.06 * a * a - 0.05)
    for v in u:
        vv = np.full_like(t, v)
        cv.ax.plot(mx + mw / 2 * (1 + t * 0.82), my + mw / 2 * (1 + vv * 0.82) * 0.82, color='#9a9a9a', lw=0.45, zorder=3)
        cv.ax.plot(mx + mw / 2 * (1 + vv * 0.82), my + mw / 2 * (1 + t * 0.82) * 0.82, color='#9a9a9a', lw=0.45, zorder=3)
        a, b = warp(t, np.full_like(t, v)); cv.ax.plot(mx + mw / 2 * (1 + a * 0.82), my + mw / 2 * (1 + b * 0.82) * 0.82, color=CH[2], lw=0.6, zorder=4)
        a, b = warp(np.full_like(t, v), t); cv.ax.plot(mx + mw / 2 * (1 + a * 0.82), my + mw / 2 * (1 + b * 0.82) * 0.82, color=CH[2], lw=0.6, zorder=4)
    cv.lines(mx - 1.0, my + mw * 0.82 + 2.2, ['order-2 polynomial map', r'per channel $\leftrightarrow$ ch1'], size=FS - 0.3)
    cv.text(mx + mw + 0.5, my + 1.0, 'ch1', size=FS - 0.6, color='#777777')
    cv.text(mx + mw + 0.5, my + 4.0, 'ch3', size=FS - 0.6, color=CH[2], weight='bold')
    # (iii) patch cut without resampling: nearest pixel centre + sub-pixel remainder
    px0, py0, n, q = 88.0, 8.6, 7, 2.55
    for i in range(n + 1):
        cv.seg([px0, px0 + n * q], [py0 + i * q] * 2, color='#b5b5b5', lw=0.35, z=3)
        cv.seg([px0 + i * q] * 2, [py0, py0 + n * q], color='#b5b5b5', lw=0.35, z=3)
    cv.ax.add_patch(Rectangle((px0 + 0.5 * q, py0 + 0.5 * q), 6 * q, 6 * q, fc='none', ec=CH[2], lw=0.9, zorder=4))
    ccx, ccy = px0 + 3.5 * q, py0 + 3.5 * q
    cv.ax.plot([ccx], [ccy], marker='+', ms=4.5, mew=0.8, color='k', zorder=6)
    dx, dy = 0.38 * q, -0.30 * q
    cv.ax.plot([ccx + dx], [ccy + dy], marker='o', ms=2.6, color=C['off'], zorder=6)
    cv.arrow(ccx, ccy, ccx + dx, ccy + dy, color=C['off'], lw=0.6, ms=4)
    cv.tag(ccx + dx + 0.6, ccy + dy - 2.6, size=FS - 0.8)
    cv.lines(px0 - 1.0, py0 + n * q + 2.2, ['15 × 15 patch on the nearest', 'pixel: no resampling'], size=FS - 0.3)
    # patches of the four channels (one frame) with their offsets
    qx, qy, qs = 63.0, 37.0, 8.6
    for c in range(4):
        x = qx + c * (qs + 2.2)
        cv.img(patch_img(D['img'][1, c]), x, qy, qs, qs, cmap='magma', ec=CH[c])
        dxx, dyy = D['doff'][c]
        cv.ax.plot([x + qs / 2 + dxx * qs / 15], [qy + qs / 2 + dyy * qs / 15], marker='+', ms=3.5, mew=0.6, color='#8fd3ff', zorder=6)
        cv.text(x + qs / 2, qy - 0.5, f'ch{c + 1}', size=FS - 0.5, ha='center', va='bottom', color=CH[c], weight='bold')
    cv.tag(qx, qy + qs + 0.8, 'sub-pixel offset kept per channel (ch1: 0)', size=FS - 0.5)


def panel_b(cv, D):
    x0, y0 = 111.5, 1.0
    cv.letter(x0, y0, 'b', 'Detector')
    # 4 channel frames -> resampled into c0 coordinates and summed
    fx, fy = 116.5, 10.0
    for c in range(4):
        cv.ax.add_patch(Rectangle((fx + c * 1.5, fy + c * 1.5), 10.8, 6.0, fc='#1e1e1e', ec=CH[c], lw=0.8, zorder=3 + c * 0.1))
    flag(cv, fx + 4.5 + 5.4, fy + 4.5 + 3.0, 2.8, color='#f2f2f2')
    sx_, sy_ = fx + 21.5, fy + 5.2
    cv.ax.add_patch(Circle((sx_, sy_), 2.6, fc='white', ec='k', lw=0.7, zorder=4))
    cv.text(sx_, sy_ + 0.15, r'$\Sigma$', size=FS + 1.2, ha='center', va='center')
    cv.arrow(fx + 15.5, sy_, sx_ - 2.7, sy_)
    cv.lines(fx - 0.5, fy + 12.6, ['registered 4-channel sum', '(ch2-ch4 resampled into ch1)'], size=FS - 0.4)
    # 6 planes (one per frame of the window)
    zx, zy = sx_ + 7.0, fy - 1.2
    for t in range(6):
        cv.img(np.full((2, 2), 0.2 + 0.1 * D['Pt'][t]), zx + t * 1.25, zy + t * 1.25, 7.0, 4.5, cmap='gray', vmin=0, vmax=1, z=3 + 0.1 * t)
    cv.arrow(sx_ + 2.7, sy_, zx - 0.2, sy_)
    cv.text(zx + 7.0 + 6 * 1.25 + 0.6, zy + 4.0, '6 planes', size=FS - 0.5)
    # detector box -> candidates
    dbx, dby = 133.0, 28.0
    cv.box(dbx, dby, 22.0, 11.5, fc='white', ec=C['net'], lw=0.7)
    cv.text(dbx + 11, dby + 1.0, 'Detector', size=FS + 0.3, ha='center', weight='bold', color=C['net'])
    cv.lines(dbx + 11, dby + 4.4, ['U-Net, as in', 'SIMCODE'], size=FS - 0.4, ha='center')
    cv.arrow(zx + 6, zy + 6 * 1.25 + 4.6, dbx + 11, dby, cs='arc3,rad=0.0')
    cx_, cy_ = 161.0, 28.0
    cv.ax.add_patch(Rectangle((cx_, cy_), 16.0, 9.0, fc='#1e1e1e', ec='#444444', lw=0.4, zorder=3))
    for (u, v) in ((3.5, 2.6), (9.8, 5.6), (12.5, 2.5), (5.2, 6.8)):
        cv.ax.add_patch(Circle((cx_ + u, cy_ + v), 1.05, fc='none', ec='#ffd23f', lw=0.6, zorder=4))
    cv.arrow(dbx + 22, dby + 5.7, cx_ - 0.2, dby + 5.7)
    cv.text(cx_ + 8, cy_ + 10.0, 'candidates', size=FS - 0.4, ha='center')
    cv.text(cx_ + 8, cy_ + 12.6, r'($p\geq0.5$)', size=FS - 0.6, ha='center', color='#555555')
    cv.text(116.0, 46.0, 'SIMCODE: the 6 frames of its one channel', size=FS - 0.5, color=C['two'], style='italic')


def panel_c(cv, D, N):
    x0, y0 = 1.0, 54.0
    cv.seg([1, W - 1], [y0 - 1.6, y0 - 1.6], color='#bbbbbb', lw=0.5)
    cv.letter(x0, y0, 'c', 'Emitter network (per candidate)')
    # input: 6 x 4 patches
    gx, gy, gs, gp = 5.0, y0 + 10.0, 3.7, 0.35
    for t in range(6):
        for c in range(4):
            cv.img(patch_img(D['img'][t, c], 0, D['img'].max() * 0.85), gx + c * (gs + gp), gy + t * (gs + gp), gs, gs, cmap='magma', ec=CH[c])
    for c in range(4):
        cv.text(gx + c * (gs + gp) + gs / 2, gy - 0.5, f'ch{c + 1}', size=FS - 1.4, ha='center', va='bottom', color=CH[c], weight='bold')
    cv.text(gx - 0.9, gy + 1.4, '1', size=FS - 1.2, ha='right'); cv.text(gx - 0.9, gy + 5 * (gs + gp) + 1.4, '6', size=FS - 1.2, ha='right')
    gb = gy + 6 * (gs + gp)
    cv.lines(gx - 2.0, gb + 0.8, ['24 patches', '(6 frames × 4', 'channels)'], size=FS - 0.5)
    # channel offsets: input marker and red bus to the two networks
    ox, oy = gx - 2.0, gb + 9.8
    cv.box(ox, oy - 0.6, 17.6, 4.4, fc='white', ec=C['off'], lw=0.7, r=0.6)
    cv.tag(ox + 0.8, oy, '4 channel offsets', size=FS - 0.8)
    bus_x = 25.5
    # prior network
    px, py, pw, ph = 30.0, y0 + 7.5, 37.0, 21.0
    cv.box(px, py, pw, ph, fc=C['netl'], ec=C['net'], lw=0.8)
    cv.text(px + 1.2, py + 1.0, 'Prior network', size=FS + 0.3, weight='bold', color=C['net'])
    cv.text(px + pw - 1.2, py + 1.2, 'frozen', size=FS - 0.8, ha='right', color=C['net'], style='italic')
    cv.lines(px + 1.2, py + 4.6, ['all 24 patches (log, linear)', '+ 4 frame means', '+ 8 coordinate maps',
                                  'CNN 96 ch → MLP', '+ 24 border bg + 8 offsets'], size=FS - 0.5)
    # window network
    wx, wy, ww, wh = 30.0, y0 + 34.5, 37.0, 22.5
    cv.box(wx, wy, ww, wh, fc=C['netl'], ec=C['net'], lw=0.8)
    cv.text(wx + 1.2, wy + 1.0, 'Window network', size=FS + 0.3, weight='bold', color=C['net'])
    cv.lines(wx + 1.2, wy + 4.6, ['4 frame means (log, linear)', '+ 4 PSF templates at the prior',
                                  '+ 8 coordinate maps', 'CNN 64 ch → window changes',
                                  '(no response to the emitter)', 'MLP → backgrounds, unc. scale'], size=FS - 0.5)
    # arrows from the input
    xin = gx + 4 * (gs + gp) + 0.2
    cv.arrow(xin, py + 8, px - 0.2, py + 8, color='k')
    cv.seg([xin, 23.0, 23.0], [gb - 2.0, gb - 2.0, wy + 6], color='k'); cv.arrow(23.0, wy + 6, wx - 0.2, wy + 6, color='k')
    # offset bus
    cv.seg([ox + 17.6, bus_x], [oy + 1.6, oy + 1.6], color=C['off'], lw=0.9, z=5, halo=True)
    cv.seg([bus_x, bus_x], [oy + 1.6, py + 15.5], color=C['off'], lw=0.9, z=5, halo=True)
    cv.arrow(bus_x, py + 15.5, px - 0.2, py + 15.5, color=C['off'], lw=0.9, ms=5)
    cv.arrow(bus_x, wy + 12.0, wx - 0.2, wy + 12.0, color=C['off'], lw=0.9, ms=5)
    # prior -> d (along the top)
    ty_ = y0 + 5.6
    cv.seg([px + pw - 8, px + pw - 8], [py, ty_], color=C['net'], lw=0.9)
    cv.arrow(px + pw - 8, ty_, 139.0, ty_, color=C['net'], lw=0.9)
    cv.text(131.0, ty_ - 0.4, r'$x, y, z\pm\sigma$', size=FS - 0.6, va='bottom', ha='center', color=C['net'])
    # prior -> p0 -> window network (templates)
    cv.arrow(px + pw / 2, py + ph, px + pw / 2, wy - 0.2, color=C['net'], lw=0.8, ms=5)
    cv.text(px + pw / 2 + 0.8, py + ph + 1.4, 'prior position', size=FS - 0.9, color=C['net'])
    # canonical windows of one channel at p0 + offset
    kx, ky, ks, kg = 74.0, y0 + 14.5, 6.0, 2.3
    cv.text(kx, ky - 6.6, 'read-out windows, per channel', size=FS - 0.4)
    cv.text(kx, ky - 3.6, 'at the prior position', size=FS - 0.6, color=C['net'])
    cv.tag(kx + 21.0, ky - 3.6, '+ offset', size=FS - 0.6)
    names = [('mf', 'matched\nfilter'), ('cr', 'interf.\ncos'), ('ci', 'interf.\nsin'), ('flat', 'patch\nsum')]
    for i, (k, lab) in enumerate(names):
        a = D['wins'][k]; v = np.abs(a).max() * (2.5 if k == 'flat' else 1.0)
        cv.img(a, kx + i * (ks + kg), ky, ks, ks, cmap='RdBu_r', vmin=-v, vmax=v)
        cv.text(kx + i * (ks + kg) + ks / 2, ky + ks + 0.6, lab, size=FS - 1.2, ha='center', linespacing=1.05)
    cv.arrow(px + pw, ky + ks / 2, kx - 0.3, ky + ks / 2, color=C['net'], lw=0.8, ms=5)

    # read-out
    rx, ry, rw = 74.0, y0 + 34.5, 31.0
    cv.box(rx, ry, rw, 13.0, fc='white', ec=C['net'], lw=0.7)
    cv.text(rx + 1.2, ry + 1.0, 'linear read-out', size=FS, weight='bold', color=C['net'])
    cv.lines(rx + 1.2, ry + 4.6, ['(window + change) · (pixels', '− background), per frame,', 'channel and window'], size=FS - 0.8)
    xm = kx + 2 * (ks + kg) - kg / 2
    cv.arrow(xm, ky + ks + 5.2, xm, ry - 0.2, color=C['net'], lw=0.8, ms=5)

    cv.arrow(wx + ww, wy + 9.5, rx - 0.2, wy + 9.5, color=C['net'], lw=0.8, ms=5)
    cv.seg([23.0, 23.0], [wy + 6, y0 + 60.0], color='#999999', lw=0.6, ls=(0, (2, 1.5)))
    cv.seg([23.0, 71.0], [y0 + 60.0, y0 + 60.0], color='#999999', lw=0.6, ls=(0, (2, 1.5)))
    cv.seg([71.0, 71.0], [y0 + 60.0, ry + 11.0], color='#999999', lw=0.6, ls=(0, (2, 1.5)))
    cv.arrow(71.0, ry + 11.0, rx - 0.2, ry + 11.0, color='#999999', lw=0.6, ls=(0, (2, 1.5)), ms=5)
    cv.text(46.0, y0 + 59.6, 'pixels of all 24 patches', size=FS - 1.0, va='bottom', color='#666666', ha='center')
    # outputs: read-out grid (6 frames x 4 channels x 4 windows) + block-diagonal covariance
    qx, qy = 110.0, y0 + 14.5
    cv.lines(qx, qy - 8.6, ['96 read-outs: 4 per', 'frame and channel'], size=FS - 0.6)
    cw, chh = 3.5, 3.0
    rng = np.random.default_rng(5)
    for c in range(4):
        cv.text(qx + c * (cw + 0.45) + cw / 2, qy - 0.4, f'ch{c + 1}', size=FS - 2.0, ha='center', va='bottom', color=CH[c], weight='bold')
    for t in range(6):
        for c in range(4):
            xx, yy = qx + c * (cw + 0.45), qy + t * (chh + 0.35)
            cv.ax.add_patch(Rectangle((xx, yy), cw, chh, fc='white', ec=CH[c], lw=0.4, zorder=3))
            hgt = np.array([0.75 * D['Pt'][t] / 1.85 + 0.15, 0.35 + 0.25 * rng.random(), 0.25 + 0.3 * rng.random(), 0.9])
            for k in range(4):
                bh = hgt[k] * (chh - 0.5)
                cv.ax.add_patch(Rectangle((xx + 0.3 + k * 0.75, yy + chh - 0.25 - bh), 0.55, bh, fc=CH[c], ec='none', zorder=4))
    cv.text(qx - 0.6, qy + 1.3, '1', size=FS - 1.4, ha='right')
    cv.text(qx + 2 * (cw + 0.45), qy + 6 * (chh + 0.35) + 0.5, '4 windows per\n(frame, channel)', size=FS - 1.1, ha='center',
            linespacing=1.05)
    cv.seg([rx + rw, 107.5], [ry + 6.5, ry + 6.5], color=C['net'], lw=0.8)
    cv.seg([107.5, 107.5], [ry + 6.5, qy + 3 * (chh + 0.35) - 0.17], color=C['net'], lw=0.8)
    cv.arrow(107.5, qy + 3 * (chh + 0.35) - 0.17, qx - 0.3, qy + 3 * (chh + 0.35) - 0.17, color=C['net'], lw=0.8, ms=5)
    # uncertainties and backgrounds (words; the analytic 4 x 4 read-out covariance per frame and channel is in the caption)
    vx, vy = 127.5, y0 + 14.5
    cv.lines(vx, vy, ['± uncertainties', '(analytic, ×', 'learned scale)', '', 'backgrounds:', 'one per channel'], size=FS - 0.8)
    cv.text(110.0, y0 + 46.0, 'SIMCODE: one 21 × 21 × 6 patch →', size=FS - 0.8, color=C['two'], style='italic')
    cv.text(110.0, y0 + 48.8, '6 intensities ± uncertainties,', size=FS - 0.8, color=C['two'], style='italic')
    cv.text(110.0, y0 + 51.6, '6 backgrounds, position (x, y, z)', size=FS - 0.8, color=C['two'], style='italic')


def panel_d(cv, D):
    x0, y0 = 140.0, 54.0
    cv.letter(x0, y0, 'd', 'Pattern-free prior', color='k')
    zl = ('−0.3', '0', '+0.3')
    ix, iy, s = 143.5, y0 + 9.0, 9.6
    a = D['inc_stack']; v = a.max()
    for i in range(3):
        cv.img(a[i], ix + i * (s + 1.5), iy, s, s, cmap='magma', vmin=0, vmax=v)
        cv.text(ix + i * (s + 1.5) + s / 2, iy + s + 0.6, f'z = {zl[i]} µm', size=FS - 1.1, ha='center')
    cv.lines(143.5, iy + s + 4.4, ['shape of the channel-summed', 'PSF (astigmatism) → z;', 'trained with random 4Pi phase',
                                   'and random pattern phase'], size=FS - 0.6)
    # output glyph
    gx, gy = 150.0, y0 + 46.0
    cv.ax.add_patch(Ellipse((gx, gy), 9.0, 6.0, fc=C['net'], alpha=0.22, ec=C['net'], lw=0.5, zorder=3))
    cv.ax.plot([gx], [gy], marker='o', ms=1.8, color=C['net'], zorder=4)
    cv.seg([gx + 7.0, gx + 7.0], [gy - 5.0, gy + 5.0], color=C['net'], lw=1.0)
    for yy in (gy - 5.0, gy + 5.0): cv.seg([gx + 6.2, gx + 7.8], [yy, yy], color=C['net'], lw=1.0)
    cv.ax.plot([gx + 7.0], [gy], marker='o', ms=1.8, color=C['net'], zorder=4)
    cv.text(gx, gy + 4.0, r'$x, y$', size=FS - 0.6, ha='center', va='top')
    cv.text(gx + 8.4, gy - 0.2, r'$z\pm\sigma_z$', size=FS - 0.6, va='center')
    cv.lines(143.5, gy + 7.5, [r'$\sigma_x, \sigma_y, \sigma_z$ (independent)', '→ position of the read-out windows (c),',
                               '   prior term of the fit (e)'], size=FS - 0.6)


def panel_e(cv, D):
    x0, y0 = 1.0, 118.0
    cv.seg([1, W - 1], [y0 - 1.6, y0 - 1.6], color='#bbbbbb', lw=0.5)
    cv.letter(x0, y0, 'e', 'Explicit solve on the read-outs (per candidate; no pixel fit)', color='k')
    # per-file estimation
    bx, by, bw, bh = 4.0, y0 + 7.0, 33.0, 53.0
    cv.box(bx, by, bw, bh, fc=C['patl'], ec=C['pat'], lw=0.8)
    cv.text(bx + 1.2, by + 1.0, 'Per file, in situ', size=FS + 0.3, weight='bold', color=C['pat'])
    cv.lines(bx + 1.2, by + 4.3, ['from the read-outs of', 'bright isolated candidates'], size=FS - 0.6, color=C['pat'])
    # SIM pattern glyph
    n = 40; yy, xx = np.mgrid[:n, :n]
    for i, ang in enumerate((0.0, np.pi / 2)):
        a = np.cos(2 * np.pi / 7 * (np.cos(ang) * xx + np.sin(ang) * yy))
        cv.img(a, bx + 1.5 + i * 8.0, by + 10.5, 7.0, 7.0, cmap='Greys_r', vmin=-1.6, vmax=1.2)
    cv.lines(bx + 18.0, by + 10.8, ['SIM pattern', r'$\vec{q}_i, \phi_{i,j}, m_i$'], size=FS - 0.5)
    # 4Pi phase field glyph: tile map + time blocks
    rng = np.random.default_rng(11)
    from scipy import ndimage as ndi
    M = ndi.gaussian_filter(rng.standard_normal((17, 31)), 3.0); M = (M - M.mean()) / M.std() * 0.9
    cv.img(M, bx + 1.5, by + 21.5, 20.0, 11.0, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    tb = np.cumsum(rng.standard_normal(30) * 0.12)[None]
    cv.img(tb, bx + 1.5, by + 33.6, 20.0, 1.8, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    cv.text(bx + 22.5, by + 22.5, '12-px', size=FS - 1.1); cv.text(bx + 22.5, by + 25.0, 'tiles', size=FS - 1.1)
    cv.text(bx + 22.5, by + 33.4, 'time', size=FS - 1.1)
    cv.lines(bx + 1.2, by + 37.4, ['4Pi phase field', '(per file; tiles × time)'], size=FS - 0.5)
    cv.text(bx + 1.2, by + 44.6, '+ an offset per angle', size=FS - 0.9, color='#555555')
    cv.arrow(bx + bw, by + 26.0, 41.0, by + 26.0, color=C['pat'], lw=1.0)
    # solve box
    sx, sy, sw, sh = 41.0, y0 + 7.0, 69.0, 53.0
    cv.box(sx, sy, sw, sh, fc=C['soll'], ec=C['sol'], lw=0.8)
    cv.text(sx + 1.2, sy + 1.0, 'Read-out model', size=FS + 0.3, weight='bold', color=C['sol'])
    cv.text(sx + 1.2, sy + 4.6, r'expected read-out = $e_{i,j}(\theta)$ × the window applied to the channel', size=FS - 0.6)
    cv.text(sx + 1.2, sy + 7.6, r"4Pi PSF at $(x, y, z)$ + offset, with the 4Pi phase of the field", size=FS - 0.6)
    cv.text(sx + 1.2, sy + 10.6, r'$e_{i,j}(\theta)=N[1+m_i\sin(\vec{q}_i\cdot(x,y)-\phi_{i,j})]$ (× on-time, if blinking)', size=FS - 0.6)
    # left glyph: channel photons vs z (interference)
    zs = D['zs']; g = D['gz'] / D['gz'].mean()
    gx, gy, gw, gh = sx + 3.0, sy + 20.0, 27.0, 12.0
    for c in range(4):
        yv = (g[:, c] - g.min()) / (g.max() - g.min())
        cv.ax.plot(gx + (zs - zs[0]) / (zs[-1] - zs[0]) * gw, gy + gh - yv * gh, color=CH[c], lw=0.8, zorder=4)
    cv.seg([gx, gx + gw], [gy + gh + 0.3, gy + gh + 0.3], color='#444444', lw=0.4)
    cv.text(gx + gw / 2, gy + gh + 1.0, r'$z$', size=FS - 0.6, ha='center')
    cv.text(gx, gy - 2.8, 'photons per channel', size=FS - 0.9, va='bottom')
    # fringe period marker
    zA = 0.0; zB = zA + LAM
    xa = gx + (zA - zs[0]) / (zs[-1] - zs[0]) * gw; xb = gx + (zB - zs[0]) / (zs[-1] - zs[0]) * gw
    cv.seg([xa, xb], [gy - 1.0, gy - 1.0], color='k', lw=0.5)
    for xx_ in (xa, xb): cv.seg([xx_, xx_], [gy - 1.6, gy - 0.4], color='k', lw=0.5)
    cv.text(xb + 0.6, gy - 1.0, 'period 257 nm', size=FS - 1.1, ha='left', va='center')
    cv.text(gx, gy + gh + 4.3, 'z from the interference phase', size=FS - 0.7)
    # right glyph: objective vs z, prior, starts
    hx, hy, hw, hh = sx + 36.0, sy + 20.0, 24.0, 12.0
    chi = D['chi2']; pri = D['prior']; tot = chi + pri
    norm = lambda a: np.clip(a / np.percentile(tot, 97), 0, 1)
    X = hx + (zs - zs[0]) / (zs[-1] - zs[0]) * hw
    cv.ax.plot(X, hy + hh - norm(chi) * hh, color='#8a8a8a', lw=0.7, zorder=4)
    cv.ax.plot(X, hy + hh - norm(pri) * hh, color=C['net'], lw=0.7, ls='--', zorder=4)
    cv.ax.plot(X, hy + hh - norm(tot) * hh, color=C['sol'], lw=1.0, zorder=5)
    cv.seg([hx, hx + hw], [hy + hh + 0.3, hy + hh + 0.3], color='#444444', lw=0.4)
    for j, js in enumerate((-0.5, 0.0, 0.5)):
        zz = D['zp'] + js * LAM
        xx_ = hx + (zz - zs[0]) / (zs[-1] - zs[0]) * hw
        cv.ax.plot([xx_], [hy + hh + 1.6], marker='^', ms=2.8, color='k' if js == 0 else '#777777', zorder=6, clip_on=False)
    cv.text(hx + hw / 2, hy + hh + 3.0, r'3 starts: prior $z$, ± half a period', size=FS - 0.9, ha='center')
    cv.text(hx + hw + 0.8, hy + 0.5, 'data', size=FS - 1.1, color='#777777')
    cv.text(hx + hw + 0.8, hy + 3.1, 'prior', size=FS - 1.1, color=C['net'])
    cv.text(hx + hw + 0.8, hy + 5.7, 'total', size=FS - 1.1, color=C['sol'])
    cv.text(hx, hy - 2.8, r'cost vs $z$: the prior picks the fringe', size=FS - 0.9, va='bottom')
    # fit line
    cv.lines(sx + 1.2, sy + 41.5, [r'LM fit of $\theta=(x, y, z, N)$, a background per channel',
                                   r'[, on / off switching]; prior on $x, y, z$; output: $x, y, z$,',
                                   r'$\sigma_x, \sigma_y, \sigma_z$, photons, $\chi^2$'], size=FS - 0.6, dy=2.75)
    cv.text(sx + sw - 1.2, sy + sh - 1.0, 'no pass 2', size=FS - 0.6, ha='right', va='bottom', color=C['sol'], style='italic')
    # inputs to the solve from c: read-outs, covariance, windows, prior + offsets
    cv.text(sx + 1.2, sy + sh - 1.0, 'from c: read-outs ± unc., windows, prior,', size=FS - 0.9, va='bottom', color='#555555')
    cv.tag(sx + 39.6, sy + sh - 1.0, 'offsets', size=FS - 0.9, va='bottom')


def panel_f(cv):
    x0, y0 = 113.0, 118.0
    cv.letter(x0, y0, 'f', 'SIMCODE vs 4Pi-SIMCODE')
    rows = [
        ('camera', '1 channel', '4 channels, mapped'),
        ('detector input', '6 frames', '6 channel sums'),
        ('network input', '21×21×6', '15×15×6×4 + offsets'),
        ('read-outs', '6 intensities', '4 per frame and channel'),
        ('uncertainties', 'per intensity', 'per frame and channel'),
        ('background', 'per frame', 'per channel'),
        ('prior position', r'$x,y,z$ (z with x, y)', r'$x,y,z$, independent'),
        ('pass 2', 'pattern-aware', 'none'),
        ('per file / movie', 'pattern', 'pattern + 4Pi phase field'),
        ('fit', r'$x, y, N$', r'$x, y, z, N$, bg per channel'),
        ('starts', '9 (lateral fringes)', r'3 ($z$, half a fringe apart)'),
        ('z from', 'pass-1 prior', 'interference phase'),
        ('selection', r'$p$, $\sigma_x, \sigma_y$', r'$\chi^2$/dof, $\sigma$, photons'),
    ]
    tx, ty = x0 + 1.0, y0 + 7.0
    c1, c2, c3 = tx, tx + 19.5, tx + 39.0
    rh = 3.3
    cv.text(c2, ty, 'SIMCODE', size=FS - 0.4, weight='bold', color=C['two'])
    cv.text(c3, ty, '4Pi-SIMCODE', size=FS - 0.4, weight='bold', color=C['sol'])
    cv.seg([tx, W - 1.5], [ty + 3.0, ty + 3.0], color='#888888', lw=0.4)
    for i, (a, b, c) in enumerate(rows):
        yy = ty + 3.6 + i * rh
        if i % 2 == 0:
            cv.ax.add_patch(Rectangle((tx - 0.5, yy - 0.35), W - 1.0 - tx, rh, fc='#f3f3f3', ec='none', zorder=1))
        cv.text(c1, yy, a, size=FS - 1.0, color='#444444')
        cv.text(c2, yy, b, size=FS - 1.0, color=C['two'])
        cv.text(c3, yy, c, size=FS - 1.0)


def build(out, m4, m2):
    F = checkpoint_facts(m4, m2)
    N = check(F)
    psf = PSFImages(F['emit']['psf'])
    D = demo(psf)
    cv = Canvas()
    panel_a(cv, D); panel_b(cv, D); panel_c(cv, D, N); panel_d(cv, D); panel_e(cv, D); panel_f(cv)
    os.makedirs(out, exist_ok=True)
    base = f'{out}/si_4pi_pipeline'
    for ext in ('pdf', 'svg'):
        cv.fig.savefig(f'{base}.{ext}')
    cv.fig.savefig(f'{base}.png', dpi=300)
    plt.close(cv.fig)
    with open(f'{base}_caption.txt', 'w') as f:
        f.write(CAPTION.format(**N, det_step=F['det']['step'], emit_it=F['emit']['it'], prior_it=F['prior']['it'],
                               phi=' / '.join(f'{v:.0f}' for v in psf.phi_c)))
    return base


CAPTION = r"""si_4pi_pipeline: caption material (generated by analysis/paper/fig_4pi_pipeline.py)

DRAFT CAPTION (symbols as in the preprint / main-text Methods; 4Pi additions defined here; see out/paper_figs_v2/NOTATION.md)
Figure S?. The 4Pi-SIMCODE pipeline and how it differs from SIMCODE. (a) Channel mapping. Each camera frame holds the four
interference channels (ch1-ch4) side by side; two of them are mirrored by the optics and are flipped back when the channels
are cut out. An order-2 polynomial map relates every channel to the reference channel ch1. For each candidate, a 15 x 15 px
patch is cut from every channel around the pixel nearest to the candidate's mapped position, so the pixels are never
resampled. The sub-pixel remainder of the mapped position is kept as the channel offset of that channel (zero for ch1).
(b) Detector. The three other channels are resampled into ch1 coordinates and summed with ch1, frame by frame. The detector,
the same U-Net as in SIMCODE (trained on 4Pi simulations), runs on these 6 summed planes of a window and proposes candidates
(detection probability p >= 0.5). SIMCODE runs it on the 6 frames of its single channel. (c) Emitter network. Its input per
candidate is the 24 patches (6 frames x 4 channels) and the four channel offsets. The offsets (red) enter the network as
coordinate maps and scalars of the prior network, as coordinate maps and PSF templates of the window network, and as the
positions of the read-out windows; the fit uses them again to place each channel's PSF (e). A frozen prior network sees all
24 patches and gives a pattern-free 3D position (d). At this position, shifted by each channel's offset, four read-out
windows are placed per channel: a matched filter, two quadrature windows on the 4Pi interference term, and the patch sum. A
window network sees only frame-symmetric inputs (frame means of each channel, the PSF templates at the prior position,
coordinate maps). It learns a bounded change of the windows that does not respond to the emitter's own image, plus one
background per channel and a scale for the uncertainties. Each read-out is the window applied to the background-subtracted
pixels of one frame and channel, which gives 96 read-outs per candidate (4 per frame and channel), with uncertainties
computed analytically from the photon and camera noise and multiplied by the learned scale. SIMCODE's emitter network instead
returns 6 intensities with their uncertainties, a background per frame and a 3D position. (d) Prior. The prior network is
trained with random 4Pi phase and random pattern phase, so its z comes from the shape of the channel-summed PSF
(astigmatism) and not from the interference. It outputs x, y and z with independent uncertainties sigma_x, sigma_y and
sigma_z. (e) Explicit fit. For each file, the SIM pattern (modulation frequency q_i, phases phi_ij and modulation depth m_i
of angle i and phase step j) and the 4Pi phase field (the 4Pi interference phase as a function of position and time: 12-px
tiles and 50-window time blocks, plus one offset per angle) are estimated in situ from the read-outs of bright, isolated
candidates. Each candidate's 96 read-outs are then fitted by Levenberg-Marquardt. The expected read-out is the modulated
intensity e_ij(theta) = N [1 + m_i sin(q_i . (x, y) - phi_ij)] times the window applied to the 4Pi PSF of that channel at
(x, y, z) plus the channel offset, with the 4Pi phase of the field at the candidate. The free parameters are
theta = (x, y, z, N), a residual background per channel and, for an emitter that switches on or off within the window, the
switching times. The network position enters the fit as a Gaussian prior on x, y and z. z follows from the interference
phase, which repeats every 257 nm. The fit starts at the prior z and half a fringe period on either side of it, and the
prior picks the fringe. There is no pass 2. (f) Summary of the differences from SIMCODE.

SYMBOLS (paper symbol / words <- code name; the FACTS below use the code names)
   ch1-ch4 <- channels c0-c3 (0-based; ch3, ch4 = the mirrored c2, c3)
   channel offset <- doff_c = (dx_c, dy_c); read-outs <- Y_tck (frame t, channel c, window k); windows <- W_ck, window
   change <- D_ck; background per channel <- bg_c; prior position <- p0; uncertainty scale <- cs
   q_i, phi_ij, m_i <- k_a, psi_t (phi_ij = psi_t - pi/2: cos -> sin convention of the preprint), m
   N (emitter intensity per window) <- F; e_ij(theta) <- N_t = F [1 + m cos(k_a . r - psi_t)]; switching times <- t_on, t_off
   4Pi phase field <- Phi0(x, y, t) = M(x, y) + d(t); offset per angle <- delta_a; fringe period 257 nm <- Lambda
   If the SI text needs symbols for them, NOTATION.md proposes: channel index c (c = 1..4), channel offset (Delta x_c, Delta y_c),
   4Pi phase field Psi(x, y, t) with per-angle offset Psi_i, fringe period Lambda (none clashes with a preprint symbol).

FACTS BEHIND EACH PANEL (code = this repository, branch 4pipipe from main 56c27e5; checkpoints read on CPU from
/home/jc/simcode2/out/4pi/models; the paper's 4Pi tables were made on 2026-11-12 by the pre-refactor driver
analysis/fourpi/real_w4pi.py (commit 41963c8), which the current simcode/processing/fourpi.py reproduces; see
out/paper_v2psf/FACTS_4PI.md section 7)

Parameter counts (simcode.detector.model.n_params on the networks built from the checkpoints; asserted in check()):
   detector (det4pi/final.pt, step {det_step}): {det:,} parameters = the SIMCODE m1809 detector ({det:,}, same class and widths)
   4Pi emitter network (head4pi_v10.pt, {emit_it} steps): {emit:,} parameters in total = window network {own:,} + frozen prior
     network {prior:,} (identical to head4pi_v3L.pt, {prior_it} steps; max weight difference 0)
   SIMCODE EmitterNet (m1809 pass 1): {emit2d:,} parameters

(a) Channel mapping (simcode/channels/layout.py, registration.py PolyMap, pass1_inference.py ChannelFront.patches;
    simcode/pipeline/configs/fourpi.py)
   - crops 360 x 200 px at raw columns 217, 652, 1420, 1855 of the 2304-px frame; channels c2 and c3 (0-based) mirrored
     left-right (layout mirror 'lr': flipped when cut, so all channels overlay c0 up to the registration map).
   - PolyMap: features u^i v^j, i + j <= order; the paper's results use order 2, one map per cell for the whole series
     (out/4pi/smlm/reg/reg_insitu_<cell>.json, 'order': 2; NPC = bead order-2 map + in-situ translation from file 10,
     MT = in-situ order-2 map from file 5). The current main config has per-file in-situ maps (registration mode 'insitu');
     not used for the paper's tables.
   - patches(): the candidate pixel (cx, cy) in c0 is mapped to channel c with the inverse map (PolyMap.inv);
     ctr = round(mapped), doff = mapped - ctr in [-0.5, 0.5]; the patch is the P x P = 15 x 15 block centred on ctr (pure
     indexing). c0: the candidate is an integer pixel, so doff_0 = (0, 0).
   - The example patches are rendered with the 4Pi PSF state of the emitter-network checkpoint (channel phases
     {phi} deg relative to c0) at illustrative offsets; schematic otherwise.

(b) Detector (simcode/channels/front.py registered_sum; processing/fourpi.py stage A; det4pi args)
   - registered_sum: c0 + channels 1..3 resampled into c0 coordinates (grid_sample, bilinear), weights 1. So the detector
     input IS resampled (only the emitter-network patches are not).
   - per window: the 6 summed planes (photo-electrons) -> Detector (simcode/detector/model.py, ch 48/96/160, stem 16,
     up = 1: the SIMCODE detector class) initialised from pretrain_detector_pass1, 20k steps on full-scene 4Pi simulations (0.01-2 /um^2).
   - select_points(k = 2048 per 20-window batch, p_min = 0.5), candidates inside the illuminated columns [55, 305) and the
     patch border; candidates with < 300 e matched-filter read-out per window (summed over frames and channels) dropped.
   - The detector's sub-pixel offset is not used by the 4Pi pipeline (patches at the integer NMS pixel; position from the prior).

(c) Emitter network (simcode/emitter/fourpi.py FourPiEmitterNet, args arch 'w', ch 64, nblk 4, bound 1.0,
    wins mf / cr / ci / flat)
   - prior network (FourPiPrior, ch 96, 4 ResBlocks): input 60 planes = norm_photons of the 24 patches + 24 linear patches
     ((y - border median) / 50) + norm_photons of the 4 frame means + 4 + 4 coordinate maps (pixel - doff_c, / 7 px);
     MLP input 96 x 16 pooled + 24 border medians (per t, c) + 8 offsets = 1568 -> 384 -> 384 -> 150 outputs (of which only
     the prior dx, dy, z, log sigma x, y, z are used by FourPiEmitterNet; the T / C read-outs of this older network are not).
   - window network: input 20 planes = norm_photons(frame mean) [4] + linear frame mean [4] + PSF template g_c at
     p0 + doff_c (incoherent part, peak-normalised) [4] + coordinate maps [8]; 4 ResBlocks of 64 ch (dilations 1, 2, 1, 2);
     dmap conv 64 -> 64 -> 16 = D for 4 channels x K = 4 windows, D = bound x tanh(.) x window amplitude, projected orthogonal
     to the emitter's image components at p0 (inc, cr, ci per channel); MLP (64 x 16 + 4 border bg) -> 128 -> 5 = 4 background
     corrections + 1 covariance scale cs = 1 + softplus(. - 3) >= 1 (one per candidate).
   - canonical windows at p0 + doff_c from the 4Pi PSF LUT: mf = u / (u . inc) x sum(inc), u = inc / (inc + 0.003);
     cr / ci = coherent cross-term components / (inc + 0.003); flat = 1.
   - Y[t, c, k] = sum_px (W + D)[c, k] (img[t, c] - bg_c); covariance per (t, c): sum_px W'_k W'_l v_px, v = model photons
     + background + read variance (1.14^2 + 0.2^2 e^2) + frame-mean excess, times cs.
   - background: bg_c = border median of the frame-mean image + 0.3 dbg sqrt(|.| + 1), one per channel for the 6 frames.
   - Where the offsets enter (red in the figure): prior coordinate maps + MLP scalars; window-network coordinate maps and
     templates; window placement W at p0 + doff_c; the solve's PSF at (x + dx_c, y + dy_c) (e).

(d) Prior (FourPiPrior.forward; simcode/simulate/fourpi_patches.py)
   - dxy = 1.5 tanh(.) px, z = 0.6 tanh(.) um, log sigma = -6 + 6 sigmoid(.) for x, y (px) and z (um): a diagonal covariance.
   - training (head4pi_v3L.pt args: eps_range pi): random SIM pattern phase per window; Phi0 uniform in [0, 2 pi) per sample
     (fourpi_patches.py), so the prior cannot use the interference; z rests on the PSF shape. The bead-fitted PSF has
     opposite astigmatism in the two arms (Noll 6: +0.085 / -0.088 waves; FACTS_4PI.md section 3); the three PSF images in
     d are the channel-summed incoherent part at z = -0.3 / 0 / +0.3 um.
   - astigmatic prior width 17-30 nm at 2700 e << Lambda / 4 ~ 62 nm (DESIGN_4PI_NET.md 6.2).

(e) Explicit solve (simcode/solve/fourpi.py FourPiSolve; processing/fourpi.py stages B-D; processing/fourpi_pattern_phase.py)
   - pattern per file: repattern_xy on the matched-filter read-outs summed over channels at the prior positions, rows with
     window total > 1500 e, nearest neighbour > 6 px, min frame > 0.25 x mean; k refined around the reference pattern, step
     phases from the slot DFT, depth = median per-emitter 3-step modulation. Full 2D wave vectors (sv.k2). No within-file
     phase drift term (SIMCODE: shared drift eps_a(t)).
   - 4Pi phase field per file: per-candidate Phi fitted with z held by the prior ('free' mode + fit_phi, phase starts 0 / 2.1
     / 4.2 rad) on up to 60 000 candidates (total > 800 e, nearest neighbour > 4 px), weights 1 / sigma_phi^2 for
     sigma_phi < 0.8 rad; Phi0(x, y, t) = M(x, y) on 12-px tiles + d(t) on 50-window blocks (alternating circular means).
     delta_a = per-angle 4Pi phase offset (MT -10.4 deg, NPC -13.9 deg).
   - model: E[Y_tck] = N_t sum_px W'_ck g_c(px; x + dx_c, y + dy_c, z, Phi0 + delta_a(t)) with the network's actual windows W'
     (exact, from the LUT); interference terms scaled by the in-situ contrast 1.56 (coh); N_t = F (1 + m cos(k_a.r - psi_t))
     ('on') or F q_t (...) with q_t the overlap of frame t with [t_on, t_off] ('kin'); residual background b_c per channel
     (prior sd 0.2 e/px); prior on (x, y, z) as pseudo-observations.
   - LM, 15 iterations; 'on': z starts prior z + (-0.5, 0, +0.5) x Lambda, Lambda = 0.257 um, best objective (data + prior)
     kept; 'kin': one z start at the 'on' solution, two kinetics starts, kept if chi^2 drops by > 9.21. No lateral
     fringe-shift starts (the SIM pattern only adds lateral information on top of the 4Pi prior).
   - output: x, y, z, sigma from the inverse Hessian, photons, chi^2 (dof 88 / 86).
   - The read-out-vs-z curves are the four channels' photons of the bead-fitted PSF at fixed Phi0 (no contrast scaling); the
     -log L glyph = chi^2 of the four channel fractions of a 2700 e window (F profiled) + an illustrative prior (sd 45 nm).

(f) SIMCODE column from fig_pipeline_v2_caption.txt / si_network_structure_v3_caption.txt (pass 1 + pass 2, 9 starts =
    prior centre + 8 fringe shifts, z from the pass-1 prior conditioned on x, y, selection p >= 0.63 and sigma <= 12 nm).
    4Pi-SIMCODE selection (analysis/fourpi/post_w4pi.py): good = chi^2/dof < 2.5, sigma_z < 25 nm, sigma_x, sigma_y < 0.12 px,
    N > 300, finite z; validated range N >= 1000 and |z - median| <= 0.3 um; the SI figures add reported-sigma cuts / a
    matched count (si_4pi_*_caption.txt). Lateral drift by DME, sample z drift by a structure-based estimator (post-processing,
    not drawn).
"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/jc/simcode2/out/paper_figs_v2')
    ap.add_argument('--models4pi', default='/home/jc/simcode2/out/4pi/models')
    ap.add_argument('--models', default='/home/jc/simcode2/models')
    a = ap.parse_args()
    print(build(a.out, a.models4pi, a.models))
