"""nr80 evaluation: are the gate scenes nanoruler-like? Draws a few batches of every gate scene exactly as simcode.benchmark.gate.gen
does and prints window photons of the targets, on-frames, ON density, background, pattern pitch / depth / step spacing, read
noise, gain and the Gaussian-equivalent PSF width (2D Gaussian + offset fitted to the summed images of bright isolated emitters).
Must be run through the preset wrapper (otherwise the scenes are those of the other setup):
  SIMCODE_GATE_CALIB=out/nanoruler/calib_nr80.json python -u analysis/nanoruler/nrwrap.py analysis/nanoruler/ev_scenecheck.py [scenes] [seed] > out/nanoruler/eval/scenecheck_nr80.txt"""
import math, sys
import numpy as np, torch
from scipy.optimize import least_squares
from simcode.benchmark import gate as GA
from simcode.simulate import acquisition as RS
from simcode.simulate.gate_layouts import CD, S

scen = (sys.argv[1] if len(sys.argv) > 1 else 'iso,pair1,pair2,rs05qm').split(',')
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 301
PXNM = 65.0
q = lambda v, p: np.percentile(v, p)
CD.a.calib = GA.gate_calib()
print(f'# gate calib {CD.a.calib}')
print('# preset 0.5nM:', {k: v for k, v in RS.PRESETS['0.5nM'].items() if k in ('k', 'spacing', 'depth', 'relint', 'bg', 'bg_sd', 'dens', 'tau', 'bright', 'bright_sd', 'z_med', 'cam', 'struct_frac')})
print(f'# hard-coded in simcode/benchmark/gate.py (NOT from the preset): iso window photons log-uniform 100-20000; pair totals 300-5000 per emitter (log-uniform, 1:1); '
      f'rs05qm window-photon median Q_MED {GA.Q_MED["0.5nM"]} log-sd {GA.Q_SD} (preset bright x 6 = {RS.PRESETS["0.5nM"]["bright"] * 6:.0f})')


def gfit(im):
    n = im.shape[0]; yy, xx = np.mgrid[:n, :n]
    f = lambda p: (p[0] * np.exp(-0.5 * ((xx - p[1]) ** 2 + (yy - p[2]) ** 2) / p[3] ** 2) + p[4] - im).ravel()
    r = least_squares(f, [im.max(), n // 2, n // 2, 1.5, np.median(im)])
    return abs(r.x[3])


for sc in scen:
    PH, ON, NN, BG, DENS, SIG, PIT, DEP, SPC, RN = [], [], [], [], [], [], [], [], [], []
    for it in (1, 2, 3):
        o = GA.gen(sc, it, seed); G = o['gt']; nb = o['nb']
        ph = G['phot'].sum(1); on = G['onf'].sum(1)
        inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
        tg = inter & (on >= 1) & (ph >= 100)
        PH.append(ph[tg].cpu().numpy()); ON.append(on[tg].cpu().numpy())
        inF = (G['x'] >= 0) & (G['x'] < S) & (G['y'] >= 0) & (G['y'] < S)
        DENS.append(float((G['onf'][inF].sum()) / (nb * 6 * S * S)))
        BG.append(float(o['bg'].mean())); RN.append(float(o['rn'].median()))
        p = o['patt']                                              # [nb, T, 5] kx ky phase depth relint
        PIT.append((2 * math.pi / torch.hypot(p[..., 0], p[..., 1])).cpu().numpy().ravel()); DEP.append(p[..., 3].cpu().numpy().ravel())
        if p.shape[1] >= 3:
            d = torch.remainder(p[:, 1, 2] - p[:, 0, 2] + math.pi, 2 * math.pi) - math.pi; SPC.append(d.abs().cpu().numpy())
        if sc == 'iso':
            fr = o['frames'].sum(1) - o['bg'].sum(1).expand_as(o['frames'].sum(1)) if o['bg'].dim() == 4 else o['frames'].sum(1)
            k = torch.nonzero(tg & (ph > 3000) & (ph < 20000))[:, 0][:40]
            for i in k.tolist():
                cx, cy = int(round(float(G['x'][i]))), int(round(float(G['y'][i])))
                im = fr[int(G['fov'][i]), cy - 7:cy + 8, cx - 7:cx + 8].cpu().numpy()
                if im.shape == (15, 15): SIG.append(gfit(im))
    ph = np.concatenate(PH); on = np.concatenate(ON); pit = np.concatenate(PIT); dep = np.concatenate(DEP)
    print(f'\n== {sc} (seed {seed}, 3 batches, {len(ph)} targets)')
    print(f'  target window photons p5 / p50 / p95: {q(ph, 5):.0f} / {q(ph, 50):.0f} / {q(ph, 95):.0f}; fraction in 800-2500: {np.mean((ph >= 800) & (ph < 2500)):.2f}; '
          f'in 200-2000: {np.mean((ph >= 200) & (ph < 2000)):.2f}')
    print(f'  on-frames per target window: mean {on.mean():.2f}, fully on (>= 5.99) {np.mean(on >= 5.99):.2f}')
    print(f'  ON density {np.mean(DENS):.2e} emitters / px / frame = {np.mean(DENS) / (PXNM * 1e-3) ** 2:.3f} / um^2 (65 nm px)')
    print(f'  bg mean {np.mean(BG):.3f} photons / px / frame; read noise median {np.mean(RN):.2f} e; gain {float(GA.rsim("0.5nM").sim.gain):.3f} e/ADU')
    print(f'  pattern pitch px: {np.unique(pit.round(3))[:6]} (mean {pit.mean():.4f} px = {pit.mean() * PXNM:.1f} nm); depth mean {dep.mean():.3f} '
          f'[{dep.min():.2f}, {dep.max():.2f}]' + (f'; phase step |d| mean {np.concatenate(SPC).mean():.3f} rad' if SPC else ''))
    if SIG: print(f'  PSF Gaussian-fit sigma (bright iso emitters, n {len(SIG)}): median {np.median(SIG):.3f} px = {np.median(SIG) * PXNM:.1f} nm')
