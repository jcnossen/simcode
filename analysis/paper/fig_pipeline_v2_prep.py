"""Data of the mini-images of Fig. 1 (analysis/paper/fig_pipeline_v2.py), CPU only, cached in
out/paper_figs_v2/cache_fig1/fig1_data.npz (+ pattern_est.pt, the estimated pattern of the 1809_2 run of 2026-11-15).

- real frames: one 6-frame window of the 1809/2 movie (Fig 3 data set) from the cached copy of its first 12012 frames
  (out/archive/optclean/m1809_first12012.tif, raw ADU), in photo-electrons with the v2 camera calibration
  (models/insitu/calib_1809_2_psf_v2.json: per-pixel offset map, gain); the window (start % 6 == 0, so frame t = angle i,
  step j as in the run) with the most bright, isolated spots among the first 3000 frames;
- pass-1 networks (m1809 detector + emitter network, the shipped checkpoints) run on that window on the CPU: detection
  probability map p, candidates, and for the brightest isolated candidate its 21 x 21 x 6 patch, 6 intensities with
  uncertainties (sqrt of the diagonal of the network covariance), 6 backgrounds, position x, y, z with sigma_x, sigma_y,
  sigma_z;
- the estimated pattern of the run (k, phases, depth, relative intensities, phase drift per 150-frame knot);
- the clean emitter-windows of the pattern-estimation plot (si_phase_decouple_cleanrows.npz: pass-1 intensities and
  positions of the run) -> normalized intensity vs pattern phase, a random subset;
- one simulated training scene of the m1809 simulator (simulate/acquisition.AcquisitionSim with the m1809 in-situ PSF,
  generic pass-2 patterns, p_std 0): 6 frames, ground-truth positions, illumination maps, for 3 FOVs;
- the SIMCODE zoom render of Fig 3 (panel I, the embedded 757 x 757 image of 1809_2_fig3_psfv2.pdf, via pdfimages).

usage: PYTHONPATH=. python analysis/paper/fig_pipeline_v2_prep.py"""
import os, sys, json, shutil, subprocess, glob
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); sys.path[:0] = [REPO]

OUT = '/home/jc/simcode2/out/paper_figs_v2/cache_fig1'
TIF = '/home/jc/simcode2/out/archive/optclean/m1809_first12012.tif'
RUN = '/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/runs/simcode_v2_psfv2_20261115'
CALIB = 'insitu/calib_1809_2_psf_v2.json'
FIG3 = '/home/jc/simcode2/out/paper_figs_v2/1809_2_fig3_psfv2.pdf'
CLEAN = '/home/jc/simcode2/out/paper_figs_v2/si_phase_decouple_cleanrows.npz'
torch.set_num_threads(8)


def frames_window():
    import tifffile
    from simcode.processing.common import load_calib
    cal = load_calib(CALIB, dev='cpu')
    with tifffile.TiffFile(TIF) as t:
        n = 3000
        mov = np.stack([t.pages[i].asarray() for i in range(n)]).astype(np.float32)
    ph = (mov - cal['offset'][None]) * cal['gain']
    from scipy.ndimage import maximum_filter, uniform_filter
    scores = []
    for f0 in range(0, n - 6, 6):
        s = uniform_filter(ph[f0:f0 + 6].sum(0), 3)
        bgv = np.median(s)
        mx = (s == maximum_filter(s, 9)) & (s > bgv + 60)
        yy, xx = np.nonzero(mx)
        ok = (xx > 12) & (xx < 108) & (yy > 12) & (yy < 108)
        scores.append((int(ok.sum()), f0))
    return [f0 for _, f0 in sorted(scores, reverse=True)], ph, cal


@torch.no_grad()
def pass1(win):
    from simcode import config as C
    from simcode.emitter import model as EM
    from simcode.detector import model as DM
    from simcode.detector.select import select_subpixel
    from simcode.simulate.simulator import Simulator
    EM.dev = 'cpu'
    det, _ = DM.load(C.model(C.DETECTOR), dev='cpu')
    det = det.float()
    em = EM.load(C.model(C.EMITTER)).float(); em.bf16 = False
    col = EM.PSFCol(Simulator(CALIB, H=32, W=32, T=6, device='cpu'))
    x = torch.tensor(win[None]).float()
    dd = det(x)
    lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
    H, W = lgm.shape[1:]
    c = select_subpixel(lgm, dd['dxy'].float(), int(640 * H * W / 64 ** 2), p_min=0.2, radius=0.7)
    keep = torch.isfinite(c['logit'])
    b, iy, ix, xs, ys, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
    nbr = EM.nbr_map(lgm, b, iy, ix, xs, ys, lg)
    s5 = torch.sigmoid(lg) >= 0.3
    b, iy, ix, xs, ys, lg = (v[s5] for v in (b, iy, ix, xs, ys, lg))
    oh = EM.apply_to_candidates(em, col, x, None, b, iy, ix, xs, ys, nbr=nbr, p=torch.sigmoid(lg))
    z = EM.emitter_z(em, oh)
    R = dict(p=torch.sigmoid(lg).numpy(), x=oh['x'].numpy(), y=oh['y'].numpy(), I=oh['I'].float().numpy(),
             sI=(oh['L'].float() ** 2).sum(-1).sqrt().numpy(), bg=oh['bg'].float().numpy(),
             sxy=oh['lsxy'].float().exp().numpy(), z=z['z'].numpy(), sz=z['lsz'].exp().numpy(),
             ix=ix.numpy(), iy=iy.numpy())
    pmap = torch.sigmoid(lgm[0]).numpy()
    return R, pmap


def pick(R, win):
    """brightest candidate with p >= 0.63, on in all 6 frames (each intensity >= 1/4 of the mean), >= 14 px from the edge and no other candidate (p >= 0.3) within 8 px"""
    n = len(R['p']); best, bv = None, -1
    for k in range(n):
        if R['p'][k] < 0.63 or R['I'][k].min() < 0.25 * R['I'][k].mean(): continue   # on through the window
        if not (14 <= R['x'][k] <= 105 and 14 <= R['y'][k] <= 105): continue
        d = np.hypot(R['x'] - R['x'][k], R['y'] - R['y'][k]); d[k] = 1e9
        if d.min() < 8: continue
        v = R['I'][k].sum()
        if v > bv: best, bv = k, v
    return best


def sim_scene():
    from simcode.simulate import acquisition as A
    A.dev = 'cpu'
    torch.manual_seed(3)
    S = A.AcquisitionSim(preset='0.5nM', H=40, W=40, T=6, npsf=1, psf_jitter=False, seed=0,
                         calib='m1809/calib_1809_2_psf_v2.json', dens=0.03)
    S.p_std = 0.0
    sc = S.scene(3, struct_frac=0.7, pair_frac=0.3)
    g = sc['gt'] if 'gt' in sc else sc
    return dict(sim_frames=sc['frames'].numpy(), sim_illum=sc['illum'].numpy(),
                sim_fov=g['fov'].numpy(), sim_x=g['x'].numpy(), sim_y=g['y'].numpy())


def main():
    os.makedirs(OUT, exist_ok=True)
    pe = f'{OUT}/pattern_est.pt'
    if not os.path.exists(pe):
        shutil.copy(f'{RUN}/pattern_est.pt', pe)
    P = torch.load(pe, weights_only=False, map_location='cpu')
    D = dict(pat_k=P['k'].numpy(), pat_phi=P['phi'].numpy(), pat_m=P['m'].numpy(), pat_logrel=P['logrel'].numpy(),
             pat_dphi=P['dphi'].numpy(), pat_knot=P['knot_frames'])
    order, ph, cal = frames_window()
    best = None                    # among the 60 windows with the most bright spots: the brightest isolated emitter on in all 6 frames
    for f0 in order[:60]:
        R, pmap = pass1(ph[f0:f0 + 6])
        k = pick(R, None)
        if k is not None and (best is None or R['I'][k].sum() > best[0]):
            best = (R['I'][k].sum(), f0, R, pmap, k)
    _, f0, R, pmap, k = best; win = ph[f0:f0 + 6]
    print('window start', f0)
    print('picked', k, {q: R[q][k] for q in R})
    cx, cy = int(R['ix'][k]), int(R['iy'][k])
    D.update(win=win, win_f0=f0, pmap=pmap, gain=cal['gain'], **{'c_' + q: v for q, v in R.items()}, pick=k,
             patch=win[:, cy - 10:cy + 11, cx - 10:cx + 11])
    # clean emitter-windows: normalized intensity vs pattern phase (sin convention phi_ij = psi - pi/2: plotted vs the
    # argument of the cos model k.r - psi_s - eps; the shift is irrelevant for the picture)
    C = np.load(CLEAN, allow_pickle=True)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(C['w']), 4000, replace=False)
    D.update(cl_w=C['w'][sel], cl_x=C['x'][sel], cl_y=C['y'][sel], cl_I=C['I'][sel], cl_phot=C['phot'][sel])
    D.update(sim_scene())
    # Fig 3 SIMCODE zoom render (panel I = the 8th 757 x 757 image)
    tmp = f'{OUT}/fig3img'
    subprocess.run(['pdfimages', '-png', '-f', '1', '-l', '1', FIG3, tmp], check=True)
    ims = sorted(glob.glob(tmp + '-*.png'))
    import matplotlib.image as mpimg
    D['render_I'] = mpimg.imread(ims[8]); D['render_E'] = mpimg.imread(ims[4]); D['render_A'] = mpimg.imread(ims[0])
    for f in ims: os.remove(f)
    np.savez_compressed(f'{OUT}/fig1_data.npz', **D)
    print('saved', f'{OUT}/fig1_data.npz')


if __name__ == '__main__':
    main()
