"""Validation of a simulated nanoruler movie (sim_nr.py): python analysis/nanoruler/sim_nr_check.py <sim_dir> [quick_smlm hdf5 of the sim]
 (a) the stored movie un-reordered to acquisition order: box-sum photons of isolated events per frame against the GT per-frame
     photons (and the same with the wrong frame mapping as a control)
 (b) mean frame level, background, ON density, on-times, fully-on window photons; quick Gaussian SMLM statistics vs the real movie
 (c) pattern per STORED step fitted from the MOVIE box sums of fully-on cycle-aligned windows (GT positions only) against the
     calibration pattern block (calib convention: step = stored frame % 6, angle = step // 3), and against the wrong convention
 (d) sim_gt.npz against evalproto.lib.gt_windows_fast (the matching used by analysis/psf3d_nets/sigcal.py) and the exact windows
CPU only."""
import json, os, sys
sys.path.insert(0, '/home/jc/simcode2'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tifffile
from simcode2.util.proctitle import set_title
import sim_nr_lib as L
set_title('nr-simchk')
D = sys.argv[1].rstrip('/')
G = dict(np.load(f'{D}/sim_gt.npz', allow_pickle=True)); NF = int(G['nf']); H, W = (int(v) for v in G['shape'])
C = json.load(open(f'{D}/calib_sim.json')); cam = np.load(C['camera']['maps']); S = np.load(f'{D}/sites.npz')
mov = tifffile.imread(f'{D}/sim_movie.tif')
print(f'# {D}: movie {mov.shape} {mov.dtype}, events {len(G["x"])}, sites {len(S["x"])}, rulers {len(S["ruler_cx"])}')
gain, off = C['camera']['gain_e_per_adu'], cam['offset_adu']
x, y, t0, t1, site = G['x'], G['y'], G['t0'], G['t1'], G['site']
rul = S['ruler'][site]
B = 7   # box half-size (15 x 15 px)


def box(frames, ev):
    """background-subtracted box-sum photons of event ev in the given STORED frames (GT mean bg per frame x static map)"""
    cx, cy = int(round(float(x[ev]))), int(round(float(y[ev])))
    sl = (slice(cy - B, cy + B + 1), slice(cx - B, cx + B + 1))
    ph = (mov[frames][(slice(None),) + sl].astype(np.float64) - off[sl]) * gain
    bg = G['bg_map'][sl][None] * (G['bg_frame'][frames] / G['bg_frame'].mean())[:, None, None]
    return (ph - bg).sum((1, 2))


# isolated events: no other event within 18 px (box 15 x 15 + PSF extent) on within [t0 - 8, t1 + 8]
iso = np.ones(len(x), bool)
from scipy.spatial import cKDTree
nb = cKDTree(np.c_[S['x'], S['y']]).query_ball_point(np.c_[S['x'], S['y']], 18.0)
for i in range(len(x)):
    o = np.nonzero(np.isin(site, nb[site[i]]))[0]; o = o[o != i]
    if ((t0[o] < t1[i] + 8) & (t1[o] > t0[i] - 8)).any(): iso[i] = False
inside = (t0 > 12) & (t1 < NF - 12)
print(f'isolated events (no other event within 18 px and 8 frames): {iso.sum()} of {len(x)}')

# ------------------------------------------------------------------ (a)
print('\n(a) un-reordered movie vs GT per-frame photons (box 15 x 15, isolated events with on-time >= 30 frames)')
sel = np.nonzero(iso & inside & (t1 - t0 >= 30))[0]
rs = np.random.RandomState(0); show = rs.choice(sel, min(6, len(sel)), replace=False)
cc, cw, sl_, lk, lkw = [], [], [], [], []
for ev in sel:
    ta = np.arange(int(np.floor(t0[ev])) - 6, int(np.ceil(t1[ev])) + 6)          # acquisition frames
    gt = np.zeros(len(ta)); m = G['fr_ev'] == ev
    gt[G['fr_acq'][m] - ta[0]] = G['fr_ph'][m]
    good = box(L.stored_of_acq(ta), ev)                                          # movie read at the stored index of each acquisition frame
    wrong = box(ta, ev)                                                          # control: stored movie taken as if in acquisition order
    cc.append(np.corrcoef(gt, good)[0, 1]); cw.append(np.corrcoef(gt, wrong)[0, 1]); sl_.append((gt * good).sum() / (gt * gt).sum())
    offm = gt == 0
    lk.append(good[offm].mean() / gt[~offm].mean()); lkw.append(wrong[offm].mean() / gt[~offm].mean())
    if ev in show:
        j = np.argmax(gt > 0)
        print(f'  event {ev:5d} site {site[ev]:3d} t0 {t0[ev]:9.2f} t1 {t1[ev]:9.2f}: corr {cc[-1]:.3f} (wrong order {cw[-1]:.3f}), slope {sl_[-1]:.3f}')
        print('     acq frame   ' + ' '.join(f'{v:6d}' for v in ta[j - 2:j + 8]))
        print('     GT photons  ' + ' '.join(f'{v:6.0f}' for v in gt[j - 2:j + 8]))
        print('     movie box   ' + ' '.join(f'{v:6.0f}' for v in good[j - 2:j + 8]))
print(f'  {len(sel)} events: corr median {np.median(cc):.3f} (p5 {np.percentile(cc, 5):.3f}); wrong frame order: median {np.median(cw):.3f}; '
      f'slope box / GT median {np.median(sl_):.3f} (GT psf_sum on the 41 px ROI {float(G["psf_sum"]):.3f})')
print(f'  mean box signal in the 12 off-frames around the event / mean on-frame GT: {np.mean(lk):+.4f} (wrong order {np.mean(lkw):+.4f})')

# ------------------------------------------------------------------ (b)
print('\n(b) statistics vs targets')
sst = np.arange(NF) % 6
mf = mov.reshape(NF, -1).mean(1)
print(f'  mean frame level {mf.mean():.3f} ADU (real movie 103.276); per stored step ' + ' '.join(f'{mf[sst == s].mean():.3f}' for s in range(6)) +
      '  (real, stored order: 103.371 103.362 103.363 103.186 103.188 103.189)')
print(f'  offset map of the crop: mean {off.mean():.3f} median {np.median(off):.2f} ADU')
sub = (mov[::7].astype(np.float32) - off[None]) * gain
med = np.median(sub, 0)
print(f'  background: GT mean {G["bg_frame"].mean():.3f} ph/px/frame (nominal 1.65 = Gaussian-fit bg; default 1.45 matches the mean level), angle factors {G["bg_frame"][sst < 3].mean() / G["bg_frame"].mean():.3f} : '
      f'{G["bg_frame"][sst >= 3].mean() / G["bg_frame"].mean():.3f} (target 1.03 : 0.97); movie per-pixel temporal median, mean over px {med.mean():.3f}, '
      f'movie mean minus offset {((mf.mean() - off.mean()) * gain):.3f} ph/px/frame (real: {(103.276 - 100.0) * 0.45:.3f} with offset 100)')
tt = np.arange(NF)
on = np.zeros(NF); np.add.at(on, G['fr_acq'], G['fr_onf'])
print(f'  ON emitters per frame {on.mean():.2f} = {on.mean() / (H * W):.3e} per px per frame (nominal 8.5e-5, default 9.5e-5; {on.mean() / (H * W) * 160000:.2f} per 400 x 400)')
dur = (t1 - t0)[(t0 >= 0) & (t1 <= NF)]
print(f'  on-time (events inside the movie, n {len(dur)}): mean {dur.mean():.1f} median {np.median(dur):.1f} p5/p25/p75/p95/p99 ' +
      ' / '.join(f'{v:.1f}' for v in np.percentile(dur, [5, 25, 75, 95, 99])) + '  (nominal real-movie figures: mean 23, median 11)')
cyc = np.arange(NF // 6)
full6 = (t0[:, None] <= 6 * cyc[None]) & (t1[:, None] >= 6 * cyc[None] + 6)
wp = G['ph6'][full6]
print(f'  fully-on cycle-aligned windows: {full6.sum()} ({full6.sum() / len(S["x"]):.0f} per site; {len(x) / len(S["x"]):.0f} events per site); GT window photons '
      f'p5/p50/p95 {np.percentile(wp, 5):.0f} / {np.median(wp):.0f} / {np.percentile(wp, 95):.0f}  (nominal 750 / 1400 / 2500; defaults tuned to the Gaussian-fit photons, see sim_nr.py)')
print(f'  per-frame fluctuation check: frame_sd {float(G["frame_sd"])}, step_sd {float(G["step_sd"])}, ang_sd {float(G["ang_sd"])}')
if len(sys.argv) > 2:
    import h5py
    def qs(path, area, nfr, name):
        Q = h5py.File(path, 'r')['locs'][:]
        g = (Q['chi2'] < 3.6) & (Q['snr'] > 8) & (Q['sx'] > 1.4) & (Q['sx'] < 2.4)
        al = g & (Q['frame'] % 6 == 0)
        pc = np.percentile(Q['photons'][g], [5, 50, 95]); pa = np.percentile(Q['photons'][al], [5, 50, 95])
        print(f'  {name:5s} good detections / window {g.sum() / nfr:6.2f} = {g.sum() / nfr / area * 1e4:.3f} per 100 x 100 px; photons p5/p50/p95 {pc[0]:.0f} / {pc[1]:.0f} / {pc[2]:.0f} '
              f'(cycle-aligned windows {pa[0]:.0f} / {pa[1]:.0f} / {pa[2]:.0f}); bg / px / 6 frames {np.median(Q["bg"][g]):.2f}; sx median {np.median(Q["sx"][g]):.3f} '
              f'(p5 {np.percentile(Q["sx"][g], 5):.3f}, p95 {np.percentile(Q["sx"][g], 95):.3f}); good fraction {g.mean():.3f}; chi2 median {np.median(Q["chi2"][g]):.3f}')
    print('  quick Gaussian SMLM (--sum 6; chi2 < 3.6, snr > 8, 1.4 < sx < 2.4); area = FOV minus the 6 px detection border')
    qs('/home/jc/simcode2/out/nanoruler/quick_smlm_w6.hdf5', 388 * 388, 30000, 'real')
    qs(sys.argv[2], (W - 12) * (H - 12), NF, 'sim')

# ------------------------------------------------------------------ (c)
print('\n(c) pattern per STORED step from the movie (fully-on cycle-aligned windows of isolated events; GT positions + GT sample drift and pattern phase drift)')
P = C['pattern']; k = np.array(P['k_rad_per_px']); ph_c = np.array(P['phase']); m_c = np.array(P['depth']); r_c = np.array(P['relint'])
ev_i, c_i = np.nonzero(full6 & iso[:, None])
Y = np.zeros((len(ev_i), 6))
for ev in np.unique(ev_i):
    m = ev_i == ev
    fr = (6 * c_i[m][:, None] + np.arange(6)[None]).ravel()
    Y[m] = box(fr, ev).reshape(-1, 6)
Yn = Y / Y.mean(1, keepdims=True)
import torch
GP = torch.load(f'{D}/gt_pattern.pt', weights_only=False)
print(f'  windows {len(Y)}; gt_pattern.pt: phi - calib phase max |diff| {float(np.abs(GP["phi"].numpy() - ph_c).max()):.2e}, patt[:, 2] - (phi[s] + dphi[a, f]) max |diff| '
      f'{float(np.abs(G["patt"][:, 2] - (GP["phi"].numpy()[np.arange(NF) % 6] + GP["dphi"].numpy()[(np.arange(NF) % 6) // 3, np.arange(NF)])).max()):.2e}')
def fitconv(ang_of_step, label):
    res = 0; out = []
    for s in range(6):
        f_ = 6 * c_i + s; a_ = ang_of_step[s]
        th = k[a_, 0] * (x[ev_i] + G['drift'][f_, 0]) + k[a_, 1] * (y[ev_i] + G['drift'][f_, 1]) - GP['dphi'][a_, f_].numpy()
        A = np.c_[np.ones(len(th)), np.cos(th), np.sin(th)]
        c = np.linalg.lstsq(A, Yn[:, s], rcond=None)[0]; res += ((Yn[:, s] - A @ c) ** 2).sum()
        out.append((c[0], np.hypot(c[1], c[2]) / c[0], np.arctan2(c[2], c[1])))
    print(f'  {label}: residual rms {np.sqrt(res / Yn.size):.4f}')
    return out
o = fitconv([0, 0, 0, 1, 1, 1], 'calib convention (angle = stored step // 3)')
for s in range(6):
    d = (o[s][2] - ph_c[s] + np.pi) % (2 * np.pi) - np.pi
    print(f'    stored step {s}: relint {o[s][0]:.3f} (calib {r_c[s]:.3f})  depth {o[s][1]:.3f} (calib {m_c[s]:.2f})  phase {o[s][2]:+.3f} (calib {(ph_c[s] + np.pi) % (2 * np.pi) - np.pi:+.3f}, diff {d:+.3f} rad)')
fitconv([0, 1, 0, 1, 0, 1], 'wrong convention (angle = stored step % 2, i.e. movie not reordered)')
# GT per-frame photons of the same windows against the calibration model (normalized 6-vectors)
key = G['fr_ev'].astype(np.int64) * NF + G['fr_frame']; od = np.argsort(key); ks = key[od]
fr = 6 * c_i[:, None] + np.arange(6)[None]
gp = G['fr_ph'][od][np.searchsorted(ks, ev_i[:, None] * NF + fr)]
s6 = np.arange(6); a6 = s6 // 3
th = k[a6, 0][None] * (x[ev_i, None] + G['drift'][fr, 0]) + k[a6, 1][None] * (y[ev_i, None] + G['drift'][fr, 1]) - GP['dphi'].numpy()[a6[None], fr]
q = r_c[None] * (1 + m_c[None] * np.cos(th - ph_c[None]))
gn = gp / gp.mean(1, keepdims=True); qn = q / q.mean(1, keepdims=True)
cr = [np.corrcoef(gn[i], qn[i])[0, 1] for i in range(len(gn))]
aw = s6 % 2; sw = s6 // 2          # wrong: stored frame j taken as acquisition pattern index j
thw = k[aw, 0][None] * x[ev_i, None] + k[aw, 1][None] * y[ev_i, None]
qw = r_c[3 * aw + sw][None] * (1 + m_c[3 * aw + sw][None] * np.cos(thw - ph_c[3 * aw + sw][None]))
crw = [np.corrcoef(gn[i], qw[i])[0, 1] for i in range(len(gn))]
print(f'  GT per-frame photons (stored order) vs calib model at the GT position: rms of normalized residual {np.sqrt(((gn - qn) ** 2).mean()):.4f}, '
      f'max |residual| {np.abs(gn - qn).max():.4f}, corr median {np.median(cr):.4f}; with the interleaved (acquisition) step order assigned to the stored frames: corr median {np.median(crw):.4f}')

# ------------------------------------------------------------------ (d)
print('\n(d) sim_gt.npz with simcode2.evalproto.lib.gt_windows_fast (as analysis/psf3d_nets/sigcal.py load / cmp)')
from simcode2.evalproto import lib as E
G['nf'] = int(G['nf'])
GW = E.gt_windows_fast(G, grid=3); GX = L.gt_windows_exact(G, grid=3)
print(f'  gt_windows_fast rows {GW.shape}, exact rows {GX.shape}; fast target flag set {int(GW[:, 5].sum())} (evalproto CROP {E.CROP}, PX {E.PX})')
kf = (GW[:, 0] * len(x) + GW[:, 1]).astype(np.int64); kx = (GX[:, 0] * len(x) + GX[:, 1]).astype(np.int64)
com, ia, ib = np.intersect1d(kf, kx, return_indices=True)
print(f'  rows in both {len(com)}; only fast {len(kf) - len(com)}; only exact {len(kx) - len(com)}')
for nm, sel_ in (('all common rows', np.ones(len(com), bool)), ('fully on (exact min on-fraction 1)', GX[ib, 6] >= 1), ('partially on', GX[ib, 6] < 1),
                 ('cycle-aligned (w % 6 == 0)', GX[ib, 0] % 6 == 0), ('w % 6 == 3', GX[ib, 0] % 6 == 3)):
    r = GW[ia, 4][sel_] / np.maximum(GX[ib, 4][sel_], 1e-9)
    print(f'    {nm:36s} n {sel_.sum():7d}  P fast / P exact: median {np.median(r):.4f}, |dev| > 5 %: {100 * np.mean(np.abs(r - 1) > 0.05):.2f} %, > 20 %: {100 * np.mean(np.abs(r - 1) > 0.2):.2f} %')
al = (GX[:, 0] % 6 == 0) & (GX[:, 6] >= 1)
p6 = G['ph6'][GX[al, 1].astype(int), (GX[al, 0] // 6).astype(int)]
print(f'  ph6 (with per-frame fluctuation) / exact P on fully-on aligned windows: median {np.median(p6 / GX[al, 4]):.4f}, sd {np.std(p6 / GX[al, 4]):.4f}')
