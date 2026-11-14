"""Site-size study, step 1: every method table of the 80 / 40 nm rulers (and the GT simulations) drift-corrected by DME v2 with ONE setting
(abl_lib.dme = simcode.util.drift.dme_v2 with its defaults on the table's own rows, time unit = one 6-frame cycle, crlb = the table's sigma).

Tables written to out/nanoruler/sitesize/tabs/<movie>__<method>.npz (nra format: nm, frame = cycle) + _drift.npy (px, corrected = raw - d[cycle])
+ _drift.json. Method names:
  pub_<m>_dme      published table WITHOUT its drift correction (g2d / simflux -nodriftc; nr40 *_nodc) + DME v2
  pub_<m>_paper    published table as published (paper's Picasso / RCC drift), no DME
  pub_<m>_paperdme published table as published + DME v2 on top (residual check)
  bl_<m>_dme       re-implemented baseline (bl_run tables, no drift correction) + DME v2
  simcode_m12_dme  SIMCODE v2 pass 1 (variant A), cycle-aligned rows, max(sx, sy) <= 12 nm, stored calibrated sigma, + DME v2
  sim tables: simcode_m12_dme (and the GT-drift residual of DME in the json).
Real-movie DME traces get the mean of the published total correction (common frame with the site lists; a constant only).
usage: CUDA_VISIBLE_DEVICES=1 python ss_tables.py [movie ...]   movies: nr80 nr40 nr_a nr_c0 s40g_a s40g_c0"""
import json, os, sys
import numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import abl_lib as A
L = A.L; PX = A.PX; NR = A.NR
OUT = os.path.join(NR, 'sitesize', 'tabs')
PUB80 = '/mnt/data/SIMFLUX/Submission Data/80 nm rulers/Sim4_1_MMStack_Pos0.omeMod0.12Gauss10/'


def h5tab(fn, shift):
    p = h5py.File(fn, 'r')['locs'][:]
    return dict(frame=p['frame'].astype(np.int64), x_nm=(p['x'] + shift[0]) * PX, y_nm=(p['y'] + shift[1]) * PX, sx_nm=p['lpx'] * PX, sy_nm=p['lpy'] * PX,
                photons=p['photons'].astype(np.float64))


def npztab(fn):
    d = np.load(fn); t = {k: np.asarray(d[k], np.float64) for k in L.FIELDS}; t['frame'] = t['frame'].astype(np.int64); return t


def locstab(fn, sel='m12'):
    T = np.load(fn); T = {k: T[k] for k in T.files}; m = A.selection(T, sel); return A.to_nra(T, m)


def gt_drift(sim, ncyc):
    G = np.load(f'{sim}/sim_gt.npz', allow_pickle=True); d = G['drift'][:ncyc * 6].reshape(ncyc, 6, 2).mean(1).astype(np.float64)
    return d - d.mean(0)


def write(movie, name, tab, ncyc, drift='own', ref_mean=None, gt=None):
    fn = f'{OUT}/{movie}__{name}.npz'
    if os.path.exists(fn): print('exists', fn); return
    k = (tab['frame'] >= 0) & (tab['frame'] < ncyc) & np.isfinite(tab['x_nm']) & np.isfinite(tab['y_nm']) & (tab['sx_nm'] > 0) & (tab['sy_nm'] > 0)
    tab = {q: np.asarray(v)[k] for q, v in tab.items()}; cyc = tab['frame']; x = tab['x_nm'] / PX; y = tab['y_nm'] / PX
    rep = dict(movie=movie, method=name, n=int(len(cyc)), ncyc=ncyc, drift=drift)
    if drift == 'own':
        d, info = A.dme(x, y, cyc, tab['sx_nm'] / PX, tab['sy_nm'] / PX, ncyc); rep['dme'] = info
        if ref_mean is not None: d = d + ref_mean
        if gt is not None:
            g = gt_drift(gt, ncyc); e = (d - d.mean(0)) - g
            rep['dme_minus_gt_rms_nm'] = (e.std(0) * PX).tolist(); rep['gt_rms_nm'] = (g.std(0) * PX).tolist()
            rep['dme_minus_gt_rms_nm_w10'] = ((A.smooth(e, 10) - e.mean(0)).std(0) * PX).tolist()
    else:
        d = np.zeros((ncyc, 2))
    rep.update(drift_rms_nm=(d.std(0) * PX).tolist(), drift_p2p_nm=(np.ptp(d, 0) * PX).tolist())
    tab['x_nm'] = (x - d[cyc, 0]) * PX; tab['y_nm'] = (y - d[cyc, 1]) * PX
    L.save_table(fn, **tab); np.save(fn[:-4] + '_drift.npy', d); A.jdump(rep, fn[:-4] + '_drift.json')
    print(f'{movie} {name}: {len(cyc)} rows, DME {rep.get("dme", {}).get("chosen", "-")}, rms {rep["drift_rms_nm"][0]:.2f} / {rep["drift_rms_nm"][1]:.2f} nm'
          + (f', DME - GT rms {rep["dme_minus_gt_rms_nm"][0]:.2f} / {rep["dme_minus_gt_rms_nm"][1]:.2f}' if 'dme_minus_gt_rms_nm' in rep else ''), flush=True)


def nr80():
    sh = json.load(open(f'{NR}/analysis/transform.json'))['stages']; nc = 5000
    pic = np.loadtxt(PUB80 + 'g2d-nodriftc_190824_132639_drift.txt')[:nc].mean(0)       # common frame: mean of the paper's Picasso trace
    for m, h, st in (('smlm', 'g2d-nodriftc.hdf5', 'g2d-nodriftc'), ('simflux', 'simflux-nodriftc.hdf5', 'simflux-nodriftc')):
        write('nr80', f'pub_{m}_dme', h5tab(PUB80 + h, sh[st]['shift_px']), nc, ref_mean=pic)
        write('nr80', f'pub_{m}_paper', npztab(f'{NR}/analysis/tab_pub_{m}.npz'), nc, drift='none')
        write('nr80', f'pub_{m}_paperdme', npztab(f'{NR}/analysis/tab_pub_{m}.npz'), nc)
        write('nr80', f'bl_{m}_dme', npztab(f'{NR}/baselines/tab_bl_{m}.npz'), nc, ref_mean=pic)
    write('nr80', 'simcode_m12_dme', locstab(f'{NR}/ablation/runs/p1/A/locs.npz'), nc, ref_mean=pic)


def nr40():
    R = f'{NR}/nr40/ref'; nc = 6666; rm = np.load(f'{R}/drift_pub_total_simflux.npy')[:nc].mean(0)
    for m in ('smlm', 'simflux'):
        write('nr40', f'pub_{m}_dme', npztab(f'{R}/tab_pub_{m}_nodc.npz'), nc, ref_mean=rm)
        write('nr40', f'pub_{m}_paper', npztab(f'{R}/tab_pub_{m}.npz'), nc, drift='none')
        write('nr40', f'pub_{m}_paperdme', npztab(f'{R}/tab_pub_{m}.npz'), nc)
        write('nr40', f'bl_{m}_dme', npztab(f'{R}/tab_bl_{m}.npz'), nc, ref_mean=rm)
    write('nr40', 'simcode_m12_dme', locstab(f'{NR}/nr40/runs/real/locs_p1.npz'), nc, ref_mean=rm)


def sims(s):
    if s in ('nr_a', 'nr_c0'):
        write(s, 'simcode_m12_dme', locstab(f'{NR}/ablation/runs/sim{s[3:]}_p1/A/locs.npz'), 2000, gt=f'{NR}/sim/{s}')
    else:
        write(s, 'simcode_m12_dme', locstab(f'{NR}/nr40/runs/{s}/locs_p1.npz'), 4000, gt=f'{NR}/nr40/sim/{s}')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); A.title('nr-sitesize')
    for mv in (sys.argv[1:] or ['nr80', 'nr40', 'nr_a', 'nr_c0', 's40g_a', 's40g_c0']):
        nr80() if mv == 'nr80' else nr40() if mv == 'nr40' else sims(mv)
