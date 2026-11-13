"""Synthetic check of nra_lib: simulated 80 nm rulers (2 and 3 sites, PAINT-like binding events lasting several cycles,
uniform background localizations, some crowded rulers), two methods on the same events:
  A: sigma = 180 / sqrt(N) nm, reported correctly
  B: half the sigma of A plus a 2 nm site-independent jitter, reported sigma = CRLB (so under-reported, ratio rising with N),
     constant offset (+7, -4) nm
Prints truth vs estimate. Output: out/nanoruler/analysis/selftest/"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nra_lib as L
O = '/home/jc/simcode2/out/nanoruler/analysis/selftest/'; os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(5); FOV = 26000.0
nr = 260; ctr = rng.uniform(300, FOV - 300, (nr, 2)); ang = rng.uniform(0, np.pi, nr); n3 = rng.random(nr) < 0.6
sx, sy, sn = [], [], []
for r in range(nr):
    u = np.array([np.cos(ang[r]), np.sin(ang[r])])
    for k in ((-1, 0, 1) if n3[r] else (-0.5, 0.5)):
        p = ctr[r] + 80.0 * k * u; sx.append(p[0]); sy.append(p[1]); sn.append(rng.poisson(rng.choice([4, 14, 14, 20])))   # events / site
fr, X, Y, N = [], [], [], []
for x0, y0, ne in zip(sx, sy, sn):
    for e in range(ne):
        t0 = rng.integers(0, 4990); ln = 1 + rng.geometric(0.2)
        for t in range(t0, min(t0 + ln, 5000)): fr.append(t); X.append(x0); Y.append(y0); N.append(rng.lognormal(np.log(1200), 0.3))
nb = 6000
fr = np.r_[fr, rng.integers(0, 5000, nb)]; X = np.r_[X, rng.uniform(0, FOV, nb)]; Y = np.r_[Y, rng.uniform(0, FOV, nb)]; N = np.r_[N, rng.lognormal(np.log(900), 0.3, nb)]
sA = 180 / np.sqrt(N); sB = 0.5 * sA; jit = 2.0
tabs = dict(A=dict(frame=fr, x_nm=X + sA * rng.normal(size=len(X)), y_nm=Y + sA * rng.normal(size=len(X)), sx_nm=sA, sy_nm=sA, photons=N),
            B=dict(frame=fr, x_nm=X + 7 + np.hypot(sB, jit) * rng.normal(size=len(X)), y_nm=Y - 4 + np.hypot(sB, jit) * rng.normal(size=len(X)),
                   sx_nm=sB, sy_nm=sB, photons=N, crlb_x_nm=sB, crlb_y_nm=sB))
for k, t in tabs.items(): L.save_table(O + 'tab_%s.npz' % k, **t)
ref = L.load_table(O + 'tab_A.npz')
c, n, sd = L.find_sites(ref['x_nm'], ref['y_nm'])
sites, rulers, st = L.build_rulers(c, n, sd)
L.save_sites(O + 'sites.json', sites, rulers, dict(stats=st))
print('simulated rulers %d (%d 3-site), sites %d; found' % (nr, n3.sum(), len(sx)), st)
from scipy.spatial import cKDTree
d, _ = cKDTree(np.c_[sx, sy]).query(np.array([[s['x'], s['y']] for s in sites])); print('auto sites within 5 nm of a true site: %d / %d' % ((d < 5).sum(), len(d)))
inb = N.copy(); inb[-nb:] = np.nan
for k in 'AB':
    tab = L.load_table(O + 'tab_%s.npz' % k); M = L.eval_method(tab, sites, rulers)
    e, ci = L.bootstrap(M, M['ruler_ok'], L.photon_edges(M, M['ruler_ok'], 4), nboot=200)
    s = tab['sx_nm'][:-nb]; tr = np.sqrt(np.median(s ** 2 + (jit ** 2 if k == 'B' else 0)))
    print('\nmethod %s: shift %s, rulers %d; true sd (median over locs of total sigma) ~ %.2f; true rms %.2f' % (
        k, np.round(M['shift'], 2), e['n_rulers'], tr, np.sqrt(np.mean(s ** 2 + (jit ** 2 if k == 'B' else 0)))))
    for q in ('dist_mean', 'dist_sd', 'site_rsd_x_median', 'site_rsd_y_median', 'site_csd_x_median', 'pooled_rsd_x', 'pooled_rsd_y', 'pooled_csd_x',
              'all_zsd_x', 'all_zsd_y', 'all_ratio_x', 'pair_rsd_x', 'pair_zsd_x', 'half_dx_rsd', 'half_expected_sd'):
        print('   %-22s %s' % (q, L.fmt(e, ci, q)))
    for i in range(4):
        p = 'bin%d_' % i; true = np.sqrt(e[p + 'sig_med_x'] ** 2 + (jit ** 2 if k == 'B' else 0))
        print('   bin %d phot %.0f: spread %s, sigma %.2f, true total %.2f, ratio %s, zsd %s' % (i, e[p + 'phot_med'], L.fmt(e, ci, p + 'rsd_x'), e[p + 'sig_med_x'], true,
                                                                                         L.fmt(e, ci, p + 'ratio_x'), L.fmt(e, ci, p + 'zsd_x')))
