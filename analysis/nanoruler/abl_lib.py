"""Nanoruler solve-variant ablation: shared helpers (paths, selections, drift, tables in the nra format).

Conventions of every abl_ script:
  - only cycle-aligned windows (table frame % 6 == 0) are used; cycle = frame // 6 (5000 cycles for the 30000-frame movie)
  - positions in movie pixels (65 nm); the nra tables are in nm in the common frame (= px x 65), frame = cycle index
  - drift traces are [n_cycles, 2] in px, mean removed; corrected = raw - drift[cycle]
Outputs under out/nanoruler/ablation/."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE, os.path.join(REPO, 'analysis', 'ablation_cov'), os.path.join(REPO, 'analysis', 'calib')):
    if p not in sys.path: sys.path.insert(0, p)
import nra_lib as L

NR = os.path.join(REPO, 'out', 'nanoruler'); OUT = os.path.join(NR, 'ablation'); PX = 65.0
SITES = os.path.join(NR, 'analysis', 'sites.json')
PICASSO_DRIFT = '/mnt/data/SIMFLUX/Submission Data/80 nm rulers/Sim4_1_MMStack_Pos0.omeMod0.12Gauss10/g2d-nodriftc_190824_132639_drift.txt'
SIGMA_MAX_NM = 12.0   # the paper's selection (simcode.strict.select): p >= 0.63, inside the run's crop, max(sx, sy) * 65 nm <= 12 nm


def title(t='nr-abl'):
    from simcode.util import proctitle
    proctitle.set_title(t)


def load_table(path):
    """a pipeline table (locs_p*.npz / locs.npz) as a dict of arrays, with 'crop' = the output region of its run (the table's
    <base>_info.json or locs_info.json next to it, else the run's pass-1 files: simcode.strict.run_crop), which the selection needs"""
    from simcode import strict as SP
    z = np.load(path); T = {k: z[k] for k in z.files}
    if 'crop' not in T:
        b = os.path.splitext(path)[0]
        for f in (b + '_info.json', os.path.join(os.path.dirname(path), 'locs_info.json')):
            if os.path.exists(f):
                c = json.load(open(f)).get('crop')
                if c: T['crop'] = np.asarray(c, np.float64); break
        else:
            T['crop'] = np.asarray(SP.run_crop(os.path.dirname(os.path.abspath(path))), np.float64)
    return T


def selection(T, sel='m12'):
    """row mask of a pipeline table (load_table): cycle-aligned AND the named selection: 'm12' = the paper's selection
    (simcode.strict.select with the 65 nm pixel and the table's crop), 'sel' / 'good' (those columns), 'all'.
    (The u rule 'u45' and the strict product 'strict' were removed 2026-11-14.)"""
    ca = (T['frame'] % 6 == 0) & np.isfinite(T['x']) & np.isfinite(T['y'])
    if sel == 'm12':
        from simcode import strict as SP
        if 'crop' not in T: raise KeyError("the table has no 'crop': load it with abl_lib.load_table")
        return ca & SP.select(T, SIGMA_MAX_NM, px_nm=PX)
    if sel in ('sel', 'good'): return ca & T[sel].astype(bool)
    if sel == 'all': return ca
    raise ValueError(f'{sel!r}: selections m12, sel, good, all (u45 / strict were removed 2026-11-14)')


def dme(x_px, y_px, cyc, sx_px, sy_px, ncyc, device=None):
    """DME v2 on the method's own localizations, time unit = one 6-frame cycle. Returns drift [ncyc, 2] px (mean removed), info."""
    from simcode.util.drift import dme_v2
    d, info = dme_v2(np.c_[x_px, y_px], cyc, np.c_[sx_px, sy_px], numframes=ncyc, return_info=True, device=device)
    d = np.asarray(d, np.float64); d = d - d.mean(0)
    return d, dict(chosen=[str(v) for v in info.get('chosen', ())], version=str(info.get('version', '')))


def picasso_drift(ncyc=None):
    d = np.loadtxt(PICASSO_DRIFT); d = d - d.mean(0)
    return d if ncyc is None else d[:ncyc]


def smooth(d, w):
    k = np.ones(w) / w
    pad = np.pad(d, ((w // 2, w - 1 - w // 2), (0, 0)), mode='edge')
    return np.stack([np.convolve(pad[:, i], k, mode='valid') for i in range(d.shape[1])], 1)


def site_spread(x_nm, y_nm, cyc, sites=None, rulers=None, radius=30.0, min_n=10):
    """median over sites of the per-site robust sd (x, y), pooled robust sd (x, y), n assigned: the cluster-sharpness check of a drift trace"""
    if sites is None: sites, rulers, _ = L.load_sites(SITES)
    tab = dict(frame=np.asarray(cyc, float), x_nm=np.asarray(x_nm, float), y_nm=np.asarray(y_nm, float), sx_nm=np.ones(len(x_nm)), sy_nm=np.ones(len(x_nm)),
               photons=np.ones(len(x_nm)))
    M = L.eval_method(tab, sites, rulers, radius, min_n)
    g = M['n'][M['L_site']] >= min_n
    return dict(site_x=float(np.nanmedian(M['S']['rsd_x'])), site_y=float(np.nanmedian(M['S']['rsd_y'])), pooled_x=float(L.rsd(M['L_rx'][g])),
                pooled_y=float(L.rsd(M['L_ry'][g])), n_assigned=int(len(M['L_site'])), n_sites=int((M['n'] >= min_n).sum()),
                half_dx_rsd=float(L.rsd(M['S']['half_dx'][np.isfinite(M['S']['half_dx'])])), half_dy_rsd=float(L.rsd(M['S']['half_dy'][np.isfinite(M['S']['half_dy'])])))


def drift_report(x_px, y_px, cyc, sx_px, sy_px, ncyc, name='', sites=SITES, log=print):
    """own DME v2 drift + its sanity checks against the published Picasso trace and the cluster spread. Returns drift, report dict."""
    S, R = L.load_sites(sites)[:2] if isinstance(sites, str) else sites
    d, info = dme(x_px, y_px, cyc, sx_px, sy_px, ncyc)
    rep = dict(n=int(len(x_px)), dme=info, drift_rms_nm=(d.std(0) * PX).tolist(), drift_p2p_nm=(np.ptp(d, 0) * PX).tolist())
    sp = lambda dd: site_spread((x_px - dd[cyc, 0]) * PX, (y_px - dd[cyc, 1]) * PX, cyc, S, R)
    rep['spread_none'] = sp(np.zeros((ncyc, 2))); rep['spread_own'] = sp(d)
    if os.path.exists(PICASSO_DRIFT) and ncyc == 5000:
        p = picasso_drift(ncyc); rep['spread_picasso'] = sp(p)
        rep['picasso_rms_nm'] = (p.std(0) * PX).tolist()
        for w in (1, 50, 250):
            a, b = smooth(d, w), smooth(p, w); e = (a - b) - (a - b).mean(0)
            rep[f'own_minus_picasso_rms_nm_w{w}'] = (e.std(0) * PX).tolist()
            rep[f'corr_w{w}'] = [float(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(2)]
    f = lambda s: f"site {s['site_x']:.2f} / {s['site_y']:.2f} pooled {s['pooled_x']:.2f} / {s['pooled_y']:.2f} half-movie centre sd {s['half_dx_rsd']:.2f} / {s['half_dy_rsd']:.2f} (n {s['n_assigned']})"
    log(f"{name}: DME v2 {info['chosen']} on {len(x_px)} locs; drift rms {rep['drift_rms_nm'][0]:.2f} / {rep['drift_rms_nm'][1]:.2f} nm, "
        f"p2p {rep['drift_p2p_nm'][0]:.1f} / {rep['drift_p2p_nm'][1]:.1f} nm")
    log(f"   spread nm (x / y), no drift correction: {f(rep['spread_none'])}")
    log(f"   own DME:                                {f(rep['spread_own'])}")
    if 'spread_picasso' in rep:
        log(f"   published Picasso trace:                {f(rep['spread_picasso'])}")
        log('   own - Picasso rms nm (x / y): raw %.2f / %.2f, 50-cycle mean %.2f / %.2f (corr %.2f / %.2f), 250-cycle mean %.2f / %.2f (corr %.2f / %.2f)' % (
            *rep['own_minus_picasso_rms_nm_w1'], *rep['own_minus_picasso_rms_nm_w50'], *rep['corr_w50'], *rep['own_minus_picasso_rms_nm_w250'], *rep['corr_w250']))
    return d, rep


def to_nra(T, m, drift=None, extra=None, sig=('sx', 'sy')):
    """pipeline table rows m -> dict in the nra input format (nm, frame = cycle); drift [ncyc, 2] px or None"""
    cyc = (T['frame'][m] // 6).astype(np.int64); x = T['x'][m].astype(np.float64); y = T['y'][m].astype(np.float64)
    if drift is not None: x = x - drift[cyc, 0]; y = y - drift[cyc, 1]
    o = dict(frame=cyc, x_nm=x * PX, y_nm=y * PX, sx_nm=T[sig[0]][m].astype(np.float64) * PX, sy_nm=T[sig[1]][m].astype(np.float64) * PX,
             photons=T['photons'][m].astype(np.float64), row=np.nonzero(m)[0])
    for k, v in (extra or {}).items(): o[k] = v
    return o


def jdump(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=float)


# ------------------------------------------------------------------ data sets of the ablation
# run = pipeline output dir (pass1_<f0>.npz, pattern_est.pt); head = captured pass-2 head outputs (abl_capture.py); front = gate front of the
# calibration fits (s1 pass 1, s2 pass 2); sim = simulated movie dir with GT; prod = the production calibration of that pass
_MC = os.path.join(NR, 'models', 'calib'); _PR = os.path.join(OUT, 'prod')
SETS = {
    'p1': dict(kind='real', run=os.path.join(NR, 'runs', 'real_p1'), front='s1', tag='est', ncyc=5000, prod=f'{_MC}/calib_sc_hd_nr80.json'),
    'p2': dict(kind='real', run=os.path.join(NR, 'runs', 'real_p1'), front='s2', tag='est2', ncyc=5000, prod=f'{_MC}/calib_p2L_nr80.json', head=f'{OUT}/head/real'),
}
# nr_c0g: control = nr_c0 with the TRUE pattern of the simulation given to the solve and to the pass-2 networks (prod/nr_c0g: links to the pass-1
# files of prod/nr_c0, pattern_est.pt = sim/nr_c0/gt_pattern.pt; see abl_all.sh stage gtpat)
for _s, _m in (('nr_a', 'nr_a'), ('nr_c0', 'nr_c0'), ('nr_c0g', 'nr_c0')):
    _k = 'sim' + _s[3:]
    SETS[_k + '_p1'] = dict(kind='sim', run=f'{_PR}/{_s}', front='s1', tag='est', ncyc=2000, prod=f'{_MC}/calib_sc_hd_nr80.json', sim=os.path.join(NR, 'sim', _m))
    SETS[_k + '_p2'] = dict(kind='sim', run=f'{_PR}/{_s}', front='s2', tag='est2', ncyc=2000, prod=f'{_MC}/calib_p2L_nr80.json', sim=os.path.join(NR, 'sim', _m),
                            head=f'{OUT}/head/{_s}')
MOVIES = {'real': dict(movie=os.path.join(NR, 'raw', 'sim4_1_reord.tif'), calib=os.path.join(NR, 'calib_nr80.json'), crop='5,5,390', chunk=1500),
          'nr_a': dict(movie=os.path.join(NR, 'sim', 'nr_a', 'sim_movie.tif'), calib=os.path.join(NR, 'sim', 'nr_a', 'calib_sim.json'), crop='5,5,246', chunk=1500),
          'nr_c0': dict(movie=os.path.join(NR, 'sim', 'nr_c0', 'sim_movie.tif'), calib=os.path.join(NR, 'sim', 'nr_c0', 'calib_sim.json'), crop='5,5,246', chunk=1500)}
