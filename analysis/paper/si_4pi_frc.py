"""SI figure si_4pi_frc_vs_frames: how many frames the real 4Pi movies need (half-split FRC vs number of frames), next to the
simulation's ground-truth FRC timeline (si_4pi_sim e). CPU only; reads the cached tables of si_4pi.py (cache_si_4pi/mt.npz,
npc.npz: drift corrected, z drift removed, registered; '4Pi-SIMCODE' = the standard selection (good, N >= 1000, sigma cuts),
'4Pi-SIMFLUX' = the published-style table) and, for the simulation, the sweep's region rows ($P4/sweep4pi/<tag>/locs_*.npz).

Estimator (ONE for the real data, the simulation and the FRC values / timeline panels of si_4pi_npc and si_4pi_mt, which read
their curves from this script's output): half-split FRC of the localizations of N frames, N = 1000 * 2^(k/4) and every T / 20 up to all T;
  x-y: simcode/evaluation/frc.py (4 nm pixels on the square grid of the data, integer rings, 5-ring smoothing), all rows of the
  figure tables in the region (no z window); x-z sections: (x, z) histograms with 5 nm pixels in 250 nm y-slabs over
  +-300 nm around the median z, correlation pooled over the slabs before normalisation (= the simulation's GT-FRC x-z
  estimator; rings of width 1 / (largest side), 5-ring smoothing); 1/7 threshold; resolution = the INTERPOLATED crossing by the sustained rule (first frequency after which the curve
  stays below 1/7 for 0.01 / nm; the rule of all 4Pi figures); the first-crossing value is stored too. Raw values per N
  (no running minimum). Three ways to choose the N frames and the two halves:
  blk     first N frames; halves = alternating blocks of b six-frame windows, b = min(500, max(1, N / 6 / 10)) (>= 10 blocks
          for every N, blocks shorter than N; 500 windows = the figures' blocks from N = 30 000 frames on). For N < 30 000 the
          blocks are shorter than the same-molecule correlation time (binding bursts / rebinding, up to ~300 windows on MT,
          see 'lag'), so both halves contain the same bursts: precision-dominated.
  ind     first N frames; halves = alternating blocks of b = min(500, N / 12) windows (two halves of N / 2 for N < 6000, the
          figures' 500-window blocks from 6000 frames on): (nearly) independent halves.
  spread  the 'ind' blocks spread evenly over the whole series (block k starts at k * T / K): same block structure, the
          frames taken from the whole series instead of its start (drift, site depletion, focus over the series).
  lag     same-molecule correlation vs time lag: pairs closer than 8 nm (x-y) vs pairs at 8-24 nm (per area), per lag bin.
  structure       MT: skeleton coverage = fraction of the skeleton (20 nm pixels; skeleton of the union of both full-series
                  tables) with >= k localizations within +-30 nm (k = 1, 3, 10); NPC: corners seen per pore = number of the 8
                  corners (45 deg sectors, per-pore rotation from the union of the full tables) with >= 1 / >= 3 localizations
                  in the ring annulus (25-85 nm), averaged over the 1459 pores, plus the valley / peak of the ring z profile
                  (select_curve_npc.pore_metrics) vs N.
  simulation      the same half-split FRC on the sweep's region rows (net_sig1 = the si_4pi_sim 4Pi-SIMCODE selection,
                  cl_pub = 4Pi-SIMFLUX), next to the GT-FRC timeline of ttf_fp.json.

usage:  export SIMCODE_CONFIG=~/simcode2/simcode.json
        python analysis/paper/si_4pi_frc.py compute     # ~20 min, 20 processes -> cache_si_4pi/frc_vs_frames.json
        python analysis/paper/si_4pi_frc.py fig         # -> out/paper_figs_v2/si_4pi_frc_vs_frames.{pdf,svg,png}, _caption.txt, _values.json
"""
import argparse, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, f'{REPO}/analysis/fourpi', f'{REPO}/analysis/fourpi_classical', f'{REPO}/analysis/fourpi_sim'):
    if p not in sys.path: sys.path.insert(0, p)
from simcode.config import root
P4 = root('fourpi')
OUT = os.path.expanduser('~/simcode2/out/paper_figs_v2')
CACHE = f'{OUT}/cache_si_4pi'
RES = f'{CACHE}/frc_vs_frames.json'
DREG = os.path.expanduser('~/simcode2/out/fourpi_sim/sweep/densregion.json')
PX = 128.0
THR = 1 / 7
PIX_XY, PIX_XZ, SLAB, ZH = 4.0, 5.0, 250.0, 300.0
BLK_MAX, NBLK_MIN = 500, 10
SIMTAGS = {'mtc_d01': 0.28, 'mtc_d03': 0.84, 'mtc_d05': 1.4}
SIM_LO, SIM_HI = 8.0 * 128, 52.0 * 128           # sweep4pi_eval LO, HI (px) -> nm
t0 = time.time()
def lg(s): print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)
def inrect(x, y, r): return (x >= r[0]) & (x < r[1]) & (y >= r[2]) & (y < r[3])


# ------------------------------------------------------------------------------------------------------------ estimator
def _pool(a, b, pix, acc):
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    df = 1.0 / (max(a.shape) * pix)
    fy = np.fft.fftfreq(a.shape[0], pix); fx = np.fft.fftfreq(a.shape[1], pix)
    k = (np.hypot(fy[:, None], fx[None]) / df).astype(int).ravel(); nmax = int(0.5 / pix / df)
    q = [np.bincount(k, v.ravel(), nmax + 1)[:nmax] for v in ((A * B.conj()).real, np.abs(A) ** 2, np.abs(B) ** 2)]
    if acc is None: return q + [df]
    return [acc[i] + q[i] for i in range(3)] + [df]


def _curve(acc):
    num, da, db, df = acc
    c = np.convolve(num / np.sqrt(np.maximum(da * db, 1e-30)), np.ones(5) / 5, 'same')
    return np.arange(len(c)) * df, c


def crossing_sustained(f, c, start=3, width=0.01):
    """interpolated 1/7 crossing, sustained rule (sweep4pi_ttf_fp.crossing_interp)"""
    if len(c) <= start: return float('nan')
    w = max(int(width / max(f[1] - f[0], 1e-12)), 1)
    for i in range(start, len(c)):
        if (c[i:min(i + w, len(c))] < THR).all():
            if i == start or c[i - 1] < THR: return float(1 / f[i])
            u = (c[i - 1] - THR) / (c[i - 1] - c[i])
            return float(1 / (f[i - 1] + u * (f[i] - f[i - 1])))
    return float(1 / f[-1])                                               # never below: Nyquist period


def crossing_first(f, c, start=3):
    b = np.nonzero(c[start:] < THR)[0]
    return float('nan') if len(b) == 0 else float(1 / f[b[0] + start])


def hist2(u, v, u0, u1, v0, v1, pix):
    nu, nv = max(int(round((u1 - u0) / pix)), 1), max(int(round((v1 - v0) / pix)), 1)
    return np.histogram2d(v, u, bins=(nv, nu), range=((v0, v0 + nv * pix), (u0, u0 + nu * pix)))[0]


def frc_xy(x, y, h, box=None):
    """the figures' x-y FRC curve (simcode/evaluation/frc.py: 4 nm pixels on the square grid of the data, integer rings, 5-ring smoothing)
    -> (sustained interpolated crossing = the value of all 4Pi figures, first crossing)"""
    from simcode.evaluation.frc import frc as _frc
    first, f, c = _frc(x[h], y[h], x[~h], y[~h], px=PIX_XY, units='nm', return_curve=True)
    return crossing_sustained(f, c), float(first)


def frc_xz(x, y, z, h, box, zc):
    acc = None
    for y0 in np.arange(box[2], box[3], SLAB):
        s = (y >= y0) & (y < y0 + SLAB)
        a = hist2(x[s & h], z[s & h], box[0], box[1], zc - ZH, zc + ZH, PIX_XZ)
        b = hist2(x[s & ~h], z[s & ~h], box[0], box[1], zc - ZH, zc + ZH, PIX_XZ)
        acc = _pool(a, b, PIX_XZ, acc)
    f, c = _curve(acc)
    return crossing_sustained(f, c), crossing_first(f, c)


def blk_of(N, mode):
    nw = max(N // 6, 1)
    if mode == 'blk': return int(min(BLK_MAX, max(1, nw // NBLK_MIN)))
    return int(min(BLK_MAX, max(1, nw // 2)))


def subset(w, N, Tw, mode):
    """rows of the first N frames ('blk', 'ind') or of the 'ind' blocks spread over the series ('spread'); -> (mask, half)"""
    b = blk_of(N, mode); nw = max(N // 6, 1)
    if mode in ('blk', 'ind'):
        m = w < nw; half = (w // b) % 2 == 0
    else:
        K = max(nw // b, 2); P = Tw / K
        k = np.floor(w / P).astype(np.int64); off = w - np.floor(k * P).astype(np.int64)
        m = (off < b) & (k < K); half = k % 2 == 0
    return m, half


def lag_excess(T, n=20000, seed=0):
    from scipy.spatial import cKDTree
    edges = np.array([1, 2, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000])
    rng = np.random.default_rng(seed); tr = cKDTree(np.c_[T['x'], T['y']]); q = rng.choice(len(T['x']), min(n, len(T['x'])), replace=False)
    H = []
    for r in (8.0, 24.0):
        L = tr.query_ball_point(np.c_[T['x'][q], T['y'][q]], r)
        dw = np.concatenate([np.abs(T['w'][np.asarray(l, int)] - T['w'][i]) for i, l in zip(q, L)])
        H.append(np.histogram(dw, edges)[0].astype(float))
    ratio = H[0] / np.maximum((H[1] - H[0]) / 8, 1e-9)                       # annulus 8-24 nm has 8x the area of r < 8 nm
    return dict(lag_edges_windows=edges.tolist(), ratio=ratio.tolist(), pairs_r8=H[0].astype(int).tolist())


# ------------------------------------------------------------------------------------------------------------ jobs
_D = {}


def _job(arg):
    key, meth, N, mode, metric = arg
    T = _D[key][meth]; box, zc, Tw = _D[key]['box'], _D[key]['zc'], _D[key]['Tw']
    m, half = subset(T['w'], N, Tw, mode)
    x, y, h = T['x'][m], T['y'][m], half[m]
    n = int(m.sum())
    if h.sum() < 50 or (~h).sum() < 50: return arg, dict(n=n, sus=float('nan'), first=float('nan'))
    if metric == 'xy': s, f = frc_xy(x, y, h, box)
    else: s, f = frc_xz(x, y, T['z'][m], h, box, zc)
    return arg, dict(n=n, sus=s, first=f)


def grid_of(T):
    """N = 1000 * 2^(k/4) and every T / 20 (dense enough at large N for a linear frames axis), up to T"""
    g = {int(round(1000 * 2 ** (k / 4))) for k in range(80) if 1000 * 2 ** (k / 4) < T} | {int(round(T * k / 20)) for k in range(1, 21)}
    return sorted(g)


def load_real():
    Z = np.load(f'{CACHE}/mt.npz'); J = json.load(open(f'{CACHE}/mt_geom.json'))
    d = json.load(open(DREG))['fixed']['sim_field_7.68um']; o = d['origin_px']; s = d['side_um'] * 1000
    regions = {}
    mt = {k: dict(x=Z[f'{k}_x'].astype(np.float64), y=Z[f'{k}_y'].astype(np.float64), z=Z[f'{k}_z'].astype(np.float64),
                  w=(Z[f'{k}_frame'] // 6).astype(np.int64)) for k in ('net', 'cl')}
    zc = float(np.median(mt['net']['z']))                                            # x-y: all rows of the figure tables; x-z: zc +- ZH (histogram range)
    # matched count as si_4pi.fig_mt: the most precise network rows (combined reported sigma), as many in boxes 0 + 1 as 4Pi-SIMFLUX keeps
    cores = J['info']['cores_nm']; inreg = lambda U: inrect(U['x'], U['y'], cores[0]) | inrect(U['x'], U['y'], cores[1])
    comb = np.sqrt((Z['net_sx'] * PX) ** 2 + (Z['net_sy'] * PX) ** 2 + (Z['net_sz'].astype(np.float64) * 1000) ** 2)
    n_m = int(inreg(mt['cl']).sum()); thr = float(np.sort(comb[inreg(mt['net'])])[n_m - 1])
    mt['net_m'] = {q: v[comb <= thr] for q, v in mt['net'].items()}
    big = (float(Z['net_x'].min()) + 500, float(Z['net_x'].max()) - 500, float(Z['net_y'].min()) + 500, float(Z['net_y'].max()) - 500)
    boxes = dict(mt_dense=(o[0] * PX, o[0] * PX + s, o[1] * PX, o[1] * PX + s), mt_box0=tuple(J['info']['cores_nm'][0]), mt_field=big)
    for nm, b in boxes.items():
        regions[nm] = dict(box=b, zc=zc, Tw=300000, T=1800000,
                           **{k: {q: v[inrect(T['x'], T['y'], b)] for q, v in T.items()} for k, T in mt.items()})
    Zn = np.load(f'{CACHE}/npc.npz')
    npc = {k: dict(x=Zn[f'{k}_x'].astype(np.float64), y=Zn[f'{k}_y'].astype(np.float64), z=Zn[f'{k}_z'].astype(np.float64),
                   w=(Zn[f'{k}_frame'] // 6).astype(np.int64)) for k in ('net', 'cl')}
    fb = Zn['fb']; cx, cy = 0.5 * (fb[0] + fb[1]), 0.5 * (fb[2] + fb[3]); b = (cx - 5000, cx + 5000, cy - 5000, cy + 5000)
    zcn = float(np.median(npc['net']['z'][inrect(npc['net']['x'], npc['net']['y'], b)]))
    npc['net_m'] = {q: v[Zn['net_matched']] for q, v in npc['net'].items()}         # matched count of si_4pi_npc
    regions['npc_center'] = dict(box=b, zc=zcn, Tw=114000, T=684000,
                                 **{k: {q: v[inrect(T['x'], T['y'], b)] for q, v in T.items()} for k, T in npc.items()})
    return regions, (Z, J, Zn)


def load_sim():
    out = {}
    for tag in SIMTAGS:
        r = {}
        for meth, fn in (('net', 'net_sig1'), ('cl', 'cl_pub')):
            L = np.load(f'{P4}/sweep4pi/{tag}/locs_{fn}.npz')
            r[meth] = dict(x=L['x'].astype(np.float64), y=L['y'].astype(np.float64), z=L['z'].astype(np.float64), w=L['w'].astype(np.int64))
        zc = float(np.median(r['net']['z']))
        out[f'sim_{tag}'] = dict(box=(SIM_LO, SIM_HI, SIM_LO, SIM_HI), zc=zc, Tw=5000, T=30000, **r)
    return out


# ------------------------------------------------------------------------------------------------------------ structure
def skeleton_ref(U, box, pix=20.0):
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.filters import threshold_otsu
    H = hist2(U[0], U[1], *box, pix); Hs = gaussian_filter(H, 1.0)
    L = np.log1p(Hs); t = threshold_otsu(L[L > 0])
    mask = remove_small_objects(L > t, max_size=29)
    sk = skeletonize(mask)
    return sk, dict(pix=pix, threshold_log=float(t), skeleton_px=int(sk.sum()), skeleton_um=float(sk.sum() * pix / 1000))


def coverage(x, y, sk, box, pix=20.0, ks=(1, 3, 10)):
    from scipy.ndimage import uniform_filter
    H = hist2(x, y, *box, pix); C = uniform_filter(H, 3, mode='constant') * 9          # counts within +-30 nm (3 x 3 px)
    v = C[sk]
    return {str(k): float(np.mean(v >= k)) for k in ks}


def npc_corners_ref(T_union, P):
    """per pore: 8-fold rotation from the union of the full tables (mean of exp(8 i theta) over the ring annulus)"""
    from scipy.spatial import cKDTree
    tr = cKDTree(np.c_[T_union[0], T_union[1]]); L = tr.query_ball_point(P[:, :2], 90.0)
    rot = np.zeros(len(P))
    for i, ((cx, cy, _), idx) in enumerate(zip(P, L)):
        idx = np.asarray(idx, int); dx, dy = T_union[0][idx] - cx, T_union[1][idx] - cy; r = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)[(r > 25) & (r < 85)]
        rot[i] = np.angle(np.exp(8j * th).sum()) / 8 if len(th) else 0.0
    return rot


def npc_corners(x, y, P, rot):
    from scipy.spatial import cKDTree
    tr = cKDTree(np.c_[x, y]); L = tr.query_ball_point(P[:, :2], 90.0)
    n1 = np.zeros(len(P)); n3 = np.zeros(len(P))
    for i, ((cx, cy, _), idx) in enumerate(zip(P, L)):
        idx = np.asarray(idx, int)
        if len(idx) == 0: continue
        dx, dy = x[idx] - cx, y[idx] - cy; r = np.hypot(dx, dy); ok = (r > 25) & (r < 85)
        k = np.round((np.arctan2(dy[ok], dx[ok]) - rot[i]) / (np.pi / 4)).astype(int) % 8
        c = np.bincount(k, minlength=8); n1[i] = (c >= 1).sum(); n3[i] = (c >= 3).sum()
    return dict(corners_ge1=float(n1.mean()), corners_ge3=float(n3.mean()), pores_ge6_corners_ge1=float(np.mean(n1 >= 6)))


def structure(regions, raw):
    import select_curve_npc as SN
    Z, J, Zn = raw
    out = {}
    for key in ('mt_dense', 'mt_box0'):
        R = regions[key]; box = R['box']
        U = (np.r_[R['net']['x'], R['cl']['x']], np.r_[R['net']['y'], R['cl']['y']])
        sk, info = skeleton_ref(U, box)
        grid = grid_of(R['T']); res = dict(ref=info, grid=grid)
        for meth in ('net', 'cl'):
            T = R[meth]
            for mode in ('ind', 'spread'):
                cv = []
                for N in grid:
                    m, _ = subset(T['w'], N, R['Tw'], mode); cv.append(coverage(T['x'][m], T['y'][m], sk, box))
                res[f'{meth}_{mode}'] = {k: [c[k] for c in cv] for k in cv[0]}
            lg(f'{key} {meth}: coverage>=3 ' + ' '.join(f'{v:.2f}' for v in res[f"{meth}_ind"]["3"][::4]))
        out[key] = res
    # NPC: all pores, the full footprint (cache rows), corners + valley / peak
    P = Zn['P']
    T = {k: dict(x=Zn[f'{k}_x'].astype(np.float64), y=Zn[f'{k}_y'].astype(np.float64), z=Zn[f'{k}_z'].astype(np.float64),
                 N=Zn[f'{k}_N'].astype(np.float64), frame=Zn[f'{k}_frame']) for k in ('net', 'cl')}
    rot = npc_corners_ref((np.r_[T['net']['x'], T['cl']['x']], np.r_[T['net']['y'], T['cl']['y']]), P)
    grid = grid_of(684000); res = dict(grid=grid, n_pores=int(len(P)))
    gz = [N for N in grid[::2]] + ([grid[-1]] if grid[-1] not in grid[::2] else [])
    res['grid_dip'] = gz
    for meth in ('net', 'cl'):
        w = T[meth]['frame'] // 6
        for mode in ('ind', 'spread'):
            cs = []
            for N in grid:
                m, _ = subset(w, N, 114000, mode); cs.append(npc_corners(T[meth]['x'][m], T[meth]['y'][m], P, rot))
            res[f'{meth}_{mode}'] = {k: [c[k] for c in cs] for k in cs[0]}
        dp = []
        for N in gz:
            m, _ = subset(w, N, 114000, 'ind'); pm, _ = SN.pore_metrics({q: v[m] for q, v in T[meth].items()}, P)
            dp.append(dict(dip=pm.get('z_dip', float('nan')), ring_z_sd=pm.get('ring_z_sd_nm', float('nan')), sep=pm.get('two_ring_sep_em2_nm', float('nan')),
                           locs_per_pore=pm.get('locs_per_pore', float('nan'))))
        res[f'{meth}_dip'] = {k: [d[k] for d in dp] for k in dp[0]}
        lg(f'npc {meth}: corners>=1 ' + ' '.join(f'{v:.2f}' for v in res[f"{meth}_ind"]["corners_ge1"][::4]) + ' | dip ' + ' '.join(f'{v:.2f}' for v in res[f'{meth}_dip']['dip']))
    out['npc'] = res
    return out


# ------------------------------------------------------------------------------------------------------------ main
def compute(a):
    from multiprocessing import get_context
    regions, raw = load_real(); lg('real tables loaded: ' + ', '.join(f"{k} {len(R['net']['x']):,} / {len(R['cl']['x']):,}" for k, R in regions.items()))
    regions.update(load_sim()); lg('sim rows loaded')
    _D.update(regions)
    jobs = []
    for key, R in regions.items():
        grid = grid_of(R['T'])
        for meth in ('net', 'cl'):
            for N in grid:
                for mode in ('blk', 'ind', 'spread'):
                    for metric in ('xy', 'xz'):
                        if key == 'mt_field' and metric == 'xz' and mode != 'ind': continue
                        jobs.append((key, meth, N, mode, metric))
                if meth == 'net' and 'net_m' in R and key in ('npc_center', 'mt_box0'): jobs.append((key, 'net_m', N, 'ind', 'xy'))   # si_4pi_npc g / si_4pi_mt f
    jobs.sort(key=lambda j: -(len(_D[j[0]][j[1]]['x']) * (3 if j[4] == 'xz' else 1) * min(j[2], 3e5)))
    lg(f'{len(jobs)} FRC jobs')
    out = {key: dict(box=list(R['box']), zc=R['zc'], T=R['T'], grid=grid_of(R['T']), n_rows={m: int(len(R[m]['x'])) for m in ('net', 'net_m', 'cl') if m in R})
           for key, R in regions.items()}
    with get_context('fork').Pool(a.procs) as pool:
        for i, (arg, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            key, meth, N, mode, metric = arg
            out[key].setdefault(f'{meth}_{mode}_{metric}', {})[str(N)] = r
            if i % 100 == 0: lg(f'{i}/{len(jobs)} {arg} {r}')
    for key in out:
        for k, v in list(out[key].items()):
            if isinstance(v, dict) and all(q.isdigit() for q in v):
                g = sorted(int(q) for q in v)
                out[key][k] = dict(N=g, sus=[v[str(n)]['sus'] for n in g], first=[v[str(n)]['first'] for n in g], n=[v[str(n)]['n'] for n in g])
    for tag in SIMTAGS:
        j = json.load(open(f'{P4}/sweep4pi/{tag}/ttf_fp.json'))
        out[f'sim_{tag}']['gt'] = {m: dict(N=j['grid'], xy=j['methods'][nm]['ttf_xy_interp'], xz=j['methods'][nm]['ttf_xz_interp'],
                                           xy_runmin=j['methods'][nm]['ttf_xy'], xz_runmin=j['methods'][nm]['ttf_xz'])
                                   for m, nm in (('net', 'net_sig1'), ('cl', 'cl_pub'))}
        out[f'sim_{tag}']['x_dense'] = SIMTAGS[tag]
    for key in ('mt_dense', 'mt_box0', 'npc_center', 'sim_mtc_d03'):
        out[key]['lag'] = {m: lag_excess(regions[key][m]) for m in ('net', 'cl')}
    json.dump(out, open(RES, 'w'), indent=1)
    lg(f'FRC -> {RES}')
    out['structure'] = structure(regions, raw)
    json.dump(out, open(RES, 'w'), indent=1)
    lg(f'-> {RES}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['compute', 'fig']); ap.add_argument('--procs', type=int, default=18)
    a = ap.parse_args()
    if a.cmd == 'compute':
        from simcode.util.proctitle import set_title; set_title('4pi_frc_frames'); compute(a)
    else:
        import si_4pi_frc_fig as F; F.main(RES)
