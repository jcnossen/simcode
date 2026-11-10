"""Filtered-ours vs SIMCODE's full (self-filtered) output at matched output size (coordinator / user request, 2026-11-10/11).
Ours pass 1 (and p2L for (i)) on the common grid; N = SIMCODE's row count on that grid. Variants, all GT-free:
  G random        random N of the p >= 0.63 rows (selection baseline)
  (i)  G top-N σ  top N by own sigma 0.5 (sx_cal + sy_cal) of the p >= 0.63 rows
  (ii) G dnll>3+σ rows with dnll > 3 (fringe-unambiguous), then top N by sigma (= a sigma threshold)
  (iii) P top-N σ / P dnll>3+σ  the same on the phase-calibrated selection (p2prior/lib.phase_thr on the stored p >= 0.5 rows)
Metrics = lib.metrics (FRC std / adj300 / gap300 / gap600, NN, revisit proxy) + selection descriptors (median own photons,
row-weighted mean event length in windows: sim = GT on-time / 6, real = linked length in the full G set); sim adds the GT
truth (lib.truth: rms, dyn, event, site) on the GT-matched rows.  same algorithm. Data = a datasets.py json spec: sim mode uses
"gt" and the sets "ours p1" / "ours p2L" (kind ours, dir + tag) and the reference (kind simcode, npz, NDI rows) with this
script's own loader (as dev); real mode reads the stored ours rows with p >= 0.5 (all window starts, for the phase-calibrated
selection) and the reference set through datasets.py. phase_thr = simcode2.strict.phase_thr (= dev p2prior/lib.phase_thr).
usage: python analysis/evalproto/filt.py <dataset.json> [out_dir]  -> filt_<dataset>.txt / .json"""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode2.evalproto import lib as E
from simcode2.strict import phase_thr
spec = sys.argv[1]; SPEC = json.load(open(spec))
mode, tag = SPEC.get('mode', 'real'), SPEC.get('name', '')
REF = SPEC.get('reference', 'SIMCODE (b)')
rng = np.random.default_rng(5); E.RNG = rng
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


def ours_sets(A, grid):
    """A: all stored rows (p >= 0.5) with x, y (corrected), xr, yr (raw), f, p, phot, sx_cal, sy_cal, [dnll]; crop applied"""
    on = A['f'] % grid == 0
    G = on & (A['p'] >= 0.63)
    P = on & phase_thr(A['p'], A['xr'], A['yr'], A['phot'])
    return G, P


def sub(A, m):
    i = np.nonzero(m)[0] if m.dtype == bool else m
    return {k: (v[i] if isinstance(v, np.ndarray) and len(v) == len(A['f']) else v) for k, v in A.items()}


def variants(A, G, P, N, has_dnll):
    sig = 0.5 * (A['sx_cal'] + A['sy_cal'])
    V = {}
    gi = np.nonzero(G)[0]
    V['G random'] = np.sort(rng.choice(gi, min(N, len(gi)), replace=False))
    for lab, base in (('G', G), ('P', P)):
        bi = np.nonzero(base)[0]
        V[f'{lab} top-N σ'] = np.sort(bi[np.argsort(sig[bi])[:N]])
        if has_dnll:
            di = bi[A['dnll'][bi] > 3]
            V[f'{lab} dnll>3+σ'] = np.sort(di[np.argsort(sig[di])[:N]])
    return V


MK = ('std', 'adj300', 'gap300', 'gap600', 'nn', 'nn12', 'rev')
fm = lambda d, keys: ' | '.join(f'{d.get(k, np.nan):.2f}' for k in keys)

if mode == 'sim':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
    import compare as C       # analysis/baselines/compare.py
    m = tag
    Gt = dict(np.load(SPEC['gt'])); NF = int(Gt['nf']); grid = 3
    GW = E.gt_windows_fast(Gt, grid=3)
    site_of = Gt['site'] if 'site' in Gt else -np.ones(len(Gt['x']), np.int64)
    dur = (Gt['t1'] - Gt['t0']) / 6.0

    def attach(D):
        gi = E.match_fast(GW, D['f'], D['xd'], D['yd']); D['gi'] = gi; g = gi >= 0
        e = np.full((len(gi), 2), np.nan); e[g] = np.c_[D['x'][g] - GW[gi[g], 2], D['y'][g] - GW[gi[g], 3]]
        P_ = np.where(g, GW[np.maximum(gi, 0), 4], np.nan); sel = g & (P_ >= 400) & (P_ < 2000)
        off = np.median(e[sel], 0); D['e'] = (e - off) * E.PX
        D['ev'] = np.where(g, GW[np.maximum(gi, 0), 1], -1).astype(np.int64)
        D['site'] = np.where(g, site_of[np.maximum(D['ev'], 0)], -1)
        D['x'] = D['x'] - off[0]; D['y'] = D['y'] - off[1]
        D['len'] = np.where(g, dur[np.maximum(D['ev'], 0)], np.nan)
        return D

    def load_ours(d, t):
        Q = {}
        for fn in sorted(glob.glob(f'{d}/pass2_{t}_*.npz')):
            d = np.load(fn)
            for k in d.files:
                if d[k].ndim: Q.setdefault(k, []).append(d[k])
        Q = {k: np.concatenate(v) for k, v in Q.items()}
        ok = np.isfinite(Q['x_hy0']) & (Q['wstart'] <= NF - 6) & (Q['x_hy0'] >= 5) & (Q['x_hy0'] < 115) & (Q['y_hy0'] >= 5) & (Q['y_hy0'] < 115)
        A = dict(f=Q['wstart'][ok].astype(np.int64), x=Q['x_hy0'][ok].astype(float), y=Q['y_hy0'][ok].astype(float), xd=Q['x_prior'][ok], yd=Q['y_prior'][ok],
                 p=Q['p'][ok], phot=Q['phot'][ok], sx_cal=Q['sx_cal'][ok], sy_cal=Q['sy_cal'][ok])
        if 'dnll' in Q: A['dnll'] = Q['dnll'][ok]
        A['xr'], A['yr'] = A['x'], A['y']
        return A
    Sd = C.load_method('sc_ndi', SPEC['sets'][REF]['npz'], NF)
    ok = (Sd['w'] % 3 == 0) & (Sd['w'] <= NF - 6) & (Sd['x'] >= 5) & (Sd['x'] < 115) & (Sd['y'] >= 5) & (Sd['y'] < 115)
    S = {k: np.asarray(v)[ok] for k, v in Sd.items()}; S['f'] = S.pop('w').astype(np.int64)
    S = attach(S)
    SETS = {k: load_ours(v['dir'], v.get('tag', 'est')) for k, v in SPEC['sets'].items() if v['kind'] == 'ours'}
else:
    grid = int(SPEC.get('grid', 3)); SETS = {}
    for nm, v in SPEC['sets'].items():
        if v['kind'] != 'ours': continue
        R = DS._ours_rows(v['dir'], v.get('tag', 'est')); ok = np.isfinite(R['x_hy0']) & np.isfinite(R['y_hy0']) & (R['p'] >= 0.5)
        d = DS._drift(v.get('drift')); f = R['wstart'][ok].astype(np.int64); ti = np.clip(f + 3, 0, len(d) - 1)
        A = dict(f=f, xr=R['x_hy0'][ok].astype(float), yr=R['y_hy0'][ok].astype(float), p=R['p'][ok], phot=R['phot'][ok],
                 sx_cal=R['sx_cal'][ok], sy_cal=R['sy_cal'][ok])
        if 'dnll' in R: A['dnll'] = R['dnll'][ok]
        A['x'] = A['xr'] - d[ti, 0]; A['y'] = A['yr'] - d[ti, 1]
        k = (A['x'] >= 5) & (A['x'] < 115) & (A['y'] >= 5) & (A['y'] < 115)
        SETS[nm] = {kk: vv[k] for kk, vv in A.items()}
    S0 = DS.load_set(SPEC['sets'][REF], None, grid); S = dict(f=S0['f'], x=S0['x'], y=S0['y'])
    k = S['f'] % grid == 0; S = {kk: v[k] for kk, v in S.items()}
N = len(S['f'])
log(f'# {tag if tag.startswith(mode) else mode + " " + tag}: grid G{grid}, N = SIMCODE rows on the grid = {N}')
hdr = '| set | rows | med photons | mean event length (win) | ' + (' truth rms | dyn | event | site | ' if mode == 'sim' else '') + ' | '.join(MK) + ' |'
log(hdr); log('|---' * (hdr.count('|') - 1) + '|')
Ms = E.metrics(S['x'], S['y'], S['f'])
T = E.truth(S, np.arange(N)) if mode == 'sim' else {}
Ls = np.nanmean(S['len']) if mode == 'sim' else np.nan
OUT['SIMCODE'] = dict(n=N, metr=Ms, truth=T)
log(f'| SIMCODE (b) all | {N} | – | {Ls:.1f} | ' + (fm(T, ('rms', 'dyn', 'event', 'site')) + ' | ' if mode == 'sim' else '') + fm(Ms, MK) + ' |')
for nm, A in SETS.items():
    G, P = ours_sets(A, grid)
    if mode == 'real':      # real event length: linked length in the full G set
        Ag = sub(A, G); ev = E.link_events(Ag['x'], Ag['y'], Ag['f'], grid=grid, r=0.75)
        u, inv, cnt = np.unique(ev, return_inverse=True, return_counts=True)
        A['len'] = np.full(len(A['f']), np.nan); A['len'][np.nonzero(G)[0]] = cnt[inv] * grid / 6.0
    V = variants(A, G, P, N, 'dnll' in A)
    V = {'G all': np.nonzero(G)[0], **V}
    for lab, i in V.items():
        B = sub(A, i)
        if mode == 'sim': B = attach(B)
        Mt = E.metrics(B['x'], B['y'], B['f'])
        T = E.truth(B, np.arange(len(i))) if mode == 'sim' else {}
        OUT[f'{nm} {lab}'] = dict(n=int(len(i)), metr=Mt, truth=T, ph=float(np.median(B['phot'])), len=float(np.nanmean(B['len'])))
        log(f'| {nm} {lab} | {len(i)} | {np.median(B["phot"]):.0f} | {np.nanmean(B["len"]):.1f} | ' + (fm(T, ('rms', 'dyn', 'event', 'site')) + ' | ' if mode == 'sim' else '') + fm(Mt, MK) + ' |')
fn = DS.out_path(spec, 'filt', sys.argv[2] if len(sys.argv) > 2 else None)
json.dump(OUT, open(fn + '.json', 'w'), default=float, indent=1); open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
