"""Nanoruler ablation, part D: pattern-model test at the position level. The same rows (variant A, the run's network outputs) are
re-solved with different pattern descriptions:
  est    the pipeline's pattern of the run (pattern_est.pt: one phase drift per orientation, 150-frame blocks, 3-block running mean)
  ps     the per-step pattern of the in-situ fit as it is (out/nanoruler/perstep/pattern_perstep.pt['pipeline']: six drifts on 1500-frame
         knots, its own per-step phases, relative intensities and (diluted regression) depths 0.79-0.84)
  psd    the per-step phases and drifts of 'ps' with the depth and relative intensities of 'est' (isolates the per-step phase model from
         the depth difference)
  shd    the in-situ SHARED fit (pattern_perstep.pt['shared_reference'], 1500-frame knots) with the depth / relative intensities of 'est'
         (control: in-situ fit versus pipeline estimate without the per-step freedom)
  d<NN>  'est' with the modulation depth multiplied by 0.NN (python abl_pattern.py depth <set>)
  k<N>   the pipeline's own estimator (simcode.pipeline.pattern_est.estimate_pattern, same rows and selection as the run) with drift
         blocks of N frames instead of 150 (k150 = reproduction check of 'est'; the measured fast common fluctuation has a correlation
         time of ~5 cycles = 30 frames)
Before use, every pattern is compared with 'est' frame by frame at the field centre (phase difference per step: mean, sd; depth).

usage: python analysis/nanoruler/abl_pattern.py make <set> [--kf 30,60,150]     -> out/nanoruler/ablation/patterns/<set>_<tag>.pt, patterns_<set>.json
       then (see abl_all.sh): abl_solve.py <set> A --pattern .. --ptag <tag>; abl_autoc.py est <set> --pattern .. --name <tag>;
       python analysis/nanoruler/abl_pattern.py link <set>   (collects runs/<set>_<tag>/A as runs/<set>_pat/<tag> for abl_nra.py --variants ...)"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
PST = os.path.join(A.NR, 'perstep', 'pattern_perstep.pt'); PD = os.path.join(A.OUT, 'patterns')


def compare(P, Q, shape=(400, 400), nfr=30000):
    from simcode.pipeline.common import pattern_frames
    f = np.arange(0, nfr - 6, 7); xc, yc = shape[1] / 2, shape[0] / 2; rad2nm = lambda a: 2 * np.pi / np.hypot(*a[0, :2]) * A.PX / (2 * np.pi)
    a0 = pattern_frames(P, f, 0, 0, 'cpu').numpy(); a1 = pattern_frames(Q, f, 0, 0, 'cpu').numpy()
    d = np.angle(np.exp(1j * ((a1[:, 0] * xc + a1[:, 1] * yc - a1[:, 2]) - (a0[:, 0] * xc + a0[:, 1] * yc - a0[:, 2]))))
    out = []
    for s in range(6):
        m = f % 6 == s; nm = rad2nm(a0[m])
        out.append(dict(step=s, dphase_mean_rad=float(d[m].mean()), dphase_sd_rad=float(d[m].std()), dphase_mean_nm=float(d[m].mean() * nm), dphase_sd_nm=float(d[m].std() * nm),
                        depth=[float(a0[m, 3].mean()), float(a1[m, 3].mean())], relint=[float(a0[m, 4].mean()), float(a1[m, 4].mean())]))
    return out


def reestimate(run, kf, crop, shape, cal):
    """the run's pattern stage (run_uncalibrated.py) with another drift block length; pitch search range of this setup as ev_run.py"""
    from simcode.pipeline import pattern_est as PE
    d = list(PE.estimate_k.__defaults__)
    if d[0] == (1.6, 2.8): d[0] = (2.9, 3.9); PE.estimate_k.__defaults__ = tuple(d)
    fs = sorted(glob.glob(f'{run}/pass1_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4])); Ls = [np.load(f) for f in fs]
    cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
    keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (i == len(Ls) - 1)) for i, L in enumerate(Ls)])
    R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
    po = cal.get('prior_off')
    if po is not None: R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]
    x0, y0, sz = crop
    R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= x0) & (R['x'] < x0 + sz) & (R['y'] >= y0) & (R['y'] < y0 + sz)
    log = []
    P, info = PE.estimate_pattern(R, shape, kf=kf, verbose=lambda s: log.append(s))
    return P, [l for l in log if 'eps angle' in l][-2:]


def make(setn, kfs):
    S = A.SETS[setn]; run = S['run']; os.makedirs(PD, exist_ok=True)
    P = torch.load(f'{run}/pattern_est.pt', weights_only=False); rep = {}
    L0 = np.load(sorted(glob.glob(f'{run}/pass1_*.npz'))[0]); crop = tuple(int(v) for v in L0['crop'])
    mv = A.MOVIES['real' if S['kind'] == 'real' else os.path.basename(S['sim'])]; cal = json.load(open(S['prod'] if S['front'] == 's1' else A.SETS['p1']['prod']))
    import tifffile
    with tifffile.TiffFile(mv['movie']) as tf: shape = tf.pages[0].shape; nfr = len(tf.pages)
    pats = {}
    if S['kind'] == 'real' and os.path.exists(PST):
        Q = torch.load(PST, weights_only=False); ps = Q['pipeline']; pats['ps'] = ps
        psd = dict(ps); m6 = torch.as_tensor(P['m']).float(); psd['m'] = m6.clone(); psd['logrel'] = torch.as_tensor(P['logrel']).float().clone()
        psd['note'] = 'per-step phases / drifts of pattern_perstep.pt, depth and relative intensities of pattern_est.pt'; pats['psd'] = psd
        sh = {k: v for k, v in Q['shared_reference'].items() if k != 'config'}; sh['m'] = m6.clone(); sh['logrel'] = psd['logrel'].clone()
        sh['note'] = 'in-situ shared fit (1500-frame knots), depth and relative intensities of pattern_est.pt'; pats['shd'] = sh
    for kf in kfs:
        Pk, lg = reestimate(run, kf, crop, shape, cal); pats[f'k{kf}'] = Pk; rep[f'k{kf}_log'] = lg; print(f'k{kf}:', *lg, sep='\n   ')
    for tag, Pq in pats.items():
        torch.save(Pq, f'{PD}/{setn}_{tag}.pt'); c = compare(P, Pq, shape, nfr); rep[tag] = c
        print(f'{tag} vs est, phase at the field centre per step: mean ' + ' '.join(f'{q["dphase_mean_nm"]:+.2f}' for q in c) + ' nm | sd over time '
              + ' '.join(f'{q["dphase_sd_nm"]:.2f}' for q in c) + ' nm | depth ' + ' '.join(f'{q["depth"][1]:.3f}' for q in c) + f' (est {c[0]["depth"][0]:.3f} / {c[3]["depth"][0]:.3f})')
    A.jdump(rep, f'{A.OUT}/patterns_{setn}.json')


def depth(setn, scales=(0.97, 0.94, 0.91)):
    """d<NN>: the run's pattern with the modulation depth multiplied by 0.NN (test of the excess at the fringe minima: the network
    intensities of the darkest frames lie above the model, abl_autoc.py branch)"""
    P = torch.load(f'{A.SETS[setn]["run"]}/pattern_est.pt', weights_only=False); os.makedirs(PD, exist_ok=True)
    for sc in scales:
        Q = dict(P); Q['m'] = torch.as_tensor(P['m']).float() * sc; Q['note'] = f'pattern_est.pt with the depth x {sc}'
        torch.save(Q, f'{PD}/{setn}_d{round(100 * sc):02d}.pt'); print(f'd{round(100 * sc):02d}: depth', Q['m'].numpy().round(3).tolist())


def link(setn):
    d = f'{A.OUT}/runs/{setn}_pat'; os.makedirs(d, exist_ok=True); tags = []
    for src, tag in [(f'{A.OUT}/runs/{setn}/A', 'est')] + [(p, os.path.basename(os.path.dirname(p))[len(setn) + 1:]) for p in sorted(glob.glob(f'{A.OUT}/runs/{setn}_*/A'))]:
        if tag == 'pat' or not os.path.exists(f'{src}/locs.npz'): continue
        if os.path.islink(f'{d}/{tag}'): os.unlink(f'{d}/{tag}')
        os.symlink(src, f'{d}/{tag}'); tags.append(tag)
    print(','.join(tags))


if __name__ == '__main__':
    a = sys.argv[1:]; A.title()
    if a[0] == 'depth': depth(a[1])
    elif a[0] == 'make': make(a[1], [int(v) for v in (a[a.index('--kf') + 1] if '--kf' in a else '30,60,150').split(',')])
    else: link(a[1])
