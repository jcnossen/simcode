"""Task 1 (2026-11-13): production pass 1 (s1) / pass 2 (s2) of a PSF-matched set on the standard gate scenes with the
trained-range step spacing (out/psfmatch/fe_<tag>, the acceptance dumps) and with the movie's measured per-step spacings
(out/p2generic/fe_<set>_real; same seeds -> same emitters). Each front solved with its production sim calibration.
Tables per scene and GT-photon bin (seeds 301-303 pooled; FOV bootstrap 95 % CI, 400 resamples):
  p2 - p1      paired RMS / CRLB on targets found by both, within each spacing
  real - trn   same front, real vs trained spacing, targets found in both (each row with its own scene's CRLB)
  recall       targets matched by a candidate with score >= 0.5 (plib.targets), and the differences with CIs
usage: python t1_eval.py <m1809|m25> <scenes>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode import config
from simcode.util.proctitle import set_title
set_title('p2gen-t1ev')
S, scen = sys.argv[1], sys.argv[2].split(',')
TAG = dict(m1809='m1809a', m25='m25a')[S]
DIRS = dict(trn=config.root('out') + f'/psfmatch/fe_{TAG}', real=config.root('out') + f'/p2generic/fe_{S}_real')
CAL = dict(s1=json.load(open(config.resolve(f'calib/calib_sc_hd_{S}.json'))), s2=json.load(open(config.resolve(f'calib/calib_p2L_{S}.json'))))
BINS = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9), (200, 2000)]
NB = 400


def boot_diff(fov, f, nboot=NB, seed=0):
    rng = np.random.default_rng(seed); uf, inv = np.unique(fov, return_inverse=True)
    v = [f(np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf))[inv]) for _ in range(nboot)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rr(e, c, w=None):
    w = np.ones(len(e)) if w is None else w
    return float(np.sqrt((w[:, None] * e ** 2).sum() / max(w.sum(), 1)) / np.sqrt((w[:, None] * c ** 2).sum() / max(w.sum(), 1)))


def front(sp, f, sc):
    R = P.load(sc, f, d=DIRS[sp]); ti, oi = P.targets(R); sol = P.solve(R, oi, CAL[f]); E = P.errs(R, ti, oi, sol)
    return R, E


def pair_tab(RA, EA, RB, EB, label):
    """paired on GT index (RA / RB index-aligned GT); per row CRLB from its own scene"""
    com, ia, ib = np.intersect1d(EA['ti'], EB['ti'], return_indices=True)
    ca, cb = RA['gt_sf_m'][com], RB['gt_sf_m'][com]
    ok = np.isfinite(ca).all(1) & np.isfinite(cb).all(1) & (np.hypot(*cb.T) / np.sqrt(2) * P.PX < 25)
    ea, eb = EA['e'][ia], EB['e'][ib]; ok &= np.isfinite(ea).all(1) & np.isfinite(eb).all(1)
    ph = RB['gt_phot'][com]; fov = RB['gt_fov'][com]; out = []
    for lo, hi in BINS:
        m = ok & (ph >= lo) & (ph < hi)
        if m.sum() < 20: continue
        a_, b_, ca_, cb_ = ea[m], eb[m], ca[m], cb[m]
        d = rr(a_, ca_) - rr(b_, cb_)
        ci = boot_diff(fov[m], lambda w: rr(a_, ca_, w) - rr(b_, cb_, w))
        nm = lambda e: float(np.sqrt(np.mean(e ** 2)) * P.PX)
        out.append(dict(bin=f'{lo:g}-{hi:g}', n=int(m.sum()), A=rr(a_, ca_), B=rr(b_, cb_), d=d, ci=ci, rel=d / rr(b_, cb_), nmA=nm(a_), nmB=nm(b_)))
        print(f'  {label:22s} {lo:>5g}-{hi:<5g} n {m.sum():5d}  RMS/CRLB A {out[-1]["A"]:.3f} B {out[-1]["B"]:.3f}  d {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] '
              f'({100 * d / out[-1]["B"]:+.1f} %)  nm A {out[-1]["nmA"]:.2f} B {out[-1]["nmB"]:.2f}')
    return out


def recall_tab(RA, EA, RB, EB, label):
    """recall (score >= 0.5 match) of the targets of both scenes (GT index-aligned), A - B with FOV-bootstrap CI"""
    tg = np.nonzero((RA['gt_target'] > 0) & (RB['gt_target'] > 0))[0]
    fa, fb = np.isin(tg, EA['ti']).astype(float), np.isin(tg, EB['ti']).astype(float)
    fov = RB['gt_fov'][tg]; ph = RB['gt_phot'][tg]; out = []
    for lo, hi in [(100, 400), (400, 1200), (1200, 4000), (4000, 1e9), (100, 1e9)]:
        m = (ph >= lo) & (ph < hi)
        if m.sum() < 20: continue
        a_, b_ = fa[m], fb[m]; d = a_.mean() - b_.mean()
        ci = boot_diff(fov[m], lambda w: (w * a_).sum() / w.sum() - (w * b_).sum() / w.sum())
        out.append(dict(bin=f'{lo:g}-{hi:g}', n=int(m.sum()), A=float(a_.mean()), B=float(b_.mean()), d=float(d), ci=ci))
        print(f'  {label:22s} {lo:>5g}-{hi:<5g} n {m.sum():5d}  recall A {a_.mean():.4f} B {b_.mean():.4f}  d {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]')
    return out


res = {}
print(f'# Task 1 {S}: trained-range spacing = {DIRS["trn"]}, measured spacing = {DIRS["real"]}; A minus B')
for sc in scen:
    F = {(sp, f): front(sp, f, sc) for sp in DIRS for f in ('s1', 's2')}
    Rt, Rr = F[('trn', 's1')][0], F[('real', 's1')][0]
    assert len(Rt['gt_x']) == len(Rr['gt_x']) and np.allclose(Rt['gt_x'], Rr['gt_x'], atol=1e-4), 'GT differs between spacings'
    sp_t = np.remainder(np.diff(Rt['patt_true'][:, :, 2], axis=1), 2 * np.pi); sp_r = np.remainder(np.diff(Rr['patt_true'][:, :, 2], axis=1), 2 * np.pi)
    print(f'\n== {sc}: targets {int((Rt["gt_target"] > 0).sum())} / {int((Rr["gt_target"] > 0).sum())}; true frame-to-frame phase step within an angle '
          f'(mod 2 pi, incl. drift) trn {np.median(sp_t[:, [0, 1, 3, 4]]):.4f} real {np.median(sp_r[:, [0, 1, 3, 4]]):.4f}')
    r = {}
    for sp in DIRS:
        r[f'p2-p1 {sp}'] = pair_tab(*F[(sp, 's2')], *F[(sp, 's1')], f'p2 - p1 [{sp}]')
    for f in ('s2', 's1'):
        r[f'real-trn {f}'] = pair_tab(*F[('real', f)], *F[('trn', f)], f'real - trn [{f}]')
    for sp in DIRS:
        r[f'recall p2-p1 {sp}'] = recall_tab(*F[(sp, 's2')], *F[(sp, 's1')], f'recall p2 - p1 [{sp}]')
    for f in ('s2', 's1'):
        r[f'recall real-trn {f}'] = recall_tab(*F[('real', f)], *F[('trn', f)], f'recall real - trn [{f}]')
    res[sc] = r
json.dump(res, open(config.root('out') + f'/p2generic/t1_{S}.json', 'w'), indent=1)
