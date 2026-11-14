"""
Cross-site lag covariance of the frame-common phase (and amplitude) error, per pair of steps.

Per datum (sample b, frame f) the leave-one-out estimate d of the phase error of its frame (pst_jitter.py). The weighted
product of d of two emitters at DIFFERENT binding sites, one in step s of cycle c and one in step s' of cycle c + L, estimates
C_ss'(L) = cov(delta_s(c), delta_s'(c + L)) free of photon noise, amplitude-fit noise and per-site (position / phase-field)
errors. Pooled:
    same    s = s'
    cross   s != s', same orientation       (leak-prone for L <= 2: the leave-one-out amplitude of one emitter uses the frame
                                             of the other; clean for L >= 3 because samples are 12 frames long)
    xori    different orientation           (amplitudes are per orientation: clean at every lag)
With delta_s = common_a + rho_s (rho = step-relative part, sum_s rho_s = 0, equal variances): C_same = C_com + var(rho),
C_cross = C_com - var(rho) / 2, so var(rho) = (2 / 3) (C_same - C_cross)  = the variance a shared drift per orientation
cannot represent, at the time scale of the lag bin.
usage: python analysis/nanoruler/pst_lagcov.py <config> [<config> ...]     -> out/nanoruler/perstep/lagcov.json
"""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L
import pst_jitter as J
from simcode.util.proctitle import set_title

BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 250), (251, 500)]
NOISE = (1400.0, 0.0, 0.058)            # weights only (V = a0 + a1 mu + c^2 mu^2 of the trimmed ML fit); estimates are unbiased for any weights
BLOCK = 250


def acc_pair(Ma, Mb, lo, hi, ncyc):
    """num / den per start cycle for the pair of data sets a (cycle c), b (cycle c + L), L in lo..hi, both orders."""
    num = np.zeros(ncyc); den = np.zeros(ncyc); same = Ma is Mb
    for Lg in range(lo, hi + 1):
        n, d = J.pairstat(Ma[0], Ma[1], Lg, Mb[0], Mb[1]); num[:len(n)] += n; den[:len(d)] += d
        if not same and Lg > 0:
            n, d = J.pairstat(Mb[0], Mb[1], Lg, Ma[0], Ma[1]); num[:len(n)] += n; den[:len(d)] += d
    return num, den


def boot(nd_list, combos, nb=300, seed=0):
    """nd_list: {name: (num[ncyc], den[ncyc])}; combos: {out name: function of dict name -> ratio}. Block bootstrap over cycles."""
    ncyc = len(next(iter(nd_list.values()))[0]); nbk = ncyc // BLOCK
    blk = {k: (v[0][:nbk * BLOCK].reshape(nbk, BLOCK).sum(1), v[1][:nbk * BLOCK].reshape(nbk, BLOCK).sum(1)) for k, v in nd_list.items()}
    val = lambda sel: {k: (v[0][sel].sum() / max(v[1][sel].sum(), 1e-30)) for k, v in blk.items()}
    full = val(slice(None)); rng = np.random.default_rng(seed)
    bs = [val(rng.integers(0, nbk, nbk)) for _ in range(nb)]
    out = {}
    for name, f in combos.items():
        b = np.array([f(q) for q in bs]); out[name] = [float(f(full)), float(b.std())]
    return out


if __name__ == '__main__':
    set_title('nr-pstep'); torch.set_num_threads(2)
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    D = L.load_data(); I, s, fr, site = D['I'], D['s'], D['fr'], D['site']
    g = np.unique(D['g'], return_inverse=True)[1]; cyc = fr // 6; ncyc = L.NFRAMES // 6
    R = dict(bins=BINS, block_cycles=BLOCK, noise_for_weights=NOISE,
             note='values [estimate, bootstrap se]; phase in rad^2, amp in relative^2; var_rel = (2/3)(same - cross)')
    for cfg in sys.argv[1:]:
        P = L.Pattern.from_state(F[cfg]['full']['state']); x, y, th = J.model_xy(D, P)
        o = J.loo(I, x, g, NOISE)
        R[cfg] = {}
        for kind in ('phase', 'amp'):
            a = o['Al'] * (y if kind == 'phase' else x)
            ok = o['ok'] & (o['Al'] > 20) & (np.abs(a) > 1e-3)
            if kind == 'phase': ok &= np.abs(y) > 0.15
            d = np.where(ok, o['r'] / np.where(ok, a, 1), 0); w = ok * np.where(ok, a, 1) ** 2 / o['var']
            M = [J.mats(d[ok & (s == st)], w[ok & (s == st)], cyc[ok & (s == st)], ncyc, np.r_[site[ok & (s == st)], site.max()][:-1] if False else site[ok & (s == st)]) for st in range(6)]
            ns = site.max() + 1
            # equal column count for all steps
            from scipy.sparse import csr_matrix
            M = [(csr_matrix(m_[0], shape=(ncyc, ns)) if m_[0].shape[1] == ns else csr_matrix((m_[0].data, m_[0].indices, m_[0].indptr), shape=(ncyc, ns)),
                  csr_matrix((m_[1].data, m_[1].indices, m_[1].indptr), shape=(ncyc, ns))) for m_ in M]
            res = {}
            for lo, hi in BINS:
                nd = {}
                z = lambda: (np.zeros(ncyc), np.zeros(ncyc))
                add = lambda k, v: nd.__setitem__(k, (nd.get(k, z())[0] + v[0], nd.get(k, z())[1] + v[1]))
                for s1 in range(6):
                    for s2 in range(s1, 6):
                        if s1 != s2 and lo == 0 and hi == 0: v = acc_pair(M[s1], M[s2], 0, 0, ncyc)
                        else: v = acc_pair(M[s1], M[s2], lo, hi, ncyc)
                        if s1 == s2: add(f'same{s1 // 3}', v); add(f'same_s{s1}', v)
                        elif s1 // 3 == s2 // 3: add(f'cross{s1 // 3}', v)
                        else: add('xori', v)
                rel = lambda a_: (lambda q: (2 / 3) * (q[f'same{a_}'] - q[f'cross{a_}']))
                com = lambda a_: (lambda q: (q[f'same{a_}'] + 2 * q[f'cross{a_}']) / 3)
                combos = {k: (lambda q, k=k: q[k]) for k in nd}
                combos.update(var_rel0=rel(0), var_rel1=rel(1), var_common0=com(0), var_common1=com(1),
                              same=lambda q: 0.5 * (q['same0'] + q['same1']), cross=lambda q: 0.5 * (q['cross0'] + q['cross1']),
                              var_rel=lambda q: (1 / 3) * (q['same0'] - q['cross0'] + q['same1'] - q['cross1']))
                res[f'{lo}-{hi}'] = boot(nd, combos)
                e = res[f'{lo}-{hi}']
                print(f'{cfg:14s} {kind:5s} lag {lo:3d}-{hi:3d}: same {e["same"][0]:+.5f}+-{e["same"][1]:.5f}  cross {e["cross"][0]:+.5f}+-{e["cross"][1]:.5f}  '
                      f'xori {e["xori"][0]:+.5f}+-{e["xori"][1]:.5f}  var_rel o0 {e["var_rel0"][0]:+.5f}+-{e["var_rel0"][1]:.5f} o1 {e["var_rel1"][0]:+.5f}+-{e["var_rel1"][1]:.5f}', flush=True)
            R[cfg][kind] = res
        json.dump(R, open(L.OUT + 'lagcov.json', 'w'), indent=1)
