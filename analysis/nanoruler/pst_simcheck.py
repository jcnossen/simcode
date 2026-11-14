"""
Semi-synthetic check of the cross-site lag covariance (pst_lagcov.py): model intensities of the real fully-on frames (fitted
amplitudes, shared pattern, noise V = a0 + a1 mu + c^2 mu^2) with an injected phase fluctuation that is common to the three
steps of an orientation, independent between orientations and AR(1)-correlated over cycles (variance VAR, correlation time TAU
cycles), optionally plus an independent per-step slow relative drift. Output: out/nanoruler/perstep/simcheck.json
"""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L, pst_jitter as J, pst_lagcov as LC
from scipy.sparse import csr_matrix
from simcode.util.proctitle import set_title

if __name__ == '__main__':
    set_title('nr-pstep'); torch.set_num_threads(2)
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    D = L.load_data(); I, s, fr, site = D['I'], D['s'], D['fr'], D['site']
    g = np.unique(D['g'], return_inverse=True)[1]; cyc = fr // 6; ncyc = L.NFRAMES // 6; ns = site.max() + 1
    P = L.Pattern.from_state(F['shared_3000']['full']['state']); x, y, th = J.model_xy(D, P)
    rel = np.exp(P.logrel.numpy() - P.logrel.numpy().mean()); m = P.m.numpy()
    A = J.loo(I, x, g, LC.NOISE)['A']
    R = {}
    for case, (VAR, TAU, REL) in dict(common_ar1=(0.003, 5.0, 0.0), none=(0.0, 5.0, 0.0), common_ar1_plus_rel=(0.003, 5.0, 0.03)).items():
        rng = np.random.default_rng(11)
        rho = math.exp(-1 / TAU); e = rng.standard_normal((ncyc, 2)); dl = np.zeros((ncyc, 2)); dl[0] = e[0]
        for c in range(1, ncyc): dl[c] = rho * dl[c - 1] + math.sqrt(1 - rho ** 2) * e[c]
        dl *= math.sqrt(VAR)
        # step-relative slow drift: smooth random curves (sd REL rad), orientation mean removed
        t = np.arange(ncyc) / ncyc; rr = np.zeros((ncyc, 6))
        if REL > 0:
            rr = np.stack([sum(rng.standard_normal() * np.sin(2 * np.pi * (k_ * t + rng.random())) for k_ in (1, 2, 3)) for _ in range(6)], 1)
            rr[:, :3] -= rr[:, :3].mean(1, keepdims=True); rr[:, 3:] -= rr[:, 3:].mean(1, keepdims=True)
            rr *= REL / rr.std()
        mu = A * rel[s] * (1 + m[s] * np.cos(th - dl[cyc, s // 3] - rr[cyc, s]))
        Is = mu + np.sqrt(J.V_of(mu, (1400.0, 1.0, 0.04))) * rng.standard_normal(len(mu))
        o = J.loo(Is, x, g, LC.NOISE)
        a = o['Al'] * y; ok = o['ok'] & (o['Al'] > 20) & (np.abs(a) > 1e-3) & (np.abs(y) > 0.15)
        d = np.where(ok, o['r'] / np.where(ok, a, 1), 0); w = ok * np.where(ok, a, 1) ** 2 / o['var']
        M = []
        for st in range(6):
            q = ok & (s == st); M.append((csr_matrix((w[q] * d[q], (cyc[q], site[q])), shape=(ncyc, ns)), csr_matrix((w[q], (cyc[q], site[q])), shape=(ncyc, ns))))
        res = dict(truth=dict(var_common=VAR, tau_cycles=TAU, rel_sd=REL, rel_var_actual=float(rr.var())))
        for lo, hi in LC.BINS:
            nd = {}
            z = lambda: (np.zeros(ncyc), np.zeros(ncyc))
            for s1 in range(6):
                for s2 in range(s1, 6):
                    v = LC.acc_pair(M[s1], M[s2], lo, hi, ncyc)
                    k = 'same' if s1 == s2 else 'cross' if s1 // 3 == s2 // 3 else 'xori'
                    nd[k] = (nd.get(k, z())[0] + v[0], nd.get(k, z())[1] + v[1])
            e_ = LC.boot(nd, dict(same=lambda q: q['same'], cross=lambda q: q['cross'], xori=lambda q: q['xori'],
                                  var_rel=lambda q: (2 / 3) * (q['same'] - q['cross'])), nb=100)
            Ls = np.arange(lo, hi + 1); e_['truth_common'] = float(VAR * np.mean(rho ** Ls))
            res[f'{lo}-{hi}'] = e_
            print(f'{case:20s} lag {lo:3d}-{hi:3d}: truth {e_["truth_common"]:.5f} same {e_["same"][0]:+.5f}+-{e_["same"][1]:.5f} cross {e_["cross"][0]:+.5f} '
                  f'xori {e_["xori"][0]:+.5f} var_rel {e_["var_rel"][0]:+.5f}+-{e_["var_rel"][1]:.5f} (truth {rr.var():.5f})', flush=True)
        R[case] = res
        json.dump(R, open(L.OUT + 'simcheck.json', 'w'), indent=1)
