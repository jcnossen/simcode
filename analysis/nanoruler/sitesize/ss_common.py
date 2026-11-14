"""Site-size study, the common-mode part of the per-site residuals: per cycle, the mean residual (localization minus its site median, the
same assignment as ss_lib) over all sites of the field; autocovariance of that series at lags 0..L (lag 0 minus the noise of the mean,
sigma_1^2 / n per cycle, sigma_1 = sigma_pair(1) of the method). A common-mode fluctuation (residual drift, shared pattern-phase jitter)
gives c(lag) > 0; its contribution to sigma_pair(Delta)^2 is c(0) - c(Delta).
  python ss_common.py <movie> <method> ...  -> out/nanoruler/sitesize/common_<movie>.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ss_lib as S, ss_run as RN

LAGS = [0, 1, 2, 3, 5, 10, 20, 50, 100, 300]


def common(tab, St, ncyc, sig1):
    x, y, g, j, sh = S.assign_sites(tab, St['sc'], St['used'], St['radius'], St['rmax'])
    n = np.bincount(j, minlength=len(St['sc'])); k = n[j] >= S.MIN_N; g, j = g[k], j[k]
    cx = np.zeros(len(St['sc'])); cy = np.zeros(len(St['sc']))
    for s in np.unique(j): cx[s] = np.median(x[g[j == s]]); cy[s] = np.median(y[g[j == s]])
    r = np.c_[x[g] - cx[j], y[g] - cy[j]]; r = np.clip(r, -25, 25); f = tab['frame'][g]
    cnt = np.bincount(f, minlength=ncyc); m = np.stack([np.bincount(f, r[:, a], minlength=ncyc) for a in range(2)], 1) / np.maximum(cnt, 1)[:, None]
    ok = cnt >= 5; mm = np.where(ok[:, None], m - m[ok].mean(0), 0.0)
    o = dict(n_cycles=int(ok.sum()), locs_per_cycle=float(cnt[ok].mean()), sigma1=sig1, acov={})
    for L in LAGS:
        a = ok[:ncyc - L] & ok[L:]; c = (mm[:ncyc - L][a] * mm[L:][a]).mean(0)
        if L == 0: c = c - (sig1 ** 2 / cnt[ok]).mean()
        o['acov'][L] = c.tolist()
    return o


if __name__ == '__main__':
    mv = sys.argv[1]; St = RN.sites_of(mv); nc = RN.NCYC[mv]; res = json.load(open(f'{RN.OUT}/res_{mv}.json')); o = dict(movie=mv, methods={})
    for m in sys.argv[2:]:
        s1 = res['methods'][m]['axes']['xy']['sigma_pair']['1'][0]; o['methods'][m] = common(RN.load(mv, m), St, nc, s1)
        print(m, f"{o['methods'][m]['locs_per_cycle']:.1f} locs / cycle;", ' '.join(f'c({L}) {v[0]:.2f}/{v[1]:.2f}' for L, v in o['methods'][m]['acov'].items()), flush=True)
    json.dump(o, open(f'{RN.OUT}/common_{mv}.json', 'w'), indent=1)
