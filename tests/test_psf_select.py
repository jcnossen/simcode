"""
The psf stage's sample draw (simcode.pipeline.psf_fit.select) and its sampling-spread summary (psf_fit.spread). CPU, no data.
Run: python tests/test_psf_select.py (or pytest tests/test_psf_select.py)
  - select depends only on each sample's own key: removing 3 samples from the list (or shuffling it) changes at most 3
    of the selected samples; the checkerboard parity is respected; another salt is another draw (overlap ~ n^2 / N);
    the global numpy random state is neither used nor changed
  - spread: sd over the draws, scaled by 1 / sqrt(1 - n / N) (and by sqrt(n / nsamples) for another fit size); None when
    every draw is the whole candidate set
"""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from simcode.pipeline import psf_fit as P


def tracks(B=6000, seed=0):
    """a tracks.npz-like dict: tracks of up to 4 consecutive 12-frame samples at random corners"""
    rs = np.random.RandomState(seed)
    nt = B // 3
    corner = rs.randint(0, 500, (nt, 2)); f0 = rs.randint(0, 30000, nt); ns = rs.randint(1, 5, nt)
    co, fr = [], []
    for i in range(nt):
        for s in range(ns[i]):
            co.append(corner[i]); fr.append(f0[i] + 12 * s + np.arange(12))
    B = len(co)
    return dict(rois=np.zeros((B, 1, 1, 1), np.uint16), corner=np.array(co), frames=np.array(fr), xy_init=rs.uniform(3, 7, (B, 2)),
                mask=np.ones((B, 12), bool))


def keys(D, idx):
    return set(map(tuple, P.sample_keys(D)[idx]))


def test_select_stable():
    D = tracks()
    B = len(D['rois'])
    st = np.random.get_state()
    for n, par in ((1500, None), (1500, 0), (800, 1)):
        a = keys(D, P.select(D, n, par))
        assert len(a) == n
        if par is not None:
            assert set(P.select(D, n, par)) <= set(P.candidates(D, par))
        rs = np.random.RandomState(1)
        for _ in range(10):
            keep = np.setdiff1d(np.arange(B), rs.choice(B, 3, replace=False))
            D2 = {k: v[keep] for k, v in D.items()}
            b = keys(D2, P.select(D2, n, par))
            assert len(a - b) <= 3 and len(b - a) == len(a - b), (n, par, len(a - b))
        perm = rs.permutation(B)
        D3 = {k: v[perm] for k, v in D.items()}
        assert keys(D3, P.select(D3, n, par)) == a                     # independent of the list order
        o = keys(D, P.select(D, n, par, salt=1))
        N = len(P.candidates(D, par))
        assert abs(len(a & o) - n * n / N) < 6 * np.sqrt(n * n / N), (len(a & o), n * n / N)
    s2 = np.random.get_state()
    assert st[0] == s2[0] and np.array_equal(st[1], s2[1]) and st[2:] == s2[2:]


def test_spread():
    reps = [dict(a=1.0, z=[0.0, 1.0], fit_s=3.0), dict(a=2.0, z=[0.0, 3.0], fit_s=5.0), dict(a=3.0, z=[0.0, 2.0], fit_s=4.0)]
    S = P.spread(reps, 750, 1000)
    assert S['fpc'] == 2.0 and abs(S['sd_draws']['a'] - 1.0) < 1e-12 and abs(S['sd']['a'] - 2.0) < 1e-12
    assert S['sd']['z'][0] == 0.0 and abs(S['sd']['z'][1] - 2.0) < 1e-12 and 'fit_s' not in S['sd'] and S['fit_s_per_draw'] == 4.0
    assert P.spread(reps, 1000, 1000)['sd']['a'] is None
    S4 = P.spread(reps, 750, 1000, nsamples=3000)          # sd for 4 x the draw size: x 1/2
    assert abs(S4['sd']['a'] - 1.0) < 1e-12 and abs(S4['sd_draw_size']['a'] - 2.0) < 1e-12 and S4['nsamples'] == 3000


if __name__ == '__main__':
    test_select_stable(); test_spread(); print('ok')
