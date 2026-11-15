"""In-situ channel registration (simcode/channels/registration.py) on a synthetic two-channel movie: astigmatic Gaussian
spots in channel 0 and the same emitters in channel 1 through a known order-2 warp plus a left-right flip of the crop
(both channels side by side in one camera frame). register_insitu must find the flip, recover the map to << 0.05 px and
give an inverse that round-trips.

    PYTHONPATH=. python tests/test_registration.py      (or pytest)
"""
import time
import numpy as np
import torch

from simcode.channels import registration as REG
from simcode.channels.layout import Channel, ChannelLayout, ChannelReader
from simcode.psf import ArrayMovie

W, H, R = 96, 96, 13
GAIN, OFF, RN = 0.5, 100.0, 1.0           # e / ADU, ADU, e
# true map channel 1 -> channel 0 in the flipped channel-1 frame: x0 = G(flip_lr(x1)); PolyMap feats [1, v, v^2, u, uv, u^2]
NORM = REG.Norm(W / 2, H / 2, W / 2, H / 2)
G = np.array([[W / 2 + 2.3, H / 2 - 1.7],
              [0.004 * W / 2, 0.996 * H / 2],      # v
              [0.10, -0.15],                       # v^2
              [1.006 * W / 2, -0.006 * H / 2],     # u
              [0.12, 0.08],                        # uv
              [-0.20, 0.12]])                      # u^2


def true_fwd(x1, y1):
    fx, fy = REG.flip_xy(np.asarray(x1, float), np.asarray(y1, float), 'lr', W, H)
    F = NORM.feats(fx, fy, 2) @ G
    return F[..., 0], F[..., 1]


def simulate(nframes=400, per_frame=10, seed=1, dev='cpu'):
    rng = np.random.default_rng(seed)
    psfs = [REG.make_psf(R, p, 0.30, (-0.6, 0.6), 0.1, dev) for p in ((1.1, 1.35, 0.15), (1.25, 1.05, 0.15))]
    E = []                       # emitters: frame, x0, y0, x1, y1 (positions in the two channel crops), z, N
    for f in range(nframes):
        # positions in the channel-1 crop; channel 0 = true_fwd of them; both inside the field
        x1, y1 = rng.uniform(12, W - 13, per_frame * 2), rng.uniform(12, H - 13, per_frame * 2)
        x0, y0 = true_fwd(x1, y1)
        ok = np.flatnonzero((x0 > 12) & (x0 < W - 13) & (y0 > 12) & (y0 < H - 13))[:per_frame]
        E.append(np.c_[np.full(len(ok), f), x0[ok], y0[ok], x1[ok], y1[ok]])
    E = np.concatenate(E)
    z = rng.uniform(-0.35, 0.35, len(E)); N = rng.uniform(1500, 5000, len(E))
    img = np.zeros((nframes, H, 2 * W), np.float32)
    h = R // 2
    for c, n in enumerate((N, N * rng.uniform(0.6, 0.9, len(N)))):
        xx, yy = E[:, 1 + 2 * c], E[:, 2 + 2 * c]
        ix, iy = np.round(xx).astype(int), np.round(yy).astype(int)
        th = torch.tensor(np.c_[xx - ix, yy - iy, z, n, np.zeros_like(n)], dtype=torch.float32, device=dev)
        mu = psfs[c].expected(th).cpu().numpy()
        for k in range(len(E)):
            img[int(E[k, 0]), iy[k] - h:iy[k] + h + 1, c * W + ix[k] - h:c * W + ix[k] + h + 1] += mu[k]
    e = rng.poisson(img + 8.0) + rng.normal(0, RN, img.shape)
    return (e / GAIN + OFF).astype(np.float32)


def run(dev=None):
    dev = dev or 'cpu'           # small fits: the CPU is faster than a (shared) GPU
    t0 = time.time()
    a = simulate(dev=dev)
    lay = ChannelLayout([Channel(name='c0', crop=(0, 0, W, H)), Channel(name='c1', crop=(W, 0, W, H))])
    rd = ChannelReader(ArrayMovie(a), lay)
    st = dict(sum=1, chunk=200, frames=0, calib_frames=200, calib_n=120, d_um=0.30, calib_p0=[1.3, 1.3, 0.15])
    pm, rep = REG.register_insitu(rd, dict(offset=OFF, gain=GAIN, readnoise_e=RN), st, pixel_um=0.1, dev=dev,
                                  log=lambda *m: None)
    s = rep['channels']['1']
    gx, gy = np.meshgrid(np.linspace(15, W - 16, 20), np.linspace(15, H - 16, 20))
    tx, ty = true_fwd(gx, gy); mx, my = pm.fwd(1, gx, gy)
    err = np.hypot(mx - tx, my - ty)
    ix, iy = pm.inv(1, tx, ty)
    rt = np.hypot(ix - gx, iy - gy)
    res = dict(flip=s['flip'], flip_signif_ratio=s['flip_signif_ratio'], n_pairs=s['n_pairs'], map_err_max=float(err.max()),
               map_err_rms=float(np.sqrt((err ** 2).mean())), inv_err_max=float(rt.max()), roundtrip_rms=s['roundtrip_rms'],
               pull=s['pull_rsd'], astig=rep['astig'], time_s=round(time.time() - t0, 1), dev=dev)
    return pm, rep, res


def test_register_insitu_synthetic(result=None):
    pm, rep, r = result or run()
    assert r['flip'] == 'lr', r
    assert r['flip_signif_ratio'] > 5, r
    assert r['map_err_max'] < 0.02, r
    assert r['inv_err_max'] < 0.02, r
    assert r['roundtrip_rms'] < 0.01, r
    assert max(r['pull']) < 1.3, r
    ck = REG.checks({0: rep})
    assert all(c['ok'] for c in ck if not c['warn_only']), ck
    import json
    json.dumps(rep)                                   # json-serialisable


if __name__ == '__main__':
    res = run()
    print(res[2])
    test_register_insitu_synthetic(res)
    print('ok')
