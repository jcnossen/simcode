"""Channel registration: polynomial maps channel c -> channel 0 (mirrored-crop pixel coordinates, x = column, y = row).

Stage 1 (beads): bead-align stacks (10 frames), per-channel Gaussian MLE on the frame mean, pairs matched through the
stored affine maps, polynomial fits of order 1..3 with leave-one-out residuals.
Stage 2 (in situ): see insitu_register() - bright isolated emitters in 6-frame per-channel sums, independent per-channel
Gaussian fits, residuals of the bead map vs field position, polynomial refit.
PolyMap: fwd(c, x, y) -> ch0 coords; inv(c, x0, y0) -> ch c coords (separately fitted inverse polynomial).
usage (bead stage): python register.py --work <4Pi work folder> --data <raw 4Pi movie folder>
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import os, sys, json, argparse
import numpy as np
import torch
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import Dcimg, crops, bead_align_file
from quickloc import affine_maps
from mle import GaussAstig, lm_fit

from simcode.pipeline.configs.fourpi import GAIN_E_PER_ADU as GAIN   # 4Pi camera gain (the same for MT and NPC)
from simcode.channels.registration import PolyMap  # noqa (the registration map class lives in the package)


def fit_poly(src, dst, order, w=None):
    A = feats(src[:, 0], src[:, 1], order)
    if w is not None: A, dst = A * w[:, None], dst * w[:, None]
    return np.linalg.lstsq(A, dst, rcond=None)[0]


def loo_resid(src, dst, order):
    A = feats(src[:, 0], src[:, 1], order)
    H = A @ np.linalg.pinv(A)
    r = dst - H @ dst
    return r / (1 - np.diag(H))[:, None]


def gauss_fit_rois(rois, v):
    m = GaussAstig(rois.shape[-1])
    d = torch.as_tensor(rois, dtype=torch.float64)
    th, ll, crlb, chi2, dev, mu = lm_fit(m, d, m.init(d), v, iters=60)
    return th.numpy(), crlb.numpy()


def bead_positions(cell, off, work, data, R=11, thr=150.0):
    """work: the 4Pi work folder, data: the folder of the raw 4Pi movies"""
    raw = Dcimg(bead_align_file(cell, work, data)).raw(0, 10)
    C = crops(raw)                                      # [10,4,200,360] ADU
    sat = C >= 65535
    E = (C - off) * GAIN
    mean = E.mean(0)
    pos, prec = [], []
    h = R // 2
    for c in range(4):
        f = ndi.gaussian_filter(mean[c], 1.5)
        pk = (f == ndi.maximum_filter(f, 9)) & (f > thr)
        ys, xs = np.nonzero(pk)
        ok = (ys >= h + 1) & (xs >= h + 1) & (ys < 200 - h - 1) & (xs < 360 - h - 1)
        ys, xs = ys[ok], xs[ok]
        rois = np.stack([mean[c, y - h:y + h + 1, x - h:x + h + 1] for y, x in zip(ys, xs)])
        nsat = np.array([sat[:, c, y - h:y + h + 1, x - h:x + h + 1].sum() for y, x in zip(ys, xs)])
        th, cr = gauss_fit_rois(rois, 1.3 / 10)
        # per-frame fits -> empirical position scatter (bead jitter + noise)
        roisf = np.stack([E[:, c, y - h:y + h + 1, x - h:x + h + 1] for y, x in zip(ys, xs)])   # [n,10,R,R]
        thf, _ = gauss_fit_rois(roisf.reshape(-1, R, R), 1.3)
        thf = thf.reshape(len(ys), 10, -1)
        sd = thf[:, :, :2].std(1).mean(1) / np.sqrt(10)
        P = np.c_[xs + th[:, 0], ys + th[:, 1], th[:, 2], th[:, 4], th[:, 5], nsat, sd, cr[:, 0]]
        pos.append(P)
    return pos


def match(ref, Q, maxd=1.5):
    d = np.hypot(ref[:, None, 0] - Q[None, :, 0], ref[:, None, 1] - Q[None, :, 1])
    j = d.argmin(1); m = d[np.arange(len(ref)), j] < maxd
    return np.nonzero(m)[0], j[m]


def bead_register(cell, off, out_dir, work, data, order_use=2):
    pos = bead_positions(cell, off, work, data)
    _, Tf = affine_maps(cell, work, data)
    rep = {}
    cf, ci = {}, {}
    for c in (1, 2, 3):
        Q = pos[c][:, :2] @ Tf[c - 1][:, :2].T + Tf[c - 1][:, 2]
        i0, ic = match(pos[0], Q)
        good = (pos[0][i0, 5] == 0) & (pos[c][ic, 5] == 0) & (pos[0][i0, 2] > 300) & (pos[c][ic, 2] > 300)
        i0, ic = i0[good], ic[good]
        src, dst = pos[c][ic, :2], pos[0][i0, :2]
        r = {}
        for o in (1, 2, 3):
            if len(src) > 3 * (o + 1) * (o + 2) / 2:
                lr = loo_resid(src, dst, o); r[o] = np.sqrt((lr ** 2).mean(0)).round(4).tolist()
        aff = dst - Q[ic]
        rep[c] = dict(npairs=int(len(src)), stored_affine_rms=np.sqrt((aff - aff.mean(0)) ** 2).mean(0).round(4).tolist(),
                      stored_affine_mean=aff.mean(0).round(4).tolist(), loo_rms_by_order=r,
                      bead_pos_noise_px=float(np.median(np.hypot(pos[0][i0, 6], pos[c][ic, 6]))))
        cf[c] = fit_poly(src, dst, order_use); ci[c] = fit_poly(dst, src, order_use)
        rep[c]['src'] = src.tolist(); rep[c]['dst'] = dst.tolist()
    pm = PolyMap(cf, ci, order_use)
    pm.save(f'{out_dir}/reg_beads_{cell}.json', source='bead-align', report={c: {k: v for k, v in rep[c].items() if k not in ('src', 'dst')} for c in rep})
    return pm, rep, pos


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True, help='4Pi work folder')
    ap.add_argument('--data', required=True, help='folder of the raw 4Pi movies')
    A = ap.parse_args(); P4 = A.work
    out = f'{P4}/smlm/reg'; os.makedirs(out, exist_ok=True)
    for cell in ('NPC', 'MT'):
        off = np.load(f'{P4}/smlm/camera/camera_{cell}.npz')['off']
        pm, rep, pos = bead_register(cell, off, out, P4, A.data)
        print(cell, 'beads per channel', [len(p) for p in pos], 'saturated beads', [int((p[:, 5] > 0).sum()) for p in pos])
        for c in rep:
            print(f'  ch{c}: pairs {rep[c]["npairs"]}, stored affine mean {rep[c]["stored_affine_mean"]} rms {rep[c]["stored_affine_rms"]}, '
                  f'LOO rms by order {rep[c]["loo_rms_by_order"]}, bead position noise {rep[c]["bead_pos_noise_px"]:.4f} px')
        np.save(f'{out}/beadpos_{cell}.npy', np.array(pos, dtype=object), allow_pickle=True)


def load_s6(cell, fn, f0, nfr, off):
    """6-frame per-channel sums [nfr/6, 4, 200, 360] (e) of one contiguous read"""
    d = Dcimg(fn)
    out = []
    for a0 in range(f0, f0 + nfr, 600):
        E = (crops(d.raw(a0, 600)) - off) * GAIN
        out.append(E.reshape(100, 6, 4, 200, 360).sum(1))
    return np.concatenate(out)


def insitu_positions(S6, pm, R=11, min_sum=800.0, iso=8):
    """per-channel independent Gaussian MLE of bright isolated emitters in 6-frame per-channel sums S6 [nw,4,H,W].
    Returns [n, 4, 7]: per channel (x, y, N, sx, sy, crlb_x, crlb_y) in that channel's coords."""
    from quickloc import Warp
    warp = Warp(lambda c, xx, yy: pm.inv(c, xx, yy))
    h = R // 2
    out = []
    for w in range(len(S6)):
        ssum = sum(warp(S6[w, c], c) for c in range(4))
        fl = ndi.gaussian_filter(ssum, 1.5) - ndi.gaussian_filter(ssum, 5)
        pk = (fl == ndi.maximum_filter(fl, 7)) & (fl > 30)
        ys, xs = np.nonzero(pk)
        for y, x in zip(ys, xs):
            if not (10 <= y < 190 and 60 <= x < 300): continue
            dd = np.hypot(ys - y, xs - x)
            if ((dd > 0) & (dd < iso)).any(): continue
            tot = ssum[y - 3:y + 4, x - 3:x + 4].sum() - 49 * np.median(ssum[y - 7:y + 8, x - 7:x + 8])
            if tot < min_sum: continue
            rows = []
            for c in range(4):
                xc, yc = pm.inv(c, float(x), float(y)); ix, iy = int(round(float(xc))), int(round(float(yc)))
                if not (h <= iy < 200 - h and h <= ix < 360 - h): break
                rows.append((c, ix, iy, S6[w, c, iy - h:iy + h + 1, ix - h:ix + h + 1]))
            if len(rows) < 4: continue
            out.append(rows)
    rois = np.stack([[r[3] for r in rr] for rr in out])               # [n,4,R,R]
    th, cr = gauss_fit_rois(rois.reshape(-1, R, R), 6 * 1.3)
    th, cr = th.reshape(-1, 4, 6), cr.reshape(-1, 4, 6)
    cx = np.array([[r[1] for r in rr] for rr in out]); cy = np.array([[r[2] for r in rr] for rr in out])
    X = cx + th[..., 0]; Y = cy + th[..., 1]
    return np.stack([X, Y, th[..., 2], th[..., 4], th[..., 5], cr[..., 0], cr[..., 1]], -1)


def clean(P):
    ok = np.isfinite(P).all((1, 2)) & (P[:, :, 5] < 0.1).all(1) & (P[:, :, 6] < 0.15).all(1) & (P[:, :, 3] < 3.8).all(1) & (P[:, :, 4] < 3.8).all(1)
    # all four channels must agree on the PSF shape (overlapping emitters / mismatched pairs give inconsistent sx, sy)
    sx, sy = P[:, :, 3], P[:, :, 4]
    ok &= (np.ptp(sx, 1) < 0.5) & (np.ptp(sy, 1) < 0.5)
    return P[ok]


def robust_fit(src, dst, order, w, nit=4, k=4.0):
    keep = np.ones(len(src), bool)
    for _ in range(nit):
        cf = fit_poly(src[keep], dst[keep], order, w[keep])
        r = dst - feats(src[:, 0], src[:, 1], order) @ cf
        s = 1.4826 * np.median(np.abs(r[keep]), 0)
        keep = (np.abs(r) < k * s).all(1)
    return cf, keep


def insitu_register(P, pm0, order=2, out_json=None, source=''):
    rep = {'n': int(len(P))}
    cf, ci = {}, {}
    rs = lambda r: (1.4826 * np.median(np.abs(r - np.median(r, 0)), 0)).round(4).tolist()
    for c in (1, 2, 3):
        src, dst = P[:, c, :2], P[:, 0, :2]
        x0, y0 = pm0.fwd(c, src[:, 0], src[:, 1]); r0 = dst - np.c_[x0, y0]
        e2 = (P[:, c, 5:7] ** 2 + P[:, 0, 5:7] ** 2)
        w = 1 / np.sqrt(e2.mean(1) + 0.02 ** 2)
        cf[c], keep = robust_fit(src, dst, order, w); ci[c], _ = robust_fit(dst, src, order, w)
        r1 = dst - feats(src[:, 0], src[:, 1], order) @ cf[c]
        exp_rms = np.sqrt(np.median(e2, 0))
        rep[c] = dict(start_map_median=np.median(r0, 0).round(4).tolist(), start_map_robust_sd=rs(r0),
                      insitu_robust_sd=rs(r1), crlb_expected=exp_rms.round(4).tolist(),
                      excess=np.sqrt(np.clip(np.array(rs(r1)) ** 2 - exp_rms ** 2, 0, None)).round(4).tolist(),
                      kept=float(keep.mean()))
    pm = PolyMap(cf, ci, order)
    if out_json: pm.save(out_json, source=source, report=rep)
    return pm, rep
