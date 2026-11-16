"""
z after the pattern solve (3D emitter network, 2026-11-14). The pattern informs x, y only; z comes from the emitter network's
3D Gaussian position prior (emitter/model.EmitterNet: z, s_z|xy, b) conditioned on the solved x, y (z_given_xy, z_columns).
Pass 2 takes z from PASS 1 (the pass-2 network's z is biased upward at low photons): each pass-2 row is linked one-to-one to
a pass-1 candidate of the same window (link_pass1: detector points within LINK_R px) and that candidate's prior is
conditioned on the pass-2 solved x, y (z_from_pass1, pass1_z_columns).
"""
import torch


def z_given_xy(o, r, r0, s):
    """z after the pattern solve: the head's 3D Gaussian prior (EmitterNet: z, s_z|xy = exp(lsz), b = bz) conditioned on the
    solved x, y. The pattern informs x, y only, so with the solve's x, y posterior (mean r [n,2], sd s [n,2], uncorrelated):
        z = z_prior + b . (r - r0)              var z = s_z|xy^2 + sum_i b_i^2 s_i^2       cov(x, z) = b_x s_x^2, cov(y, z) = b_y s_y^2
    r0 = the prior x, y the solve used (prior_off removed). -> z [n] px (relative to the calibration's focal plane), sd [n],
    cov [n,3] = (xy, xz, yz) px^2 (torch_psf Dataset 'cov'; xy = 0: the solve's x, y are uncorrelated)."""
    bz = o['bz'].float(); s2 = s.float() ** 2
    z = o['z'].float() + (bz * (r.float() - r0.float())).sum(1)
    var = o['lsz'].float().exp() ** 2 + (bz ** 2 * s2).sum(1)
    cov = torch.stack([torch.zeros_like(z), bz[:, 0] * s2[:, 0], bz[:, 1] * s2[:, 1]], 1)
    return z, var.sqrt(), cov


def z_columns(o, so):
    """table columns of z after the solve (z_given_xy on the solve's r, s_cal and prior r_prior): z_hy0 (px, relative to the
    calibration's focal plane), sz_cal (sd px), cov_xz, cov_yz (px^2), z_prior (the head's z, px)"""
    z, sz, cov = z_given_xy(o, so['r'], so['r_prior'], so['s_cal'])
    return dict(z_hy0=z, sz_cal=sz, cov_xz=cov[:, 1], cov_yz=cov[:, 2], z_prior=o['z'].float())


LINK_R = 1.0         # px: pass-2 row <-> pass-1 candidate, detector points of the same window (gate scenes: median 0.12-0.19 px)
Z_COLS = ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior')


def link_pass1(k1, x1, y1, k2, x2, y2, r=LINK_R, rounds=10):
    """one-to-one link of pass-2 rows to pass-1 candidates: same key (window start; FOV in the gate dumps), detector points
    within r px; iterated mutual nearest neighbours (each round links the mutual nearest pairs among the rows not yet
    linked, until no new pair). Exact distances (simcode.util.spatial_hash, float64). -> j [n2] int64 index into the pass-1
    rows, -1 = no pass-1 partner"""
    import numpy as np
    from simcode.util import spatial_hash as SH
    a = lambda v, t=np.float64: np.asarray(v.cpu().numpy() if torch.is_tensor(v) else v).astype(t)
    k1, x1, y1, k2, x2, y2 = a(k1, np.int64), a(x1), a(y1), a(k2, np.int64), a(x2), a(y2)
    j = np.full(len(x2), -1, np.int64)
    f1 = np.ones(len(x1), bool); f2 = np.ones(len(x2), bool)
    for _ in range(rounds):
        i1, i2 = np.nonzero(f1)[0], np.nonzero(f2)[0]
        if not len(i1) or not len(i2): break
        d21, j21 = SH.nearest(x1[i1], y1[i1], k1[i1], x2[i2], y2[i2], k2[i2], r)
        _, j12 = SH.nearest(x2[i2], y2[i2], k2[i2], x1[i1], y1[i1], k1[i1], r)
        q = np.nonzero(np.isfinite(d21))[0]
        q = q[j12[j21[q]] == q]
        if not len(q): break
        j[i2[q]] = i1[j21[q]]; f2[i2[q]] = False; f1[i1[j21[q]]] = False
    return j


def z_from_pass1(p1, j, r, s):
    """z of the pass-2 rows from PASS 1 (2026-11-14: the pass-2 head's z is biased upward at low photons, the pass-1 z is
    better): the pass-1 head's 3D prior of the linked candidate (link_pass1) conditioned on the pass-2 solved x, y
    (z_given_xy). p1: pass-1 rows, tensors z [n1], lsz [n1], bz [n1,2], r0 [n1,2] = the pass-1 prior x, y with the pass-1
    calibration's prior_off removed, in the frame of r; j [n2] (-1: no partner); r, s [n2,2] = the pass-2 solved x, y and
    calibrated sd. -> dict of Z_COLS [n2] (z_prior = the pass-1 head's z), NaN where j < 0"""
    j = torch.as_tensor(j, device=r.device, dtype=torch.long)
    ok = j >= 0; jj = j.clamp(min=0)
    if len(p1['z']) == 0:
        nn = torch.full((len(j),), float('nan'), device=r.device)
        return {k: nn.clone() for k in Z_COLS}
    o = dict(z=p1['z'][jj], lsz=p1['lsz'][jj], bz=p1['bz'][jj])
    z, sz, cov = z_given_xy(o, r, p1['r0'][jj], s)
    nan = lambda v: torch.where(ok, v.float(), torch.full_like(v.float(), float('nan')))
    return dict(z_hy0=nan(z), sz_cal=nan(sz), cov_xz=nan(cov[:, 1]), cov_yz=nan(cov[:, 2]), z_prior=nan(o['z']))


def pass1_z_columns(P1, cal1, ws, xd, yd, r, s):
    """the z columns of the pipeline's pass-2 rows (processing/pass2_inference.Pass2Front / processing/inference_fast.Pass2FrontFast): P1 = the chunk's stored pass-1 rows
    (pass1_<f0>.npz: wstart, x_det, y_det, x, y, z, sz, bzx, bzy; global px), cal1 = the pass-1 solve calibration (path or
    dict; its prior_off); ws [n2] window starts, xd, yd [n2] detector points, r [n2,2] solved x, y, s [n2,2] calibrated sd
    of the pass-2 rows (global px, tensors). -> dict of Z_COLS + p1_row (index into P1; -1 = no partner, z NaN)"""
    import json, numpy as np
    from simcode.processing.pass1_inference import z_inputs
    cal1 = json.load(open(cal1)) if isinstance(cal1, str) else cal1
    j = link_pass1(P1['wstart'], P1['x_det'], P1['y_det'], ws, xd, yd)
    tt = lambda v: torch.as_tensor(np.asarray(v), device=r.device, dtype=torch.float32)
    p1 = z_inputs(P1, np.arange(len(P1['x'])), tt)
    p1['r0'] = torch.stack([tt(P1['x']), tt(P1['y'])], 1) - tt(cal1['prior_off'])[None]
    Z = z_from_pass1(p1, j, r.float(), s.float())
    Z['p1_row'] = torch.as_tensor(j, device=r.device)
    return Z
