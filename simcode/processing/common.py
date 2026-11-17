"""
Shared pieces of the per-window pipeline (simcode/processing): the read plan of the chunks, the rows of the pass-1 chunk
files, camera conversion and the PSF model. Movie reading: torch_psf.movie (TIFF) and simcode/io/movie.py (dcimg,
Prefetch). Only the camera and PSF entries of the calibration (calib json or psf.zip, simcode.psf.read_calib) are read
here; its 'pattern' entry is never used.
"""
import json, os
import numpy as np, torch

PX_NM = 108.33
# stored-candidate detection cut of the pass-1 / pass-2 front-ends: the head and the solve run on the detector candidates with
# p >= CAND_PMIN (production value since 2026-11-10; the phase-calibrated selection needs candidates below p 0.63)
CAND_PMIN = 0.3


def chunk_read_plan(movie, chunks, chunk, nf, crop, margin, pad=12):
    """the movie.read() calls of Pass1FrontFast / Pass2FrontFast.run_chunk for these chunk starts (same l0 / l1 rule)"""
    X0, Y0, S = region(movie, crop, margin)
    plan = []
    for f0 in chunks:
        f1 = min(f0 + chunk, nf); own1 = min(f1, movie.n)
        l0 = max(f0 - pad, 0) // 6 * 6; l1 = min(own1 + pad, movie.n)
        plan.append((l0, l1, Y0, X0, S))
    return plan


def load_pass1_rows(out, chunks, crop=None, prior_off=None, cols=('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')):
    """the pass-1 rows of the chunk files pass1_<f0>.npz of a run, each window once: windows loaded by two chunks (padding)
    are taken from one chunk only (start in [own0 - 3, own1 - 3); the last chunk keeps its tail). prior_off (px): the
    constant prior-position bias of the head, subtracted from x, y. crop (x0, y0, size): adds 'out' = central half (time
    centroid) & inside the crop, recomputed globally. Also chunk, row: the chunk start and row index of each row in its file."""
    Ls = [np.load(f'{out}/pass1_{f0}.npz') for f0 in chunks]
    keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1]))
                           for L, f0 in zip(Ls, chunks)])
    R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
    # the source of each row: chunk start f0 and row index in pass1_<f0>.npz (the brightness-noise fit reads the rows' covariances)
    R['chunk'] = np.concatenate([np.full(len(L['wstart']), f0, np.int64) for L, f0 in zip(Ls, chunks)])[keep]
    R['row'] = np.concatenate([np.arange(len(L['wstart'])) for L in Ls])[keep]
    if prior_off is not None:
        R['x'] = R['x'] - prior_off[0]; R['y'] = R['y'] - prior_off[1]
    if crop is not None:
        x0c, y0c, szc = crop
        R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= x0c) & (R['x'] < x0c + szc) & (R['y'] >= y0c) & (R['y'] < y0c + szc)
    return R


def load_calib(path, dev='cuda'):
    """camera maps + PSF from a calibration (calib json or psf.zip; camera / PSF entries only)."""
    from simcode.config import resolve
    path = resolve(path)
    from simcode import psf as spsf
    from simcode.psf import spline
    c = spsf.read_calib(path)
    cam = c['camera']; maps = spsf.camera_maps(path, c)
    psf = spsf.PSF.load(path, roisize=11, oversample=3, device=dev)   # opt-in calib PSF terms: angle average
    zmax = 0.2
    lut = spline(psf, zmax)
    return dict(offset=maps['offset_adu'].astype(np.float32), rn2=(maps['readnoise_e'] ** 2).astype(np.float32),
                gain=float(cam['gain_e_per_adu']), lut=lut, focus=float(c['focus_um']), zmax=zmax,
                px_nm=c['pixelsize_um'] * 1e3)


def region(movie, crop, margin):
    """crop = (x0, y0, size) of the output region; the loaded region adds `margin` px (clipped to the sensor)."""
    x0, y0, sz = crop
    H, W = movie.shape
    X0, Y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    S = min(sz + x0 - X0 + margin, W - X0, H - Y0)
    return X0, Y0, S


def pattern_frames(P, frames_abs, X0, Y0, dev='cuda'):
    """per-frame pattern [F,5] = (kx, ky, phase, depth, relint) in loaded-region coordinates from a pattern dict in the
    common file format (k [2,2], phi [6], m [6], logrel [6], dphi [2,K] on knot_frames; illumination
    relint (1 + m cos(k.r - phi_s - dphi_a(t))), s = frame % 6, a = s // 3)."""
    kvec = torch.as_tensor(P['k'], device=dev).float(); phi = torch.as_tensor(P['phi'], device=dev).float()
    m = torch.as_tensor(P['m'], device=dev).float().clamp(0, 1)
    lr = torch.as_tensor(P['logrel'], device=dev).float(); rel = torch.exp(lr - lr.mean())
    dphi = torch.as_tensor(P['dphi'], device=dev).float(); kf = float(P['knot_frames'])
    f = torch.as_tensor(frames_abs, device=dev).long(); s = f % 6
    ang = torch.as_tensor(P.get('angle_of_step', [0, 0, 0, 1, 1, 1]), device=dev).long()[s]
    tt = f.float() / kf; i0 = tt.floor().long().clamp(0, dphi.shape[1] - 2); w = (tt - i0).clamp(0, 1)
    drift = dphi[ang, i0] * (1 - w) + dphi[ang, i0 + 1] * w
    kk = kvec[ang]
    return torch.stack([kk[:, 0], kk[:, 1], phi[s] + drift - (kk[:, 0] * X0 + kk[:, 1] * Y0), m[s], rel[s]], -1)
