"""
Shared pieces of the per-window pipeline (simcode/pipeline): the read plan of the chunks, the rows of the pass-1 chunk
files, camera conversion and the PSF model. Movie reading itself is simcode/fileio/movie.py (Movie = TiffMovie,
PrefetchMovie = Prefetch). Only the camera and PSF entries of the calibration json are read here; its 'pattern' entry is
never used.
"""
import json, os
import numpy as np, torch
from simcode.psf.vectorial import VectorialPSF
from simcode.fileio.movie import TiffMovie, Prefetch

PX_NM = 108.33
# stored-candidate detection cut of the pass-1 / pass-2 front-ends: the head and the solve run on the detector candidates with
# p >= CAND_PMIN (production value since 2026-11-10; the phase-calibrated selection needs candidates below p 0.63)
CAND_PMIN = 0.3


# the readers live in simcode/fileio/movie.py (shared with the 4Pi pipelines); the old names stay valid
Movie = TiffMovie
PrefetchMovie = Prefetch


def chunk_read_plan(movie, chunks, chunk, nf, crop, margin, pad=12):
    """the movie.read() calls of Pass1SCFast / Pass2P2Fast.run_chunk for these chunk starts (same l0 / l1 rule)"""
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
    centroid) & inside the crop, recomputed globally."""
    Ls = [np.load(f'{out}/pass1_{f0}.npz') for f0 in chunks]
    keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1]))
                           for L, f0 in zip(Ls, chunks)])
    R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
    if prior_off is not None:
        R['x'] = R['x'] - prior_off[0]; R['y'] = R['y'] - prior_off[1]
    if crop is not None:
        x0c, y0c, szc = crop
        R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= x0c) & (R['x'] < x0c + szc) & (R['y'] >= y0c) & (R['y'] < y0c + szc)
    return R


def camera_maps_path(path, c=None):
    """camera-map npz of a calib json: 'camera.maps', relative to the json's directory unless absolute"""
    c = c if c is not None else json.load(open(path))
    m = c['camera']['maps']
    return m if os.path.isabs(m) else os.path.join(os.path.dirname(os.path.abspath(path)), m)


def load_calib(path, dev='cuda'):
    """camera maps + PSF from a calib json (camera / PSF entries only)."""
    from simcode.config import resolve
    path = resolve(path)
    c = json.load(open(path))
    cam = c['camera']; maps = np.load(camera_maps_path(path, c))
    psf = VectorialPSF(11, pixelsize=c['pixelsize_um'], NA=c['NA'], n_imm=c['n_imm'], n_med=c['n_med'],
                       wavelength=c['wavelength_um'], oversample=3, device=dev)
    with torch.no_grad():
        psf.zcoef.copy_(torch.tensor(c['zernike_waves'], device=dev)); psf.apod.copy_(torch.tensor(c['apod'], device=dev))
    from simcode.psf.vectorial import set_psf_extras
    set_psf_extras(psf, c)   # (2026-11-11) opt-in calib PSF terms (angle average); absent = unchanged
    from simcode.psf.lut import PSFLUT
    zmax = 0.2
    lut = PSFLUT(psf, c['focus_um'], zmax)
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
