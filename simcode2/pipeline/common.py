"""
Shared I/O for the uncalibrated per-window pipeline (simcode2/pipeline): movie reading (multi-part Andor SOLIS TIFFs,
absolute frame index), camera conversion and the PSF model. Only the camera and PSF entries of the calibration json
are read here; its 'pattern' entry is never used.
"""
import json, os
import numpy as np, tifffile, torch
from simcode2.psf.vectorial import VectorialPSF

PX_NM = 108.33


class Movie:
    """Absolute-frame access to movie.tif + movie_X1.tif + ... (pattern step = absolute frame % 6)."""
    def __init__(self, path):
        self.parts = [path] + [path.replace('.tif', f'_X{i}.tif') for i in range(1, 10)
                               if os.path.exists(path.replace('.tif', f'_X{i}.tif'))]
        self.tifs = [tifffile.TiffFile(p) for p in self.parts]
        self.bounds = np.cumsum([0] + [len(t.pages) for t in self.tifs])
        self.n = int(self.bounds[-1])
        self.shape = self.tifs[0].pages[0].shape

    def read(self, f0, f1, y0, x0, S):
        out = np.empty((f1 - f0, S, S), np.float32)
        for i, fi in enumerate(range(f0, f1)):
            j = int(np.searchsorted(self.bounds, fi, side='right') - 1)
            out[i] = self.tifs[j].pages[int(fi - self.bounds[j])].asarray()[y0:y0 + S, x0:x0 + S]
        return out


def camera_maps_path(path, c=None):
    """camera-map npz of a calib json: 'camera.maps', relative to the json's directory unless absolute"""
    c = c if c is not None else json.load(open(path))
    m = c['camera']['maps']
    return m if os.path.isabs(m) else os.path.join(os.path.dirname(os.path.abspath(path)), m)


def load_calib(path, dev='cuda'):
    """camera maps + PSF from a calib json (camera / PSF entries only)."""
    from simcode2.config import resolve
    path = resolve(path)
    c = json.load(open(path))
    cam = c['camera']; maps = np.load(camera_maps_path(path, c))
    psf = VectorialPSF(11, pixelsize=c['pixelsize_um'], NA=c['NA'], n_imm=c['n_imm'], n_med=c['n_med'],
                       wavelength=c['wavelength_um'], oversample=3, device=dev)
    with torch.no_grad():
        psf.zcoef.copy_(torch.tensor(c['zernike_waves'], device=dev)); psf.apod.copy_(torch.tensor(c['apod'], device=dev))
    from simcode2.psf.lut import PSFLUT
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
