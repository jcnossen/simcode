"""Stage c1 (spots): pattern-free Gaussian SMLM on moving frame sums, the spot finder of the automatic pipeline
(docs/AUTO_PIPELINE.md section 3). The same kind of input as the SIMCODE-pipeline smlm table that the earlier in-situ recipe
used (6-frame moving sums, one window per frame start); the procedure is archive-2026-11-15:analysis/nanoruler/quick_smlm.py as a function:
smoothed (sigma 1.5 px) difference to the per-chunk temporal median of the sums, local maxima (7 x 7) above thr x robust
noise, Poisson Levenberg-Marquardt fit of x, y, I, bg, sigma of a pixel-integrated Gaussian on an R x R ROI
(torch_psf fit_gaussian_free_sigma).
Coordinates: x = column, y = row, pixel centres at integers (as the pipeline and simcode/calibration).
Chunks of 600 window starts are the units of simcode.io.work.Units (resumable); merged into a Picasso-style hdf5 'locs'."""
import os
import numpy as np, torch, torch.nn.functional as Fn
from simcode.psf import fit_gaussian_free_sigma

DT = [('frame', 'u4'), ('x', 'f4'), ('y', 'f4'), ('photons', 'f4'), ('bg', 'f4'), ('sx', 'f4'), ('lpx', 'f4'), ('lpy', 'f4'),
      ('chi2', 'f4'), ('snr', 'f4')]


def find_chunk(reader, c0, c1, gain, offset, nsum=6, thr=5.0, R=11, sigma0=1.3, dev='cuda'):
    """window starts [c0, c1) -> structured array (DT)"""
    h = R // 2
    f1 = min(c1 + nsum - 1, reader.n)
    mov = torch.tensor(reader.read(c0, f1), device=dev)
    ph = (mov - offset) * gain
    if nsum > 1:
        cs = torch.cat([torch.zeros_like(ph[:1]), ph.cumsum(0)]); ph = cs[nsum:] - cs[:-nsum]
    if len(ph) == 0: return np.zeros(0, DT)
    g1 = torch.exp(-torch.arange(-4, 5, device=dev).float() ** 2 / (2 * 1.5 ** 2)); g1 /= g1.sum()
    med = ph[::6].median(0).values
    d = ph - med
    sm = Fn.conv2d(Fn.conv2d(d[:, None], g1[None, None, :, None], padding=(4, 0)), g1[None, None, None, :], padding=(0, 4))[:, 0]
    noise = 1.4826 * (sm - sm.median()).abs().median()
    mx = Fn.max_pool2d(sm[:, None], 7, 1, 3)[:, 0]
    pk = (sm == mx) & (sm > thr * noise)
    pk[:, :h + 1] = False; pk[:, -h - 1:] = False; pk[:, :, :h + 1] = False; pk[:, :, -h - 1:] = False
    f, iy, ix = pk.nonzero(as_tuple=True)
    if len(f) == 0: return np.zeros(0, DT)
    oy, ox = torch.meshgrid(torch.arange(-h, h + 1, device=dev), torch.arange(-h, h + 1, device=dev), indexing='ij')
    roi = ph[f[:, None, None], iy[:, None, None] + oy[None], ix[:, None, None] + ox[None]].clamp(min=0)
    P_, C_, X_ = [], [], []
    for s in range(0, len(f), 4096):
        p, cr, chi = fit_gaussian_free_sigma(roi[s:s + 4096], sigma0); P_.append(p); C_.append(cr); X_.append(chi)
    p = torch.cat(P_); cr = torch.cat(C_); chi = torch.cat(X_)
    L = np.zeros(len(f), DT)
    L['frame'] = (f + c0).cpu().numpy(); L['x'] = (ix + p[:, 0]).cpu().numpy(); L['y'] = (iy + p[:, 1]).cpu().numpy()
    L['photons'] = p[:, 2].cpu().numpy(); L['bg'] = p[:, 3].cpu().numpy(); L['sx'] = p[:, 4].cpu().numpy()
    L['lpx'] = cr[:, 0].cpu().numpy(); L['lpy'] = cr[:, 1].cpu().numpy(); L['chi2'] = chi.cpu().numpy()
    L['snr'] = (sm[f, iy, ix] / noise).cpu().numpy()
    return L


def run(movie, work, gain, offset, maxframe=30000, nsum=6, thr=5.0, R=11, sigma0=1.3, chunk=600, claim=False, dev='cuda', log=print):
    from simcode.io.movie import open_movie
    from simcode.io.work import Units
    M = open_movie(movie)
    N = min(maxframe, M.n - nsum + 1)
    keys = list(range(0, N, chunk))
    U = Units(work, 'spots', pattern='spots_{key}.npz', claim=claim)
    for k in U.todo(keys):
        L = find_chunk(M, k, min(k + chunk, N), gain, offset, nsum, thr, R, sigma0, dev)
        U.save(k, locs=L)
        U.record(k, dict(n=int(len(L))))
        log(f'spots {k}: {len(L)}')
    return np.concatenate([np.load(U.path(k))['locs'] for k in keys])


def write_hdf5(L, path):
    import h5py
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as hf: hf.create_dataset('locs', data=L)
    os.replace(tmp, path)


def good(L, chi2_max=3.6, snr_min=8.0, sigma_range=(0.75, 1.35)):
    """good-fit filter (as archive-2026-11-15:analysis/nanoruler/insitu_any.sh): chi2, snr, sigma within a range around the median sigma of
    the bright fits. Returns (mask, median sigma)"""
    b = L['snr'] > snr_min
    s_med = float(np.median(L['sx'][b])) if b.any() else float('nan')
    m = (L['chi2'] < chi2_max) & b & (L['sx'] > sigma_range[0] * s_med) & (L['sx'] < sigma_range[1] * s_med)
    return m, s_med
