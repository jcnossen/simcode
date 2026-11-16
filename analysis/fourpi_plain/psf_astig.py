"""Astigmatism of the plain-4Pi PSF model (public 4Pi-SMS bead stack fit, PSF4Pi state) vs the bead data: the
astigmatic focus separation (elliptical Gaussian on the phase-averaged 4-channel sum, stage z axis) of the model with
the astigmatism Zernikes (Noll 5, 6; both arms) scaled by f. Target: the data / model separation ratio of the bead fit
(out/fourpi_refactor/public_4pi/results/astig_cmp_<tag>.json: elliptical Gaussian on the bead ROIs and on the fitted
model at the beads' own positions and phases; the model is 12-15 % short). The separation here (model at the centre,
phase-averaged) differs in absolute value from that procedure, so the ratio is matched: sep(f) / sep(1) = data / model.
Writes the state with that scale (CPU, a few minutes).

  python analysis/fourpi_plain/psf_astig.py <psf4pi_<tag>.json> <astig_cmp_<tag>.json> <out_state.json> [--scales 1,1.1,1.2,1.3]
"""
import argparse, json, copy
import numpy as np, torch
from scipy.optimize import least_squares
from simcode.fourpi.coherent_psf import psf_from_state

R = 25
yy, xx = np.mgrid[-(R // 2):R // 2 + 1, -(R // 2):R // 2 + 1].astype(float)


def gfit(im):
    def f(p):
        x0, y0, sx, sy, N, b = p
        return (N / (2 * np.pi * sx * sy) * np.exp(-(xx - x0) ** 2 / (2 * sx ** 2) - (yy - y0) ** 2 / (2 * sy ** 2)) + b - im).ravel()
    return least_squares(f, [0, 0, 1.5, 1.5, im.sum(), 0], bounds=([-4, -4, 0.5, 0.5, 1e-6, -1], [4, 4, 8, 8, 1e3, 1])).x


def curve(z, p):
    s0, c, dd, A, B = p; u = (z - c) / dd
    return s0 * np.sqrt(np.maximum(1 + u ** 2 + A * u ** 3 + B * u ** 4, 1e-6))


def scaled(st, f):
    s = copy.deepcopy(st); zi = s['meta']['zidx']
    for k in ('zt', 'zb'):
        z = list(s[k])
        for j in (5, 6): z[zi.index(j)] *= f
        s[k] = z
    return s


def separation(st, fit, zst_nm):
    """stage-axis focus separation (nm) of the phase-averaged model for the fit's stage -> model z mapping"""
    s = dict(st); s['meta'] = dict(st['meta'], R=R)
    m = psf_from_state(s, device='cpu').eval()
    z = float(np.mean(fit['z0'])) + fit['args']['s'] * fit['zscale'] * (zst_nm - zst_nm.mean()) / 1000.0
    pos = torch.tensor(np.c_[np.zeros_like(z), np.zeros_like(z), z], dtype=torch.float32)
    with torch.no_grad():
        g = sum(m(pos, phi_extra=torch.full((len(z),), ph)) for ph in (0.0, np.pi)) / 2      # incoherent part
    S = g.sum(1).numpy()
    sg = np.array([gfit(im)[2:4] for im in S])
    fx = least_squares(lambda p: curve(zst_nm, p) - sg[:, 0], [1.2, 800, 400, 0, 0]).x
    fy = least_squares(lambda p: curve(zst_nm, p) - sg[:, 1], [1.2, 1300, 400, 0, 0]).x
    return float(fy[1] - fx[1]), sg


def main():
    p = argparse.ArgumentParser(); p.add_argument('psf'); p.add_argument('astig'); p.add_argument('out')
    p.add_argument('--scales', default='1.0,1.1,1.2,1.3')
    a = p.parse_args()
    torch.set_num_threads(16)
    fit = json.load(open(a.psf)); st = fit['psf']
    ac = json.load(open(a.astig))
    ratio = float(np.mean(ac['data']['sep_nm']) / np.mean(ac['model']['sep_nm']))
    zst = np.arange(0, 2001, 40.0)
    rows = []
    for f in [float(v) for v in a.scales.split(',')]:
        sep, _ = separation(scaled(st, f), fit, zst)
        rows.append((f, sep)); print(f'astig scale {f:.3f}: separation {sep:.1f} nm', flush=True)
    fs, ss = np.array(rows).T
    data_sep = ratio * ss[fs == 1.0][0]
    print(f'data / model separation ratio of the bead fit {ratio:.4f} -> target {data_sep:.1f} nm', flush=True)
    f_best = float(np.interp(abs(data_sep), np.abs(ss), fs)) if np.all(np.diff(np.abs(ss)) > 0) else float(fs[np.argmin(np.abs(np.abs(ss) - abs(data_sep)))])
    sep_best, _ = separation(scaled(st, f_best), fit, zst)
    out = dict(fit, psf=scaled(st, f_best), astig_scale=f_best, astig_sep_nm=dict(target=data_sep, ratio=ratio, model_scaled=sep_best, scan=rows),
               note='Noll 5 / 6 of both arms x astig_scale (analysis/fourpi_plain/psf_astig.py)')
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'astig scale {f_best:.3f}: separation {sep_best:.1f} nm (data {data_sep:.1f}) -> {a.out}')


if __name__ == '__main__':
    main()
