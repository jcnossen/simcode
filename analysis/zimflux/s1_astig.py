"""S1 step 2: astigmatic PSF estimation on the cycle sums (paper 3.6.1): Poisson LM (sCMOS read-noise shift) with the
calibrated vectorial PSF (LUT), 3 z starts (z_e 0.0 / 0.15 / 0.3 um), best likelihood kept. Also per-frame intensities
N_k, b_k with the position fixed (paper 3.5.1), and the Fisher CRLB of (x, y, z, N, b).
Sign test (mode 'signtest'): fits the 5000 brightest spots with the astigmatism coefficient as calibrated (+) and flipped (-)
and compares the log-likelihood (pins the astigmatism orientation of the image axes).
usage: python s1_astig.py <gpu> [signtest|fit] [astig_sign=+1]"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/jc/simcode2/analysis/zimflux")
from s1_common import *
from simcode.fourpi.lm import lm_poisson_j, fisher_j
dev = torch.device(f'cuda:{sys.argv[1]}'); mode = sys.argv[2] if len(sys.argv) > 2 else 'fit'
sgn = float(sys.argv[3]) if len(sys.argv) > 3 and mode == 'fit' else 1.0
Z = np.load(WORK + 'rois.npz'); cam = np.load(WORK + 'camera.npz')
rois, corner, cyc, box = Z['rois'], Z['corner'], Z['cycle'], Z['boxstat']
rn = torch.tensor(cam['rnvar'], device=dev, dtype=torch.float32)
ar = torch.arange(P, device=dev)


def vpatch(cn):
    cn = torch.as_tensor(cn, device=dev).long()
    return rn[(cn[:, 0, None] + ar)[:, :, None], (cn[:, 1, None] + ar)[:, None, :]].reshape(len(cn), -1)


def fit_astig(fj, d, v, iters=(25, 25)):
    n = d.shape[0]
    img = d.view(n, P, P)
    edge = torch.cat([img[:, 0], img[:, -1], img[:, 1:-1, 0], img[:, 1:-1, -1]], 1)
    b0 = edge.median(1).values.clamp(min=0.5)
    s = (img - b0[:, None, None]).clamp(min=0); N0 = s.sum((1, 2)).clamp(min=50)
    o = ar.float() - 7.5
    x0 = ((s * o[None, None, :]).sum((1, 2)) / N0).clamp(-3, 3); y0 = ((s * o[None, :, None]).sum((1, 2)) / N0).clamp(-3, 3)
    lo = torch.tensor([-7.0, -7.0, ZLO, 1.0, 0.01], device=dev); hi = torch.tensor([7.0, 7.0, ZHI, 1e6, 1e4], device=dev)
    best = None
    for z0 in (0.0, 0.15, 0.3):
        th0 = torch.stack([x0, y0, torch.full_like(x0, z0), N0 * 1.3, b0], 1)
        th, ll, mu = lm_poisson_j(fj, d, th0, v, lo, hi, iters=iters[0])
        if best is None: best = [th, ll]
        else:
            k = ll > best[1]; best[0] = torch.where(k[:, None], th, best[0]); best[1] = torch.where(k, ll, best[1])
    th, ll, mu = lm_poisson_j(fj, d, best[0], v, lo, hi, iters=iters[1])
    return th, ll, mu


def run(lut, idx, B=8192):
    fj = astig_model(lut); out = {k: [] for k in ('th', 'll', 'dev', 'crlb', 'Nk', 'bk')}
    for i0 in range(0, len(idx), B):
        ii = idx[i0:i0 + B]
        R = torch.tensor(rois[ii], device=dev).float(); n = len(ii)
        d = R.sum(1).view(n, -1); v = 3 * vpatch(corner[ii])
        th, ll, mu = fit_astig(fj, d, v)
        dv = 2 * ((d + v).clamp(min=1e-3) * torch.log((d + v).clamp(min=1e-3) / (mu + v)) - (d - mu)).sum(1) / (P * P)
        F = fisher_j(fj, th, v); cr = torch.sqrt(torch.diagonal(torch.linalg.pinv(F), dim1=1, dim2=2).clamp(min=0))
        # per-frame intensities with the position fixed: mu_k = N_k h + b_k
        h = lut.evaluate(th[:, :3]).view(n, -1)
        Nk, bk = [], []
        for k in range(3):
            dk = R[:, k].reshape(n, -1); vk = vpatch(corner[ii])
            fk = lambda t_: (t_[:, 0, None] * h + t_[:, 1, None], torch.stack([h, torch.ones_like(h)], -1))
            t0 = torch.stack([th[:, 3] / 3, th[:, 4] / 3], 1)
            tk, _, _ = lm_poisson_j(fk, dk, t0, vk, torch.tensor([-1e4, 1e-3], device=dev), None, iters=10)
            Nk.append(tk[:, 0]); bk.append(tk[:, 1])
        for k, a in (('th', th), ('ll', ll), ('dev', dv), ('crlb', cr), ('Nk', torch.stack(Nk, 1)), ('bk', torch.stack(bk, 1))):
            out[k].append(a.cpu().numpy())
        print(i0, flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


if mode == 'convtest':   # PSF convention: calibrated / Z6 flipped / transposed pupil, and focus shifts of the best
    idx = np.argsort(-box)[:3000]; res = {}
    cases = ([(None, 0.0), ('flip', 0.0), ('T', 0.0)] if len(sys.argv) <= 4 else []) + [(v_, s_) for v_ in (sys.argv[3:4] or ['T']) for s_ in [float(a) for a in (sys.argv[4] if len(sys.argv) > 4 else '-0.06,-0.03,0.03,0.06').split(',')]]
    for v_, s_ in cases:
        lut, f, zern = make_lut(dev, 1.0, variant=v_, focus_shift=s_)
        r = run(lut, idx); ok = (np.abs(r['th'][:, :2]) < 6).all(1) & (r['th'][:, 2] > ZLO + 0.005)
        key = f'{v_}_{s_:+.2f}'
        res[key] = dict(ll_sum_all=float(r['ll'].sum()), dev_median=float(np.median(r['dev'])), n_ok=int(ok.sum()),
                        z_p5_50_95=np.percentile(r['th'][ok, 2], [5, 50, 95]).round(3).tolist(), focus=f,
                        crlbz_median_nm=float(np.median(r['crlb'][ok, 2]) * 1000))
        print(key, res[key], flush=True)
    json.dump(res, open(WORK + 'astig_convtest.json', 'w'), indent=1)
elif mode == 'signtest':
    idx = np.argsort(-box)[:5000]
    res = {}
    for s_ in (1.0, -1.0):
        lut, f, zern = make_lut(dev, s_)
        r = run(lut, idx)
        ok = (np.abs(r['th'][:, :2]) < 6).all(1)
        res[s_] = dict(ll_sum=float(r['ll'][ok].sum()), dev_median=float(np.median(r['dev'][ok])),
                       z_p5_50_95=np.percentile(r['th'][ok, 2], [5, 50, 95]).round(3).tolist(), focus=f)
        print(s_, res[s_], flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(WORK + 'astig_signtest.json', 'w'), indent=1)
else:
    var = sys.argv[4] if len(sys.argv) > 4 else None; fs = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    lut, f, zern = make_lut(dev, sgn, variant=None if var == 'none' else var, focus_shift=fs)
    json.dump({'zernikes_waves': {str(k): v for k, v in zern.items()}, 'focus_um': f, 'variant': var, 'focus_shift': fs},
              open(WORK + 'psf_used.json', 'w'), indent=1)
    r = run(lut, np.arange(len(cyc)))
    np.savez(WORK + 'astig.npz', **r, corner=corner, cycle=cyc, boxstat=box, focus=f, astig_sign=sgn)
    print('done', len(cyc))
