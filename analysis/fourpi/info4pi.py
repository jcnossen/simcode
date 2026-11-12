"""Stage-2 information check: does the 24-intensity representation (6 frames x 4 channels) lose information against the
pixel-level CRLB of a 6-frame 4Pi-SIMFLUX window?

PSF: the bead-fitted coherent 4Pi model (out/4pi/psf4pi/NPC_f_dxy15.json), ROI 15 px. Window: 6 frames (2 angles x 3
steps, k = 3.84 rad/px along x / y, depth m), N photons per window, bg per pixel per frame per channel, read-noise
variance v. Fisher information J = sum (d mu)(d mu)^T / (mu + v), parameters (x, y, z, N, bg_c x 4).

Representations:
  pix      : all 24 x 225 pixels, Phi0 known (the true bound for a known 4Pi phase field)
  pix_u    : pixels, Phi0 unknown per emitter (extra nuisance)    -> what per-emitter z needs without a field map
  sum_pix  : channel-summed, frame-summed image (astigmatic shape only; the pattern-free z prior)
  int      : 24 linear read-outs I_tc = sum_q w_cq (y - bg), w_c = matched filter of channel c (fixed at the truth),
             variance sum w^2 (mu + v); parameters (x, y, z, N) (bg known for the read-out)
  int+sum  : int + sum_pix (additive approximation: read-out amplitude and shape information treated as independent)
Output: out/4pi/sim/info4pi.json + table on stdout.
"""
import json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.psf4pi import PSF4Pi

OUT = f'{REPO}/out/4pi/sim'


def load_psf(tag='NPC_f_dxy15', R=15, dev='cuda'):
    s = json.load(open(f'{REPO}/out/4pi/psf4pi/{tag}.json'))['psf']
    s['meta']['R'] = R
    return PSF4Pi.from_state(s, device=dev).eval()


def pattern(x, y, m=0.9, k=3.84):
    """[B,6] per-frame excitation fractions (sum 1 over the window when m terms average out)"""
    out = []
    for a in range(2):
        r = x if a == 0 else y
        for s in range(3):
            out.append((1 + m * torch.cos(k * r - 2 * math.pi * s / 3)) / 6)
    return torch.stack(out, -1)


def fisher(J_rows, var):
    """J_rows [B, M, P] derivatives, var [B, M] -> [B, P, P]"""
    return torch.einsum('bmp,bm,bmq->bpq', J_rows, 1 / var, J_rows)


def crlb(F, idx):
    return torch.sqrt(torch.diagonal(torch.linalg.inv(F), dim1=-2, dim2=-1)[:, idx])


def run(N=2700.0, bg=0.8, v=1.3, m=0.9, zs=np.arange(-0.4, 0.401, 0.05), nxy=8, seed=0, dev='cuda'):
    psf = load_psf(dev=dev)
    g = torch.Generator().manual_seed(seed)
    Z = torch.tensor(np.repeat(zs, nxy), dtype=torch.float32)
    B = len(Z)
    X = (torch.rand(B, generator=g) - 0.5); Y = (torch.rand(B, generator=g) - 0.5)
    th = torch.stack([X, Y, Z, torch.full((B,), N), torch.full((B,), bg)[:, None].expand(B, 4)[:, 0],
                      torch.zeros(B)], 1).to(dev)     # x, y, z, N, bg, phi0

    def model(t):
        x, y, z, n, b, p0 = t.unbind(1)
        G = psf(torch.stack([x, y, z], 1), phi_extra=p0)              # [B,4,R,R]
        P = pattern(x, y, m)                                           # [B,6]
        mu = n[:, None, None, None, None] * P[:, :, None, None, None] * G[:, None] + b[:, None, None, None, None]
        return mu, G, P                                                # mu [B,6,4,R,R]

    with torch.no_grad():
        mu, G, P = model(th)
    Bn = mu.shape[0]
    # Jacobian by central differences (6 parameters); per-channel bg derivatives are indicators
    eps = torch.tensor([1e-3, 1e-3, 1e-4, 1e-1, 1e-3, 1e-3], device=dev)
    with torch.no_grad():
        mu0 = mu.detach()
        Jp = []
        for p in range(6):
            tp = th.detach().clone(); tm = th.detach().clone()
            tp[:, p] += eps[p]; tm[:, p] -= eps[p]
            Jp.append((model(tp)[0] - model(tm)[0]) / (2 * eps[p]))
        Jp = torch.stack(Jp, -1)                                     # [B,6,4,R,R,6]
        var = mu0 + v
        # replace the global bg derivative by per-channel bg derivatives
        Jbg = torch.zeros(*mu0.shape, 4, device=dev)
        for c in range(4):
            Jbg[:, :, c, ..., c] = 1.0
        Jfull = torch.cat([Jp[..., :4], Jbg, Jp[..., 5:6]], -1)     # x y z N bg0..3 phi0  -> 9
        rows = Jfull.reshape(Bn, -1, 9); vr = var.reshape(Bn, -1)
        F = fisher(rows, vr)
        res = {}
        Fk = F[:, :8, :8]
        res['pix'] = crlb(Fk, [0, 1, 2])
        res['pix_u'] = crlb(F, [0, 1, 2])
        # summed image (all frames, all channels): mu_s = N g_sum + 24 bg
        mus = mu0.sum((1, 2)); Js = Jp.sum((1, 2))                    # [B,R,R,6]
        Js_full = torch.stack([Js[..., 0], Js[..., 1], Js[..., 2], Js[..., 3], torch.full_like(Js[..., 0], 24.0)], -1)
        Fs = fisher(Js_full.reshape(Bn, -1, 5), (mus + 24 * v).reshape(Bn, -1))
        res['sum_pix'] = crlb(Fs, [0, 1, 2])
        # intensity read-outs with matched-filter weights per channel (at the truth, fixed)
        gs = G.detach()                                               # [B,4,R,R]
        w = gs / (gs.sum((2, 3), keepdim=True) * (mu0.mean(1) + v))  # matched filter ~ g / var
        w = w / (w * gs).sum((2, 3), keepdim=True)                    # unbiased for the channel amplitude
        Im_J = torch.einsum('bcuv,btcuvp->btcp', w, Jp[..., :4])       # d E[I_tc] / d (x, y, z, N)
        Ivar = torch.einsum('bcuv,btcuv->btc', w * w, var)
        Fi = fisher(Im_J.reshape(Bn, 24, 4), Ivar.reshape(Bn, 24))
        res['int'] = crlb(Fi, [0, 1, 2])
        Fs_marg = torch.linalg.inv(torch.linalg.inv(Fs)[:, :4, :4])  # bg marginalised
        res['int+sum'] = crlb(Fi + Fs_marg, [0, 1, 2])

        def readout_F(ws):
            # ws: list of weight maps [B,4,R,R]; read-outs per (t, c, r); covariance between read-outs of one (t, c)
            W = torch.stack(ws, 2)                                                 # [B,4,nr,R,R]
            Jm = torch.einsum('bcruv,btcuvp->btcrp', W, Jp[..., :4])               # [B,6,4,nr,4]
            Cv = torch.einsum('bcruv,btcuv,bcsuv->btcrs', W, var, W)               # [B,6,4,nr,nr]
            Ci = torch.linalg.inv(Cv)
            return torch.einsum('btcrp,btcrs,btcsq->bpq', Jm, Ci, Jm)
        # classical aperture read-out (r = 3 px disc, the published-style per-channel photometry)
        yy, xx = torch.meshgrid(torch.arange(15, device=dev) - 7.0, torch.arange(15, device=dev) - 7.0, indexing='ij')
        ap = ((xx[None] - th[:, 0, None, None]) ** 2 + (yy[None] - th[:, 1, None, None]) ** 2 <= 9.0).float()
        Fa = readout_F([ap[:, None].expand(-1, 4, -1, -1)])
        res['ap3+sum'] = crlb(Fa + Fs_marg, [0, 1, 2])
        # two read-outs per channel: matched filter + the Phi-derivative filter (d g_c / d Phi0 / var)
        dphi = Jp[..., 5].mean(1) / (mu0.mean(1) + v)                              # [B,4,R,R]
        F2 = readout_F([w, dphi])
        res['int2+sum'] = crlb(F2 + Fs_marg, [0, 1, 2])
        # two read-outs: matched filter + a centre disc (r <= rc px) -- a GT-definable second target
        for rc in (1.0, 1.5, 2.0):
            cd = ((xx[None] - th[:, 0, None, None]) ** 2 + (yy[None] - th[:, 1, None, None]) ** 2 <= rc * rc).float()
            Fc = readout_F([w, cd[:, None].expand(-1, 4, -1, -1)])
            res[f'int+c{rc:g}+sum'] = crlb(Fc + Fs_marg, [0, 1, 2])
        Fc = readout_F([ap[:, None].expand(-1, 4, -1, -1), ((xx[None] - th[:, 0, None, None]) ** 2 + (yy[None] - th[:, 1, None, None]) ** 2 <= 2.25).float()[:, None].expand(-1, 4, -1, -1)])
        res['ap3+c1.5+sum'] = crlb(Fc + Fs_marg, [0, 1, 2])
        # three GT-definable read-outs: matched filter + Gaussian centre (sd 1, the current C) + Gaussian annulus
        r2c = (xx[None] - th[:, 0, None, None]) ** 2 + (yy[None] - th[:, 1, None, None]) ** 2
        gc = torch.exp(-r2c / 2.0)[:, None].expand(-1, 4, -1, -1)
        for r0 in (1.5, 2.0, 2.5):
            ga = torch.exp(-(r2c.sqrt() - r0) ** 2 / (2 * 0.5 ** 2))[:, None].expand(-1, 4, -1, -1)
            res[f'int+c+ann{r0:g}+sum'] = crlb(readout_F([w, gc, ga]) + Fs_marg, [0, 1, 2])
        res['int+c+sum (gauss sd 1)'] = crlb(readout_F([w, gc]) + Fs_marg, [0, 1, 2])
        # the intensity representation with Phi0 unknown needs the field map; also report SMLM-only lateral (no pattern)
    px_nm = 128.0
    tab = {}
    for k, r in res.items():
        r = r.cpu().numpy().reshape(len(zs), nxy, 3)
        tab[k] = dict(x_nm=(np.median(r[..., 0], 1) * px_nm).tolist(), y_nm=(np.median(r[..., 1], 1) * px_nm).tolist(),
                      z_nm=(np.median(r[..., 2], 1) * 1000).tolist())
    return dict(N=N, bg=bg, v=v, m=m, z_um=list(map(float, zs)), crlb=tab)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    allr = []
    for N in (1500.0, 2700.0, 6000.0):
        r = run(N=N)
        allr.append(r)
        print(f'\nN = {N:.0f} e per window, bg 0.8 e/px/frame/ch, m 0.9; CRLB (nm, 128 nm px), median over x,y; z (um) columns')
        print('rep      ' + ' '.join(f'{z:+.2f}' for z in r['z_um'][::2]))
        for k, t in r['crlb'].items():
            print(f'{k:8s} z ' + ' '.join(f'{v:5.1f}' for v in t['z_nm'][::2]))
            print(f'{"":8s} x ' + ' '.join(f'{v:5.1f}' for v in t['x_nm'][::2]))
    json.dump(allr, open(f'{OUT}/info4pi.json', 'w'), indent=1)
