"""Fit the coherent two-arm 4Pi PSF (simcode2/fourpi/psf4pi.py) to a bead z-stack, per channel (61 slices x 4 channels).

Data: out/4pi/smlm/zcal/zstack_rois_<cell>.npy ([61,4,21,21] e, registered channel ROIs around the bead).
Per slice: x, y (px), N (common to the 4 channels), bg per channel; z_k = z0 + s * zstage_k (s = +-1 tried).
Global: Zernikes of both arms (bottom initialised at minus the top), channel phases / transmissions, arm ratio,
coherence, bottom-arm offset (dx, dy, dz), blur; optional per-slice 4Pi phase drift as a smooth polynomial in slice
index (--drift_order). Loss: Poisson NLL with the sCMOS variance shift v (mu + v ~ Poisson).

Usage: python analysis/fourpi/fit_psf4pi.py --cell NPC [--mirror y] [--s -1] [--iters 1500] [--tag base]
Output: out/4pi/psf4pi/<cell>_<tag>.json (+ _state.pt) and a figure.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.psf4pi import PSF4Pi

OUT = f'{REPO}/out/4pi/psf4pi'


def run(a):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    dev = a.device
    D = torch.tensor(np.load(f'{REPO}/out/4pi/smlm/zcal/zstack_rois_{a.cell}.npy'), device=dev)   # [K,4,R,R]
    K, _, R, _ = D.shape
    zc = json.load(open(f'{REPO}/out/4pi/smlm/zcal/zcal_{a.cell}.json'))
    base = json.load(open(f'{REPO}/out/4pi/smlm/psf/{a.cell}_base.json'))['bead']
    zst = torch.tensor(zc['zstage'], device=dev, dtype=torch.float32)
    m = PSF4Pi(R, px=a.px, NA=a.NA, n=a.n, lam=a.lam, npupil=a.npupil, os_=2, device=dev, mirror=a.mirror)
    zt = torch.tensor(base['psf']['zcoef1'], device=dev)
    with torch.no_grad():
        m.top.zcoef.copy_(zt); m.bot.zcoef.copy_(-zt if a.bot_init == 'neg' else zt)
        m.logblur.fill_(math.log(0.1))
    h = R // 2
    x = torch.tensor(zc['x'], device=dev); y = torch.tensor(zc['y'], device=dev)   # GaussAstig x, y are relative to the ROI centre
    x = torch.nn.Parameter(x.float()); y = torch.nn.Parameter(y.float())
    z0 = torch.nn.Parameter(torch.tensor(float(base['z0']) if a.s == base['s'] else -float(base['z0']), device=dev))
    bg0 = D[..., [0, -1], :].median(-1).values.median(-1).values.clamp(min=0.05)            # [K,4]
    lbg = torch.nn.Parameter(torch.log(bg0))
    lN = torch.nn.Parameter(torch.log((D - bg0[..., None, None]).sum((1, 2, 3)).clamp(min=100)))
    dr = torch.nn.Parameter(torch.zeros(a.drift_order, device=dev))
    lzs = torch.nn.Parameter(torch.zeros((), device=dev))                 # log z scale (stage z -> model z)
    v = a.var
    with torch.no_grad():
        lzs.fill_(math.log(a.zscale0))
        ph = torch.tensor([0.0, 270.0, 180.0, 90.0] if a.s > 0 else [0.0, 90.0, 180.0, 270.0], device=dev)
        m.phi.copy_(torch.deg2rad(ph))
    tk = (torch.arange(K, device=dev) - (K - 1) / 2) / ((K - 1) / 2)
    tb = time.time()

    def model():
        pos = torch.stack([x, y, z0 + a.s * torch.exp(lzs) * zst], 1)
        pe = sum(dr[i] * tk ** (i + 1) for i in range(a.drift_order)) if a.drift_order else None
        return m(pos, phi_extra=pe, intensity=torch.exp(lN), bg=torch.exp(lbg))

    def loss():
        mu = model().clamp(min=1e-3)
        return ((mu + v) - (D + v).clamp(min=0) * torch.log(mu + v)).sum()

    # stage 0: grid over the global 4Pi phase offset (all other parameters at their start values)
    best = None
    with torch.no_grad():
        ph0 = m.phi.clone()
        for g0 in np.arange(0, 360, 15):
            m.phi.copy_(ph0 + math.radians(g0)); Lg = loss().item()
            if best is None or Lg < best[0]: best = (Lg, g0)
        m.phi.copy_(ph0 + math.radians(best[1]))
    print(f'phase grid: best offset {best[1]} deg, L {best[0]:.1f}', flush=True)

    zern = [m.top.zcoef, m.bot.zcoef]
    other = [m.phi, m.tau_logit, m.a_log, m.gamma_logit, m.logblur]
    stages = [('A', a.iters // 3, [dict(params=other, lr=a.lr), dict(params=[x, y, lbg, lN], lr=a.lr)]),
              ('B', a.iters - a.iters // 3,
               [dict(params=other, lr=a.lr / 3), dict(params=[x, y, lbg, lN], lr=a.lr / 3),
                dict(params=zern, lr=a.lr_z), dict(params=[m.d_bot, z0, dr] + ([lzs] if a.zscale else []), lr=a.lr / 10)])]
    it = 0
    for nm, n_it, groups in stages:
        opt = torch.optim.Adam(groups)
        for _ in range(n_it):
            L = loss()
            opt.zero_grad(); L.backward(); opt.step()
            with torch.no_grad():   # projection onto the physical box
                m.a_log.clamp_(math.log(1 / a.a_max), math.log(a.a_max))
                m.d_bot[:2].clamp_(-a.dxy_max, a.dxy_max); m.d_bot[2].clamp_(-a.dz_max, a.dz_max)
                lzs.clamp_(-0.3, 0.3)
            if it % 200 == 0:
                print(f'{nm} it {it} L {L.item():.1f} a {math.exp(m.a_log.item()):.3f} g {torch.sigmoid(m.gamma_logit).item():.3f} '
                      f'phi {np.degrees(m.phi.detach().cpu().numpy()).round(1)} blur {math.exp(m.logblur.item()):.3f} '
                      f'dbot {m.d_bot.detach().cpu().numpy().round(3)} z0 {z0.item():.3f} zs {math.exp(lzs.item()):.3f} t {time.time() - tb:.0f}s', flush=True)
            it += 1
    with torch.no_grad():
        mu = model().clamp(min=1e-3)
        Dp = (D + v).clamp(min=1e-6)
        dev_ = 2 * ((mu + v) - Dp + Dp * torch.log(Dp / (mu + v)))
        dev_slice = (dev_.sum((1, 2, 3)) / 2).cpu().numpy()
        dev_ch = (dev_.sum((0, 2, 3)) / 2).cpu().numpy()
    st = m.state()
    out = dict(args=vars(a), loss=float(L.item()), deviance_half=float(dev_slice.sum()), dev_slice=dev_slice.tolist(),
               dev_channel=dev_ch.tolist(), zscale=math.exp(lzs.item()), npix=int(D.numel()), z0=float(z0.item()), s=a.s, drift=dr.tolist(),
               x=x.tolist(), y=y.tolist(), N=torch.exp(lN).tolist(), bg=torch.exp(lbg).tolist(), psf=st)
    fn = f'{OUT}/{a.cell}_{a.tag}'
    json.dump(out, open(fn + '.json', 'w'))
    np.save(fn + '_mu.npy', mu.cpu().numpy())
    print(f'DONE {a.cell} {a.tag}: deviance/2 {dev_slice.sum():.0f} over {D.numel()} px ({dev_slice.sum() / D.numel():.3f}/px); per channel {dev_ch.round(0)}')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--tag', default='base')
    p.add_argument('--mirror', default='y'); p.add_argument('--s', type=float, default=-1.0)
    p.add_argument('--bot_init', default='neg'); p.add_argument('--drift_order', type=int, default=0)
    p.add_argument('--iters', type=int, default=1500); p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--px', type=float, default=0.128); p.add_argument('--NA', type=float, default=1.35)
    p.add_argument('--n', type=float, default=1.406); p.add_argument('--lam', type=float, default=0.60)
    p.add_argument('--npupil', type=int, default=48); p.add_argument('--var', type=float, default=1.3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--a_max', type=float, default=2.0); p.add_argument('--dxy_max', type=float, default=0.5)
    p.add_argument('--dz_max', type=float, default=0.15); p.add_argument('--zscale', type=int, default=1)
    p.add_argument('--zscale0', type=float, default=1.034); p.add_argument('--lr_z', type=float, default=0.002)
    run(p.parse_args())
