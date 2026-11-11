"""Stage-2 sim benchmark, classical side: isolated on-through emitters in 6-frame x 4-channel windows from the
bead-fitted 4Pi PSF (simcode2/fourpi/sim4pi.py), known pattern and known 4Pi phase field Phi0.

Estimators:
  smlm   : astigmatic MLE on the channel- and frame-summed image (incoherent part of the 4Pi PSF): x, y, z, N, bg
  ap3    : published-style 4Pi read-out: r = 3 px aperture photometry per channel (frames summed, at the smlm
           position), z = least squares of the 4 channel fractions against the model curve, searched within
           +-Lambda/2 of the smlm z (Phi0 known); x, y from smlm
  pix    : pixel MLE on all 24 sub-images (x, y, z, F, bg_c x 4) with the true pattern and Phi0, 9 z starts
           (the classical optimum)
CRLBs: pixel Fisher of the 24-image model (pix) and of the summed image (smlm), at the truth.
Output: out/4pi/sim/bench_classical.json + table.
"""
import json, math, os, sys, time
import numpy as np, torch
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi, DELTA_A, K_PAT
from simcode2.fourpi.lm import lm_poisson, fisher, crlb, lm_poisson_j, fisher_j

OUT = f'{REPO}/out/4pi/sim'
V = 1.14 ** 2 + 0.2 ** 2


def main(n=6000, seed=11, dev='cuda'):
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    sim = Sim4Pi(lut, dev=dev, seed=seed)
    S = sim.sample(n, p_nb=0.0, p_on=1.0)
    img = S['img']                                                     # [n,6,4,P,P]
    P = 15; ang = torch.tensor([0, 0, 0, 1, 1, 1], device=dev)
    doff = S['doff']; m = S['m']; eps = S['eps']; Phi0 = S['Phi0']

    def mu_pix(th):     # th: x, y, z, F, bg0..3
        x, y, z, F = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
        pat = sim.pattern(x, y, m, eps)                                # [n,6]
        out = []
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            g = torch.stack([lut.evaluate_ch(pc, Phi0 + DELTA_A[a], c) for a in range(2)], 1)[:, ang]      # [n,6,P,P]
            out.append(F[:, None, None, None] * pat[:, :, None, None] * g + th[:, 4 + c, None, None, None])
        return torch.stack(out, 2).reshape(len(th), -1)

    kv = torch.tensor(K_PAT, device=dev)

    def mu_pix_j(th):   # analytic Jacobian: x, y, z, F, bg0..3
        x, y, z, F = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
        nn_ = len(th)
        pat = sim.pattern(x, y, m, eps)                                # [n,6]
        dpx = torch.zeros_like(pat); dpy = torch.zeros_like(pat)
        for a in range(2):
            r = x if a == 0 else y
            for s_ in range(3):
                dv = -m * K_PAT[a] * torch.sin(K_PAT[a] * r - 2 * math.pi * s_ / 3 - eps[:, a])
                (dpx if a == 0 else dpy)[:, 3 * a + s_] = dv
        mus, Js = [], []
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            G = [lut.evaluate_ch_grad(pc, Phi0 + DELTA_A[a], c) for a in range(2)]
            g, gx, gy, gz = (torch.stack([G[0][k], G[1][k]], 1)[:, ang] for k in range(4))     # [n,6,P,P]
            Fp = F[:, None, None, None]
            mu_c = Fp * pat[:, :, None, None] * g + th[:, 4 + c, None, None, None]
            jx = Fp * (dpx[:, :, None, None] * g + pat[:, :, None, None] * gx)
            jy = Fp * (dpy[:, :, None, None] * g + pat[:, :, None, None] * gy)
            jz = Fp * pat[:, :, None, None] * gz
            jF = pat[:, :, None, None] * g
            jb = [torch.full_like(g, 1.0 if cc == c else 0.0) for cc in range(4)]
            mus.append(mu_c); Js.append(torch.stack([jx, jy, jz, jF] + jb, -1))
        mu = torch.stack(mus, 2).reshape(nn_, -1)
        J = torch.stack(Js, 2).reshape(nn_, -1, 8)
        return mu, J

    def inc_sum(x, y, z):
        tot = 0
        for c in range(4):
            tot = tot + lut.luts[0][c].evaluate(torch.stack([x + doff[:len(x), c, 0], y + doff[:len(x), c, 1], z], 1))
        return tot

    dsum = img.sum((1, 2)).reshape(n, -1)

    def mu_sum(th):     # x, y, z, N, bg (per summed pixel)
        g = inc_sum(th[:, 0], th[:, 1], th[:, 2])
        return (th[:, 3, None, None] * g + th[:, 4, None, None]).reshape(len(th), -1)

    t0 = time.time()
    # --- smlm: z starts
    bg0 = img.sum((1, 2))[:, [0, -1]].median(-1).values.median(-1).values
    N0 = (dsum - bg0[:, None]).sum(-1).clamp(min=50)
    best = None
    for zs in (-0.3, 0.0, 0.3):
        th0 = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.full((n,), zs, device=dev), N0, bg0], 1)
        th, ll, _ = lm_poisson(mu_sum, dsum, th0, 24 * V, eps=[1e-3, 1e-3, 1e-3, 1.0, 1e-2],
                               lo=torch.tensor([-3, -3, -0.6, 1, 0.0], device=dev), hi=torch.tensor([3, 3, 0.6, 1e6, 1e3], device=dev))
        if best is None:
            best = (th, ll)
        else:
            b = ll > best[1]; best = (torch.where(b[:, None], th, best[0]), torch.where(b, ll, best[1]))
    th_s = best[0]
    print(f'smlm done {time.time() - t0:.0f}s', flush=True)
    # --- ap3
    yy, xx = sim.oy, sim.ox
    dfr = img.sum(1) - img.sum(1)[..., [0, -1], :].median(-1).values.median(-1).values[..., None, None]   # [n,4,P,P]
    ap = (((xx[None] - th_s[:, 0, None, None]) ** 2 + (yy[None] - th_s[:, 1, None, None]) ** 2) <= 9).float()
    A = (dfr * ap[:, None]).sum((-1, -2)); fa = A / A.sum(1, keepdim=True)
    zg = torch.arange(-0.6, 0.6001, 0.004, device=dev)
    Lam = 0.246
    best_z = torch.zeros(n, device=dev); best_e = torch.full((n,), 1e9, device=dev)
    for zv in zg:
        sel = (zv - th_s[:, 2]).abs() <= Lam / 2
        if not sel.any(): continue
        zz = torch.full((n,), float(zv), device=dev)
        fr = 0
        mp = torch.zeros(n, 4, device=dev)
        for a in range(2):
            for c in range(4):
                g = lut.evaluate_ch(torch.stack([th_s[:, 0] + doff[:, c, 0], th_s[:, 1] + doff[:, c, 1], zz], 1), Phi0 + DELTA_A[a], c)
                mp[:, c] += (g * ap).sum((-1, -2))
        mp = mp / mp.sum(1, keepdim=True)
        e = ((fa - mp) ** 2).sum(1)
        upd = sel & (e < best_e)
        best_z = torch.where(upd, zz, best_z); best_e = torch.where(upd, e, best_e)
    z_ap = best_z
    print(f'ap3 done {time.time() - t0:.0f}s', flush=True)
    # --- pixel MLE, 9 z starts
    d = img.reshape(n, -1)
    bgc = img[..., [0, -1], :].median(-1).values.median(-1).values.median(1).values   # [n,4]
    F0 = (img.sum((2, 3, 4)) - (bgc.sum(1) * P * P)[:, None]).mean(1).clamp(min=20)
    best = None
    lo = torch.tensor([-3, -3, -0.6, 1, 0, 0, 0, 0], device=dev, dtype=torch.float32)
    hi = torch.tensor([3, 3, 0.6, 1e5, 50, 50, 50, 50], device=dev, dtype=torch.float32)
    for dzs in (-0.123, 0.0, 0.123):    # fringe starts around the smlm z (spacing Lambda / 2)
        th0 = torch.cat([th_s[:, :2], (th_s[:, 2:3] + dzs).clamp(-0.55, 0.55), F0[:, None], bgc], 1)
        th, ll, _ = lm_poisson_j(mu_pix_j, d, th0, V, lo=lo, hi=hi, iters=25)
        if best is None:
            best = (th, ll)
        else:
            b = ll > best[1]; best = (torch.where(b[:, None], th, best[0]), torch.where(b, ll, best[1]))
    th_p = best[0]
    print(f'pix done {time.time() - t0:.0f}s', flush=True)
    # --- CRLBs at the truth
    tt = torch.cat([torch.stack([S['x'], S['y'], S['z'], S['F']], 1), S['bg'][:, 0, :, 0, 0]], 1)
    cr_p = crlb(fisher_j(mu_pix_j, tt, V))
    ts = torch.stack([S['x'], S['y'], S['z'], S['Nt'].sum(1) * 0 + 0, S['bg'][:, 0].sum((1, 2, 3)) / (P * P)], 1)
    # summed model: N = total photons in the window with the 4-channel PSF sum normalised per frame: N = sum_t Nt
    ts[:, 3] = S['Nt'].sum(1)
    cr_s = torch.cat([crlb(fisher(mu_sum, ts[i:i + 1000], 24 * V, [1e-3, 1e-3, 1e-3, 1.0, 1e-2])) for i in range(0, n, 1000)])
    Nw = S['Nt'].sum(1)
    res = dict(n=n, bins=[], note='window photons = sum over 6 frames of the main emitter (4 channels)')
    edges = [300, 1000, 2000, 4000, 8000, 16000]
    tr = torch.stack([S['x'], S['y'], S['z']], 1)
    L = ['| window photons | n | smlm x / z RMS (CRLB) | ap3 z RMS | pix x / y / z RMS (CRLB) | pix z fringe err | pix z RMS/CRLB |', '|---|---|---|---|---|---|---|']
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        s = (Nw >= lo_) & (Nw < hi_)
        if s.sum() < 50: continue
        rms = lambda e: float(torch.sqrt((e[s] ** 2).mean()))
        rob = lambda e: float(1.4826 * (e[s] - e[s].median()).abs().median())
        ex_s = (th_s[:, 0] - tr[:, 0]) * 128; ez_s = (th_s[:, 2] - tr[:, 2]) * 1000
        ez_a = (z_ap - tr[:, 2]) * 1000
        ex_p = (th_p[:, 0] - tr[:, 0]) * 128; ey_p = (th_p[:, 1] - tr[:, 1]) * 128; ez_p = (th_p[:, 2] - tr[:, 2]) * 1000
        fr = float(((ez_p[s]).abs() > 100).float().mean())
        good = s & (ez_p.abs() <= 100)
        rz = float(torch.sqrt((ez_p[good] ** 2).mean()) / (cr_p[good, 2] * 1000).pow(2).mean().sqrt())
        b = dict(lo=lo_, hi=hi_, n=int(s.sum()), smlm_x=rms(ex_s), smlm_z=rms(ez_s), smlm_x_crlb=float((cr_s[s, 0] * 128).pow(2).mean().sqrt()),
                 smlm_z_crlb=float((cr_s[s, 2] * 1000).pow(2).mean().sqrt()), ap3_z=rms(ez_a), ap3_z_robust=rob(ez_a),
                 ap3_fringe=float((ez_a[s].abs() > 100).float().mean()),
                 pix_x=rms(ex_p), pix_y=rms(ey_p), pix_z=float(torch.sqrt((ez_p[good] ** 2).mean())), pix_fringe=fr,
                 pix_x_crlb=float((cr_p[s, 0] * 128).pow(2).mean().sqrt()), pix_y_crlb=float((cr_p[s, 1] * 128).pow(2).mean().sqrt()),
                 pix_z_crlb=float((cr_p[s, 2] * 1000).pow(2).mean().sqrt()), pix_z_rms_over_crlb=rz)
        res['bins'].append(b)
        L.append(f"| {lo_}-{hi_} | {b['n']} | {b['smlm_x']:.1f} / {b['smlm_z']:.1f} ({b['smlm_x_crlb']:.1f} / {b['smlm_z_crlb']:.1f}) | "
                 f"{b['ap3_z']:.1f} (robust {b['ap3_z_robust']:.1f}, fringe {b['ap3_fringe']:.3f}) | {b['pix_x']:.2f} / {b['pix_y']:.2f} / {b['pix_z']:.2f} "
                 f"({b['pix_x_crlb']:.2f} / {b['pix_y_crlb']:.2f} / {b['pix_z_crlb']:.2f}) | {fr:.3f} | {rz:.2f} |")
    res['table'] = L
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f'{OUT}/bench_classical.json', 'w'), indent=1)
    torch.save(dict(S={k: v for k, v in S.items() if torch.is_tensor(v) and k not in ('mu', 'bg', 'img')}, th_s=th_s, z_ap=z_ap, th_p=th_p,
                    cr_p=cr_p, cr_s=cr_s, seed=seed, n=n), f'{OUT}/bench_classical.pt')
    print('\n'.join(L))
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
