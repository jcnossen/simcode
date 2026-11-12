"""Evaluation of a simulated 4Pi-SIMFLUX movie against ground truth: network pipeline vs classical, renders.

GT unit = one binding event within one 6-frame window (photons summed over the window, position = the site).
Usable region: candidates need a full 15 px patch, so GT and localizations are restricted to x, y in [8, 42) px.
Matching: per window, greedy nearest within 0.5 px (64 nm) laterally; recall = matched GT / GT (by GT window photons),
precision = matched / localizations. Errors x, y (nm, after removing the global mean offset, which is reported), z;
fringe error = |dz| > 100 nm. CRLB per matched GT unit: pixel Fisher information of an isolated emitter with its true
per-frame photons (kinetics and pattern included), parameters x, y, z, photon scale, bg x 4.
Classical on the same detections: (a) summed SMLM: MLE of the registered 4-channel 6-frame sum with the incoherent part
of the 4Pi PSF (x, y, z, N, bg); (b) published-style 4Pi z: r = 3 px aperture photometry per channel (frames summed,
at the SMLM position mapped into each channel), z = least squares of the channel fractions against the model curve
within +-Lambda/2 of the SMLM z, given the TRUE 4Pi phase field (an advantage the real method does not have).
Usage: CUDA_VISIBLE_DEVICES=2 python analysis/fourpi/simeval4pi.py --tag d1 --pattern iter
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
from simmovie4pi import reg_map, phi0_field, GAIN, OFF, PX
from simcode.fourpi.lut4pi import LUT4Pi
from simcode.fourpi.lm import lm_poisson_j, fisher_j, crlb

LO, HI = 8.0, 42.0
EDGES = (300, 1000, 2000, 4000, 8000, 30000)


def gt_units(d, T):
    g = np.load(f'{d}/gt_frames.npz')
    w = g['frame'] // 6
    key = g['event'].astype(np.int64) * 10000 + w
    u, inv = np.unique(key, return_inverse=True)
    N = np.bincount(inv, g['N'])
    first = np.zeros(len(u), int); first[inv[::-1]] = np.arange(len(inv))[::-1]
    U = dict(w=u % 10000, event=u // 10000, N=N, x=g['x'][first], y=g['y'][first], z=g['z'][first])
    # per-frame photons [n,6] for the CRLB
    Nt = np.zeros((len(u), 6)); np.add.at(Nt, (inv, g['frame'] % 6), g['N'])
    U['Nt'] = Nt
    return U


def match(wg, xg, yg, wl, xl, yl, r=0.5):
    mg = np.full(len(wg), -1); ml = np.full(len(wl), -1)
    og = np.argsort(wg); ol = np.argsort(wl)
    sg = np.searchsorted(wg[og], np.arange(wg.max() + 2)); sl = np.searchsorted(wl[ol], np.arange(wg.max() + 2))
    for w in range(wg.max() + 1):
        a = og[sg[w]:sg[w + 1]]; b = ol[sl[w]:sl[w + 1]] if w + 1 < len(sl) else ol[sl[w]:]
        if len(a) == 0 or len(b) == 0: continue
        D = np.hypot(xg[a][:, None] - xl[b][None], yg[a][:, None] - yl[b][None])
        while True:
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] > r: break
            mg[a[i]] = b[j]; ml[b[j]] = a[i]; D[i, :] = 9e9; D[:, j] = 9e9
    return mg, ml


def classical(d, meta, F, lut, dev):
    """summed SMLM MLE on roi_sum + published-style aperture z (oracle phase field)"""
    H, W = meta['H'], meta['W']
    roi = torch.tensor(F['roi_sum'], device=dev)                        # [n,15,15], centred on (cx, cy)
    n = len(roi)
    v = 24 * 1.3

    def inc(x, y, z):
        g = 0
        for c in range(4):
            g = g + lut.luts[0][c].evaluate(torch.stack([x, y, z], 1))
        return g

    def fj(th):
        x, y, z, N, b = th.unbind(1)
        out = [L.evaluate(torch.stack([x, y, z], 1), True) for L in [lut.luts[0][c] for c in range(4)]]
        g = sum(o[0] for o in out); gx = sum(o[1] for o in out); gy = sum(o[2] for o in out); gz = sum(o[3] for o in out)
        mu = N[:, None, None] * g + b[:, None, None]
        J = torch.stack([N[:, None, None] * gx, N[:, None, None] * gy, N[:, None, None] * gz, g, torch.ones_like(g)], -1)
        return mu.reshape(len(th), -1), J.reshape(len(th), -1, 5)

    dd = roi.reshape(n, -1)
    bg0 = roi[:, [0, -1]].median(-1).values.median(-1).values
    N0 = (dd - bg0[:, None]).sum(1).clamp(min=50)
    best = None
    lo = torch.tensor([-3, -3, -0.6, 1, 0], device=dev, dtype=torch.float32); hi = torch.tensor([3, 3, 0.6, 1e6, 1e3], device=dev, dtype=torch.float32)
    for zs in (-0.3, 0.0, 0.3):
        th0 = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.full((n,), zs, device=dev), N0, bg0], 1)
        th, ll, _ = lm_poisson_j(fj, dd, th0, v, lo=lo, hi=hi, iters=30)
        if best is None: best = (th, ll)
        else:
            b = ll > best[1]; best = (torch.where(b[:, None], th, best[0]), torch.where(b, ll, best[1]))
    th_s = best[0]
    cr = crlb(fisher_j(fj, th_s, v))
    mu_f, _ = fj(th_s)
    chi2px = (((dd - mu_f) ** 2) / (mu_f.clamp(min=1e-6) + v)).mean(1)              # Pearson chi2 per pixel
    # convergence: 10 more LM iterations from the solution must not move it (|dx|, |dy| < 0.01 px, |dz| < 2 nm)
    th_c, _, _ = lm_poisson_j(fj, dd, th_s, v, lo=lo, hi=hi, iters=10)
    conv = ((th_c[:, :2] - th_s[:, :2]).abs().max(1).values < 0.01) & ((th_c[:, 2] - th_s[:, 2]).abs() < 0.002)
    at_bound = (th_s[:, :2].abs() >= 2.95).any(1) | (th_s[:, 2].abs() >= 0.595) | (th_s[:, 3] <= 1.5) | (th_s[:, 4] <= 1e-3) | (th_s[:, 4] >= 999)
    # aperture photometry per channel from the movie (frames summed), at the SMLM position mapped into each channel
    raw = np.load(f'{d}/movie.npy', mmap_mode='r'); Wd = W
    xs = F['cx'] + th_s[:, 0].cpu().numpy(); ys = F['cy'] + th_s[:, 1].cpu().numpy(); w = F['w']
    A = np.zeros((n, 4))
    yy, xx = np.mgrid[0:H, 0:W]
    order = np.argsort(w)
    for wv in np.unique(w):
        ii = order[np.searchsorted(w[order], wv):np.searchsorted(w[order], wv, 'right')]
        r = np.asarray(raw[6 * wv:6 * wv + 6]).astype(np.float32)
        cr_ = np.stack([r[:, :, 0:Wd], r[:, :, Wd:2 * Wd], r[:, :, 2 * Wd:3 * Wd][:, :, ::-1], r[:, :, 3 * Wd:][:, :, ::-1]], 1)
        Es = ((cr_ - OFF) * GAIN).sum(0)                                    # [4,H,W]
        for c in range(4):
            mx, my = reg_map(c, xs[ii], ys[ii], H, W)
            bgc = np.median(Es[c])
            for q, (u, v_) in enumerate(zip(mx, my)):
                m_ = (xx - u) ** 2 + (yy - v_) ** 2 <= 9.0
                A[ii[q], c] = (Es[c][m_] - bgc).sum()
    fa = A / np.maximum(A.sum(1, keepdims=True), 1e-6)
    Phi0 = phi0_field(xs, ys, 6 * w + 3, H, W, meta['T'])
    fa_t = torch.tensor(fa, device=dev, dtype=torch.float32)
    zg = torch.arange(-0.6, 0.6001, 0.004, device=dev)
    o = torch.arange(15, device=dev, dtype=torch.float32) - 7
    oy, ox = torch.meshgrid(o, o, indexing='ij')
    best_z = th_s[:, 2].clone(); best_e = torch.full((n,), 1e9, device=dev)
    Ph = torch.tensor(Phi0, device=dev, dtype=torch.float32)
    delta = math.radians(meta['delta_deg'])
    fx = torch.tensor(xs - np.round(xs), device=dev, dtype=torch.float32); fy = torch.tensor(ys - np.round(ys), device=dev, dtype=torch.float32)
    ap = (((ox[None] - fx[:, None, None]) ** 2 + (oy[None] - fy[:, None, None]) ** 2) <= 9).float()
    for zv in zg:
        sel = (zv - th_s[:, 2]).abs() <= 0.123
        if not sel.any(): continue
        zz = torch.full((n,), float(zv), device=dev)
        mp = torch.zeros(n, 4, device=dev)
        for aa, dl in ((0, 0.0), (1, delta)):
            for c in range(4):
                g = lut.evaluate_ch(torch.stack([fx, fy, zz], 1), Ph + dl, c)
                mp[:, c] += (g * ap).sum((-1, -2))
        mp = mp / mp.sum(1, keepdim=True)
        e = ((fa_t - mp) ** 2).sum(1)
        upd = sel & (e < best_e)
        best_z = torch.where(upd, zz, best_z); best_e = torch.where(upd, e, best_e)
    ap_edge = ((best_z - th_s[:, 2]).abs() >= 0.12) | (best_z.abs() >= 0.595)          # aperture z at its search limit
    return dict(chi2px=chi2px.cpu().numpy(), conv=conv.cpu().numpy(), at_bound=at_bound.cpu().numpy(), ap_edge=ap_edge.cpu().numpy(),
                x=xs, y=ys, z=th_s[:, 2].cpu().numpy(), z_ap=best_z.cpu().numpy(), sx=cr[:, 0].cpu().numpy(), sy=cr[:, 1].cpu().numpy(),
                sz=cr[:, 2].cpu().numpy(), N=th_s[:, 3].cpu().numpy(), w=w)


def crlb_units(U, idx, meta, lut, dev):
    """pixel CRLB (x, y, z) of isolated emitters with their true per-frame photons; bg = the simulated mean"""
    from simmovie4pi import BG
    n = len(idx); x0 = torch.tensor(U['x'][idx] - np.round(U['x'][idx]), device=dev, dtype=torch.float32)
    y0 = torch.tensor(U['y'][idx] - np.round(U['y'][idx]), device=dev, dtype=torch.float32)
    Nt = torch.tensor(U['Nt'][idx], device=dev, dtype=torch.float32)
    Phi = torch.tensor(phi0_field(U['x'][idx], U['y'][idx], 6 * U['w'][idx] + 3, meta['H'], meta['W'], meta['T']), device=dev, dtype=torch.float32)
    k = meta['k']; psi = np.radians(meta['psi_deg']); m = meta['m']; delta = math.radians(meta['delta_deg'])
    xa, ya = torch.tensor(U['x'][idx], device=dev, dtype=torch.float32), torch.tensor(U['y'][idx], device=dev, dtype=torch.float32)
    ang = [0, 0, 0, 1, 1, 1]
    # pattern factor and its derivative at the truth: N_t = F q_t P_t -> dN_t/dx = N_t dP/dx / P
    dlogP = []
    for t in range(6):
        a = ang[t]; arg = k[a][0] * xa + k[a][1] * ya - psi[a][t % 3]
        P = 1 + m * torch.cos(arg)
        dlogP.append(torch.stack([-m * k[a][0] * torch.sin(arg) / P, -m * k[a][1] * torch.sin(arg) / P], -1))
    dlogP = torch.stack(dlogP, 1)                                        # [n,6,2]
    bg = torch.tensor(BG, device=dev)

    def fj(th, sl):
        Phi_, Nt_, dlogP_ = Phi[sl], Nt[sl], dlogP[sl]
        x, y, z, s = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
        mus, Js = [], []
        for c in range(4):
            G = [lut.evaluate_ch_grad(torch.stack([x, y, z], 1), Phi_ + (delta if aa else 0.0), c) for aa in range(2)]
            mu_t, J_t = [], []
            for t in range(6):
                g, gx, gy, gz, _ = G[ang[t]]
                Ns = (s * Nt_[:, t])[:, None, None]
                mu_t.append(Ns * g + th[:, 4 + c, None, None])
                jb = [torch.full_like(g, 1.0 if cc == c else 0.0) for cc in range(4)]
                J_t.append(torch.stack([Ns * (gx + dlogP_[:, t, 0, None, None] * g), Ns * (gy + dlogP_[:, t, 1, None, None] * g), Ns * gz,
                                        Nt_[:, t, None, None] * g] + jb, -1))
            mus.append(torch.stack(mu_t, 1)); Js.append(torch.stack(J_t, 1))
        return torch.stack(mus, 2).reshape(len(th), -1), torch.stack(Js, 2).reshape(len(th), -1, 8)

    out = []
    for i in range(0, n, 2000):
        sl = slice(i, i + 2000)
        th = torch.cat([torch.stack([x0[sl], y0[sl], torch.tensor(U['z'][idx][sl], device=dev, dtype=torch.float32),
                                     torch.ones_like(x0[sl])], 1), bg[None].expand(len(x0[sl]), 4)], 1)
        out.append(crlb(fisher_j(lambda t_: fj(t_, sl), th, 1.14 ** 2))[:, :3])
    return torch.cat(out).cpu().numpy()


def render_xy(x, y, z, ax, title, zr, sr=6.0):
    from matplotlib.colors import hsv_to_rgb
    from scipy.ndimage import gaussian_filter
    W = int((HI - LO) * PX * 1000 / sr)
    ix = np.clip(((x - LO) * PX * 1000 / sr).astype(int), 0, W - 1); iy = np.clip(((y - LO) * PX * 1000 / sr).astype(int), 0, W - 1)
    c = np.zeros((W, W)); zs = np.zeros((W, W)); np.add.at(c, (iy, ix), 1); np.add.at(zs, (iy, ix), z)
    cs = gaussian_filter(c, 0.8); zz = gaussian_filter(zs, 0.8) / np.maximum(cs, 1e-9)
    v = np.clip(cs / max(np.percentile(cs[cs > 0], 99.5), 1e-9), 0, 1)
    ax.imshow(hsv_to_rgb(np.stack([np.clip((zz - zr[0]) / (zr[1] - zr[0]), 0, 1) * 0.75, np.ones_like(v), v], -1)))
    ax.set_title(title, fontsize=9); ax.axis('off')


def render_xz(x, z, ax, title, sr=6.0, zr=(-0.35, 0.35)):
    from scipy.ndimage import gaussian_filter
    W = int((HI - LO) * PX * 1000 / sr); Hz = int((zr[1] - zr[0]) * 1000 / sr)
    ix = ((x - LO) * PX * 1000 / sr).astype(int); iz = ((z - zr[0]) * 1000 / sr).astype(int)
    ok = (ix >= 0) & (ix < W) & (iz >= 0) & (iz < Hz)
    c = np.zeros((Hz, W)); np.add.at(c, (iz[ok], ix[ok]), 1)
    c = gaussian_filter(c, 0.8)
    ax.imshow(c[::-1], cmap='magma', vmax=max(np.percentile(c[c > 0], 99.5), 1e-9) if (c > 0).any() else 1, aspect='equal')
    ax.set_title(title, fontsize=9); ax.set_yticks([0, Hz // 2, Hz - 1]); ax.set_yticklabels([f'{zr[1]*1000:.0f}', '0', f'{zr[0]*1000:.0f}'], fontsize=7)
    ax.set_xticks([]); ax.set_ylabel('z (nm)', fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='d1'); p.add_argument('--model', default='v2'); p.add_argument('--pattern', default='iter')
    p.add_argument('--sz_max', type=float, default=0.025); p.add_argument('--chi2_max', type=float, default=4.0); p.add_argument('--sxy_max', type=float, default=0.12)
    a = p.parse_args()
    from simcode.util.proctitle import auto; auto('4pi-sim')
    d = f'{P4}/simmovie/{a.tag}'; dev = 'cuda'
    meta = json.load(open(f'{d}/meta.json'))
    st = json.load(open(f'{P4}/psf4pi/{meta["psf"]}.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    U = gt_units(d, meta['T'])
    inr = lambda x, y: (x >= LO) & (x < HI) & (y >= LO) & (y < HI)
    gsel = inr(U['x'], U['y']) & (U['N'] >= 300)
    G = {k: v[gsel] for k, v in U.items()}
    L = dict(np.load(f'{d}/locs_SIM_000_{a.model}_{a.pattern}.npz'))  # pattern tag may carry the _phiO suffix
    F = dict(np.load(f'{d}/front_SIM_000_{a.model}.npz'))
    t0 = time.time()
    fc = f'{d}/classical_{a.model}_q.npz'      # with fit-quality fields (chi2, bounds, convergence)
    if os.path.exists(fc):
        C = dict(np.load(fc))
    else:
        C = classical(d, meta, F, lut, dev); np.savez(fc, **C)
    print(f'classical done {time.time() - t0:.0f}s', flush=True)
    area_um2 = ((HI - LO) * PX) ** 2
    res = dict(tag=a.tag, pattern=a.pattern, dens_per_um2_frame=meta['dens_actual'],
               gt_units_per_window=float(gsel.sum() / (meta['T'] // 6)), area_um2=area_um2,
               candidates_per_window=float(inr(L['x'], L['y']).sum() / (meta['T'] // 6)))
    tabs = {}
    sets = {'network (head4pi v2 + explicit solve)': dict(x=L['x'], y=L['y'], z=L['z'], ok=(L['sz'] < a.sz_max) & (L['sx'] < a.sxy_max) & (L['sy'] < a.sxy_max) & (L['N'] > 300)
                                                          & (L['good'] if 'good' in L else True), w=L['w']),   # 'good': the linear pipeline's chi2 selection (simlin4pi.py)
            'network, all candidates': dict(x=L['x'], y=L['y'], z=L['z'], ok=np.ones(len(L['x']), bool), w=L['w']),
            'classical summed SMLM (astig z), unfiltered': dict(x=C['x'], y=C['y'], z=C['z'], ok=(C['sx'] < a.sxy_max) & (C['sy'] < a.sxy_max) & (C['N'] > 300), w=C['w']),
            'classical + published-style aperture z (oracle phase field), unfiltered': dict(x=C['x'], y=C['y'], z=C['z_ap'], ok=(C['sx'] < a.sxy_max) & (C['sy'] < a.sxy_max) & (C['N'] > 300), w=C['w'])}
    # standard classical rejections: chi2/px <= 4 (Pearson, i.e. chi2 <= 4 ROI^2), no parameter at a bound, converged,
    # and the network's uncertainty thresholds (sigma_xy < sxy_max px, N > 300; sigma_z < sz_max um or, as a variant,
    # without the sigma_z cut because the astigmatic CRLB alone is 15-60 nm)
    qok = (C['chi2px'] <= a.chi2_max) & ~C['at_bound'] & C['conv'] & (C['sx'] < a.sxy_max) & (C['sy'] < a.sxy_max) & (C['N'] > 300)
    sets['classical summed SMLM (astig z), filtered'] = dict(x=C['x'], y=C['y'], z=C['z'], ok=qok, w=C['w'])
    sets['classical summed SMLM (astig z), filtered + sigma_z cut'] = dict(x=C['x'], y=C['y'], z=C['z'], ok=qok & (C['sz'] < a.sz_max), w=C['w'])
    sets['classical + published-style aperture z (oracle phase field), filtered'] = dict(x=C['x'], y=C['y'], z=C['z_ap'], ok=qok & ~C['ap_edge'], w=C['w'])
    print('classical rows %d: chi2/px > %.0f %.3f, at bound %.3f, not converged %.3f, pass all + sigma_xy %.3f' % (
        len(C['x']), a.chi2_max, np.mean(C['chi2px'] > a.chi2_max), np.mean(C['at_bound']), np.mean(~C['conv']), np.mean(qok)), flush=True)
    # CRLB for all GT units in the region (isolated-emitter bound)
    cr = crlb_units(G, np.arange(len(G['x'])), meta, lut, dev) * np.array([PX * 1000, PX * 1000, 1000])
    for nm, S in sets.items():
        ok = S['ok'] & inr(S['x'], S['y'])
        xl, yl, zl, wl = S['x'][ok], S['y'][ok], S['z'][ok], S['w'][ok]
        mg, ml = match(G['w'], G['x'], G['y'], wl, xl, yl)
        mm = mg >= 0
        ex = (xl[mg[mm]] - G['x'][mm]) * PX * 1000; ey = (yl[mg[mm]] - G['y'][mm]) * PX * 1000; ez = (zl[mg[mm]] - G['z'][mm]) * 1000
        bias = [float(np.median(ex)), float(np.median(ey)), float(np.median(ez))]
        rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sg = (G['N'] >= lo) & (G['N'] < hi)
            sm = sg[mm]
            if sm.sum() < 30: continue
            rb = lambda e: float(1.4826 * np.median(np.abs(e[sm] - np.median(e[sm]))))
            gz = sm & (np.abs(ez) <= 100)
            rows.append(dict(lo=lo, hi=hi, n_gt=int(sg.sum()), recall=float(mm[sg].mean()),
                             x_rob=rb(ex), y_rob=rb(ey), z_rob=rb(ez),
                             x_rms=float(np.sqrt(np.mean((ex[gz] - bias[0]) ** 2))), y_rms=float(np.sqrt(np.mean((ey[gz] - bias[1]) ** 2))),
                             z_rms=float(np.sqrt(np.mean((ez[gz] - bias[2]) ** 2))), fringe=float(np.mean(np.abs(ez[sm]) > 100)),
                             crlb=[float(np.sqrt(np.mean(cr[mm][sm][:, k] ** 2))) for k in range(3)]))
        prec = float((ml >= 0).mean()) if len(ml) else 0.0
        tabs[nm] = dict(n_locs=int(ok.sum()), precision=prec, bias_nm=bias, rows=rows)
        print(f'\n{nm}: {ok.sum()} locs, precision {prec:.3f}, offset x/y/z {bias[0]:.1f}/{bias[1]:.1f}/{bias[2]:.1f} nm')
        for r in rows:
            print(f"  {r['lo']}-{r['hi']} e: n {r['n_gt']} recall {r['recall']:.2f} | robust x/y/z {r['x_rob']:.1f}/{r['y_rob']:.1f}/{r['z_rob']:.1f} "
                  f"| RMS (no fringe err) {r['x_rms']:.1f}/{r['y_rms']:.1f}/{r['z_rms']:.1f} | CRLB {r['crlb'][0]:.1f}/{r['crlb'][1]:.1f}/{r['crlb'][2]:.1f} | fringe {r['fringe']:.3f}")
    res['methods'] = tabs
    json.dump(res, open(f'{d}/eval_{a.model}_{a.pattern}_filtered.json', 'w'), indent=1)
    # renders: GT, network, classical summed SMLM, published-style z
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    zr = (-0.3, 0.3)
    fig, axs = plt.subplots(2, 4, figsize=(18, 7.2), gridspec_kw=dict(height_ratios=[3.2, 1]))
    panels = [('ground truth (sites of GT units)', G['x'], G['y'], G['z'])]
    for nm in ('network (head4pi v2 + explicit solve)', 'classical summed SMLM (astig z), filtered', 'classical + published-style aperture z (oracle phase field), filtered'):
        S = sets[nm]; ok = S['ok'] & inr(S['x'], S['y'])
        panels.append((nm, S['x'][ok], S['y'][ok], S['z'][ok]))
    ystrip = (24.0, 28.0)
    for i, (nm, x, y, z) in enumerate(panels):
        render_xy(x, y, z, axs[0, i], f'{nm}\n{len(x)} locs, hue z {zr[0]*1000:.0f}..{zr[1]*1000:.0f} nm', zr)
        s = (y >= ystrip[0]) & (y < ystrip[1])
        render_xz(x[s], z[s], axs[1, i], f'x-z, strip y {ystrip[0]:.0f}-{ystrip[1]:.0f} px')
        if i == 0:
            for yy_ in ystrip:
                axs[0, 0].axhline((yy_ - LO) * PX * 1000 / 6.0, color='w', lw=0.5, ls='--')
    fig.suptitle(f'4Pi-SIMFLUX simulation {a.tag}: {meta["dens_actual"]:.2f} emitters/um^2/frame, {meta["T"]} frames, '
                 f'{(HI-LO)*PX:.2f} x {(HI-LO)*PX:.2f} um usable (6 nm render px)', fontsize=10)
    plt.tight_layout(); fn = f'{d}/render_{a.model}_{a.pattern}_filtered.png'; plt.savefig(fn, dpi=130); plt.close()
    print('render ->', fn)


if __name__ == '__main__':
    main()
