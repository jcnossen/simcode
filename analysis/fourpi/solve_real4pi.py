"""4Pi-SIMFLUX explicit solve on the real-data head outputs (analysis/fourpi/front4pi.py):
  1. SIM pattern per file: k, step phases, depth from the classical pattern estimate (out/4pi/smlm/diag/pattern_<cell>.json)
     or re-estimated from the network intensities (--repattern: sum over channels of T, prior positions, emitters
     on through the window).
  2. 4Pi phase per candidate with z held by the network's (astigmatic) prior: solve 'free' with Phi free.
  3. In-situ 4Pi phase field Phi0(x, y, t) = M(x, y) + d(t): circular regression of the per-candidate phases,
     M on a smoothed tile grid (tile `tile` px, Gaussian smoothing `msmooth` tiles), d per block of `tblk` windows.
  4. Final solve with Phi0 from the field: 'on' (pattern + constant photons) and 'free' (per-frame photons; x, y from the
     prior); 'on' is kept if chi2_on - chi2_free < `dchi` (5 extra dof), else 'free' (lateral = prior).
Output: out/4pi/net/locs_<cell>_<files>_<model>.npz (x, y px in ch0 crop coords, z um, sigmas, photons, chi2, mode,
Phi, Phi0, window, frame) and a json summary.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/solve_real4pi.py --cell NPC --files 10 --model v2
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, kin_starts, best_of

OUT = f'{REPO}/out/4pi/net'
DELTA = {'MT': math.radians(-10.4), 'NPC': math.radians(-13.9)}   # per-angle 4Pi phase offset (SMLM_REPORT 5.2)


def load_front(cell, fi, model, dev):
    d = np.load(f'{OUT}/front_{cell}_{fi:03d}_{model}.npz')
    o = {k: torch.tensor(d[k], device=dev) for k in ('T', 'C', 'bg', 'dxy', 'z', 'ls')}
    Lr = torch.tensor(d['L'], device=dev)
    L = torch.zeros(*Lr.shape[:-1], 2, 2, device=dev)
    L[..., 0, 0] = Lr[..., 0]; L[..., 1, 1] = Lr[..., 1]; L[..., 1, 0] = Lr[..., 2]
    o['L'] = L
    meta = {k: d[k] for k in ('w', 'cx', 'cy', 'dog', 'nn')}
    meta['doff'] = torch.tensor(d['doff'], device=dev)
    return o, meta


def pattern_psi(pat, cx, cy, dev):
    """per-candidate per-frame phases psi_eff [n,6] for local coordinates, depth m [n,6], main-axis k (2)"""
    psi, m, kk = [], [], []
    for a in range(2):
        A = pat[f'angle{a}']
        k = np.array(A['k']); ps = np.radians(A['psi_deg'])
        kr = k[0] * cx + k[1] * cy
        for s in range(3):
            psi.append(ps[s] - kr); m.append(np.full(len(cx), A['m_median']))
        kk.append(k[a])
    return (torch.tensor(np.stack(psi, 1), device=dev, dtype=torch.float32),
            torch.tensor(np.stack(m, 1), device=dev, dtype=torch.float32), tuple(kk))


def repattern(o, meta, ref, nmax=60000):
    """per-file SIM pattern from the network intensities: per-frame channel sums of bright, isolated, on-through
    candidates at the prior positions; k refined around the reference, step phases psi from the slot DFT, depth =
    median per-emitter 3-step modulation (as the classical estimate)."""
    from pattern import refine_k, slotZ, emitter_phase
    I = o['T'].sum(2).cpu().numpy()                                          # [n,6]
    X = np.stack([meta['cx'] + o['dxy'][:, 0].cpu().numpy(), meta['cy'] + o['dxy'][:, 1].cpu().numpy()], 1)
    mI = I.mean(1)
    sel = (mI * 6 > 1500) & (meta['nn'] > 6) & (I.min(1) > 0.25 * mI)
    idx = np.nonzero(sel)[0][:nmax]
    out = dict(n=int(len(idx)))
    for a in range(2):
        I3 = I[idx, 3 * a:3 * a + 3]
        k = refine_k(I3, X[idx], np.array(ref[f'angle{a}']['k']), span=0.02, n=21)
        Z = slotZ(I3, X[idx], k); psi = -np.angle(Z)
        phi, m, B = emitter_phase(I3, psi)
        out[f'angle{a}'] = dict(k=k.tolist(), psi_deg=np.degrees(psi).tolist(), m_median=float(np.nanmedian(np.clip(m, 0, 1.5))),
                                 R=float(np.abs(np.mean(np.exp(1j * (phi - X[idx] @ k))))))
    return out


def solve_all(sv, o, idx, Phi0, doff, m, psi, chunk=4096, kin0s=None, **kw):
    res = []
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        if kin0s is not None: kw['kin0'] = kin0s[j]
        res.append(sv.solve({k: v[j] for k, v in o.items()}, Phi0[j], doff[j], m[j], psi[j], **kw))
    return {k: torch.cat([r[k] for r in res]) for k in res[0]}


def phase_field(X, Y, W, phi, wgt, tile=12.0, msmooth=1.0, tblk=50, nit=4, H=200, Wd=360):
    """Phi0 = M(x, y) + d(t) by alternating circular means. Returns a function (x, y, w) -> Phi0 and diagnostics"""
    from scipy import ndimage as ndi
    tx = np.clip((X / tile).astype(int), 0, int(Wd / tile)); ty = np.clip((Y / tile).astype(int), 0, int(H / tile))
    nx, ny = int(Wd / tile) + 1, int(H / tile) + 1
    tb = (W // tblk).astype(int); nb = tb.max() + 1
    d = np.zeros(nb); M = np.zeros((ny, nx))
    z = np.exp(1j * phi) * wgt
    for _ in range(nit):
        r = z * np.exp(-1j * d[tb])
        G = np.zeros((ny, nx), complex); np.add.at(G, (ty, tx), r)
        G = ndi.gaussian_filter(G.real, msmooth) + 1j * ndi.gaussian_filter(G.imag, msmooth)
        M = np.angle(G)
        r = z * np.exp(-1j * M[ty, tx])
        B = np.zeros(nb, complex); np.add.at(B, tb, r)
        Bs = ndi.gaussian_filter1d(B.real, 1.0) + 1j * ndi.gaussian_filter1d(B.imag, 1.0)
        d = np.angle(Bs)
    res = np.angle(np.exp(1j * (phi - M[ty, tx] - d[tb])))
    R = float(np.abs(np.sum(wgt * np.exp(1j * res))) / np.sum(wgt))

    def f(x, y, w):
        ix = np.clip((x / tile).astype(int), 0, nx - 1); iy = np.clip((y / tile).astype(int), 0, ny - 1)
        return M[iy, ix] + d[np.clip(w // tblk, 0, nb - 1).astype(int)]
    return f, dict(R=R, M=M, d=d, tile=tile, tblk=tblk, res=res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--files', type=int, nargs='+', required=True)
    p.add_argument('--model', default='v2'); p.add_argument('--psf', default='NPC_f_dxy15')
    p.add_argument('--tile', type=float, default=12.0); p.add_argument('--tblk', type=int, default=50)
    p.add_argument('--dchi', type=float, default=7.81)   # kin vs free: 3 extra dof
    p.add_argument('--repattern', type=int, default=1)
    a = p.parse_args()
    dev = 'cuda'
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15)
    pat = json.load(open(f'{REPO}/out/4pi/smlm/diag/pattern_{a.cell}.json'))
    tb = Solve4Pi(lut); tb.build_table()          # tabulated read-out sums (same for every file)
    summ = {}
    for fi in a.files:
        t0 = time.time()
        o, meta = load_front(a.cell, fi, a.model, dev)
        n = len(meta['w'])
        cx, cy = meta['cx'].astype(np.float64), meta['cy'].astype(np.float64)
        patf = repattern(o, meta, pat) if a.repattern else pat
        psi, m, kk = pattern_psi(patf, cx, cy, dev)
        sv = Solve4Pi(lut, k=kk, delta=(0.0, DELTA[a.cell])); sv.tab, sv.tab_lim = tb.tab, tb.tab_lim
        idx = torch.arange(n, device=dev)
        # 2. per-candidate phase with z from the prior
        rp = solve_all(sv, o, idx, torch.zeros(n, device=dev), meta['doff'], m, psi, mode='free', fit_phi=True)
        phi = rp['th'][:, -1].cpu().numpy(); sphi = rp['sig'][:, -1].cpu().numpy()
        X = cx + o['dxy'][:, 0].cpu().numpy(); Y = cy + o['dxy'][:, 1].cpu().numpy()
        Tsum = o['T'].sum((1, 2)).cpu().numpy()
        good = (sphi < 0.8) & (Tsum > 800) & (meta['nn'] > 4)
        wgt = good / np.maximum(sphi, 0.05) ** 2
        f, diag = phase_field(X, Y, meta['w'], phi, wgt, tile=a.tile, tblk=a.tblk)
        Phi0 = torch.tensor(f(X, Y, meta['w']), device=dev, dtype=torch.float32)
        print(f'{a.cell} {fi}: {n} candidates, phase field R {diag["R"]:.3f} ({good.sum()} used), {time.time() - t0:.0f}s', flush=True)
        # 4. final solve
        rfr = solve_all(sv, o, idx, Phi0, meta['doff'], m, psi, mode='free')
        ks = [kin_starts({'th': rfr['th'][i:i + 4096]}, m[i:i + 4096], psi[i:i + 4096], sv) for i in range(0, n, 4096)]
        r1 = solve_all(sv, o, idx, Phi0, meta['doff'], m, psi, mode='kin', kin0s=torch.cat([k[0] for k in ks]))
        r2 = solve_all(sv, o, idx, Phi0, meta['doff'], m, psi, mode='kin', kin0s=torch.cat([k[1] for k in ks]))
        ron = best_of(r1, r2)
        use_on = (ron['chi2'] - rfr['chi2'] < a.dchi)
        th_xy = torch.where(use_on[:, None], ron['th'][:, :2], rfr['th'][:, :2])
        sg_xy = torch.where(use_on[:, None], ron['sig'][:, :2], rfr['sig'][:, :2])
        z = torch.where(use_on, ron['th'][:, 2], rfr['th'][:, 2]); sz = torch.where(use_on, ron['sig'][:, 2], rfr['sig'][:, 2])
        tw = (torch.minimum(ron['th'][:, 5], torch.tensor(6.0, device=dev)) - torch.maximum(ron['th'][:, 4], torch.tensor(0.0, device=dev))).clamp(min=0)
        N = torch.where(use_on, ron['th'][:, 3] * tw, rfr['th'][:, 3:9].sum(1))
        out = dict(x=cx + th_xy[:, 0].cpu().numpy(), y=cy + th_xy[:, 1].cpu().numpy(), z=z.cpu().numpy(),
                   sx=sg_xy[:, 0].cpu().numpy(), sy=sg_xy[:, 1].cpu().numpy(), sz=sz.cpu().numpy(), N=N.cpu().numpy(),
                   ton=ron['th'][:, 4].cpu().numpy(), toff=ron['th'][:, 5].cpu().numpy(), chi2_on=ron['chi2'].cpu().numpy(), chi2_free=rfr['chi2'].cpu().numpy(), on=use_on.cpu().numpy(),
                   phi=phi, sphi=sphi, Phi0=Phi0.cpu().numpy(), zprior=o['z'].cpu().numpy(), szprior=torch.exp(o['ls'][:, 2]).cpu().numpy(),
                   w=meta['w'], frame=meta['w'] * 6, nn=meta['nn'], Tsum=Tsum, file_index=fi)
        fn = f'{OUT}/locs_{a.cell}_{fi:03d}_{a.model}.npz'
        np.savez(fn, **out, M=diag['M'], d=diag['d'])
        res = diag['res'][good]
        summ[fi] = dict(pattern={k: patf[k] for k in ('angle0', 'angle1')}, n=n, on_frac=float(use_on.float().mean()), field_R=diag['R'], n_field=int(good.sum()),
                        z_med=float(np.median(out['z'])), sz_med=float(np.median(out['sz'][good])) * 1000,
                        sx_med_nm=float(np.median(out['sx'][good & out['on']])) * 128, t_s=time.time() - t0)
        print(f'{a.cell} {fi}: {summ[fi]} -> {fn}', flush=True)
        fs = f'{OUT}/locs_{a.cell}_{a.model}_summary.json'      # merged per file (resumable runs)
        old = json.load(open(fs)) if os.path.exists(fs) else {}
        old[str(fi)] = summ[fi]; json.dump(old, open(fs, 'w'), indent=1)


if __name__ == '__main__':
    main()
