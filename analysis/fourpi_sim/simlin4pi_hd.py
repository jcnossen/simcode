"""COPY of analysis/fourpi/simlin4pi.py (hd head experiment, 2026-11-15): --head may be a checkpoint path, --name sets the
output tag, --out the output directory (default: the movie directory, as the original). Nothing else changed.
Original docstring:
Linear fixed-window read-out + exact-window solve on a simulated 4Pi movie (simulation gate of the read-out of
out/4pi/REVIEW_NET.md section 5). Candidates, prior (x, y, z, sigma) and patches come from a front file written by
simfront4pi.py --save_img 1 (det4pi or DoG candidates, any head: only its prior is used).

Steps: one background per channel (border median of the frame-mean patch) -> K linear read-outs per (frame, channel) at
the prior position (linread4pi) -> SIM pattern (true, or estimated from the per-frame read-out sums: slot DFT at the prior
positions, refined once at the solved positions) -> 4Pi phase field (true, or in situ: per-candidate phase with z held by
the prior, tile map + time blocks) -> explicit solve with the exact window model and a residual background per channel:
on-through vs kinetics, kinetics kept if it lowers chi2 by > 9.21 (the main pipeline's rule; no free fallback).
Output: locs_SIM_000_<front>_<pattern>lin[_phiO].npz in the table format of the real pipeline (+ chi2, dof, good), so simeval4pi.py and
showcase4pi_eval.py run on it (--model <front> --pattern <pattern>lin[_phiO]).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/simlin4pi.py --tag iso --front v3L_det --pattern oracle --phi oracle
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fourpi'))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.psf import fourpi_state
from simcode.processing import fourpi_pattern_phase as SR
from simcode.processing.fourpi_pattern_phase import repattern_xy
from simcode.fourpi.spline_table import LUT4Pi
from simcode.solve.fourpi import kin_starts, best_of
from simcode.solve.fourpi import FourPiSolve
from simcode.emitter.fourpi import linear_readout, border_bg

WINS = (('mf', 0.003), ('cr', 0.003), ('ci', 0.003), ('flat', 0.0))
dev = 'cuda'
SB = 0.2      # prior sd of the residual background per channel (e/px); x sqrt(K) for K-summed movies


def solve_all(sv, D, idx, Phi0, m, psi, chunk=2000, kin0s=None, sb=None, **kw):
    """D: dict(Y, Lb, Ws, W, dxy, z, ls, doff). Exact window model + background nuisance."""
    res = []
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        if kin0s is not None: kw['kin0'] = kin0s[j]
        oo = dict(Y=D['Y'][j], Lb=D['Lb'][j], dxy=D['dxy'][j], z=D['z'][j], ls=D['ls'][j], W=D['W'][j], Wsum=D['Ws'][j],
                  sb=torch.full((len(j), 4), sb if sb is not None else SB, device=dev))
        res.append(sv.solve(oo, Phi0[j], D['doff'][j], m[j], psi[j], **kw))
    return {k: torch.cat([r[k] for r in res]) for k in res[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='iso'); p.add_argument('--front', default='v3L_det'); p.add_argument('--psf', default='NPC_f_dxy15')
    p.add_argument('--tile', type=float, default=6.0); p.add_argument('--tblk', type=int, default=25)
    p.add_argument('--dnll', type=float, default=9.21)
    p.add_argument('--phi', default='field'); p.add_argument('--pattern', default='iter')
    p.add_argument('--chi2_max', type=float, default=1.6, help="'good': chi2 / dof below this")
    p.add_argument('--img_cpu', type=int, default=0, help='1: keep the candidate patches in host memory (large movies; batches go to the GPU)')
    p.add_argument('--ksum', type=int, default=1, help='K-summed movie: read variance K x in the read-out covariance')
    p.add_argument('--head', default='', help='learned-window head (v10 ...): its intensities, windows and covariance instead of the fixed read-out')
    p.add_argument('--name', default=''); p.add_argument('--out', default='')
    p.add_argument('--work', required=True, help='4Pi work folder')
    a = p.parse_args(); P4 = a.work
    from simcode.util.proctitle import set_title; set_title('4pi-simlin')
    global SB; SB = 0.2 * math.sqrt(a.ksum)
    d = f'{P4}/simmovie/{a.tag}'
    meta = json.load(open(f'{d}/meta.json'))
    st = fourpi_state(a.psf, P4)
    lut = LUT4Pi(st, P=15)
    t0 = time.time()
    F = np.load(f'{d}/front_SIM_000_{a.front}.npz')
    assert 'img' in F.files, 'run simfront4pi.py with --save_img 1'
    img = torch.tensor(F['img'].astype(np.float32), device='cpu' if a.img_cpu else dev)
    n = img.shape[0]
    D = dict(dxy=torch.tensor(F['dxy'], device=dev), z=torch.tensor(F['z'], device=dev), ls=torch.tensor(F['ls'], device=dev),
             doff=torch.tensor(F['doff'], device=dev))
    w = F['w']; nn = F['nn']; cx, cy = F['cx'].astype(np.float64), F['cy'].astype(np.float64)
    bm = border_bg(img).to(dev)
    Ys, Ls, Wss, Wl = [], [], [], []
    if a.head:
        from simcode.emitter.fourpi import load as load_head
        net, _ = load_head(a.head, work=P4); bl = []; pri = {k: [] for k in ('dxy', 'z', 'ls')}
        net.vread = a.ksum * 1.14 ** 2 + (0.2 * a.ksum) ** 2
        nbr = None
        if getattr(net, 'nbr_in', False):   # pass 0: the head's prior positions -> map of the other candidates of each window
            from simcode.emitter.fourpi import candidate_neighbours, neighbour_map
            with torch.no_grad():
                p0 = torch.cat([net(img[i:i + 1000], D['doff'][i:i + 1000])['dxy'] for i in range(0, n, 1000)]).cpu().numpy()
            ndx, ndy, nval = candidate_neighbours(F['w'], F['cx'] + p0[:, 0], F['cy'] + p0[:, 1])
            # offsets relative to the candidate PIXEL (the patch centre): add the candidate's own prior offset
            nbr = (torch.tensor(ndx + p0[:, :1], device=dev), torch.tensor(ndy + p0[:, 1:2], device=dev), torch.tensor(nval, device=dev))
        with torch.no_grad():
            for i in range(0, n, 1000):
                j = slice(i, i + 1000)
                oh = net(img[j], D['doff'][j], **({'nbr': neighbour_map(nbr[0][j], nbr[1][j], nbr[2][j], D['doff'][j])} if nbr is not None else {}))
                Ys.append(oh['Y']); Ls.append(oh['Lb']); Wss.append(oh['Wsum']); Wl.append(oh['W']); bl.append(oh['bgc'])
                for k in pri: pri[k].append(oh[k])
        D.update({k: torch.cat(v) for k, v in pri.items()})     # the head's own prior (trained for these conditions)
        bm = torch.cat(bl)
    else:
        for i in range(0, n, 2000):
            j = slice(i, i + 2000)
            Y, Lb, Ws, W = linear_readout(img[j].to(dev), D['doff'][j], lut, WINS, D['dxy'][j], D['z'][j], bm[j], return_w=True)
            Ys.append(Y); Ls.append(Lb); Wss.append(Ws); Wl.append(W)
    D.update(Y=torch.cat(Ys), Lb=torch.cat(Ls), Ws=torch.cat(Wss), W=torch.cat(Wl))
    I = D['Y'][..., 0].sum(2).cpu().numpy(); mI = I.mean(1)                    # per-frame photons (matched-filter read-out)
    pdxy = D['dxy'].detach().cpu().numpy(); Xp = np.stack([cx + pdxy[:, 0], cy + pdxy[:, 1]], 1)
    ref = {f'angle{i}': dict(k=meta['k'][i], psi_deg=meta['psi_deg'][i], m_median=0.85) for i in range(2)}
    if a.pattern == 'oracle':
        patf = {f'angle{i}': dict(k=meta['k'][i], psi_deg=meta['psi_deg'][i], m_median=meta['m']) for i in range(2)}
    else:
        patf = repattern_xy(I, Xp, ref, (mI * 6 > 800) & (nn > 3) & (I.min(1) > 0.25 * mI))

    def setpat(patf):
        ps, m, _ = SR.pattern_psi(patf, cx, cy, dev)
        # full 2D wave vector in the solve; pattern_psi's psi already holds -k . (cx, cy)
        sv.k2 = tuple(tuple(patf[f'angle{i}']['k']) for i in range(2))
        return ps, m
    sv = FourPiSolve(lut, WINS, delta=(0.0, math.radians(meta['delta_deg']))); sv.tab_lim = None
    psi, m = setpat(patf)
    idx = torch.arange(n, device=dev)
    # ---- 4Pi phase field
    rp = solve_all(sv, D, idx, torch.zeros(n, device=dev), m, psi, mode='free', fit_phi=True)
    phi = rp['th'][:, -5].cpu().numpy(); sphi = rp['sig'][:, -5].cpu().numpy()
    Tsum = I.sum(1)
    good_f = (sphi < 0.8) & (Tsum > 800) & (nn > 4)
    f, diag = SR.phase_field(Xp[:, 0], Xp[:, 1], w, phi, good_f / np.maximum(sphi, 0.05) ** 2, tile=a.tile, tblk=a.tblk, H=meta['H'], Wd=meta['W'])
    Phi0 = torch.tensor(f(Xp[:, 0], Xp[:, 1], w), device=dev, dtype=torch.float32)
    from simmovie4pi import phi0_field
    Ptrue = phi0_field(Xp[:, 0], Xp[:, 1], 6 * w + 3, meta['H'], meta['W'], meta['T'])
    dph = Phi0.cpu().numpy() - Ptrue
    field_err = float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * dph)))))))
    if a.phi == 'oracle':
        Phi0 = torch.tensor(Ptrue, device=dev, dtype=torch.float32)

    def final(psi, m):
        ron = solve_all(sv, D, idx, Phi0, m, psi, mode='on')
        rfr = solve_all(sv, D, idx, Phi0, m, psi, mode='free')
        ks = [kin_starts({'th': rfr['th'][i:i + 4096]}, m[i:i + 4096], psi[i:i + 4096], sv) for i in range(0, n, 4096)]
        r1 = solve_all(sv, D, idx, Phi0, m, psi, mode='kin', kin0s=torch.cat([k[0] for k in ks]))
        r2 = solve_all(sv, D, idx, Phi0, m, psi, mode='kin', kin0s=torch.cat([k[1] for k in ks]))
        rk = best_of(r1, r2)
        return ron, rk, (ron['chi2'] - rk['chi2']) > a.dnll
    ron, rk, usek = final(psi, m)
    if a.pattern == 'iter':   # one refinement at the solved positions of bright on-through rows
        Xs = np.stack([cx + ron['th'][:, 0].cpu().numpy(), cy + ron['th'][:, 1].cpu().numpy()], 1)
        ok = (~usek).cpu().numpy() & (Tsum > 800) & (nn > 3) & ((ron['chi2'] / 88).cpu().numpy() < a.chi2_max)
        patf = repattern_xy(I, Xs, patf, ok)
        psi, m = setpat(patf)
        ron, rk, usek = final(psi, m)
    pick = lambda A, B: torch.where(usek.reshape(-1, *([1] * (A.dim() - 1))), A, B)
    th = pick(rk['th'][:, :3], ron['th'][:, :3]); sg = pick(rk['sig'][:, :3], ron['sig'][:, :3])
    chi2 = pick(rk['chi2'], ron['chi2']); dof = torch.where(usek, torch.tensor(86.0, device=dev), torch.tensor(88.0, device=dev))
    tw = (torch.minimum(rk['th'][:, 5], torch.tensor(6.0, device=dev)) - torch.maximum(rk['th'][:, 4], torch.tensor(0.0, device=dev))).clamp(min=0)
    Pt, _, _ = sv.pattern(ron['th'][:, 0], ron['th'][:, 1], m, psi)
    N = torch.where(usek, rk['th'][:, 3] * tw, ron['th'][:, 3] * 6)             # photons per window (pattern average)
    good = (chi2 / dof < a.chi2_max)
    out = dict(x=cx + th[:, 0].cpu().numpy(), y=cy + th[:, 1].cpu().numpy(), z=th[:, 2].cpu().numpy(),
               sx=sg[:, 0].cpu().numpy(), sy=sg[:, 1].cpu().numpy(), sz=sg[:, 2].cpu().numpy(), N=N.cpu().numpy(),
               on=np.ones(n, bool), kin=usek.cpu().numpy(), chi2_on=ron['chi2'].cpu().numpy(), chi2_free=rk['chi2'].cpu().numpy(),
               chi2=chi2.cpu().numpy(), dof=dof.cpu().numpy(), good=good.cpu().numpy(),
               phi=phi, sphi=sphi, Phi0=Phi0.cpu().numpy(), zprior=D['z'].detach().cpu().numpy(), szprior=np.exp(D['ls'].detach().cpu().numpy()[:, 2]),
               xprior=Xp[:, 0], yprior=Xp[:, 1], w=w, nn=nn, Tsum=Tsum,
               bg=(bm + pick(rk['th'][:, -4:], ron['th'][:, -4:])).cpu().numpy())
    tag = f"{a.pattern}{a.name or a.head or 'lin'}" + ('' if a.phi == 'field' else '_phiO')
    od = a.out or d; os.makedirs(od, exist_ok=True)
    np.savez(f'{od}/locs_SIM_000_{a.front}_{tag}.npz', **out, M=diag['M'], d=diag['d'])
    summ = dict(n=n, kin_frac=float(usek.float().mean()), good_frac=float(good.float().mean()), field_R=diag['R'],
                field_err_deg=field_err, pattern=patf, t_s=time.time() - t0)
    json.dump(summ, open(f'{od}/solve_summary_{a.front}_{tag}.json', 'w'), indent=1)
    print({k: v for k, v in summ.items() if k != 'pattern'}, {k: (v['psi_deg'], v['m_median']) for k, v in patf.items() if k.startswith('angle')}, flush=True)


if __name__ == '__main__':
    main()
