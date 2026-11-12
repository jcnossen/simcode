"""4Pi-SIMFLUX network pipeline on raw dcimg with a learned-window head (v10 ...): one read per file, all in memory.

Per file:
  A. read (threaded) -> camera offset / gain -> det4pi candidates on the registered 4-channel sum (6 planes per window)
     -> unwarped 15 x 15 channel patches (kept in host RAM as float16; never written to disk) -> head: prior, intensities
  B. SIM pattern of the file from the per-frame photon read-outs at the prior positions (slot DFT, as solve_real4pi)
  C. 4Pi phase field: per-candidate phase (z held by the prior) on bright isolated candidates -> tile map + time blocks
  D. per chunk from RAM: head (intensities Y, windows W', covariance) -> explicit solve with the window model and a
     residual background per channel: on-through vs kinetics (kept if chi2 drops by > 9.21; no free fallback)
Network intensities -> explicit solve; no pixel fit. Output out/4pi/net/locs_<cell>_<file>_<head>.npz (+ summary json).
Usage: CUDA_VISIBLE_DEVICES=1 python analysis/fourpi/real_w4pi.py --cell NPC --files 10 --head v10 [--nframes 3600]
"""
import argparse, json, math, os, sys, time, threading, queue
import numpy as np, torch
import torch.nn.functional as F
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from dc import Dcimg, movie_files, file_index
from front4pi import Front, reader, XLIM, P
import solve_real4pi as SR
from simsolve4pi import repattern_xy
from simcode2.fourpi.solve4pi import kin_starts, best_of
from simcode2.fourpi.solvek4pi import SolveK
from simcode2.fourpi.loadhead import load_head

OUT = f'{REPO}/out/4pi/net'
dev = 'cuda'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--files', type=int, nargs='+', required=True)
    p.add_argument('--head', default='v10'); p.add_argument('--det', default=f'{REPO}/out/4pi/models/det4pi/final.pt')
    p.add_argument('--nframes', type=int, default=0); p.add_argument('--chunk', type=int, default=600)
    p.add_argument('--p_min', type=float, default=0.5); p.add_argument('--tile', type=float, default=12.0); p.add_argument('--tblk', type=int, default=50)
    p.add_argument('--dnll', type=float, default=9.21); p.add_argument('--chi2_max', type=float, default=2.5)
    p.add_argument('--coh', type=float, default=1.0, help='interference-contrast scale of the PSF model in the solve')
    p.add_argument('--iters', type=int, default=20); p.add_argument('--batch', type=int, default=3000)
    p.add_argument('--min_photons', type=float, default=300.0, help='candidates below this photon read-out per window are not kept (the good selection needs N > 300)')
    p.add_argument('--tag', default='', help='suffix of the output name'); p.add_argument('--nphi', type=int, default=60000)
    a = p.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-real-w')
    os.makedirs(OUT, exist_ok=True)
    net, ck = load_head(a.head); lut = net.lut
    from simcode2.det.up2 import load as dload, select as dselect
    det, _ = dload(a.det)
    fr = Front(a.cell)
    pat0 = json.load(open(f'{REPO}/out/4pi/smlm/diag/pattern_{a.cell}.json'))
    sv = SolveK(lut, net.wins, delta=(0.0, SR.DELTA[a.cell])); sv.tab_lim = None; sv.coh_scale = a.coh
    h = P // 2
    for fi in a.files:
        fn = [f for f in movie_files(a.cell) if file_index(f) == fi][0]
        nfr = a.nframes or Dcimg(fn).nframes
        q = queue.Queue(maxsize=2)
        threading.Thread(target=reader, args=(fn, 0, nfr, a.chunk, q), daemon=True).start()
        acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'img', 'dxy', 'z', 'ls', 'I', 'pdet', 'nn')}
        t0 = time.time()
        # ---- A: read, detect, patches, head (prior + photon read-outs)
        while True:
            it = q.get()
            if it is None: break
            a0, raw = it
            with torch.no_grad():
                E = fr.to_e(raw); nW = E.shape[0] // 6
                if nW == 0: continue
                Rs = E[:nW * 6, 0].clone()
                for c in (1, 2, 3):
                    Rs += F.grid_sample(E[:nW * 6, c:c + 1], fr.grid[c].expand(nW * 6, -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
                Rf = Rs.reshape(nW, 6, 200, 360); ccs = []
                for b0 in range(0, nW, 20):                      # detector in batches of 20 windows (memory)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        dout = det(Rf[b0:b0 + 20])
                    c1 = dselect({k: v.float() for k, v in dout.items()}, 1, k=2048, p_min=a.p_min)
                    c1['b'] = c1['b'] + b0; ccs.append(c1)
                cc = {k: torch.cat([c1[k] for c1 in ccs]) for k in ('b', 'iy', 'ix', 'logit')}
                okc = torch.isfinite(cc['logit']) & (cc['ix'] >= XLIM[0]) & (cc['ix'] < XLIM[1]) & (cc['iy'] >= h + 1) & (cc['iy'] < 200 - h - 1)
                w, cy, cx, pd = cc['b'][okc], cc['iy'][okc], cc['ix'][okc], torch.sigmoid(cc['logit'][okc])
                if len(w) == 0: continue
                dd = torch.full((len(w),), 99.0, device=dev)
                for wv in torch.unique(w):
                    s = torch.nonzero(w == wv)[:, 0]
                    if len(s) > 1:
                        pts = torch.stack([cx[s], cy[s]], 1).float(); D = torch.cdist(pts, pts); D.fill_diagonal_(99.0); dd[s] = D.min(1).values
                img, doff, ok = fr.patches(E, w, cy, cx)
                sel = torch.nonzero(ok)[:, 0]
                img = img[sel]; doff = doff[sel]
                outs = [net(img[i:i + a.batch], doff[i:i + a.batch]) for i in range(0, len(sel), a.batch)]
                o = {k: torch.cat([u[k] for u in outs]) for k in ('Y', 'dxy', 'z', 'ls')}
                kp = o['Y'][..., 0].sum((1, 2)) >= a.min_photons
                o = {k: v[kp] for k, v in o.items()}; img = img[kp]; doff = doff[kp]; sel = sel[kp]
            for k, v in (('w', w[sel] + a0 // 6), ('cx', cx[sel]), ('cy', cy[sel]), ('doff', doff), ('img', img.half()), ('dxy', o['dxy']), ('z', o['z']),
                         ('ls', o['ls']), ('I', o['Y'][..., 0].sum(2)), ('pdet', pd[sel]), ('nn', dd[sel])):
                acc[k].append(v.cpu().numpy())
        R = {k: np.concatenate(v) for k, v in acc.items()}; del acc
        n = len(R['w']); cx, cy = R['cx'].astype(np.float64), R['cy'].astype(np.float64)
        tA = time.time() - t0
        print(f'{a.cell} {fi}: {n} candidates ({n / (nfr // 6):.1f} per window), patches {R["img"].nbytes / 1e9:.2f} GB in RAM, read + head {tA:.0f}s', flush=True)
        I = R['I']; mI = I.mean(1); Tsum = I.sum(1)
        Xp = np.stack([cx + R['dxy'][:, 0], cy + R['dxy'][:, 1]], 1)
        # ---- B: pattern
        selp = (mI * 6 > 1500) & (R['nn'] > 6) & (I.min(1) > 0.25 * mI)
        patf = repattern_xy(I, Xp, pat0, selp)
        psi, m, _ = SR.pattern_psi(patf, cx, cy, dev)
        sv.k2 = tuple(tuple(patf[f'angle{i}']['k']) for i in range(2))
        tdoff = torch.tensor(R['doff'], device=dev)

        def chunk_out(j):
            with torch.no_grad():
                o = net(torch.tensor(R['img'][j].astype(np.float32), device=dev), tdoff[j])
            return dict(Y=o['Y'], Lb=o['Lb'], dxy=o['dxy'], z=o['z'], ls=o['ls'], W=o['W'], Wsum=o['Wsum'],
                        sb=torch.full((len(j), 4), 0.2, device=dev)), o

        # ---- C: phase field from bright isolated candidates
        cand = np.nonzero((Tsum > 800) & (R['nn'] > 4))[0]
        if len(cand) > a.nphi: cand = np.sort(np.random.default_rng(0).choice(cand, a.nphi, replace=False))
        phi = np.zeros(len(cand)); sphi = np.zeros(len(cand))
        for i in range(0, len(cand), a.batch):
            j = cand[i:i + a.batch]; oo, _ = chunk_out(j); jt = torch.tensor(j, device=dev)
            rp = sv.solve(oo, torch.zeros(len(j), device=dev), tdoff[jt], m[jt], psi[jt], mode='free', fit_phi=True, iters=a.iters)
            phi[i:i + len(j)] = rp['th'][:, -5].cpu().numpy(); sphi[i:i + len(j)] = rp['sig'][:, -5].cpu().numpy()
        gphi = sphi < 0.8
        f, diag = SR.phase_field(Xp[cand, 0], Xp[cand, 1], R['w'][cand], phi, gphi / np.maximum(sphi, 0.05) ** 2, tile=a.tile, tblk=a.tblk)
        Phi0 = torch.tensor(f(Xp[:, 0], Xp[:, 1], R['w']), device=dev, dtype=torch.float32)
        tC = time.time() - t0 - tA
        print(f'{a.cell} {fi}: pattern psi {[np.round(patf[f"angle{i}"]["psi_deg"], 1).tolist() for i in range(2)]} m {patf["angle0"]["m_median"]:.2f}/{patf["angle1"]["m_median"]:.2f}'
              f' R {patf["angle0"]["R"]:.2f}/{patf["angle1"]["R"]:.2f} ({int(selp.sum())} emitters); phase field R {diag["R"]:.3f} ({int(gphi.sum())} used), {tC:.0f}s', flush=True)
        # ---- D: final solve
        cols = {k: [] for k in ('th', 'sig', 'chi2', 'kin', 'chi2_on', 'chi2_kin', 'N', 'ton', 'toff', 'bg', 'cs')}
        for i in range(0, n, a.batch):
            j = np.arange(i, min(i + a.batch, n)); oo, o = chunk_out(j); jt = torch.tensor(j, device=dev)
            arg = (Phi0[jt], tdoff[jt], m[jt], psi[jt])
            ron = sv.solve(oo, *arg, mode='on', iters=a.iters)
            rfr = sv.solve(oo, *arg, mode='free', iters=a.iters, zstarts=(0.0,))
            k1, k2 = kin_starts({'th': torch.cat([ron['th'][:, :3], rfr['th'][:, 3:]], 1)}, m[jt], psi[jt], sv)
            rk = best_of(sv.solve(oo, *arg, mode='kin', kin0=k1, iters=a.iters), sv.solve(oo, *arg, mode='kin', kin0=k2, iters=a.iters))
            usek = (ron['chi2'] - rk['chi2']) > a.dnll
            pick = lambda A, B: torch.where(usek.reshape(-1, *([1] * (A.dim() - 1))), A, B)
            tw = (torch.minimum(rk['th'][:, 5], torch.tensor(6.0, device=dev)) - torch.maximum(rk['th'][:, 4], torch.tensor(0.0, device=dev))).clamp(min=0)
            for k, v in (('th', pick(rk['th'][:, :3], ron['th'][:, :3])), ('sig', pick(rk['sig'][:, :3], ron['sig'][:, :3])), ('chi2', pick(rk['chi2'], ron['chi2'])),
                         ('kin', usek), ('chi2_on', ron['chi2']), ('chi2_kin', rk['chi2']), ('N', torch.where(usek, rk['th'][:, 3] * tw, ron['th'][:, 3] * 6)),
                         ('ton', rk['th'][:, 4]), ('toff', rk['th'][:, 5]), ('bg', o['bgc'] + pick(rk['th'][:, -4:], ron['th'][:, -4:])), ('cs', o['cs'])):
                cols[k].append(v.cpu().numpy())
            if (i // a.batch) % 10 == 0:
                print(f'{a.cell} {fi}: solved {i + len(j)}/{n}, {time.time() - t0:.0f}s', flush=True)
        C = {k: np.concatenate(v) for k, v in cols.items()}
        dof = np.where(C['kin'], 86.0, 88.0); good = C['chi2'] / dof < a.chi2_max
        out = dict(x=cx + C['th'][:, 0], y=cy + C['th'][:, 1], z=C['th'][:, 2], sx=C['sig'][:, 0], sy=C['sig'][:, 1], sz=C['sig'][:, 2], N=C['N'],
                   on=np.ones(n, bool), kin=C['kin'], ton=C['ton'], toff=C['toff'], chi2=C['chi2'], dof=dof, good=good, chi2_on=C['chi2_on'], chi2_free=C['chi2_kin'],
                   Phi0=Phi0.cpu().numpy(), zprior=R['z'], szprior=np.exp(R['ls'][:, 2]), xprior=Xp[:, 0], yprior=Xp[:, 1], cx=R['cx'], cy=R['cy'],
                   w=R['w'], frame=R['w'] * 6, nn=R['nn'], Tsum=Tsum, pdet=R['pdet'], bg=C['bg'], cs=C['cs'], file_index=fi)
        fo = f'{OUT}/locs_{a.cell}_{fi:03d}_{a.head}{a.tag}.npz'
        if __import__('shutil').disk_usage(OUT).free < 10e9:
            print('WARNING: < 10 GB free, output not written', flush=True); continue
        np.savez(fo + '.tmp.npz', **out, M=diag['M'], d=diag['d']); os.replace(fo + '.tmp.npz', fo)
        g2 = good & (Tsum > 2000) & (Tsum < 4000)
        summ = dict(n=n, per_window=n / (nfr // 6), good_frac=float(good.mean()), kin_frac=float(C['kin'].mean()), field_R=diag['R'],
                    chi2_dof_median=float(np.median(C['chi2'] / dof)), pattern={k: patf[k] for k in ('angle0', 'angle1')},
                    sig_med_nm_2_4k=[float(np.median(out['sx'][g2]) * 128), float(np.median(out['sy'][g2]) * 128), float(np.median(out['sz'][g2]) * 1000)],
                    coh=a.coh, t_read_head=tA, t_field=tC, t_total=time.time() - t0)
        fs = f'{OUT}/locs_{a.cell}_{a.head}{a.tag}_summary.json'
        old = json.load(open(fs)) if os.path.exists(fs) else {}
        old[str(fi)] = summ; json.dump(old, open(fs, 'w'), indent=1)
        print(f'{a.cell} {fi}: good {summ["good_frac"]:.3f}, kinetics {summ["kin_frac"]:.3f}, chi2/dof median {summ["chi2_dof_median"]:.2f}, sigma 2-4k e {np.round(summ["sig_med_nm_2_4k"], 2).tolist()} nm, total {summ["t_total"]:.0f}s -> {fo}', flush=True)
        del R


if __name__ == '__main__':
    main()
