"""4Pi-SIMCODE: the 4Pi-SIMFLUX network pipeline on raw dcimg files with the learned-window head (v10 ...). One read per
file, all candidate patches in host memory.

Per file (process_file):
  A. read (prefetched in a background thread) -> camera offset / gain -> det4pi candidates on the registered 4-channel sum
     (6 planes per window) -> unwarped 15 x 15 channel patches (kept in host RAM as float16; never written to disk)
     -> head: prior, intensities
  B. SIM pattern of the file from the per-frame photon read-outs at the prior positions (slot DFT)
  C. 4Pi phase field: per-candidate phase (z held by the prior) on bright isolated candidates -> tile map + time blocks
  D. per chunk from RAM: head (intensities Y, windows W', covariance) -> explicit solve with the window model and a
     residual background per channel: on-through vs kinetics (kept if chi2 drops by > 9.21; no free fallback)
Network intensities -> explicit solve; no pixel fit.

Files are the units of the shared work driver (simcode/io/work.py): finished files are skipped, outputs are written
atomically, the disk is checked before a file starts, and with --claim 1 several copies of the command (one per GPU:
python -m simcode.io.launch --gpus 1,2 -- python -m simcode.fourpi.run --cell MT --all --claim 1 ...) share the files.

Output: <4Pi work dir>/net/locs_<cell>_<file>_<head><tag>.npz, a record locs_<cell>_<file>_<head><tag>.json per file and
the merged locs_<cell>_<head><tag>_summary.json.
Usage: CUDA_VISIBLE_DEVICES=1 python -m simcode.fourpi.run --cell NPC --files 10 --head v10 [--nframes 3600]
"""
import argparse, json, os, time
import numpy as np, torch
from simcode.fourpi.setup import P4, LAYOUT, XLIM, P, DELTA, movie_file, movie_files, file_index
from simcode.fourpi.front import Front, nn_dist
from simcode.fourpi import insitu as IS
from simcode.fourpi.solve4pi import kin_starts, best_of
from simcode.fourpi.solvek4pi import SolveK
from simcode.fourpi.loadhead import load_head
from simcode.io.movie import DcimgMovie, Prefetch
from simcode.io.work import Units, atomic_json

dev = 'cuda'


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cell', default='NPC')
    p.add_argument('--files', type=int, nargs='+', default=None, help='file numbers of the movie')
    p.add_argument('--all', action='store_true', help='all files of the cell')
    p.add_argument('--order', default='fwd', choices=['fwd', 'rev'])
    p.add_argument('--claim', type=int, default=0, help='1: claim each file (several workers on one output directory)')
    p.add_argument('--redo', type=int, default=0)
    p.add_argument('--min_free_gb', type=float, default=10.0, help='a file is not started with less free space on the output file system')
    p.add_argument('--out', default=f'{P4}/net')
    p.add_argument('--head', default='v10'); p.add_argument('--det', default=f'{P4}/models/det4pi/final.pt')
    p.add_argument('--nframes', type=int, default=0); p.add_argument('--chunk', type=int, default=600)
    p.add_argument('--p_min', type=float, default=0.5); p.add_argument('--tile', type=float, default=12.0); p.add_argument('--tblk', type=int, default=50)
    p.add_argument('--dnll', type=float, default=9.21); p.add_argument('--chi2_max', type=float, default=2.5)
    p.add_argument('--coh', type=float, default=1.0, help='interference-contrast scale of the PSF model in the solve')
    p.add_argument('--iters', type=int, default=15); p.add_argument('--batch', type=int, default=1500)
    p.add_argument('--min_photons', type=float, default=300.0, help='candidates below this photon read-out per window are not kept (the good selection needs N > 300)')
    p.add_argument('--regions', default='', help='json with boxes (x0, x1, y0, y1 on the candidate pixel): the final solve only for candidates inside (pattern and phase field still from the whole field)')
    p.add_argument('--tag', default='', help='suffix of the output name'); p.add_argument('--nphi', type=int, default=60000)
    return p.parse_args(argv)


def setup(a):
    """networks, front end and solver of a cell (the same for every file)"""
    net, ck = load_head(a.head)
    from simcode.detector.model import load as dload
    from simcode.detector.select import select_points as dselect
    det, _ = dload(a.det)
    fr = Front(a.cell)
    pat0 = json.load(open(f'{P4}/smlm/diag/pattern_{a.cell}.json'))
    sv = SolveK(net.lut, net.wins, delta=(0.0, DELTA[a.cell])); sv.tab_lim = None; sv.coh_scale = a.coh
    return dict(net=net, det=det, dselect=dselect, fr=fr, pat0=pat0, sv=sv)


def process_file(rd, ctx, a, fi, nfr):
    """one file: rd = reader of the channel crops (DcimgMovie with the layout; any reader with the same read()), nfr frames
    -> (table dict, phase-field arrays dict(M, d), summary dict)"""
    net, det, dselect, fr, pat0, sv = (ctx[k] for k in ('net', 'det', 'dselect', 'fr', 'pat0', 'sv'))
    h = P // 2
    plan = [(a0, min(a0 + a.chunk, nfr)) for a0 in range(0, nfr, a.chunk)]
    pf = Prefetch(rd, plan)
    acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'img', 'dxy', 'z', 'ls', 'I', 'pdet', 'nn')}
    t0 = time.time()
    # ---- A: read, detect, patches, head (prior + photon read-outs)
    for a0, a1 in plan:
        raw = pf.read(a0, a1)
        with torch.no_grad():
            E = fr.to_e(raw); nW = E.shape[0] // 6
            if nW == 0: continue
            Rs = fr.registered_sum(E[:nW * 6])
            Rf = Rs.reshape(nW, 6, 200, 360); ccs = []
            for b0 in range(0, nW, 20):                      # detector in batches of 20 windows (memory)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dout = det(Rf[b0:b0 + 20])
                c1 = dselect({k: v.float() for k, v in dout.items()}, k=2048, p_min=a.p_min)
                c1['b'] = c1['b'] + b0; ccs.append(c1)
            cc = {k: torch.cat([c1[k] for c1 in ccs]) for k in ('b', 'iy', 'ix', 'logit')}
            okc = torch.isfinite(cc['logit']) & (cc['ix'] >= XLIM[0]) & (cc['ix'] < XLIM[1]) & (cc['iy'] >= h + 1) & (cc['iy'] < 200 - h - 1)
            w, cy, cx, pd = cc['b'][okc], cc['iy'][okc], cc['ix'][okc], torch.sigmoid(cc['logit'][okc])
            if len(w) == 0: continue
            dd = nn_dist(w, cx, cy)
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
    pf.close()
    R = {k: np.concatenate(v) for k, v in acc.items()}; del acc
    n = len(R['w']); cx, cy = R['cx'].astype(np.float64), R['cy'].astype(np.float64)
    tA = time.time() - t0
    print(f'{a.cell} {fi}: {n} candidates ({n / (nfr // 6):.1f} per window), patches {R["img"].nbytes / 1e9:.2f} GB in RAM, read + head {tA:.0f}s', flush=True)
    I = R['I']; mI = I.mean(1); Tsum = I.sum(1)
    Xp = np.stack([cx + R['dxy'][:, 0], cy + R['dxy'][:, 1]], 1)
    # ---- B: pattern
    selp = (mI * 6 > 1500) & (R['nn'] > 6) & (I.min(1) > 0.25 * mI)
    patf = IS.repattern_xy(I, Xp, pat0, selp)
    psi, m, _ = IS.pattern_psi(patf, cx, cy, dev)
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
    f, diag = IS.phase_field(Xp[cand, 0], Xp[cand, 1], R['w'][cand], phi, gphi / np.maximum(sphi, 0.05) ** 2, tile=a.tile, tblk=a.tblk)
    Phi0 = torch.tensor(f(Xp[:, 0], Xp[:, 1], R['w']), device=dev, dtype=torch.float32)
    tC = time.time() - t0 - tA
    print(f'{a.cell} {fi}: pattern psi {[np.round(patf[f"angle{i}"]["psi_deg"], 1).tolist() for i in range(2)]} m {patf["angle0"]["m_median"]:.2f}/{patf["angle1"]["m_median"]:.2f}'
          f' R {patf["angle0"]["R"]:.2f}/{patf["angle1"]["R"]:.2f} ({int(selp.sum())} emitters); phase field R {diag["R"]:.3f} ({int(gphi.sum())} used), {tC:.0f}s', flush=True)
    if a.regions:   # keep only the candidates inside the boxes for the final solve
        bx = [b for b in json.load(open(a.regions))['boxes'] if b.get('save', True)]
        inb = np.zeros(n, bool)
        for b in bx: inb |= (R['cx'] >= b['x0']) & (R['cx'] < b['x1']) & (R['cy'] >= b['y0']) & (R['cy'] < b['y1'])
        ii = np.nonzero(inb)[0]
        R = {k: v[ii] for k, v in R.items()}; n = len(ii); cx, cy = cx[ii], cy[ii]; Xp = Xp[ii]; Tsum = Tsum[ii]; I = I[ii]
        jt_ = torch.tensor(ii, device=dev); psi, m, Phi0, tdoff = psi[jt_], m[jt_], Phi0[jt_], tdoff[jt_]
        print(f'{a.cell} {fi}: {n} candidates inside the {len(bx)} boxes', flush=True)
    # ---- D: final solve
    cols = {k: [] for k in ('th', 'sig', 'chi2', 'kin', 'chi2_on', 'chi2_kin', 'N', 'ton', 'toff', 'bg', 'cs')}
    for i in range(0, n, a.batch):
        j = np.arange(i, min(i + a.batch, n)); oo, o = chunk_out(j); jt = torch.tensor(j, device=dev)
        arg = (Phi0[jt], tdoff[jt], m[jt], psi[jt])
        ron = sv.solve(oo, *arg, mode='on', iters=a.iters)
        o2 = dict(oo, z0=ron['th'][:, 2], xy0=ron['th'][:, :2])     # free / kinetics start at the on-through solution (one z start)
        # kinetics starts from the per-frame photon read-outs (matched-filter read-out summed over channels); no free solve
        k1, k2 = kin_starts({'th': torch.cat([ron['th'][:, :3], oo['Y'][..., 0].sum(2).clamp(min=0) / 0.85], 1)}, m[jt], psi[jt], sv)
        rk = best_of(sv.solve(o2, *arg, mode='kin', kin0=k1, iters=a.iters, zstarts=(0.0,)), sv.solve(o2, *arg, mode='kin', kin0=k2, iters=a.iters, zstarts=(0.0,)))
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
    g2 = good & (Tsum > 2000) & (Tsum < 4000)
    summ = dict(n=n, per_window=n / (nfr // 6), good_frac=float(good.mean()), kin_frac=float(C['kin'].mean()), field_R=diag['R'],
                chi2_dof_median=float(np.median(C['chi2'] / dof)), pattern={k: patf[k] for k in ('angle0', 'angle1')},
                sig_med_nm_2_4k=[float(np.median(out['sx'][g2]) * 128), float(np.median(out['sy'][g2]) * 128), float(np.median(out['sz'][g2]) * 1000)],
                coh=a.coh, t_read_head=tA, t_field=tC, t_total=time.time() - t0)
    return out, dict(M=diag['M'], d=diag['d']), summ


def merge_summary(U, keys, path):
    """merged summary json = its previous content + the per-file records (so that workers never rewrite each other's entry)"""
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(k): v for k, v in U.records(keys).items()})
    atomic_json(path, old, indent=1)


def main(argv=None):
    a = args(argv)
    from simcode.util.proctitle import auto; auto('4pi-real-w')
    files = sorted(file_index(f) for f in movie_files(a.cell)) if a.all else (a.files or [])
    if not files: raise SystemExit('give --files <numbers> or --all')
    U = Units(a.out, 'locs', pattern='{stage}_{cell}_{key:03d}_{head}{tag}.npz', cell=a.cell, head=a.head, tag=a.tag,
              claim=a.claim, redo=a.redo, min_free_gb=a.min_free_gb, title=f'4pi-w {a.cell}')
    fs = f'{a.out}/locs_{a.cell}_{a.head}{a.tag}_summary.json'
    ctx = None
    for fi in U.todo(files, order=a.order):
        if ctx is None: ctx = setup(a)
        rd = DcimgMovie(movie_file(a.cell, fi), layout=LAYOUT)
        nfr = a.nframes or rd.n
        out, fld, summ = process_file(rd, ctx, a, fi, nfr)
        fo = U.save(fi, **out, **fld)
        U.record(fi, summ); merge_summary(U, files, fs)
        print(f'{a.cell} {fi}: good {summ["good_frac"]:.3f}, kinetics {summ["kin_frac"]:.3f}, chi2/dof median {summ["chi2_dof_median"]:.2f}, sigma 2-4k e {np.round(summ["sig_med_nm_2_4k"], 2).tolist()} nm, total {summ["t_total"]:.0f}s -> {fo}', flush=True)
        del out, rd
    if ctx is not None: merge_summary(U, files, fs)


if __name__ == '__main__':
    main()
