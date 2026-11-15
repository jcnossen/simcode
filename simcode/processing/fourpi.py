"""4Pi-SIMCODE processing of the files of a 4Pi acquisition (the process stage of the fourpi / fourpi_plain set-ups,
simcode/setups/fourpi.py): the learned-window emitter network (v10 ...), one read per file, all candidate patches in host
memory. Network intensities -> explicit solve; no pixel fit. Windows of T frames (fourpi.window: 4Pi-SIMFLUX 6 = 2
angles x 3 steps; plain 4Pi 1); fourpi.pattern = false (plain 4Pi): no SIM pattern (stage B skipped, pattern factor 1).

Per file (process_file):
  A. read (prefetched in a background thread) -> camera offset / gain -> 4Pi detector candidates on the registered
     4-channel sum (T planes per window; the file's registration map) -> unwarped P x P channel patches (host RAM,
     float16; never written to disk) -> emitter network: prior, intensities
  B. SIM pattern of the file from the per-frame photon read-outs at the prior positions (slot DFT; patterns only)
  C. 4Pi phase field: per-candidate phase (z held by the prior) on bright isolated candidates -> tile map + time blocks
  D. per chunk from RAM: network (intensities Y, windows W', covariance) -> explicit solve with the window model and a
     residual background per channel: on-through vs kinetics (T > 1; kept if chi2 drops by > dnll; no free fallback)

Files are the units of the shared work driver (simcode/io/work.py): finished files are skipped, outputs are written
atomically, the disk is checked before a file starts, and with --claim 1 several workers (one per GPU, python -m
simcode.io.launch) share the files.

  python -m simcode.processing.fourpi <config_resolved.json | acquisition config> [--files 5 6] [--nframes 3600]
         [--claim 1] [--redo 1] [--out DIR] [--set fourpi.coh=1.56 ...]
Output: <out>/process/locs_<file>.npz (<out> of the config, or --out), a record locs_<file>.json per file and the merged
locs_summary.json. Normally started by the pipeline: python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --only fourpi_MT
"""
import argparse, json, os, time
from types import SimpleNamespace
import numpy as np, torch
from simcode.channels.front import nn_dist
from simcode.processing import fourpi_pattern_phase as PP
from simcode.solve.fourpi import FourPiSolve, kin_starts, best_of
from simcode.io.movie import Prefetch
from simcode.io.work import Units, atomic_json

dev = 'cuda'


def settings(c):
    """the processing settings of a resolved config: its 'fourpi' section + the acquisition name (log prefix)"""
    return SimpleNamespace(**c['fourpi'], cell=c['name'])


def setup_of(c):
    from simcode.setups import get
    return get(c['setup'])


def context(c):
    """networks and solver of an acquisition (the same for every file)"""
    from simcode.emitter.fourpi import load as load_emitter
    from simcode.detector.model import load as dload
    from simcode.detector.select import select_points as dselect
    f = c['fourpi']
    net, ck = load_emitter(f['emitter'])
    det, _ = dload(f['detector'])
    if net.T != f['window'] or det.T != f['window']:
        raise ValueError(f"{c['name']}: networks for {net.T} / {det.T} frames, config fourpi.window = {f['window']}")
    pat0 = json.load(open(f['pattern_ref'])) if f['pattern'] else None
    sv = FourPiSolve(net.lut, net.wins, **setup_of(c).solve_kw(c, net)); sv.tab_lim = None; sv.coh_scale = f['coh']
    return dict(net=net, det=det, dselect=dselect, pat0=pat0, sv=sv, px_nm=c['optics']['pixel_um'] * 1000)


def process_file(rd, fr, ctx, a, fi, nfr):
    """one file: rd = channel reader (read(f0, f1) -> [n, 4, H, W] ADU), fr = its ChannelFront (camera, registration),
    nfr frames -> (table dict, phase-field arrays dict(M, d), summary dict)"""
    net, det, dselect, pat0, sv = (ctx[k] for k in ('net', 'det', 'dselect', 'pat0', 'sv'))
    P, XLIM, H, W = fr.P, fr.xlim, fr.H, fr.W
    T = a.window
    h = P // 2
    plan = [(a0, min(a0 + a.chunk, nfr)) for a0 in range(0, nfr, a.chunk)]
    pf = Prefetch(rd, plan)
    acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'img', 'dxy', 'z', 'ls', 'I', 'pdet', 'nn')}
    t0 = time.time()
    # ---- A: read, detect, patches, network (prior + photon read-outs)
    for a0, a1 in plan:
        raw = pf.read(a0, a1)
        with torch.no_grad():
            E = fr.to_e(raw); nW = E.shape[0] // T
            if nW == 0: continue
            Rs = fr.registered_sum(E[:nW * T])
            Rf = Rs.reshape(nW, T, H, W); ccs = []
            for b0 in range(0, nW, a.det_batch):             # detector in batches of det_batch windows (memory)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dout = det(Rf[b0:b0 + a.det_batch])
                c1 = dselect({k: v.float() for k, v in dout.items()}, k=2048, p_min=a.p_min)
                c1['b'] = c1['b'] + b0; ccs.append(c1)
            cc = {k: torch.cat([c1[k] for c1 in ccs]) for k in ('b', 'iy', 'ix', 'logit')}
            okc = torch.isfinite(cc['logit']) & (cc['ix'] >= XLIM[0]) & (cc['ix'] < XLIM[1]) & (cc['iy'] >= h + 1) & (cc['iy'] < H - h - 1)
            w, cy, cx, pd = cc['b'][okc], cc['iy'][okc], cc['ix'][okc], torch.sigmoid(cc['logit'][okc])
            if len(w) == 0: continue
            dd = nn_dist(w, cx, cy)
            img, doff, ok = fr.patches(E, w, cy, cx, window=T)
            sel = torch.nonzero(ok)[:, 0]
            img = img[sel]; doff = doff[sel]
            outs = [net(img[i:i + a.batch], doff[i:i + a.batch]) for i in range(0, len(sel), a.batch)]
            o = {k: torch.cat([u[k] for u in outs]) for k in ('Y', 'dxy', 'z', 'ls')}
            kp = o['Y'][..., 0].sum((1, 2)) >= a.min_photons
            o = {k: v[kp] for k, v in o.items()}; img = img[kp]; doff = doff[kp]; sel = sel[kp]
        for k, v in (('w', w[sel] + a0 // T), ('cx', cx[sel]), ('cy', cy[sel]), ('doff', doff), ('img', img.half()), ('dxy', o['dxy']), ('z', o['z']),
                     ('ls', o['ls']), ('I', o['Y'][..., 0].sum(2)), ('pdet', pd[sel]), ('nn', dd[sel])):
            acc[k].append(v.cpu().numpy())
    pf.close()
    R = {k: np.concatenate(v) for k, v in acc.items()}; del acc
    n = len(R['w']); cx, cy = R['cx'].astype(np.float64), R['cy'].astype(np.float64)
    tA = time.time() - t0
    print(f'{a.cell} {fi}: {n} candidates ({n / (nfr // T):.1f} per window), patches {R["img"].nbytes / 1e9:.2f} GB in RAM, read + network {tA:.0f}s', flush=True)
    I = R['I']; mI = I.mean(1); Tsum = I.sum(1)
    Xp = np.stack([cx + R['dxy'][:, 0], cy + R['dxy'][:, 1]], 1)
    # ---- B: pattern
    if a.pattern:
        selp = (mI * 6 > 1500) & (R['nn'] > 6) & (I.min(1) > 0.25 * mI)
        patf = PP.repattern_xy(I, Xp, pat0, selp)
        psi, m, _ = PP.pattern_psi(patf, cx, cy, dev)
        sv.k2 = tuple(tuple(patf[f'angle{i}']['k']) for i in range(2))
    else:   # plain 4Pi: pattern factor 1 (the solve ignores m, psi)
        psi = m = torch.zeros(n, T, device=dev); patf = None
    tdoff = torch.tensor(R['doff'], device=dev)

    def chunk_out(j):
        with torch.no_grad():
            o = net(torch.tensor(R['img'][j].astype(np.float32), device=dev), tdoff[j])
        return dict(Y=o['Y'], Lb=o['Lb'], dxy=o['dxy'], z=o['z'], ls=o['ls'], W=o['W'], Wsum=o['Wsum'],
                    sb=torch.full((len(j), 4), a.sb, device=dev)), o

    # ---- C: phase field from bright isolated candidates
    cand = np.nonzero((Tsum > 800) & (R['nn'] > 4))[0]
    if len(cand) > a.nphi: cand = np.sort(np.random.default_rng(0).choice(cand, a.nphi, replace=False))
    phi = np.zeros(len(cand)); sphi = np.zeros(len(cand))
    for i in range(0, len(cand), a.batch):
        j = cand[i:i + a.batch]; oo, _ = chunk_out(j); jt = torch.tensor(j, device=dev)
        rp = sv.solve(oo, torch.zeros(len(j), device=dev), tdoff[jt], m[jt], psi[jt], mode='free', fit_phi=True, iters=a.iters, Lam=a.z_period_um)
        phi[i:i + len(j)] = rp['th'][:, -5].cpu().numpy(); sphi[i:i + len(j)] = rp['sig'][:, -5].cpu().numpy()
    gphi = sphi < 0.8
    f, diag = PP.phase_field(Xp[cand, 0], Xp[cand, 1], R['w'][cand], phi, gphi / np.maximum(sphi, 0.05) ** 2, tile=a.tile, tblk=a.tblk, H=H, Wd=W)
    Phi0 = torch.tensor(f(Xp[:, 0], Xp[:, 1], R['w']), device=dev, dtype=torch.float32)
    tC = time.time() - t0 - tA
    pmsg = (f'pattern psi {[np.round(patf[f"angle{i}"]["psi_deg"], 1).tolist() for i in range(2)]} m {patf["angle0"]["m_median"]:.2f}/{patf["angle1"]["m_median"]:.2f}'
            f' R {patf["angle0"]["R"]:.2f}/{patf["angle1"]["R"]:.2f} ({int(selp.sum())} emitters); ') if a.pattern else ''
    print(f'{a.cell} {fi}: {pmsg}phase field R {diag["R"]:.3f} ({int(gphi.sum())} used), {tC:.0f}s', flush=True)
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
        ron = sv.solve(oo, *arg, mode='on', iters=a.iters, Lam=a.z_period_um)
        if T > 1:
            o2 = dict(oo, z0=ron['th'][:, 2], xy0=ron['th'][:, :2])     # free / kinetics start at the on-through solution (one z start)
            # kinetics starts from the per-frame photon read-outs (matched-filter read-out summed over channels); no free solve
            k1, k2 = kin_starts({'th': torch.cat([ron['th'][:, :3], oo['Y'][..., 0].sum(2).clamp(min=0) / 0.85], 1)}, m[jt], psi[jt], sv)
            rk = best_of(sv.solve(o2, *arg, mode='kin', kin0=k1, iters=a.iters, zstarts=(0.0,)), sv.solve(o2, *arg, mode='kin', kin0=k2, iters=a.iters, zstarts=(0.0,)))
        else:   # one frame: no kinetics model (the on fraction is part of F)
            nan = torch.full((len(j),), float('nan'), device=dev)
            rk = dict(th=torch.cat([ron['th'][:, :4], nan[:, None], nan[:, None], ron['th'][:, 4:]], 1), sig=ron['sig'], chi2=nan)
        usek = (ron['chi2'] - rk['chi2']) > a.dnll
        pick = lambda A, B: torch.where(usek.reshape(-1, *([1] * (A.dim() - 1))), A, B)
        tw = (torch.minimum(rk['th'][:, 5], torch.tensor(float(T), device=dev)) - torch.maximum(rk['th'][:, 4], torch.tensor(0.0, device=dev))).clamp(min=0)
        for k, v in (('th', pick(rk['th'][:, :3], ron['th'][:, :3])), ('sig', pick(rk['sig'][:, :3], ron['sig'][:, :3])), ('chi2', pick(rk['chi2'], ron['chi2'])),
                     ('kin', usek), ('chi2_on', ron['chi2']), ('chi2_kin', rk['chi2']), ('N', torch.where(usek, rk['th'][:, 3] * tw, ron['th'][:, 3] * T)),
                     ('ton', rk['th'][:, 4]), ('toff', rk['th'][:, 5]), ('bg', o['bgc'] + pick(rk['th'][:, -4:], ron['th'][:, -4:])), ('cs', o['cs'])):
            cols[k].append(v.cpu().numpy())
        if (i // a.batch) % 10 == 0:
            print(f'{a.cell} {fi}: solved {i + len(j)}/{n}, {time.time() - t0:.0f}s', flush=True)
    C = {k: np.concatenate(v) for k, v in cols.items()}
    nY = T * 4 * net.K; dof = np.where(C['kin'], nY - 10.0, nY - 8.0); good = C['chi2'] / dof < a.chi2_max
    out = dict(x=cx + C['th'][:, 0], y=cy + C['th'][:, 1], z=C['th'][:, 2], sx=C['sig'][:, 0], sy=C['sig'][:, 1], sz=C['sig'][:, 2], N=C['N'],
               on=np.ones(n, bool), kin=C['kin'], ton=C['ton'], toff=C['toff'], chi2=C['chi2'], dof=dof, good=good, chi2_on=C['chi2_on'], chi2_free=C['chi2_kin'],
               Phi0=Phi0.cpu().numpy(), zprior=R['z'], szprior=np.exp(R['ls'][:, 2]), xprior=Xp[:, 0], yprior=Xp[:, 1], cx=R['cx'], cy=R['cy'],
               w=R['w'], frame=R['w'] * T, nn=R['nn'], Tsum=Tsum, pdet=R['pdet'], bg=C['bg'], cs=C['cs'], file_index=fi)
    g2 = good & (Tsum > 2000) & (Tsum < 4000)
    px = ctx['px_nm']
    summ = dict(n=n, per_window=n / (nfr // T), good_frac=float(good.mean()), kin_frac=float(C['kin'].mean()), field_R=diag['R'],
                chi2_dof_median=float(np.median(C['chi2'] / dof)), pattern={k: patf[k] for k in ('angle0', 'angle1')} if a.pattern else None,
                sig_med_nm_2_4k=[float(np.median(out['sx'][g2]) * px), float(np.median(out['sy'][g2]) * px), float(np.median(out['sz'][g2]) * 1000)],
                coh=a.coh, t_read_network=tA, t_field=tC, t_total=time.time() - t0)
    return out, dict(M=diag['M'], d=diag['d']), summ


def output_units(c, out=None, claim=None, redo=None):
    """the file driver of the process stage: <out>/process/locs_<file><tag>.npz"""
    f = c['fourpi']
    return Units(out or os.path.join(c['out'], 'process'), 'locs', pattern='{stage}_{key:03d}{tag}.npz', tag=f['tag'],
                 claim=f['claim'] if claim is None else claim, redo=f['redo'] if redo is None else redo,
                 min_free_gb=f['min_free_gb'], title=f'4pi {c["name"]}')


def merge_summary(U, keys, path):
    """merged summary json = its previous content + the per-file records (so that workers never rewrite each other's entry)"""
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(k): v for k, v in U.records(keys).items()})
    atomic_json(path, old, indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('config', help='resolved config json of the pipeline (<out>/config_resolved.json) or an acquisition config')
    p.add_argument('--only', default='', help='acquisition name (a config file with several acquisitions)')
    p.add_argument('--files', type=int, nargs='+', default=None, help='file numbers (default: fourpi.files of the config)')
    p.add_argument('--nframes', type=int, default=None)
    p.add_argument('--claim', type=int, default=None); p.add_argument('--redo', type=int, default=None)
    p.add_argument('--out', default=None, help='output directory (default <out>/process)')
    p.add_argument('--set', nargs='*', default=[], help='overrides of the fourpi section: key=value (json values)')
    a = p.parse_args(argv)
    from simcode.util.proctitle import auto; auto('4pi-process')
    from simcode.pipeline import acquisition_config as AC
    L = AC.load_all(a.config)
    if a.only: L = [c for c in L if c['name'] == a.only]
    if len(L) != 1: raise SystemExit(f'{a.config}: {len(L)} acquisitions; give --only <name>')
    c = L[0]
    for kv in a.set:
        k, v = kv.split('=', 1); c['fourpi'][k] = json.loads(v)
    if a.nframes is not None: c['fourpi']['nframes'] = a.nframes
    S = setup_of(c)
    files = a.files or S.files(c)
    U = output_units(c, a.out, a.claim, a.redo)
    fs = os.path.join(U.out, f'locs{c["fourpi"]["tag"]}_summary.json')
    st = settings(c); ctx = None
    for fi in U.todo(files, order=c['fourpi']['order']):
        if ctx is None: ctx = context(c)
        rd = S.reader(c, S.movie_file(c, fi))
        fr = S.front(c, fi, dev)
        nfr = c['fourpi']['nframes'] or rd.n
        out, fld, summ = process_file(rd, fr, ctx, st, fi, nfr)
        fo = U.save(fi, **out, **fld)
        U.record(fi, summ); merge_summary(U, files, fs)
        print(f'{st.cell} {fi}: good {summ["good_frac"]:.3f}, kinetics {summ["kin_frac"]:.3f}, chi2/dof median {summ["chi2_dof_median"]:.2f}, sigma 2-4k e {np.round(summ["sig_med_nm_2_4k"], 2).tolist()} nm, total {summ["t_total"]:.0f}s -> {fo}', flush=True)
        del out, rd
    if ctx is not None: merge_summary(U, files, fs)


if __name__ == '__main__':
    main()
