"""Classical 4Pi-SIMFLUX on a simulated movie (analysis/fourpi/simmovie4pi.py format) with the SAME candidates as the
network (the front file of simfront4pi.py: w, cx, cy), patches cut from the movie exactly as simfront4pi does.
Pattern and 4Pi phase field are estimated from the data (no truth used), then evaluated against GT with
analysis/fourpi/simeval4pi.py's matching / CRLB. An oracle-Phi0 row isolates the phase-field error.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_classical/sim_cl.py --dir out/4pi/simmovie/d03 [--tile 6 --tblk 25]
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p_ in (REPO, f'{REPO}/analysis/fourpi', os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path: sys.path.insert(0, p_)
from simcode.fourpi.setup import P4
from simcode.fourpi.lut4pi import LUT4Pi
from simmovie4pi import reg_map, phi0_field, GAIN, OFF, PX
from simeval4pi import gt_units, match, crlb_units, LO, HI, EDGES
from cl4pi import Classic4Pi
import mov4pi_cl as M

P = 15


def patches(d, meta, F, dev):
    H, W = meta['H'], meta['W']; h = P // 2
    raw = np.load(f'{d}/movie.npy', mmap_mode='r')
    w, cx, cy = F['w'].astype(np.int64), F['cx'].astype(np.int64), F['cy'].astype(np.int64)
    imgs = np.zeros((len(w), 6, 4, P, P), np.float32); doff = np.zeros((len(w), 4, 2), np.float32)
    mxy = np.zeros((len(w), 4, 2))
    for c in range(4):
        mx, my = reg_map(c, cx.astype(float), cy.astype(float), H, W) if c else (cx.astype(float), cy.astype(float))
        mxy[:, c, 0], mxy[:, c, 1] = mx, my
    ctr = np.round(mxy); doff[:] = mxy - ctr
    order = np.argsort(w)
    for wv in np.unique(w):
        ii = order[np.searchsorted(w[order], wv):np.searchsorted(w[order], wv, 'right')]
        r = np.asarray(raw[6 * wv:6 * wv + 6]).astype(np.float32)
        cr = np.stack([r[:, :, 0:W], r[:, :, W:2 * W], r[:, :, 2 * W:3 * W][:, :, ::-1], r[:, :, 3 * W:][:, :, ::-1]], 1)
        E = (cr - OFF) * GAIN                                                   # [6,4,H,W]
        for c in range(4):
            ix = (ctr[ii, c, 0] - h).astype(int).clip(0, W - P); iy = (ctr[ii, c, 1] - h).astype(int).clip(0, H - P)
            for q, (a_, b_) in enumerate(zip(ix, iy)):
                imgs[ii[q], :, c] = E[:, c, b_:b_ + P, a_:a_ + P]
    assert np.abs(doff - F['doff']).max() < 1e-3, 'doff mismatch with the front file'
    return imgs, doff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=f'{P4}/simmovie/d03'); ap.add_argument('--model', default='v2')
    ap.add_argument('--tile', type=float, default=6.0); ap.add_argument('--tblk', type=int, default=25)
    ap.add_argument('--chunk', type=int, default=1024); ap.add_argument('--out', default='')
    ap.add_argument('--front', default=''); ap.add_argument('--field', default='poly'); ap.add_argument('--order', type=int, default=3); ap.add_argument('--scale', type=float, default=1.0)
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('4pi-classic')
    dev = 'cuda'; d = a.dir; tag = os.path.basename(d.rstrip('/'))
    out = a.out or f'{P4}/classical/sim/{tag}_{a.field}'; os.makedirs(out, exist_ok=True)
    meta = json.load(open(f'{d}/meta.json')); H, W, T = meta['H'], meta['W'], meta['T']
    F = dict(np.load(a.front or f'{d}/front_SIM_000_{a.model}.npz'))
    st = json.load(open(f'{P4}/psf4pi/{meta["psf"]}.json'))['psf']
    lut = LUT4Pi(st, P=P, dev=dev)
    c4 = Classic4Pi(lut, delta=(0.0, math.radians(meta['delta_deg'])), dev=dev)
    t0 = time.time()
    imgs, doff = patches(d, meta, F, dev)
    n = len(imgs); print(f'{tag}: {n} candidates, patches {time.time() - t0:.0f}s', flush=True)
    A = {}
    for i in range(0, n, a.chunk):
        r = M.stage_a(c4, torch.tensor(imgs[i:i + a.chunk], device=dev), torch.tensor(doff[i:i + a.chunk], device=dev))
        for k, v in r.items(): A.setdefault(k, []).append(v)
    A = {k: np.concatenate(v) for k, v in A.items()}
    cx, cy, w = F['cx'].astype(np.float64), F['cy'].astype(np.float64), F['w'].astype(np.int64)
    X, Y = cx + A['x0'], cy + A['y0']
    print(f'stage A {time.time() - t0:.0f}s', flush=True)
    # pattern from the data
    fitok = A['conv'] & ~A['at_bound'] & (A['chi2px'] < 4) & (A['llr'] < M.FILTERS['llr_max'])
    psel = fitok & (A['Nsum'] > 1000) & (A['R'] < 0.15) & (F['nn'] > 5)
    kref = [[3.8387, -0.0238], [0.0031, 3.8511]]      # nominal (MT grating), refined on the data
    pat = M.estimate_pattern(A, X, Y, kref, psel)
    lat = M.lateral_np(A, pat, cx, cy, dev, c4)
    pat['m'] = float(np.median(np.clip(lat['MD'][psel], 0, 1.5)))
    print('pattern', json.dumps({k: (v if not isinstance(v, dict) else {kk: np.round(vv, 3).tolist() for kk, vv in v.items()}) for k, v in pat.items()}), flush=True)
    print('  truth k', meta['k'], 'psi', meta['psi_deg'], 'm', meta['m'], flush=True)
    # 4Pi phase field
    fsel = fitok & (A['Nsum'] > 1500) & (A['sz_ast'] < 0.06) & (A['C_a'] > 0.4) & (A['C_a'] < 1.2) & (F['nn'] > 4)
    f, s, diag = M.field_fit(c4, A, X, Y, w, fsel, a.tile, a.tblk, H, W, mode=a.field, order=a.order, scales=(a.scale,))
    Phi0 = f(X, Y, w)
    Ptrue = phi0_field(X, Y, 6 * w + 3, H, W, T)
    ferr = np.degrees(np.angle(np.exp(1j * (Phi0 - Ptrue))))
    print(f'field: scale {s}, R {diag["R"]:.3f} ({fsel.sum()} used); field error vs truth: mean {np.mean(ferr):.1f} deg, '
          f'circ sd {np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * np.radians(ferr - np.mean(ferr)))))))):.1f} deg', flush=True)
    z4a = M.z4pi(c4, A, Phi0, s)
    B, Bo = {}, {}
    for i in range(0, n, a.chunk):
        sl = slice(i, i + a.chunk)
        im = torch.tensor(imgs[sl], device=dev); df = torch.tensor(doff[sl], device=dev)
        As = {k: v[sl] for k, v in A.items()}; Ls = {k: v[sl] for k, v in lat.items()}
        r = M.stage_b(c4, im, df, As, Ls, Phi0[sl], z4a[sl], s, pat, cx[sl], cy[sl])
        for k, v in r.items(): B.setdefault(k, []).append(v)
        ro = M.stage_b(c4, im, df, As, Ls, Ptrue[sl], M.z4pi(c4, As, Ptrue[sl], s), s, pat, cx[sl], cy[sl], do_pix=False)
        for k, v in ro.items(): Bo.setdefault(k, []).append(v)
    B = {k: np.concatenate(v) for k, v in B.items()}; Bo = {k: np.concatenate(v) for k, v in Bo.items()}
    print(f'stage B {time.time() - t0:.0f}s', flush=True)
    Tb = dict(A, **{k: v for k, v in lat.items() if k in ('MD', 'alpha', 'salpha')}, sxs=lat['sxy'][:, 0], sys=lat['sxy'][:, 1], **B)
    fl = M.flags(Tb)
    keep, rows = M.flag_table(fl)
    pix_rej = ~B['convp'] | B['boundp'] | (B['chi2p'] > 4)
    tab = dict(w=w, cx=cx, cy=cy, x=cx + lat['xy'][:, 0], y=cy + lat['xy'][:, 1], z=B['z4'], sx=lat['sxy'][:, 0], sy=lat['sxy'][:, 1], sz=B['sz4'],
               xp=cx + B['xp'], yp=cy + B['yp'], zp=B['zp'], sxp=B['sxp'], syp=B['syp'], szp=B['szp'], x0=X, y0=Y, z_ast=A['z_ast'],
               z_oracle=Bo['z4'], N=A['Nsum'], keep=keep, keep_pix=keep & ~pix_rej, Phi0=Phi0, Phi0_true=Ptrue,
               **{('flag_' + str(j)): m for j, m in enumerate(fl.values())})
    np.savez(f'{out}/cl_{tag}.npz', **tab, **{('A_' + k): v for k, v in A.items()}, **{('B_' + k): v for k, v in B.items()})
    # ---- GT evaluation
    U = gt_units(d, T)
    inr = lambda x, y: (x >= LO) & (x < HI) & (y >= LO) & (y < HI)
    gsel = inr(U['x'], U['y']) & (U['N'] >= 300)
    G = {k: v[gsel] for k, v in U.items()}
    cr = crlb_units(G, np.arange(len(G['x'])), meta, lut, dev) * np.array([PX * 1000, PX * 1000, 1000])
    sets = {'classical published-style (SIMFLUX two-step + 4Pi phase z), filtered': (tab['x'], tab['y'], tab['z'], keep),
            'classical published-style, unfiltered': (tab['x'], tab['y'], tab['z'], np.ones(n, bool)),
            'classical published-style, filtered, ORACLE phase field (z only)': (tab['x'], tab['y'], tab['z_oracle'], keep),
            'classical 24-image pixel MLE, filtered (+ own chi2/bound/conv)': (tab['xp'], tab['yp'], tab['zp'], tab['keep_pix']),
            'classical 24-image pixel MLE, unfiltered': (tab['xp'], tab['yp'], tab['zp'], np.ones(n, bool)),
            'classical summed fit only (x0, y0, z_ast), filtered': (X, Y, A['z_ast'], keep)}
    res = dict(tag=tag, n_cand=n, pattern=pat, field=dict(scale=s, R=diag['R'], err_mean_deg=float(np.mean(ferr)),
               err_rsd_deg=float(1.4826 * np.median(np.abs(ferr - np.median(ferr))))), filters=rows, kept=float(keep.mean()), methods={})
    L = [f'# {tag}: {meta["dens_actual"]:.2f} emitters/um^2/frame, {n} candidates (network front, shared detection)',
         f'pattern est: k0 {np.round(pat["angle0"]["k"], 4)}, k1 {np.round(pat["angle1"]["k"], 4)}, psi0 {np.round(pat["angle0"]["psi_deg"], 1)}, '
         f'psi1 {np.round(pat["angle1"]["psi_deg"], 1)}, m {pat["m"]:.3f} (truth k {meta["k"]}, psi {meta["psi_deg"]}, m {meta["m"]})',
         f'phase field: z_ast scale {s}, R {diag["R"]:.3f}, error vs truth mean {np.mean(ferr):.1f} deg, robust sd {res["field"]["err_rsd_deg"]:.1f} deg',
         '', 'filters (fraction of candidates removed: alone | sequential):']
    for k, al, sq in rows: L.append(f'  {k}: {al:.3f} | {sq:.3f}')
    L.append(f'  kept {keep.mean():.3f}; pixel MLE additionally rejects {np.mean(pix_rej & keep):.3f}')
    for nm, (x, y, z, ok) in sets.items():
        ok = ok & inr(x, y) & np.isfinite(z)
        mg, ml = match(G['w'], G['x'], G['y'], w[ok], x[ok], y[ok])
        mm = mg >= 0
        ex = (x[ok][mg[mm]] - G['x'][mm]) * PX * 1000; ey = (y[ok][mg[mm]] - G['y'][mm]) * PX * 1000; ez = (z[ok][mg[mm]] - G['z'][mm]) * 1000
        bias = [float(np.median(e)) for e in (ex, ey, ez)]
        prec = float((ml >= 0).mean()) if len(ml) else 0.0
        L.append(f'\n{nm}: {ok.sum()} locs, precision {prec:.3f}, offset x/y/z {bias[0]:.1f}/{bias[1]:.1f}/{bias[2]:.1f} nm')
        rws = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sg = (G['N'] >= lo) & (G['N'] < hi); sm = sg[mm]
            if sm.sum() < 30: continue
            rb = lambda e: float(1.4826 * np.median(np.abs(e[sm] - np.median(e[sm]))))
            gz = sm & (np.abs(ez) <= 100)
            r = dict(lo=lo, hi=hi, n_gt=int(sg.sum()), recall=float(mm[sg].mean()), x_rob=rb(ex), y_rob=rb(ey), z_rob=rb(ez),
                     z_rms=float(np.sqrt(np.mean((ez[gz] - bias[2]) ** 2))), fringe=float(np.mean(np.abs(ez[sm]) > 100)),
                     crlb=[float(np.sqrt(np.mean(cr[mm][sm][:, q] ** 2))) for q in range(3)])
            rws.append(r)
            L.append(f"  {lo}-{hi} e: n {r['n_gt']} recall {r['recall']:.2f} | robust x/y/z {r['x_rob']:.1f}/{r['y_rob']:.1f}/{r['z_rob']:.1f} "
                     f"| z RMS (no fringe err) {r['z_rms']:.1f} | CRLB {r['crlb'][0]:.1f}/{r['crlb'][1]:.1f}/{r['crlb'][2]:.1f} | fringe {r['fringe']:.3f}")
        res['methods'][nm] = dict(n_locs=int(ok.sum()), precision=prec, bias_nm=bias, rows=rws)
    json.dump(res, open(f'{out}/eval_cl_{tag}.json', 'w'), indent=1, default=float)
    open(f'{out}/eval_cl_{tag}.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
