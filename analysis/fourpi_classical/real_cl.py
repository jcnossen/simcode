"""Classical 4Pi-SIMFLUX on the real movies, per file.

Sources (same candidates as the network either way):
  --src patches : out/4pi/patches/<cell>/patches_<file>.npy + meta_<file>.npz (v3 pass-1 candidates; README there),
                  camera offset map out/4pi/patches/camera_<cell>.npz; streamed (mmap), read twice (stage A, stage B).
  --src dcimg   : a short direct read (--nframes) through analysis/fourpi/front4pi.Front (identical detection and
                  patch cut as the network's pass 1); patches kept in RAM.
Per file: stage A (summed fit, per-frame counts, moment phase) -> pattern from the data -> SIMFLUX two-step ->
4Pi phase field (cubic XY polynomial + tile residual + block drift) -> z_4pi -> z_ast scale check (regression of z_ast
on z_4pi; if off by > 5 %, redo the field with the corrected scale) -> stage B (calibration at z_4pi) -> filters ->
pixel MLE on the kept rows.
Output: out/4pi/classical/real/cl_<cell>_<file>.npz (+ _summary.json)
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_classical/real_cl.py --cell NPC --files 10 --src dcimg --nframes 3600
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p_ in (REPO, f'{REPO}/analysis/fourpi', os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path: sys.path.insert(0, p_)
from simcode.fourpi.setup import P4
from simcode.fourpi.lut4pi import LUT4Pi
from cl4pi import Classic4Pi
import mov4pi_cl as M

OUT = f'{P4}/classical/real'
PDIR = f'{P4}/patches'
DELTA = {'MT': math.radians(-10.4), 'NPC': math.radians(-13.9)}
GAIN = 0.23
P = 15


def nn_dist(w, cx, cy):
    """px to the nearest other candidate in the same window"""
    out = np.full(len(w), 99.0)
    o = np.argsort(w, kind='stable'); ws = w[o]
    b = np.r_[0, np.nonzero(np.diff(ws))[0] + 1, len(ws)]
    for i0, i1 in zip(b[:-1], b[1:]):
        ii = o[i0:i1]
        if len(ii) < 2: continue
        D = np.hypot(cx[ii, None] - cx[None, ii], cy[ii, None] - cy[None, ii]); np.fill_diagonal(D, 99)
        out[ii] = D.min(1)
    return out


class PatchSource:
    def __init__(self, cell, fi, dev):
        self.fn = f'{PDIR}/{cell}/patches_{fi:03d}.npy'
        self.meta = dict(np.load(f'{PDIR}/{cell}/meta_{fi:03d}.npz'))
        self.off = torch.tensor(np.load(f'{PDIR}/camera_{cell}.npz')['offset_adu'], device=dev, dtype=torch.float32)
        self.dev = dev
        self.n = len(self.meta['w'])

    def info(self):
        m = self.meta
        return m['w'].astype(np.int64), m['cx'].astype(np.float64), m['cy'].astype(np.float64), m['doff'].astype(np.float32)

    def chunks(self, size, rows=None):
        arr = np.load(self.fn, mmap_mode='r')
        idx = np.arange(self.n) if rows is None else rows
        o = torch.arange(P, device=self.dev)
        for i in range(0, len(idx), size):
            j = idx[i:i + size]
            adu = torch.tensor(np.asarray(arr[j]).astype(np.float32), device=self.dev)          # [n,6,4,P,P]
            cor = torch.tensor(self.meta['corner'][j].astype(np.int64), device=self.dev)         # [n,4,2]
            Y = cor[:, :, 1, None] + o[None, None]; X = cor[:, :, 0, None] + o[None, None]       # [n,4,P]
            c = torch.arange(4, device=self.dev)
            off = self.off[c[None, :, None, None], Y[:, :, :, None], X[:, :, None, :]]           # [n,4,P,P]
            yield j, (adu - off[:, None]) * GAIN, torch.tensor(self.meta['doff'][j], device=self.dev)


class DcimgSource:
    def __init__(self, cell, fi, nframes, dev, chunk=600):
        from dc import Dcimg, movie_files, file_index
        from front4pi import Front
        fn = [f for f in movie_files(cell) if file_index(f) == fi][0]
        fr = Front(cell, dev); d = Dcimg(fn)
        imgs, doffs, ws, cxs, cys = [], [], [], [], []
        for a0 in range(0, nframes, chunk):
            with torch.no_grad():
                E = fr.to_e(d.raw(a0, min(chunk, nframes - a0)))
                w, cy, cx, dog = fr.detect(E)
                img, doff, ok = fr.patches(E, w, cy, cx)
            s = torch.nonzero(ok)[:, 0]
            imgs.append(img[s].cpu().numpy().astype(np.float32)); doffs.append(doff[s].cpu().numpy().astype(np.float32))
            ws.append((w[s] + a0 // 6).cpu().numpy()); cxs.append(cx[s].cpu().numpy()); cys.append(cy[s].cpu().numpy())
            del E
        self.img = np.concatenate(imgs); self.doff = np.concatenate(doffs)
        self.w = np.concatenate(ws).astype(np.int64); self.cx = np.concatenate(cxs).astype(np.float64); self.cy = np.concatenate(cys).astype(np.float64)
        self.n = len(self.w); self.dev = dev

    def info(self):
        return self.w, self.cx, self.cy, self.doff

    def chunks(self, size, rows=None):
        idx = np.arange(self.n) if rows is None else rows
        for i in range(0, len(idx), size):
            j = idx[i:i + size]
            yield j, torch.tensor(self.img[j], device=self.dev), torch.tensor(self.doff[j], device=self.dev)


def process(c4, src, cell, fi, a, dev, A=None, pix_mask=None, extra=None, tA=None):
    t0 = time.time()
    w, cx, cy, doff = src.info(); n = len(w)
    nn = nn_dist(w, cx, cy)
    if A is None:
        A = {}
        for j, img, df in src.chunks(a.chunk):
            r = M.stage_a(c4, img, df)
            for k, v in r.items(): A.setdefault(k, []).append(v)
        A = {k: np.concatenate(v) for k, v in A.items()}
        tA = time.time() - t0
    else:
        t0 = t0 - (tA or 0.0)
    X, Y = cx + A['x0'], cy + A['y0']
    fitok = A['conv'] & ~A['at_bound'] & (A['chi2px'] < 4) & (A['llr'] < M.FILTERS['llr_max'])
    # pattern
    ref = json.load(open(f'{P4}/smlm/diag/pattern_{cell}.json'))
    kref = [ref['angle0']['k'], ref['angle1']['k']]
    psel = fitok & (A['Nsum'] > 1500) & (A['R'] < 0.15) & (nn > 5)
    pat = M.estimate_pattern(A, X, Y, kref, psel)
    lat = M.lateral_np(A, pat, cx, cy, dev, c4)
    pat['m'] = float(np.median(np.clip(lat['MD'][psel], 0, 1.5)))
    for aa in range(2):
        ps = np.array(pat[f'angle{aa}']['psi_deg'])
        pat[f'angle{aa}']['steps_deg'] = [float((ps[0] - ps[1]) % 360), float((ps[1] - ps[2]) % 360)]
    # 4Pi phase field (iterate the z_ast scale once if needed)
    fsel = fitok & (A['Nsum'] > 1500) & (A['sz_ast'] < 0.06) & (A['C_a'] > 0.4) & (A['C_a'] < 1.6) & (nn > 4)
    s = 1.0; hist = []
    for it in range(3):
        f, _, diag = M.field_fit(c4, A, X, Y, w, fsel, a.tile, a.tblk, 200, 360, mode=a.field, order=a.order, scales=(s,))
        Phi0 = f(X, Y, w)
        z4 = M.z4pi(c4, A, Phi0, s)
        g = fsel & (np.abs(A['z_ast'] - z4) < 0.1)
        slope = float(np.polyfit(z4[g], A['z_ast'][g], 1)[0]) if g.sum() > 100 else 1.0
        hist.append(dict(scale=s, R=diag['R'], slope_zast_on_z4=slope, n=int(fsel.sum())))
        if abs(slope - 1) < 0.05 or it == 2: break
        s = s / slope
    # stage B (calibration at z4) for all
    B = {}
    for j, img, df in src.chunks(a.chunk):
        As = {k: v[j] for k, v in A.items()}; Ls = {k: v[j] for k, v in lat.items()}
        r = M.stage_b(c4, img, df, As, Ls, Phi0[j], z4[j], s, pat, cx[j], cy[j], do_pix=False)
        for k, v in r.items(): B.setdefault(k, []).append(v)
    B = {k: np.concatenate(v) for k, v in B.items()}
    Tb = dict(A, MD=lat['MD'], sxs=lat['sxy'][:, 0], sys=lat['sxy'][:, 1], **B)
    fl = M.flags(Tb)
    keep, rows = M.flag_table(fl)
    tB = time.time() - t0 - tA
    # pixel MLE on the kept rows
    P_ = {k: np.full(n, np.nan, np.float32) for k in ('xp', 'yp', 'zp', 'Fp', 'sxp', 'syp', 'szp', 'chi2p')}
    P_.update(convp=np.zeros(n, bool), boundp=np.zeros(n, bool))
    if a.pix:
        K = torch.tensor([pat['angle0']['k'], pat['angle1']['k']], device=dev, dtype=torch.float32)
        rk = np.nonzero(keep & (pix_mask if pix_mask is not None else True))[0]
        for j, img, df in src.chunks(a.chunk, rows=rk):
            pe = torch.tensor(M.psi_eff(pat, cx[j], cy[j]), device=dev, dtype=torch.float32)
            st = torch.tensor(np.c_[lat['xy'][j], B['z4'][j]], device=dev, dtype=torch.float32)
            p = c4.pixel_mle(img, df, st, torch.full((len(j),), pat['m'], device=dev), pe, K, torch.tensor(Phi0[j], device=dev, dtype=torch.float32),
                             refine=(8, 12))
            for k_, q in (('xp', 0), ('yp', 1), ('zp', 2), ('Fp', 3)): P_[k_][j] = p['th'][:, q].cpu().numpy()
            for k_, q in (('sxp', 0), ('syp', 1), ('szp', 2)): P_[k_][j] = p['crlb'][:, q].cpu().numpy()
            P_['chi2p'][j] = p['chi2px'].cpu().numpy(); P_['convp'][j] = p['conv'].cpu().numpy(); P_['boundp'][j] = p['at_bound'].cpu().numpy()
    keep_pix = keep & P_['convp'] & ~P_['boundp'] & (P_['chi2p'] <= 4)
    tP = time.time() - t0 - tA - tB
    tab = dict(w=w, frame=6 * w, cx=cx, cy=cy, x=cx + lat['xy'][:, 0], y=cy + lat['xy'][:, 1], z=B['z4'], sx=lat['sxy'][:, 0], sy=lat['sxy'][:, 1],
               sz=B['sz4'], N=A['Nsum'], bg=A['bgsum'], x0=X, y0=Y, z_ast=A['z_ast'], sx0=A['sx0'], sy0=A['sy0'], sz_ast=A['sz_ast'],
               xp=cx + P_['xp'], yp=cy + P_['yp'], zp=P_['zp'], sxp=P_['sxp'], syp=P_['syp'], szp=P_['szp'], Fp=P_['Fp'], chi2p=P_['chi2p'],
               keep=keep, keep_pix=keep_pix, Phi0=Phi0, phi=B['phi'], C=B['C'], R=A['R'], MD=lat['MD'], chi2px=A['chi2px'], llr=A['llr'],
               conv=A['conv'], at_bound=A['at_bound'], Nt=A['Nt'], nn=nn, file_index=np.full(n, fi),
               **({'row': src.meta['row']} if hasattr(src, 'meta') else {}),
               **({'box': src.meta['box']} if hasattr(src, 'meta') and 'box' in src.meta else {}))
    tab['flags'] = np.stack(list(fl.values()), 1)
    if pix_mask is not None: tab['pix_region'] = pix_mask
    if extra: tab.update(extra)
    fn = f'{OUT}/cl_{cell}_{fi:03d}{a.tag}.npz'
    np.savez(fn, **tab)
    g = keep
    summ = dict(cell=cell, file=fi, n=n, kept=float(keep.mean()), kept_pix=float(keep_pix.mean()), filters=rows, flag_names=list(fl.keys()),
                pattern=pat, field=hist, field_mode=a.field,
                med_sx_nm=float(np.median(tab['sx'][g]) * 128), med_sy_nm=float(np.median(tab['sy'][g]) * 128), med_sz_nm=float(np.median(tab['sz'][g]) * 1000),
                med_szp_nm=float(np.nanmedian(tab['szp'][keep_pix]) * 1000) if keep_pix.any() else None,
                med_N=float(np.median(tab['N'][g])), z_p2_p98=np.percentile(tab['z'][g], [2, 98]).tolist(), C_med=float(np.median(B['C'][fitok])),
                t_stageA_s=tA, t_stageB_s=tB, t_pix_s=tP)
    json.dump(summ, open(fn.replace('.npz', '_summary.json'), 'w'), indent=1, default=float)
    print(f"{cell} {fi}: {n} cand, kept {summ['kept']:.3f} (pix {summ['kept_pix']:.3f}); pattern m {pat['m']:.3f} steps "
          f"{np.round(pat['angle0']['steps_deg'], 1)} {np.round(pat['angle1']['steps_deg'], 1)}; field {hist}; "
          f"med sigma {summ['med_sx_nm']:.1f}/{summ['med_sy_nm']:.1f}/{summ['med_sz_nm']:.1f} nm; t A {tA:.0f} B {tB:.0f} pix {tP:.0f} s -> {fn}", flush=True)
    for k, al, sq in rows: print(f'    {k}: {al:.3f} | {sq:.3f}')
    return summ


def ready_files(cell, min_age=90):
    """patch files that are completely written: meta present, npy size = header + n x 6 x 4 x 15 x 15 x 2 bytes, older than
    min_age s"""
    import glob, re
    out = []
    for fm in sorted(glob.glob(f'{PDIR}/{cell}/meta_[0-9][0-9][0-9].npz')):
        fi = int(re.findall(r'meta_(\d+)', fm)[0]); fp = f'{PDIR}/{cell}/patches_{fi:03d}.npy'
        if not os.path.exists(fp) or time.time() - max(os.path.getmtime(fp), os.path.getmtime(fm)) < min_age: continue
        try:
            n = len(np.load(fm)['w'])
            with open(fp, 'rb') as fh:
                ver = np.lib.format.read_magic(fh); shp = np.lib.format._read_array_header(fh, ver)[0]; hdr = fh.tell()
            if shp[0] == n and os.path.getsize(fp) == hdr + int(np.prod(shp)) * 2: out.append(fi)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='NPC'); ap.add_argument('--files', type=int, nargs='*')
    ap.add_argument('--src', default='patches'); ap.add_argument('--nframes', type=int, default=3600)
    ap.add_argument('--psf', default='NPC_f_dxy15'); ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--field', default='poly+tile'); ap.add_argument('--order', type=int, default=3)
    ap.add_argument('--tile', type=float, default=12.0); ap.add_argument('--tblk', type=int, default=50)
    ap.add_argument('--pix', type=int, default=1); ap.add_argument('--tag', default='')
    ap.add_argument('--skip_done', type=int, default=1)
    ap.add_argument('--follow', type=int, default=0, help='poll out/4pi/patches and process files as they complete')
    ap.add_argument('--cells', nargs='*', default=['NPC', 'MT'])
    ap.add_argument('--order_npc', type=int, default=3); ap.add_argument('--order_mt', type=int, default=1); ap.add_argument('--hours', type=float, default=16)
    a = ap.parse_args()
    if a.follow: return follow(a)
    from simcode.util.proctitle import auto; auto('4pi-classic')
    dev = 'cuda'; os.makedirs(OUT, exist_ok=True)
    st = json.load(open(f'{P4}/psf4pi/{a.psf}.json'))['psf']
    c4 = Classic4Pi(LUT4Pi(st, P=P, dev=dev), delta=(0.0, DELTA[a.cell]), dev=dev)
    for fi in (a.files or []):
        if a.skip_done and os.path.exists(f'{OUT}/cl_{a.cell}_{fi:03d}{a.tag}.npz'):
            print(f'{a.cell} {fi}: done, skipped'); continue
        src = PatchSource(a.cell, fi, dev) if a.src == 'patches' else DcimgSource(a.cell, fi, a.nframes, dev)
        process(c4, src, a.cell, fi, a, dev)
        del src; torch.cuda.empty_cache()


def follow(a):
    from simcode.util.proctitle import auto; auto('4pi-classic')
    from dc import movie_files, file_index
    dev = 'cuda'; os.makedirs(OUT, exist_ok=True)
    st = json.load(open(f'{P4}/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=P, dev=dev)
    c4s = {c: Classic4Pi(lut, delta=(0.0, DELTA[c]), dev=dev) for c in a.cells}
    expect = {c: sorted(file_index(f) for f in movie_files(c)) for c in a.cells}
    t_end = time.time() + 3600 * a.hours
    while time.time() < t_end:
        todo = [(c, fi) for c in a.cells for fi in ready_files(c)
                if not os.path.exists(f'{OUT}/cl_{c}_{fi:03d}{a.tag}.npz') and not os.path.exists(f'{OUT}/cl_{c}_{fi:03d}{a.tag}.failed')]
        for c, fi in todo:
            try:
                a.order = a.order_mt if c == 'MT' else a.order_npc    # MT: only two 6 um boxes -> linear XY + tiles
                process(c4s[c], PatchSource(c, fi, dev), c, fi, a, dev)
            except Exception as e:
                print(f'{c} {fi}: FAILED {e!r}', flush=True)
                open(f'{OUT}/cl_{c}_{fi:03d}{a.tag}.failed', 'w').write(repr(e))
            torch.cuda.empty_cache()
        done = {c: [fi for fi in expect[c] if os.path.exists(f'{OUT}/cl_{c}_{fi:03d}{a.tag}.npz')] for c in a.cells}
        if all(len(done[c]) == len(expect[c]) for c in a.cells):
            print('all files done', flush=True); break
        if not todo:
            print(time.strftime('%H:%M'), 'waiting; done', {c: len(v) for c, v in done.items()}, 'of', {c: len(v) for c, v in expect.items()}, flush=True)
            time.sleep(120)


if __name__ == '__main__':
    main()
